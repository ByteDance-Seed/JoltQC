# Copyright 2025 ByteDance Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Generate kernels for incremental DFT
"""

from functools import lru_cache
from pathlib import Path

import cupy as cp
import numpy as np

from jqc.constants import COORD_STRIDE, NPRIM_MAX, PRIM_STRIDE

code_path = Path(__file__).resolve().parent

__all__ = ["gen_rho_kernel", "gen_vv10_kernel", "gen_vxc_kernel", "vv10nlc"]

compile_options = ("-std=c++17", "--use_fast_math", "--minimal")
nthreads = 256

with open(f"{code_path}/cuda/eval_rho.cu") as f:
    eval_rho_cuda_code = f.read()


@lru_cache(maxsize=None)
def gen_rho_kernel(ang, nprim, dtype, ndim=1, print_log=False):
    """
    ndim:
        1: LDA
        4: GGA
        5: mGGA
    """
    if dtype == np.float64:
        dtype_cuda = "double"
    elif dtype == np.float32:
        dtype_cuda = "float"
    else:
        raise RuntimeError("Data type is not supported")

    li, lj = ang
    npi, npj = nprim
    const = f"""
using DataType = {dtype_cuda};
constexpr int li = {li};
constexpr int lj = {lj};
constexpr int npi = {npi};
constexpr int npj = {npj};
constexpr int ndim = {ndim};
constexpr int deriv = {1 if ndim > 1 else 0};
constexpr int nthreads = {nthreads};
// Inject constants to match host-side layout
#define NPRIM_MAX {NPRIM_MAX}
// PRIM_STRIDE here matches host scalar stride; device uses prim_stride = PRIM_STRIDE/2
#define PRIM_STRIDE {PRIM_STRIDE}
#define COORD_STRIDE {COORD_STRIDE}
"""

    script = const + eval_rho_cuda_code
    mod = cp.RawModule(code=script, options=compile_options)
    kernel = mod.get_function("eval_rho")

    if print_log:
        print(
            f"Type: ({li}{lj}), \
primitives: ({npi}{npj}), \
registers: {kernel.num_regs:3d}, \
local memory: {kernel.local_size_bytes:4d} Bytes"
        )

    def fun(*args):
        ngrids = args[-1]
        blocks = ngrids // nthreads
        if ngrids % nthreads != 0:
            raise RuntimeError(f"The number of grids must be divisible by {nthreads}")
        try:
            kernel(
                (blocks,),
                (nthreads, 1),
                args,
            )
        except cp.cuda.runtime.CUDARuntimeError as e:
            print("CUDA Runtime Error in eval_rho kernel:", e)

    return script, mod, fun


with open(f"{code_path}/cuda/eval_vxc.cu") as f:
    eval_vxc_cuda_code = f.read()


@lru_cache(maxsize=None)
def gen_vxc_kernel(ang, nprim, dtype, ndim=1, print_log=False):
    """
    ndim:
        1: LDA
        4: GGA
        5: mGGA
    """
    if dtype == np.float64:
        dtype_cuda = "double"
    elif dtype == np.float32:
        dtype_cuda = "float"
    else:
        raise RuntimeError("Data type is not supported")

    li, lj = ang
    npi, npj = nprim
    const = f"""
using DataType = {dtype_cuda};
constexpr int li = {li};
constexpr int lj = {lj};
constexpr int npi = {npi};
constexpr int npj = {npj};
constexpr int ndim = {ndim};
constexpr int deriv = {1 if ndim > 1 else 0};
constexpr int nthreads = {nthreads};
// Inject constants to match host-side layout
#define NPRIM_MAX {NPRIM_MAX}
// PRIM_STRIDE here matches host scalar stride; device uses prim_stride = PRIM_STRIDE/2
#define PRIM_STRIDE {PRIM_STRIDE}
#define COORD_STRIDE {COORD_STRIDE}
"""

    script = const + eval_vxc_cuda_code
    mod = cp.RawModule(code=script, options=compile_options)
    kernel = mod.get_function("eval_vxc")

    if print_log:
        print(
            f"Type: ({li}{lj}), \
primitives: ({npi}{npj}), \
registers: {kernel.num_regs:3d}, \
local memory: {kernel.local_size_bytes:4d} Bytes"
        )

    def fun(*args):
        ngrids = args[-1]
        blocks = ngrids // nthreads
        if ngrids % nthreads != 0:
            raise RuntimeError(f"The number of grids must be divisible by {nthreads}")
        try:
            kernel(
                (blocks,),
                (nthreads, 1),
                args,
            )
        except cp.cuda.runtime.CUDARuntimeError as e:
            print("CUDA Runtime Error in eval_vxc kernel:", e)

    return script, mod, fun


with open(f"{code_path}/cuda/estimate_log_aovalue.cu") as f:
    estimate_aovalue_script = f.read()


def estimate_log_aovalue(
    grid_coords, coords, coeff_exp, ang, nprim, log_cutoff=-36.8, shell_base=0
):
    """Estimate the maximum log-value of atomic orbitals on grid blocks, 256 grids per block.

    This function performs a pre-screening step to identify which atomic
    orbital shells are significant on different blocks of the integration grid.
    For each block of grids, it computes the maximum value of each shell and
    compares it to a cutoff to build a list of significant shells.

    Parameters
    ----------
    grid_coords : cp.ndarray
        Grid coordinates, shape (3, ngrids).
    coords : cp.ndarray
        Shell coordinates, shape (nbas, 4).
    coeff_exp : cp.ndarray
        Coefficients and exponents of primitive Gaussians.
    ang : int
        Angular momentum for each shell.
    nprim : int
        Number of primitives for each shell.
    log_cutoff : float, optional
        Logarithm of the cutoff value for screening. Shells with a maximum
        log-value below this on a grid block are ignored. Default is -36.8.

    Returns
    -------
    logidx : cp.ndarray (structured)
        A (nblocks, nbas) structured array with fields:
        - 'log' (float32): max log-value per kept shell in a block
        - 'idx' (int32): shell index corresponding to the log value
        Only the first nnz_per_block[block] entries in each row are valid.
    nnz_per_block : cp.ndarray
        A (nblocks,) array with the count of significant shells for each
        grid block.
    """
    grid_coords = cp.asarray(grid_coords, dtype=np.float64)
    coords = cp.asarray(coords, dtype=np.float32)
    coeff_exp = cp.asarray(coeff_exp, dtype=np.float32)
    ngrids = grid_coords.shape[1]
    assert ngrids % nthreads == 0

    nbas = coords.shape[0]
    nblocks = ngrids // 256
    # Structured dtype matching `struct LogIdx { float log; int idx; }`
    logidx_dtype = np.dtype([("log", np.float32), ("idx", np.int32)], align=True)
    logidx = cp.empty((nblocks, nbas), dtype=logidx_dtype)

    nnz_per_block = cp.empty((nblocks), dtype=np.int32)
    const = f"""
constexpr int ang = {ang};
constexpr int nprim = {nprim};
// Inject constants to match host-side layout
#define NPRIM_MAX {NPRIM_MAX}
// PRIM_STRIDE here matches host scalar stride; device uses prim_stride = PRIM_STRIDE/2
#define PRIM_STRIDE {PRIM_STRIDE}
#define COORD_STRIDE {COORD_STRIDE}
"""
    script = const + estimate_aovalue_script
    mod = cp.RawModule(code=script, options=compile_options)
    kernel = mod.get_function("estimate_log_aovalue")
    kernel(
        (ngrids // nthreads,),
        (nthreads,),
        (
            grid_coords,
            ngrids,
            coords,
            coeff_exp,
            nbas,
            np.int32(shell_base),
            logidx,
            nnz_per_block,
            np.float32(log_cutoff),
        ),
    )
    return logidx, nnz_per_block


with open(f"{code_path}/cuda/vv10.cu") as f:
    vv10_script = f.read()


@lru_cache(maxsize=None)
def gen_vv10_kernel(dtype=np.float64, print_log=False):
    """
    Generate VV10 non-local correlation kernel

    Parameters
    ----------
    dtype : numpy.dtype
        Data type for intermediate calculations (float32 or float64)
        Note: All inputs/outputs are always double precision
    print_log : bool, optional
        Whether to print kernel information

    Returns
    -------
    script : str
        CUDA source code
    mod : cupy.RawModule
        Compiled CUDA module
    fun : callable
        Kernel function wrapper
    """
    if dtype == np.float64:
        dtype_cuda = "double"
    elif dtype == np.float32:
        dtype_cuda = "float"
    else:
        raise RuntimeError("Data type is not supported")

    const = f"""
using DataType = {dtype_cuda};
#define NG_PER_BLOCK      {nthreads}
"""

    script = const + vv10_script
    mod = cp.RawModule(code=script, options=compile_options)
    kernel = mod.get_function("vv10_kernel")

    if print_log:
        print(
            f"VV10 kernel - registers: {kernel.num_regs:3d}, "
            f"local memory: {kernel.local_size_bytes:4d} Bytes"
        )

    def fun(Fvec, Uvec, Wvec, vvcoords, coords, W0p, W0, K, Kp, RpW, vvngrids, ngrids):
        """
        VV10 kernel wrapper

        Parameters
        ----------
        Fvec, Uvec, Wvec : cupy.ndarray
            Output arrays for F, U, W values
        vvcoords : cupy.ndarray
            VV coordinates (3, vvngrids)
        coords : cupy.ndarray
            Grid coordinates (3, ngrids)
        W0p, W0, K, Kp, RpW : cupy.ndarray
            VV10 parameters
        vvngrids : int
            Number of VV grids
        ngrids : int
            Number of integration grids
        """
        # Assume 256-aligned grids (guaranteed by padding in vv10nlc)
        blocks = ngrids // nthreads
        assert (
            ngrids % nthreads == 0
        ), f"VV10 grids must be divisible by {nthreads} (got {ngrids})"
        try:
            kernel(
                (blocks,),
                (nthreads,),
                (
                    Fvec,
                    Uvec,
                    Wvec,
                    vvcoords,
                    coords,
                    W0p,
                    W0,
                    K,
                    Kp,
                    RpW,
                    vvngrids,
                    ngrids,
                ),
            )
        except cp.cuda.runtime.CUDARuntimeError as e:
            print("CUDA Runtime Error in vv10 kernel:", e)
            raise

    return script, mod, fun


# CuPy fused operations for VV10 non-local correlation
# These functions combine multiple elementwise operations into optimized CUDA kernels


def _ensure_256_alignment(data_arrays, alignment=256):
    """
    Ensure data arrays have 256-aligned length by padding with zeros

    Parameters
    ----------
    data_arrays : list of cupy.ndarray
        Arrays to pad (will be modified in place)
    alignment : int
        Required alignment (default: 256)

    Returns
    -------
    int
        Original length before padding
    """

    # Determine original length from first array
    first_arr = data_arrays[0]
    if first_arr.ndim == 1:
        original_length = len(first_arr)
    elif (
        first_arr.ndim == 2 and first_arr.shape[0] == 3
    ):  # coordinate arrays (3, ngrids)
        original_length = first_arr.shape[1]
    else:
        raise ValueError(f"Unsupported array shape: {first_arr.shape}")

    remainder = original_length % alignment

    if remainder == 0:
        return original_length

    # Calculate aligned size
    aligned_length = ((original_length + alignment - 1) // alignment) * alignment
    padding_needed = aligned_length - original_length

    # Pad all arrays with zeros
    for i, arr in enumerate(data_arrays):
        if arr.ndim == 1:
            padding = cp.zeros(padding_needed, dtype=arr.dtype)
            data_arrays[i] = cp.concatenate([arr, padding])
        elif arr.ndim == 2 and arr.shape[0] == 3:  # coordinate arrays (3, ngrids)
            padding = cp.zeros((3, padding_needed), dtype=arr.dtype)
            data_arrays[i] = cp.concatenate([arr, padding], axis=1)
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}")

    return original_length


@cp.fuse()
def _compute_gradient_magnitude(grad_x, grad_y, grad_z):
    """
    Compute the magnitude squared of density gradient: |∇ρ|²

    Parameters
    ----------
    grad_x, grad_y, grad_z : cupy.ndarray
        Density gradient components

    Returns
    -------
    cupy.ndarray
        Gradient magnitude squared: grad_x² + grad_y² + grad_z²
    """
    return grad_x**2 + grad_y**2 + grad_z**2


@cp.fuse()
def _compute_vv10_w0_parameters(c_vv, gradient_mag_sq, density, pi_factor):
    """
    Compute VV10 W0 parameters with fused operations

    Parameters
    ----------
    c_vv : float
        VV10 C parameter
    gradient_mag_sq : cupy.ndarray
        Gradient magnitude squared |∇ρ|²
    density : cupy.ndarray
        Electron density ρ
    pi_factor : float
        4π/3 constant

    Returns
    -------
    tuple[cupy.ndarray, cupy.ndarray]
        (W0_intermediate, W0) where:
        - W0_intermediate = C * (|∇ρ|/ρ²)²
        - W0 = √(W0_intermediate + (4π/3)ρ)
    """
    # Fuse: division, squaring, and scaling in one operation
    w0_intermediate = c_vv * (gradient_mag_sq / (density * density)) ** 2
    # Fuse: addition and square root
    w0_final = (w0_intermediate + pi_factor * density) ** 0.5
    return w0_intermediate, w0_final


@cp.fuse()
def _compute_vv10_k_parameters(k_vv, density):
    """
    Compute VV10 K parameters and derivatives

    Parameters
    ----------
    k_vv : float
        VV10 K coefficient
    density : cupy.ndarray
        Electron density ρ

    Returns
    -------
    tuple[cupy.ndarray, cupy.ndarray]
        (K, dK_dρ) where:
        - K = K_vv * ρ^(1/6)
        - dK_dρ = K/6 (derivative of K with respect to ρ)
    """
    k_parameter = k_vv * (density ** (1.0 / 6.0))
    dk_drho = k_parameter / 6.0
    return k_parameter, dk_drho


@cp.fuse()
def _compute_vv10_w0_derivatives(
    pi_factor, density, w0_intermediate, w0_final, gradient_mag_sq
):
    """
    Compute derivatives of W0 with respect to density and gradient

    Parameters
    ----------
    pi_factor : float
        4π/3 constant
    density : cupy.ndarray
        Electron density ρ
    w0_intermediate : cupy.ndarray
        W0 intermediate value
    w0_final : cupy.ndarray
        Final W0 value
    gradient_mag_sq : cupy.ndarray
        Gradient magnitude squared

    Returns
    -------
    tuple[cupy.ndarray, cupy.ndarray]
        (dW0_dρ, dW0_d|∇ρ|²) derivatives
    """
    # Derivative with respect to density
    dw0_drho = (0.5 * pi_factor * density - 2.0 * w0_intermediate) / w0_final
    # Derivative with respect to gradient magnitude squared
    dw0_dgrad = w0_intermediate * density / (gradient_mag_sq * w0_final)
    return dw0_drho, dw0_dgrad


@cp.fuse()
def _compute_vv10_final_results(
    beta_constant, kernel_f, kernel_u, kernel_w, dk_drho, dw0_drho, dw0_dgrad
):
    """
    Compute final VV10 exchange-correlation energy and potential

    Parameters
    ----------
    beta_constant : float
        VV10 β constant
    kernel_f, kernel_u, kernel_w : cupy.ndarray
        Kernel output values from CUDA computation
    dk_drho : cupy.ndarray
        K parameter derivative
    dw0_drho, dw0_dgrad : cupy.ndarray
        W0 parameter derivatives

    Returns
    -------
    tuple[cupy.ndarray, cupy.ndarray, cupy.ndarray]
        (exc, vxc_rho, vxc_grad) where:
        - exc: exchange-correlation energy density
        - vxc_rho: potential derivative w.r.t. density
        - vxc_grad: potential derivative w.r.t. gradient
    """
    # Exchange-correlation energy density
    exc_density = beta_constant + 0.5 * kernel_f

    # Potential derivative with respect to density
    vxc_density = (
        beta_constant + kernel_f + 1.5 * (kernel_u * dk_drho + kernel_w * dw0_drho)
    )

    # Potential derivative with respect to gradient
    vxc_gradient = 1.5 * kernel_w * dw0_dgrad

    return exc_density, vxc_density, vxc_gradient


def vv10nlc(rho, coords, vvrho, vvweight, vvcoords, nlc_pars, dtype=np.float64):
    """
    VV10 non-local correlation functional using optimized CUDA kernel implementation

    This function implements the VV10 (Vydrov-Van Voorhis 2010) non-local correlation
    functional with performance optimizations including:
    - Density thresholding to reduce computational cost
    - CuPy fused operations for improved memory bandwidth
    - Custom CUDA kernel for the double-loop VV10 computation

    Parameters
    ----------
    rho : cupy.ndarray, shape (4, ngrids)
        Density and its derivatives for GGA:
        - rho[0]: electron density ρ
        - rho[1:4]: gradient components ∇ρ = (∂ρ/∂x, ∂ρ/∂y, ∂ρ/∂z)
    coords : cupy.ndarray, shape (ngrids, 3)
        Integration grid coordinates
    vvrho : cupy.ndarray, shape (4, ngrids)
        Density for VV evaluation (typically same as rho)
    vvweight : cupy.ndarray, shape (ngrids,)
        Integration weights for VV grid points
    vvcoords : cupy.ndarray, shape (ngrids, 3)
        VV grid coordinates (typically same as coords)
    nlc_pars : tuple(float, float)
        VV10 parameters (b, C) where:
        - b: range separation parameter
        - C: correlation strength parameter
    dtype : numpy.dtype, optional
        Computation precision for CUDA kernel (default: float64)

    Returns
    -------
    exc : cupy.ndarray, shape (ngrids,)
        VV10 exchange-correlation energy density
    vxc : cupy.ndarray, shape (2, ngrids)
        VV10 exchange-correlation potential:
        - vxc[0]: potential derivative w.r.t. density
        - vxc[1]: potential derivative w.r.t. gradient magnitude

    Notes
    -----
    The implementation follows the GPU4PySCF reference with optimizations:
    1. Threshold-based screening (thresh=1e-10) for numerical stability
    2. Fused elementwise operations using CuPy JIT compilation
    3. Memory-efficient coordinate handling and late allocation strategy
    4. Strategic memory release of large arrays when no longer needed
    5. Output arrays allocated only at the end to minimize memory footprint
    6. Just-in-time computation of derivatives to reduce memory pressure
    """

    # ===== Initialization and Constants =====
    thresh = 1e-10
    original_grid_size = rho[0, :].size

    # ===== Grid Thresholding and Data Preparation =====
    # Outer grid: apply density threshold
    threshind_raw = rho[0, :] >= thresh
    coords_thresh = coords[threshind_raw]
    outer_density = rho[0, :][threshind_raw]
    # Compute gradient magnitude squared |∇ρ|² for outer grid
    outer_grad_mag_sq = _compute_gradient_magnitude(
        rho[1, :][threshind_raw], rho[2, :][threshind_raw], rho[3, :][threshind_raw]
    )

    # Inner grid: apply density threshold
    innerthreshind_raw = vvrho[0, :] >= thresh
    vvcoords_thresh = vvcoords[innerthreshind_raw]
    vvweight_thresh = vvweight[innerthreshind_raw]
    inner_density = vvrho[0, :][innerthreshind_raw]
    inner_density_weighted = inner_density * vvweight_thresh
    # Compute gradient magnitude squared |∇ρ'|² for inner grid
    inner_grad_mag_sq = _compute_gradient_magnitude(
        vvrho[1, :][innerthreshind_raw],
        vvrho[2, :][innerthreshind_raw],
        vvrho[3, :][innerthreshind_raw],
    )

    # ===== VV10 Parameter Calculation =====
    # Mathematical constants
    Pi = cp.pi
    Pi43 = 4.0 * Pi / 3.0

    # Extract VV10 functional parameters
    Bvv, Cvv = nlc_pars
    Kvv = Bvv * 1.5 * Pi * ((9.0 * Pi) ** (-1.0 / 6.0))
    Beta = ((3.0 / (Bvv * Bvv)) ** (0.75)) / 32.0

    # ===== VV10 Kernel Parameter Computation =====
    # Compute VV10 parameters for inner grid (integration points)
    _, W0p = _compute_vv10_w0_parameters(Cvv, inner_grad_mag_sq, inner_density, Pi43)
    Kp, _ = _compute_vv10_k_parameters(Kvv, inner_density)

    # Compute VV10 parameters for outer grid (evaluation points)
    W0tmp, W0 = _compute_vv10_w0_parameters(Cvv, outer_grad_mag_sq, outer_density, Pi43)
    K, dKdR = _compute_vv10_k_parameters(Kvv, outer_density)

    # ===== CUDA Kernel Preparation and Execution =====
    # Convert coordinates to Fortran order for CUDA kernel compatibility
    vvcoords_f = cp.asarray(vvcoords_thresh, order="F")
    coords_f = cp.asarray(coords_thresh, order="F")

    # Ensure coordinate format is (3, ngrids) for our kernel
    if vvcoords_f.shape[1] == 3:
        vvcoords_f = vvcoords_f.T
    if coords_f.shape[1] == 3:
        coords_f = coords_f.T

    # Pad arrays for 256-alignment
    arrays_to_pad = [coords_f, W0, K, dKdR]
    _ensure_256_alignment(arrays_to_pad, alignment=256)
    coords_f, W0, K, dKdR = arrays_to_pad

    vv_arrays_to_pad = [vvcoords_f, W0p, Kp, inner_density_weighted]
    _ensure_256_alignment(vv_arrays_to_pad, alignment=256)
    vvcoords_f, W0p, Kp, inner_density_weighted = vv_arrays_to_pad

    ngrids_thresh = coords_f.shape[1]
    vvngrids_thresh = vvcoords_f.shape[1]

    # Allocate output arrays for kernel results (with padding)
    kernel_F = cp.empty(ngrids_thresh, dtype=cp.float64)
    kernel_U = cp.empty(ngrids_thresh, dtype=cp.float64)
    kernel_W = cp.empty(ngrids_thresh, dtype=cp.float64)

    # Execute VV10 CUDA kernel for double-loop integration
    _, _, vv10_kernel_fun = gen_vv10_kernel(dtype=cp.float32)
    vv10_kernel_fun(
        kernel_F,
        kernel_U,
        kernel_W,
        vvcoords_f,
        coords_f,
        W0p,
        W0,
        K,
        Kp,
        inner_density_weighted,
        vvngrids_thresh,
        ngrids_thresh,
    )

    # Release large coordinate arrays that are no longer needed
    del vvcoords_f, coords_f

    # ===== Final Result Computation =====
    # Compute W0 derivatives and use non-padded portions
    ngrids_orig = len(outer_density)
    dW0dR, dW0dG = _compute_vv10_w0_derivatives(
        Pi43, outer_density, W0tmp[:ngrids_orig], W0[:ngrids_orig], outer_grad_mag_sq
    )

    # Compute final exchange-correlation energy and potential using fused operations
    exc_val, vxc0_val, vxc1_val = _compute_vv10_final_results(
        Beta,
        kernel_F[:ngrids_orig],
        kernel_U[:ngrids_orig],
        kernel_W[:ngrids_orig],
        dKdR[:ngrids_orig],
        dW0dR,
        dW0dG,
    )

    # ===== Memory Allocation and Result Assignment =====
    # Allocate output arrays only when needed to minimize memory footprint
    exc = cp.zeros(original_grid_size, dtype=cp.float64)
    vxc = cp.zeros([2, original_grid_size], dtype=cp.float64)

    # Assign computed results to full-size output arrays at original thresholded indices
    exc[threshind_raw] = exc_val
    vxc[0, threshind_raw] = vxc0_val
    vxc[1, threshind_raw] = vxc1_val

    return exc, vxc
