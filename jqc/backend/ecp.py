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

"""
JoltQC ECP (Effective Core Potential) kernel implementations
JIT-compiled CUDA kernels for ECP integral evaluation

Supports two kernel types:
- Type1: Optimized for low angular momentum combinations (li+lj <= 2)
- Type2: General implementation for higher angular momentum combinations

Following JoltQC constexpr injection pattern for compile-time optimization.
"""

import os
from typing import Any, Dict

import cupy as cp
import numpy as np
from pyscf import gto

from jqc.constants import COORD_STRIDE, PRIM_STRIDE, MAX_L_ECP

__all__ = [
    "ecp_generator",
    "get_ecp",
    "get_ecp_ip",
    "get_ecp_ipip",
    "make_ecp_tasks",
    "sort_ecp_basis",
]


def get_cp_type(precision: str):
    """Get CuPy data type for given precision"""
    if precision == 'fp32':
        return cp.float32
    elif precision == 'mixed':
        return cp.float32  # Use FP32 as base for mixed precision
    else:  # fp64
        return cp.float64


# Global cache for compiled kernels
_ecp_kernel_cache = {}

def _get_compile_options():
    # Compilation options following JoltQC style
    # Note: ECP derivative normalization is handled in basis coefficients; no extra scaling.
    return ("-std=c++17", "--use_fast_math", "--minimal")

def _compile_ecp_type2_kernel(li: int, lj: int, lc: int, precision: str = 'fp64'):
    """
    Compile ECP Type2 kernel for specific angular momentum combination

    Args:
        li: Angular momentum of first basis function
        lj: Angular momentum of second basis function
        lc: Angular momentum of ECP center
        precision: Floating point precision ('fp64', 'fp32', 'mixed')

    Returns:
        Compiled kernel function
    """

    dtype = get_cp_type(precision)
    dtype_cuda = "float" if dtype == cp.float32 else "double"

    cache_key = (li, lj, lc, 'type2', precision)

    if cache_key in _ecp_kernel_cache:
        return _ecp_kernel_cache[cache_key]

    # Create constexpr injection string following 1q1t pattern
    const_injection = f"""
using DataType = {dtype_cuda};
constexpr int LI = {li};
constexpr int LJ = {lj};
constexpr int LC = {lc};
"""

    # Read the CUDA source
    cuda_source = ""
    for filename in ['ecp.h', 'prelude.cuh', 'gauss_chebyshev.cu', 'bessel.cu', 'cart2sph.cu', 'common.cu', 'type2_ang_nuc.cu', 'ecp_type2.cu']:
        cuda_file = os.path.join(os.path.dirname(__file__), 'ecp', filename)
        with open(cuda_file) as f:
            cuda_source += f.read()
            cuda_source += "\n"  # Ensure separation between files

    # Replace DataType typedef and inject constexpr values
    cuda_source = cuda_source.replace(
        'using DataType = double;',
        ''  # Remove the original declaration
    )

    # Inject constexpr values at the beginning of the source
    cuda_source = const_injection + '\n' + cuda_source

    # Replace float-specific constants for fp32
    if precision == 'fp32':
        cuda_source = cuda_source.replace('M_PI', 'M_PI_F')

    # Compile module following JoltQC pattern
    # Inject layout constants via -D to avoid string duplication
    opts = (*_get_compile_options(),
            f"-DPRIM_STRIDE={PRIM_STRIDE}", f"-DCOORD_STRIDE={COORD_STRIDE}")
    mod = cp.RawModule(code=cuda_source, options=opts)
    kernel = mod.get_function('type2_cart')
    
    # Create wrapper function following JoltQC style
    def kernel_wrapper(*args):
        # Extract nprim values from additional arguments (last 2 arguments)
        ntasks = args[4]  # Number of tasks
        npi = args[-2]    # npi is second-to-last argument
        npj = args[-1]    # npj is last argument
        block_size = 128
        grid_size = ntasks
        # Pass original args (excluding npi, npj) plus npi, npj to kernel
        kernel_args = args[:-2] + (npi, npj)
        kernel((grid_size,), (block_size,), kernel_args)

    # Cache the compiled kernel
    _ecp_kernel_cache[cache_key] = kernel_wrapper

    return kernel_wrapper


def _compile_ecp_type1_kernel(li: int, lj: int, precision: str = 'fp64'):
    """
    Compile ECP Type1 kernel for specific angular momentum combination

    Args:
        li: Angular momentum of first basis function
        lj: Angular momentum of second basis function
        precision: Floating point precision ('fp64', 'fp32', 'mixed')

    Returns:
        Compiled kernel function
    """

    dtype = get_cp_type(precision)
    dtype_cuda = "float" if dtype == cp.float32 else "double"

    cache_key = (li, lj, precision)

    if cache_key in _ecp_kernel_cache:
        return _ecp_kernel_cache[cache_key]

    # Create constexpr injection string following 1q1t pattern
    const_injection = f"""
using DataType = {dtype_cuda};
constexpr int LI = {li};
constexpr int LJ = {lj};
"""

    # Read the CUDA source
    cuda_source = ""
    for filename in ['ecp.h', 'prelude.cuh', 'gauss_chebyshev.cu', 'bessel.cu', 'cart2sph.cu', 'common.cu', 'type1_ang_nuc.cu', 'ecp_type1.cu']:
        cuda_file = os.path.join(os.path.dirname(__file__), 'ecp', filename)
        with open(cuda_file) as f:
            cuda_source += f.read()
            cuda_source += "\n"  # Ensure separation between files

    # Replace DataType typedef and inject constexpr values
    cuda_source = cuda_source.replace(
        'using DataType = double;',
        ''  # Remove the original declaration
    )

    # Inject constexpr values at the beginning of the source
    cuda_source = const_injection + '\n' + cuda_source

    # Replace float-specific constants for fp32
    if precision == 'fp32':
        cuda_source = cuda_source.replace('M_PI', 'M_PI_F')

    # Compile module following JoltQC pattern
    opts = (*_get_compile_options(),
            f"-DPRIM_STRIDE={PRIM_STRIDE}", f"-DCOORD_STRIDE={COORD_STRIDE}")
    mod = cp.RawModule(code=cuda_source, options=opts)
    kernel = mod.get_function('type1_cart')
    # print(li, lj, npi, npj, kernel.num_regs, kernel.local_size_bytes, kernel.shared_size_bytes/1024)
    if (kernel.local_size_bytes > 2048):
        Warning.warn(f"High local memory usage detected in type1_cart kernel: {kernel.local_size_bytes} bytes")
    
    # Create wrapper function following JoltQC style
    def kernel_wrapper(*args):
        # Extract nprim values from additional arguments (last 2 arguments)
        ntasks = args[4]  # Number of tasks
        npi = args[-2]    # npi is second-to-last argument
        npj = args[-1]    # npj is last argument
        block_size = 128
        grid_size = ntasks
        # Pass original args (excluding npi, npj) plus npi, npj to kernel
        kernel_args = args[:-2] + (npi, npj)
        kernel((grid_size,), (block_size,), kernel_args)

    # Cache the compiled kernel
    _ecp_kernel_cache[cache_key] = kernel_wrapper

    return kernel_wrapper


def _compile_ecp_type1_ip_kernel(li: int, lj: int, precision: str = 'fp64'):
    """
    Compile ECP Type1 IP (first derivative) kernel for specific angular momentum combination

    Args:
        li: Angular momentum of first basis function
        lj: Angular momentum of second basis function
        precision: Floating point precision ('fp64', 'fp32', 'mixed')

    Returns:
        Compiled IP kernel function
    """

    dtype = get_cp_type(precision)
    dtype_cuda = "float" if dtype == cp.float32 else "double"

    cache_key = (li, lj, 'type1_ip', precision)

    if cache_key in _ecp_kernel_cache:
        return _ecp_kernel_cache[cache_key]

    # Create constexpr injection
    const_injection = f"""
using DataType = {dtype_cuda};
constexpr int LI = {li};
constexpr int LJ = {lj};
"""

    # Read the CUDA source
    cuda_source = ""
    for filename in ['ecp.h', 'prelude.cuh', 'gauss_chebyshev.cu', 'bessel.cu', 'cart2sph.cu', 'common.cu',
                     'type1_ang_nuc.cu', 'ecp_type1.cu', 'ecp_type1_ip.cu']:
        cuda_file = os.path.join(os.path.dirname(__file__), 'ecp', filename)
        with open(cuda_file) as f:
            cuda_source += f.read()
            cuda_source += "\n"

    # Replace DataType typedef and inject constexpr values
    cuda_source = cuda_source.replace('using DataType = double;', '')
    cuda_source = const_injection + '\n' + cuda_source

    if precision == 'fp32':
        cuda_source = cuda_source.replace('M_PI', 'M_PI_F')

    # Compile module
    opts = (*_get_compile_options(),
            f"-DPRIM_STRIDE={PRIM_STRIDE}", f"-DCOORD_STRIDE={COORD_STRIDE}")
    mod = cp.RawModule(code=cuda_source, options=opts)
    kernel = mod.get_function('type1_cart_ip1')
    if (kernel.local_size_bytes > 2048):
        Warning.warn(f"High local memory usage detected in type1_cart_ip1 kernel: {kernel.local_size_bytes} bytes")
    def kernel_wrapper(*args):
        # Extract nprim values from additional arguments (last 2 arguments)
        ntasks = args[4]  # Number of tasks
        npi = args[-2]    # npi is second-to-last argument
        npj = args[-1]    # npj is last argument
        block_size = 128
        grid_size = ntasks
        # Pass original args (excluding npi, npj) plus npi, npj to kernel
        kernel_args = args[:-2] + (npi, npj)
        kernel((grid_size,), (block_size,), kernel_args)

    _ecp_kernel_cache[cache_key] = kernel_wrapper
    return kernel_wrapper


def _compile_ecp_type2_ip_kernel(li: int, lj: int, lc: int, precision: str = 'fp64'):
    """
    Compile ECP Type2 IP (first derivative) kernel for specific angular momentum combination

    Args:
        li: Angular momentum of first basis function
        lj: Angular momentum of second basis function
        lc: Angular momentum of ECP center
        precision: Floating point precision ('fp64', 'fp32', 'mixed')

    Returns:
        Compiled IP kernel function
    """

    dtype = get_cp_type(precision)
    dtype_cuda = "float" if dtype == cp.float32 else "double"

    cache_key = (li, lj, lc, 'type2_ip', precision)

    if cache_key in _ecp_kernel_cache:
        return _ecp_kernel_cache[cache_key]

    # Create constexpr injection
    const_injection = f"""
using DataType = {dtype_cuda};
constexpr int LI = {li};
constexpr int LJ = {lj};
constexpr int LC = {lc};
"""

    # Read the CUDA source
    cuda_source = ""
    for filename in ['ecp.h', 'prelude.cuh', 'gauss_chebyshev.cu', 'bessel.cu', 'cart2sph.cu', 'common.cu',
                     'type2_ang_nuc.cu', 'ecp_type2.cu', 'ecp_type2_ip.cu']:
        cuda_file = os.path.join(os.path.dirname(__file__), 'ecp', filename)
        with open(cuda_file) as f:
            cuda_source += f.read()
            cuda_source += "\n"

    # Replace DataType typedef and inject constexpr values
    cuda_source = cuda_source.replace('using DataType = double;', '')
    cuda_source = const_injection + '\n' + cuda_source

    if precision == 'fp32':
        cuda_source = cuda_source.replace('M_PI', 'M_PI_F')

    # Compile module
    opts = (*_get_compile_options(),
            f"-DPRIM_STRIDE={PRIM_STRIDE}", f"-DCOORD_STRIDE={COORD_STRIDE}")
    mod = cp.RawModule(code=cuda_source, options=opts)
    kernel = mod.get_function('type2_cart_ip1')
    if (kernel.local_size_bytes > 2048):
        Warning.warn(f"High local memory usage detected in type2_cart_ip1 kernel: {kernel.local_size_bytes} bytes")
    def kernel_wrapper(*args):
        # Extract nprim values from additional arguments (last 2 arguments)
        ntasks = args[4]  # Number of tasks
        npi = args[-2]    # npi is second-to-last argument
        npj = args[-1]    # npj is last argument
        block_size = 128
        grid_size = ntasks
        # Pass original args (excluding npi, npj) plus npi, npj to kernel
        kernel_args = args[:-2] + (npi, npj)
        kernel((grid_size,), (block_size,), kernel_args)

    _ecp_kernel_cache[cache_key] = kernel_wrapper
    return kernel_wrapper


def _compile_ecp_type1_ipip_kernel(li: int, lj: int, variant: str, precision: str = 'fp64'):
    """
    Compile ECP Type1 IPIP (second derivative) kernel for specific angular momentum combination

    Args:
        li: Angular momentum of first basis function
        lj: Angular momentum of second basis function
        variant: IPIP variant ('ipipv' or 'ipvip')
        npi: Number of primitives for shell i
        npj: Number of primitives for shell j
        precision: Floating point precision ('fp64', 'fp32', 'mixed')

    Returns:
        Compiled IPIP kernel function
    """

    dtype = get_cp_type(precision)
    dtype_cuda = "float" if dtype == cp.float32 else "double"

    cache_key = (li, lj, variant, 'type1_ipip', precision)

    if cache_key in _ecp_kernel_cache:
        return _ecp_kernel_cache[cache_key]

    # Create constexpr injection
    const_injection = f"""
using DataType = {dtype_cuda};
constexpr int LI = {li};
constexpr int LJ = {lj};
"""

    # Read the CUDA source
    cuda_source = ""
    for filename in ['ecp.h', 'prelude.cuh', 'gauss_chebyshev.cu', 'bessel.cu', 'cart2sph.cu', 'common.cu',
                     'type1_ang_nuc.cu', 'ecp_type1.cu', 'ecp_type1_ip.cu']:
        cuda_file = os.path.join(os.path.dirname(__file__), 'ecp', filename)
        with open(cuda_file) as f:
            cuda_source += f.read()
            cuda_source += "\n"

    # Replace DataType typedef and inject constexpr values
    cuda_source = cuda_source.replace('using DataType = double;', '')
    cuda_source = const_injection + '\n' + cuda_source

    if precision == 'fp32':
        cuda_source = cuda_source.replace('M_PI', 'M_PI_F')

    # Compile module
    opts = (*_get_compile_options(),
            f"-DPRIM_STRIDE={PRIM_STRIDE}", f"-DCOORD_STRIDE={COORD_STRIDE}")
    mod = cp.RawModule(code=cuda_source, options=opts)
    kernel_name = f'type1_cart_{variant}'
    kernel = mod.get_function(kernel_name)
    if (kernel.local_size_bytes > 2048):
        Warning.warn(f"High local memory usage detected in {kernel_name} kernel: {kernel.local_size_bytes} bytes")
    def kernel_wrapper(*args):
        # Extract nprim values from additional arguments (last 2 arguments)
        ntasks = args[4]  # Number of tasks
        npi = args[-2]    # npi is second-to-last argument
        npj = args[-1]    # npj is last argument
        block_size = 128
        grid_size = ntasks
        # Pass original args (excluding npi, npj) plus npi, npj to kernel
        kernel_args = args[:-2] + (npi, npj)
        kernel((grid_size,), (block_size,), kernel_args)

    _ecp_kernel_cache[cache_key] = kernel_wrapper
    return kernel_wrapper


def _compile_ecp_type2_ipip_kernel(li: int, lj: int, lc: int, variant: str, precision: str = 'fp64'):
    """
    Compile ECP Type2 IPIP (second derivative) kernel for specific angular momentum combination

    Args:
        li: Angular momentum of first basis function
        lj: Angular momentum of second basis function
        lc: Angular momentum of ECP center
        variant: IPIP variant ('ipipv' or 'ipvip')
        npi: Number of primitives for shell i
        npj: Number of primitives for shell j
        precision: Floating point precision ('fp64', 'fp32', 'mixed')

    Returns:
        Compiled IPIP kernel function
    """

    dtype = get_cp_type(precision)
    dtype_cuda = "float" if dtype == cp.float32 else "double"

    cache_key = (li, lj, lc, variant, 'type2_ipip', precision)

    if cache_key in _ecp_kernel_cache:
        return _ecp_kernel_cache[cache_key]

    # Create constexpr injection
    const_injection = f"""
using DataType = {dtype_cuda};
constexpr int LI = {li};
constexpr int LJ = {lj};
constexpr int LC = {lc};
"""

    # Read the CUDA source
    cuda_source = ""
    for filename in ['ecp.h', 'prelude.cuh', 'gauss_chebyshev.cu', 'bessel.cu', 'cart2sph.cu', 'common.cu',
                     'type2_ang_nuc.cu', 'ecp_type2.cu', 'ecp_type2_ip.cu']:
        cuda_file = os.path.join(os.path.dirname(__file__), 'ecp', filename)
        with open(cuda_file) as f:
            cuda_source += f.read()
            cuda_source += "\n"

    # Replace DataType typedef and inject constexpr values
    cuda_source = cuda_source.replace('using DataType = double;', '')
    cuda_source = const_injection + '\n' + cuda_source

    if precision == 'fp32':
        cuda_source = cuda_source.replace('M_PI', 'M_PI_F')

    # Compile module
    opts = (*_get_compile_options(),
            f"-DPRIM_STRIDE={PRIM_STRIDE}", f"-DCOORD_STRIDE={COORD_STRIDE}")
    mod = cp.RawModule(code=cuda_source, options=opts)
    kernel_name = f'type2_cart_{variant}'
    kernel = mod.get_function(kernel_name)

    if (kernel.local_size_bytes > 2048):
        Warning.warn(f"High local memory usage detected in {kernel_name} kernel: {kernel.local_size_bytes} bytes")
    def kernel_wrapper(*args):
        # Extract nprim values from additional arguments (last 2 arguments)
        ntasks = args[4]  # Number of tasks
        npi = args[-2]    # npi is second-to-last argument
        npj = args[-1]    # npj is last argument
        block_size = 128
        grid_size = ntasks
        # Pass original args (excluding npi, npj) plus npi, npj to kernel
        kernel_args = args[:-2] + (npi, npj)
        kernel((grid_size,), (block_size,), kernel_args)

    _ecp_kernel_cache[cache_key] = kernel_wrapper
    return kernel_wrapper


def get_ecp_ip(mol_or_basis_layout, ip_type='ip', ecp_atoms=None, precision: str = 'fp64') -> cp.ndarray:
    """
    Calculate ECP first derivatives (IP integrals)

    Args:
        mol_or_basis_layout: Either mol (backward compatibility) or BasisLayout object
        ip_type: Type of IP integrals ('ip' supported)
        ecp_atoms: List of ECP atom indices (None for all ECP atoms)
        precision: Floating point precision

    Returns:
        ECP IP integral array [n_ecp_atoms, 3, nao_orig, nao_orig] where first dimension
        contains derivatives w.r.t. nuclear coordinates for each ECP atom
    """
    # Validate ip_type parameter
    if ip_type != 'ip':
        raise ValueError(f"Invalid ip_type: {ip_type}. Only 'ip' is supported.")

    # Handle both mol and basis_layout inputs for backward compatibility
    if hasattr(mol_or_basis_layout, '_mol'):
        basis_layout = mol_or_basis_layout
    else:
        from jqc.pyscf.basis import BasisLayout
        basis_layout = BasisLayout.from_mol(mol_or_basis_layout)

    dtype = get_cp_type(precision)
    mol = basis_layout._mol
    splitted_mol = basis_layout.splitted_mol

    # Robust fallback for spherical basis to avoid heavy sph2cart transforms
    if not mol.cart:
        h_ip = mol.intor('ECPscalar_ipnuc_sph')  # (3, nao, nao)
        nao = h_ip.shape[-1]
        # Determine ECP atoms
        if hasattr(mol, '_ecpbas') and len(mol._ecpbas) > 0:
            all_ecp_atoms = sorted(set(mol._ecpbas[:, gto.ATOM_OF]))
            n_ecp_atoms = len(all_ecp_atoms) if ecp_atoms is None else len(ecp_atoms)
        else:
            n_ecp_atoms = 0
        out = cp.zeros((max(n_ecp_atoms, 1), 3, nao, nao), dtype=dtype)
        out[0] = cp.asarray(h_ip, dtype=dtype)
        return out

    if not hasattr(mol, '_ecpbas') or len(mol._ecpbas) == 0:
        nao_orig = mol.nao
        if ecp_atoms is None:
            return cp.zeros((0, 3, nao_orig, nao_orig), dtype=dtype)
        else:
            return cp.zeros((len(ecp_atoms), 3, nao_orig, nao_orig), dtype=dtype)

    # Determine ECP atoms
    all_ecp_atoms = sorted(set(mol._ecpbas[:, gto.ATOM_OF]))
    if ecp_atoms is None:
        ecp_atoms = all_ecp_atoms
    else:
        # Validate requested ecp_atoms exist
        ecp_atoms = [a for a in ecp_atoms if a in all_ecp_atoms]

    # Store original nao to exclude padding from task generation
    nao_orig = mol.nao

    # Use basis layout group information like gpu4pyscf
    group_key, group_offset = basis_layout.group_info
    n_groups = basis_layout.ngroups

    # Sort ECP basis using the same mechanism as gpu4pyscf
    sorted_ecpbas, uniq_lecp, lecp_counts, ecp_loc = sort_ecp_basis(mol._ecpbas)

    # Create ECP group offsets
    lecp_offsets = np.append(0, np.cumsum(lecp_counts))
    n_ecp_groups = len(uniq_lecp)

    # Use splitted_mol for computation (like gpu4pyscf)
    atm = cp.asarray(splitted_mol._atm, dtype=cp.int32)
    env = cp.asarray(splitted_mol._env, dtype=dtype)
    ao_loc = cp.asarray(basis_layout.ao_loc, dtype=cp.int32)
    nao = int(ao_loc[-1])

    ecpbas = cp.asarray(sorted_ecpbas, dtype=cp.int32)
    ecploc = cp.asarray(ecp_loc, dtype=cp.int32)

    # Initialize result matrix in splitted_mol basis
    # IP integrals have 3 components for each ECP atom (x,y,z derivatives)
    # For now, keep backward compatibility with flattened format for kernel interface
    n_ecp_atoms = len(ecp_atoms)
    mat1_flat = cp.zeros((3*n_ecp_atoms, nao, nao), dtype=dtype)

    # Kernel types are selected automatically by lk; no env toggles

    # Compute ECP IP integrals for each basis group combination and ECP type
    # Use full (i,j) combinations to capture both i- and j-side contributions
    for i in range(n_groups):
        for j in range(n_groups):
            for k in range(n_ecp_groups):
                li = group_key[i, 0]
                lj = group_key[j, 0]
                lk = uniq_lecp[k]
                npi = group_key[i, 1]
                npj = group_key[j, 1]

                # Generate tasks for this group combination
                ish0, ish1 = group_offset[i], group_offset[i + 1]
                jsh0, jsh1 = group_offset[j], group_offset[j + 1]
                ksh0, ksh1 = lecp_offsets[k], lecp_offsets[k + 1]

                tasks = []
                for ish in range(ish0, ish1):
                    if basis_layout.pad_id[ish]:  # Skip padded shells
                        continue
                    for jsh in range(jsh0, jsh1):
                        if basis_layout.pad_id[jsh]:  # Skip padded shells
                            continue
                        for ksh in range(ksh0, ksh1):
                            tasks.append([ish, jsh, ksh])

                if not tasks:
                    continue

                tasks = cp.asarray(tasks, dtype=cp.int32, order='F')
                ntasks = len(tasks)

                # Choose appropriate IP kernel based on ECP type
                if lk < 0:
                    _compile_ecp_type1_ip_kernel(li, lj, precision)(
                        mat1_flat, ao_loc, nao, tasks, ntasks, ecpbas, ecploc,
                        basis_layout.coords, basis_layout.ce, atm, env, npi, npj
                    )
                else:
                    # Type2 IP kernel for semi-local channels
                    _compile_ecp_type2_ip_kernel(li, lj, lk, precision)(
                        mat1_flat, ao_loc, nao, tasks, ntasks, ecpbas, ecploc,
                        basis_layout.coords, basis_layout.ce, atm, env, npi, npj
                    )
    result = cp.zeros((n_ecp_atoms, 3, nao_orig, nao_orig), dtype=dtype)

    # Map ECP atoms and transform each component from flattened format
    for i in range(n_ecp_atoms):
        for comp in range(3):
            flat_idx = 3*i + comp
            result[i, comp] = basis_layout.dm_to_mol(mat1_flat[flat_idx:flat_idx+1].reshape(1, nao, nao))[0]

    return result


def get_ecp_ipip(mol_or_basis_layout, ip_type='ipipv', ecp_atoms=None, precision: str = 'fp64') -> cp.ndarray:
    """
    Calculate ECP second derivatives (IPIP integrals)

    Args:
        mol_or_basis_layout: Either mol (backward compatibility) or BasisLayout object
        ip_type: IPIP variant ('ipipv' or 'ipvip')
        ecp_atoms: List of ECP atom indices (None for all ECP atoms)
        precision: Floating point precision

    Returns:
        ECP IPIP integral array [n_ecp_atoms, 9, nao_orig, nao_orig] where first dimension
        contains second derivatives w.r.t. nuclear coordinates for each ECP atom
    """
    # Validate ip_type parameter
    if ip_type not in ['ipipv', 'ipvip']:
        raise ValueError(f"Invalid ip_type: {ip_type}. Supported types: 'ipipv', 'ipvip'")

    # Handle both mol and basis_layout inputs for backward compatibility
    if hasattr(mol_or_basis_layout, '_mol'):
        basis_layout = mol_or_basis_layout
    else:
        from jqc.pyscf.basis import BasisLayout
        basis_layout = BasisLayout.from_mol(mol_or_basis_layout)

    dtype = get_cp_type(precision)
    mol = basis_layout._mol
    splitted_mol = basis_layout.splitted_mol

    if not hasattr(mol, '_ecpbas') or len(mol._ecpbas) == 0:
        nao_orig = mol.nao
        if ecp_atoms is None:
            return cp.zeros((0, 9, nao_orig, nao_orig), dtype=dtype)
        else:
            return cp.zeros((len(ecp_atoms), 9, nao_orig, nao_orig), dtype=dtype)

    # Determine ECP atoms
    all_ecp_atoms = sorted(set(mol._ecpbas[:, gto.ATOM_OF]))
    if ecp_atoms is None:
        ecp_atoms = all_ecp_atoms
    else:
        # Validate requested ecp_atoms exist
        ecp_atoms = [a for a in ecp_atoms if a in all_ecp_atoms]

    # Store original nao to exclude padding from task generation
    nao_orig = mol.nao

    # Use basis layout group information like gpu4pyscf
    group_key, group_offset = basis_layout.group_info
    n_groups = basis_layout.ngroups

    # Sort ECP basis using the same mechanism as gpu4pyscf
    sorted_ecpbas, uniq_lecp, lecp_counts, ecp_loc = sort_ecp_basis(mol._ecpbas)

    # Create ECP group offsets
    lecp_offsets = np.append(0, np.cumsum(lecp_counts))
    n_ecp_groups = len(uniq_lecp)

    # Use splitted_mol for computation (like gpu4pyscf)
    atm = cp.asarray(splitted_mol._atm, dtype=cp.int32)
    env = cp.asarray(splitted_mol._env, dtype=dtype)
    ao_loc = cp.asarray(basis_layout.ao_loc, dtype=cp.int32)
    nao = int(ao_loc[-1])

    ecpbas = cp.asarray(sorted_ecpbas, dtype=cp.int32)
    ecploc = cp.asarray(ecp_loc, dtype=cp.int32)

    # Initialize result matrix in splitted_mol basis
    # IPIP integrals have 9 components for each ECP atom (xx,xy,xz,yx,yy,yz,zx,zy,zz derivatives)
    # For now, keep backward compatibility with flattened format for kernel interface
    n_ecp_atoms = len(ecp_atoms)
    mat1_flat = cp.zeros((9*n_ecp_atoms, nao, nao), dtype=dtype)

    # Compute ECP IPIP integrals for each basis group combination and ECP type
    # Use full (i,j) combinations to capture both i- and j-side contributions
    for i in range(n_groups):
        for j in range(n_groups):
            for k in range(n_ecp_groups):
                li = group_key[i, 0]
                lj = group_key[j, 0]
                lk = uniq_lecp[k]
                npi = group_key[i, 1]
                npj = group_key[j, 1]

                # Generate tasks for this group combination
                ish0, ish1 = group_offset[i], group_offset[i + 1]
                jsh0, jsh1 = group_offset[j], group_offset[j + 1]
                ksh0, ksh1 = lecp_offsets[k], lecp_offsets[k + 1]

                tasks = []
                for ish in range(ish0, ish1):
                    if basis_layout.pad_id[ish]:  # Skip padded shells
                        continue
                    for jsh in range(jsh0, jsh1):
                        if basis_layout.pad_id[jsh]:  # Skip padded shells
                            continue
                        for ksh in range(ksh0, ksh1):
                            tasks.append([ish, jsh, ksh])

                if not tasks:
                    continue

                tasks = cp.asarray(tasks, dtype=cp.int32, order='F')
                ntasks = len(tasks)

                # Choose appropriate IPIP kernel based on ECP type
                if lk < 0:
                    _compile_ecp_type1_ipip_kernel(li, lj, ip_type, precision)(
                        mat1_flat, ao_loc, nao, tasks, ntasks, ecpbas, ecploc,
                        basis_layout.coords, basis_layout.ce, atm, env, npi, npj
                    )
                else:
                    # Type2 IPIP kernel for semi-local channels
                    _compile_ecp_type2_ipip_kernel(li, lj, lk, ip_type, precision)(
                        mat1_flat, ao_loc, nao, tasks, ntasks, ecpbas, ecploc,
                        basis_layout.coords, basis_layout.ce, atm, env, npi, npj
                    )

    result = cp.zeros((n_ecp_atoms, 9, nao_orig, nao_orig), dtype=dtype)

    # Map ECP atoms and transform each component from flattened format
    for i in range(n_ecp_atoms):
        for comp in range(9):
            flat_idx = 9*i + comp
            result[i, comp] = basis_layout.dm_to_mol(mat1_flat[flat_idx:flat_idx+1].reshape(1, nao, nao))[0]

    return result


def sort_ecp_basis(ecpbas):
    """
    Sort ECP basis based on angular momentum and atom ID
    Similar to gpu4pyscf's sort_ecp_basis function

    Args:
        ecpbas: ECP basis array

    Returns:
        Tuple of (sorted_ecpbas, uniq_l, l_counts, ecp_loc)
    """
    from pyscf import gto

    # Remove SO Type basis functions (keep only standard ECP basis)
    not_so_type = ecpbas[:, gto.SO_TYPE_OF] == 0
    ecpbas = ecpbas[not_so_type]

    # Sort ECP basis based on angular momentum and atom_id
    l_atm = ecpbas[:, [gto.ANG_OF, gto.ATOM_OF]]

    uniq_l_atm, inv_idx, l_atm_counts = np.unique(
        l_atm, return_inverse=True, return_counts=True, axis=0)
    sorted_idx = np.argsort(inv_idx.ravel(), kind='stable').astype(np.int32)

    # Sort basis inplace
    ecpbas = ecpbas[sorted_idx]

    # Assign contiguous ECP atom IDs into the last column (slot 7)
    # so kernels can accumulate per-ECP-atom derivative blocks
    atom_ids = ecpbas[:, gto.ATOM_OF]
    uniq_atoms = np.unique(atom_ids)
    atom_id_map = {atm: i for i, atm in enumerate(uniq_atoms.tolist())}
    if ecpbas.shape[1] >= 8:
        ecpbas[:, 7] = np.vectorize(atom_id_map.get)(atom_ids)

    # Group ECP basis based on angular momentum and atom id
    # Each group contains basis with multiple power order
    ecp_loc = np.append(0, np.cumsum(l_atm_counts))

    # Further group based on angular momentum for counting
    uniq_l, l_counts = np.unique(uniq_l_atm[:, 0], return_counts=True, axis=0)

    return ecpbas, uniq_l, l_counts, ecp_loc


def make_ecp_tasks(l_ctr_offsets, lecp_ctr_offsets):
    """
    Generate ECP integral tasks for given angular momentum groups
    Similar to gpu4pyscf's make_tasks function
    """
    tasks = {}
    n_groups = len(l_ctr_offsets) - 1
    n_ecp_groups = len(lecp_ctr_offsets) - 1

    # TODO: Add screening here
    for i in range(n_groups):
        for j in range(i, n_groups):
            for k in range(n_ecp_groups):
                ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
                jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
                ksh0, ksh1 = lecp_ctr_offsets[k], lecp_ctr_offsets[k+1]
                grid = np.meshgrid(
                    np.arange(ish0, ish1),
                    np.arange(jsh0, jsh1),
                    np.arange(ksh0, ksh1))
                grid = np.stack(grid, axis=-1).reshape(-1, 3)
                idx = grid[:, 0] <= grid[:, 1]
                tasks[i, j, k] = grid[idx]
    return tasks



def get_ecp(mol_or_basis_layout, precision: str = 'fp64') -> cp.ndarray:
    """
    Calculate total ECP integrals summed over all angular momentum combinations
    Following gpu4pyscf's grouped computation pattern

    Args:
        mol_or_basis_layout: Either mol (backward compatibility) or BasisLayout object
        precision: Floating point precision

    Returns:
        Total ECP integral matrix [nao_orig, nao_orig] in original molecule basis
    """
    # Handle both mol and basis_layout inputs for backward compatibility
    if hasattr(mol_or_basis_layout, '_mol'):
        # This is a BasisLayout object
        basis_layout = mol_or_basis_layout
    else:
        # This is a mol object - create BasisLayout
        from jqc.pyscf.basis import BasisLayout
        basis_layout = BasisLayout.from_mol(mol_or_basis_layout)

    dtype = get_cp_type(precision)
    mol = basis_layout._mol
    splitted_mol = basis_layout.splitted_mol

    if not hasattr(mol, '_ecpbas') or len(mol._ecpbas) == 0:
        nao_orig = mol.nao
        return cp.zeros((nao_orig, nao_orig), dtype=dtype)

    # Store original nao to exclude padding from task generation
    nao_orig = mol.nao

    # Use basis layout group information like gpu4pyscf
    group_key, group_offset = basis_layout.group_info
    n_groups = basis_layout.ngroups

    # Sort ECP basis using the same mechanism as gpu4pyscf
    sorted_ecpbas, uniq_lecp, lecp_counts, ecp_loc = sort_ecp_basis(mol._ecpbas)

    # Create ECP group offsets
    lecp_offsets = np.append(0, np.cumsum(lecp_counts))
    n_ecp_groups = len(uniq_lecp)

    # Use splitted_mol for computation (like gpu4pyscf)
    atm = cp.asarray(splitted_mol._atm, dtype=cp.int32)
    env = cp.asarray(splitted_mol._env, dtype=dtype)
    # Use AO offsets with padding entries (dims for padded shells are zero)
    # This matches how tasks index shells in the grouped layout
    ao_loc = cp.asarray(basis_layout.ao_loc, dtype=cp.int32)
    nao = int(ao_loc[-1])

    ecpbas = cp.asarray(sorted_ecpbas, dtype=cp.int32)
    ecploc = cp.asarray(ecp_loc, dtype=cp.int32)

    # Initialize result matrix in splitted_mol basis
    mat1 = cp.zeros((nao, nao), dtype=dtype)

    # Compute ECP integrals for each basis group combination and ECP type
    # Following gpu4pyscf's triple loop pattern
    for i in range(n_groups):
        for j in range(i, n_groups):  # Only upper triangle
            for k in range(n_ecp_groups):
                li = group_key[i, 0]
                lj = group_key[j, 0]
                lk = uniq_lecp[k]
                npi = group_key[i, 1]  # Number of primitives for shell i
                npj = group_key[j, 1]  # Number of primitives for shell j

                # Generate tasks for this group combination
                ish0, ish1 = group_offset[i], group_offset[i + 1]
                jsh0, jsh1 = group_offset[j], group_offset[j + 1]
                ksh0, ksh1 = lecp_offsets[k], lecp_offsets[k + 1]

                tasks = []
                for ish in range(ish0, ish1):
                    if basis_layout.pad_id[ish]:  # Skip padded shells
                        continue
                    for jsh in range(jsh0, jsh1):
                        if basis_layout.pad_id[jsh]:  # Skip padded shells
                            continue
                        if ish <= jsh:  # Only upper triangle
                            for ksh in range(ksh0, ksh1):
                                tasks.append([ish, jsh, ksh])

                if not tasks:
                    continue

                tasks = cp.asarray(tasks, dtype=cp.int32, order='F')
                ntasks = len(tasks)
                # Choose appropriate kernel based on ECP type
                if lk < 0:
                    _compile_ecp_type1_kernel(li, lj, precision)(
                        mat1, ao_loc, nao, tasks, ntasks, ecpbas, ecploc,
                        basis_layout.coords, basis_layout.ce,
                        atm, env, npi, npj
                    )
                else:
                    # Type2 kernel for semi-local channels
                    _compile_ecp_type2_kernel(li, lj, lk, precision)(
                        mat1, ao_loc, nao, tasks, ntasks, ecpbas, ecploc,
                        basis_layout.coords, basis_layout.ce, atm, env, npi, npj
                    )

    # Transform result from splitted_mol basis back to original mol basis
    # Kernels already write symmetric contributions; no host-side symmetrization needed
    result = basis_layout.dm_to_mol(mat1.reshape(1, nao, nao))[0]
    return result


def ecp_generator(mol_or_basis_layout, precision: str = 'fp64') -> Dict[str, Any]:
    """
    Generate ECP integral evaluation kernels

    Args:
        mol_or_basis_layout: Either mol (backward compatibility) or BasisLayout object
        precision: Floating point precision

    Returns:
        Dictionary containing ECP evaluation function and metadata
    """
    # Handle both mol and basis_layout inputs for backward compatibility
    if hasattr(mol_or_basis_layout, '_mol'):
        # This is a BasisLayout object
        basis_layout = mol_or_basis_layout
        mol = basis_layout._mol
    else:
        # This is a mol object - create BasisLayout
        from jqc.pyscf.basis import BasisLayout
        mol = mol_or_basis_layout
        basis_layout = BasisLayout.from_mol(mol)

    def _ecp_eval():
        return get_ecp(basis_layout, precision)

    return {
        'eval_func': _ecp_eval,
        'precision': precision,
        'kernel_type': 'ecp_type1',
        'mol_hash': hash(str(mol.atom) + str(mol.basis))
    }


def precompile_ecp_kernels(precision: str = 'fp64'):
    """
    Precompile common ECP kernels to avoid JIT overhead

    Args:
        precision: Floating point precision
    """
    # Precompile Type1 kernels for low angular momentum
    max_l_type1 = 2
    for li in range(max_l_type1 + 1):
        for lj in range(li, max_l_type1 + 1):
            try:
                _compile_ecp_type1_kernel(li, lj, precision)
                print(f"Compiled ECP Type1 kernel for L=({li},{lj}) with {precision}")
            except Exception as e:
                print(f"Failed to compile ECP Type1 kernel for L=({li},{lj}): {e}")

    # Precompile Type2 kernels for higher angular momentum
    for li in range(MAX_L_ECP + 1):
        for lj in range(li, MAX_L_ECP + 1):
            if li + lj > max_l_type1:  # Only for combinations not covered by Type1
                for lc in range(MAX_L_ECP + 1):
                    try:
                        _compile_ecp_type2_kernel(li, lj, lc, precision)
                        print(f"Compiled ECP Type2 kernel for L=({li},{lj},{lc}) with {precision}")
                    except Exception as e:
                        print(f"Failed to compile ECP Type2 kernel for L=({li},{lj},{lc}): {e}")
