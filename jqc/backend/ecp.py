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
import warnings
from typing import Any, Dict

import cupy as cp
import numpy as np
from pyscf import gto

from jqc.constants import BASIS_STRIDE, MAX_L_ECP

__all__ = [
    "ecp_generator",
    "get_ecp",
    "get_ecp_ip",
    "get_ecp_ipip",
    "make_ecp_tasks",
    "sort_ecp_basis",
]


def get_cp_type(precision: str):
    """Return CuPy dtype for supported precision.

    Only double precision (fp64) is supported.
    """
    if precision != "fp64":
        raise ValueError(
            f"Unsupported precision '{precision}'. Only 'fp64' (double precision) is supported."
        )
    return cp.float64


# Global cache for compiled kernels
_ecp_kernel_cache = {}


def _get_compile_options():
    # Compilation options following JoltQC style
    # Note: ECP derivative normalization is handled in basis coefficients; no extra scaling.
    return ("-std=c++17", "--use_fast_math", "--minimal")


def _estimate_type1_shared_memory(li: int, lj: int, precision: str = "fp64") -> int:
    """
    Estimate shared memory requirements for Type1 ECP kernels

    Args:
        li: Angular momentum of first basis function
        lj: Angular momentum of second basis function
        precision: Floating point precision ('fp64', 'fp32', 'mixed')

    Returns:
        Estimated shared memory size in bytes
    """
    # Only double precision is supported
    dtype_size = 8

    # Actual memory layout in type1_cart kernel:
    # 1. rad_ang: LIJ1^3 doubles
    # 2. rad_all: LIJ1^2 doubles
    # 3. fi: 3*nfi doubles
    # 4. fj: 3*nfj doubles

    lij1 = li + lj + 1
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2

    rad_ang_size = lij1 * lij1 * lij1
    rad_all_size = lij1 * lij1
    fi_size = 3 * nfi
    fj_size = 3 * nfj

    total_doubles = rad_ang_size + rad_all_size + fi_size + fj_size

    return total_doubles * dtype_size


def _estimate_type1_ip_shared_memory(li: int, lj: int, precision: str = "fp64") -> int:
    """
    Estimate shared memory requirements for Type1 IP ECP kernels

    Args:
        li: Angular momentum of first basis function
        lj: Angular momentum of second basis function
        precision: Floating point precision ('fp64', 'fp32', 'mixed')

    Returns:
        Estimated shared memory size in bytes
    """
    dtype_size = 8

    # IP kernels need gctr_smem for accumulation
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    gctr_smem_size = 3 * nfi * nfj  # 3 components for derivatives

    # IP kernels also need buf for intermediate calculations
    # buf size: 3 * nfi_max * nfj_max where nfi_max = (LI+2)*(LI+3)/2, nfj_max = (LJ+1)*(LJ+2)/2
    nfi_max = (li + 2) * (li + 3) // 2
    nfj_max = (lj + 1) * (lj + 2) // 2
    buf_size = 3 * nfi_max * nfj_max

    # Plus base kernel memory
    base_memory = _estimate_type1_shared_memory(li + 1, lj, precision) // dtype_size

    total_doubles = base_memory + gctr_smem_size + buf_size
    total_bytes = total_doubles * dtype_size

    # Raise error if shared memory exceeds 96KB limit
    if total_bytes > 96 * 1024:
        raise RuntimeError(
            f"Shared memory requirement {total_bytes} bytes ({total_bytes/1024:.1f} KB) "
            f"exceeds 96KB limit for angular momentum combination. "
            f"Consider using smaller basis sets or enabling global memory fallback."
        )

    return total_bytes


def _estimate_type1_ipip_shared_memory(
    li: int, lj: int, variant: str, precision: str = "fp64"
) -> int:
    """
    Estimate shared memory requirements for Type1 IPIP ECP kernels

    Args:
        li: Angular momentum of first basis function
        lj: Angular momentum of second basis function
        variant: IPIP variant ('ipipv' or 'ipvip')
        precision: Floating point precision ('fp64', 'fp32', 'mixed')

    Returns:
        Estimated shared memory size in bytes
    """
    dtype_size = 8

    # Actual memory layout in Type1 IPIP kernels:
    # [buf1][buf][kernel_shared_mem] - all separate and additive

    if variant == "ipipv":
        # ipipv: buf1 uses (LI+3)*(LI+4)/2 * (LJ+1)*(LJ+2)/2
        nfi2_max = (li + 3) * (li + 4) // 2  # LI+2 for buf1
        nfj_max = (lj + 1) * (lj + 2) // 2  # LJ for buf1
        buf1_size = nfi2_max * nfj_max

        # buf uses 3 * (LI+2)*(LI+3)/2 * (LJ+1)*(LJ+2)/2
        nfi1_max = (li + 2) * (li + 3) // 2  # LI+1 for buf
        buf_size = 3 * nfi1_max * nfj_max

        # kernel_shared_mem uses type1_cart_kernel with LI+2, LJ
        kernel_memory = (
            _estimate_type1_shared_memory(li + 2, lj, precision) // dtype_size
        )

    else:  # ipvip
        # ipvip: buf1 uses (LI+2)*(LI+3)/2 * (LJ+2)*(LJ+3)/2
        nfi1_max = (li + 2) * (li + 3) // 2  # LI+1 for buf1
        nfj1_max = (lj + 2) * (lj + 3) // 2  # LJ+1 for buf1
        buf1_size = nfi1_max * nfj1_max

        # buf uses 3 * (LI+1)*(LI+2)/2 * (LJ+2)*(LJ+3)/2
        nfi_max = (li + 1) * (li + 2) // 2  # LI for buf
        buf_size = 3 * nfi_max * nfj1_max

        # kernel_shared_mem uses type1_cart_kernel with LI+1, LJ+1
        kernel_memory = (
            _estimate_type1_shared_memory(li + 1, lj + 1, precision) // dtype_size
        )

    # Total memory: buf1 + buf + kernel_memory (all separate)
    total_doubles = buf1_size + buf_size + kernel_memory

    return total_doubles * dtype_size


def _estimate_type2_shared_memory(
    li: int, lj: int, lc: int, precision: str = "fp64"
) -> int:
    """
    Estimate shared memory requirements for Type2 ECP kernels

    Args:
        li: Angular momentum of first basis function
        lj: Angular momentum of second basis function
        lc: Angular momentum of ECP center
        precision: Floating point precision ('fp64', 'fp32', 'mixed')

    Returns:
        Estimated shared memory size in bytes
    """
    dtype_size = 8

    # Actual memory layout in type2_cart kernel:
    # 1. omegai: LI1*(LI1+1)*(LI1+2)/6 * BLKI
    # 2. omegaj: LJ1*(LJ1+1)*(LJ1+2)/6 * BLKJ
    # 3. rad_all: (LI+LJ+1) * LIC1 * LJC1
    # 4. angi: LI1*nfi*LIC1
    # 5. angj: LJ1*nfj*LJC1
    # 6. fi: 3*nfi
    # 7. fj: 3*nfj

    LI1 = li + 1
    LJ1 = lj + 1
    LIC1 = li + lc + 1
    LJC1 = lj + lc + 1
    LCC1 = 2 * lc + 1

    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2

    # BLKI and BLKJ as defined in the kernel
    BLKI = (LIC1 + 1) // 2 * LCC1
    BLKJ = (LJC1 + 1) // 2 * LCC1

    # Actual array sizes
    omegai_size = LI1 * (LI1 + 1) * (LI1 + 2) // 6 * BLKI
    omegaj_size = LJ1 * (LJ1 + 1) * (LJ1 + 2) // 6 * BLKJ
    rad_all_size = (li + lj + 1) * LIC1 * LJC1
    angi_size = LI1 * nfi * LIC1
    angj_size = LJ1 * nfj * LJC1
    fi_size = 3 * nfi
    fj_size = 3 * nfj

    total_doubles = (
        omegai_size
        + omegaj_size
        + rad_all_size
        + angi_size
        + angj_size
        + fi_size
        + fj_size
    )

    return total_doubles * dtype_size


def _estimate_type2_ip_shared_memory(
    li: int, lj: int, lc: int, precision: str = "fp64"
) -> int:
    """
    Estimate shared memory requirements for Type2 IP ECP kernels

    Args:
        li: Angular momentum of first basis function
        lj: Angular momentum of second basis function
        lc: Angular momentum of ECP center
        precision: Floating point precision ('fp64', 'fp32', 'mixed')

    Returns:
        Estimated shared memory size in bytes
    """
    dtype_size = 8

    # IP kernels need gctr_smem for accumulation
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    gctr_smem_size = 3 * nfi * nfj  # 3 components for derivatives

    # IP kernels also need buf for intermediate calculations
    # buf size: 3 * NFI_MAX * NFJ_MAX where NFI_MAX = (LI+2)*(LI+3)/2, NFJ_MAX = (LJ+1)*(LJ+2)/2
    nfi_max = (li + 2) * (li + 3) // 2
    nfj_max = (lj + 1) * (lj + 2) // 2
    buf_size = 3 * nfi_max * nfj_max

    # Plus base kernel memory with higher angular momentum for derivatives
    base_memory = _estimate_type2_shared_memory(li + 1, lj, lc, precision) // dtype_size

    total_doubles = gctr_smem_size + buf_size + base_memory
    total_bytes = total_doubles * dtype_size

    # Raise error if shared memory exceeds 96KB limit
    if total_bytes > 96 * 1024:
        raise RuntimeError(
            f"Shared memory requirement {total_bytes} bytes ({total_bytes/1024:.1f} KB) "
            f"exceeds 96KB limit for angular momentum combination. "
            f"Consider using smaller basis sets or enabling global memory fallback."
        )

    return total_bytes


def _estimate_type2_ipip_shared_memory(
    li: int, lj: int, lc: int, variant: str, precision: str = "fp64"
) -> int:
    """
    Estimate shared memory requirements for Type2 IPIP ECP kernels

    Args:
        li: Angular momentum of first basis function
        lj: Angular momentum of second basis function
        lc: Angular momentum of ECP center
        variant: IPIP variant ('ipipv' or 'ipvip')
        precision: Floating point precision ('fp64', 'fp32', 'mixed')

    Returns:
        Estimated shared memory size in bytes
    """
    dtype_size = 8

    # Actual memory layout in IPIP kernels:
    # [buf1][kernel_shared_mem (which overlaps with buf space)]
    # The buf array reuses the kernel_shared_mem space, so we take the maximum of the two

    if variant == "ipipv":
        # ipipv: buf1 uses (LI+3)*(LI+4)/2 * (LJ+1)*(LJ+2)/2
        nfi2_max = (li + 3) * (li + 4) // 2  # LI+2 for buf1
        nfj_max = (lj + 1) * (lj + 2) // 2  # LJ for buf1
        buf1_size = nfi2_max * nfj_max

        # buf reuses kernel space, size is 3*(LI+2)*(LI+3)/2 * (LJ+1)*(LJ+2)/2
        nfi1_max = (li + 2) * (li + 3) // 2
        buf_size = 3 * nfi1_max * nfj_max

        # kernel_shared_mem uses type2_cart_kernel with LI+2, LJ, LC
        kernel_memory = (
            _estimate_type2_shared_memory(li + 2, lj, lc, precision) // dtype_size
        )

    else:  # ipvip
        # ipvip: buf1 uses (LI+2)*(LI+3)/2 * (LJ+2)*(LJ+3)/2
        nfi1_max = (li + 2) * (li + 3) // 2  # LI+1 for buf1
        nfj1_max = (lj + 2) * (lj + 3) // 2  # LJ+1 for buf1
        buf1_size = nfi1_max * nfj1_max

        # buf reuses kernel space, size is 3*(LI+1)*(LI+2)/2 * (LJ+2)*(LJ+3)/2
        nfi_max = (li + 1) * (li + 2) // 2
        buf_size = 3 * nfi_max * nfj1_max

        # kernel_shared_mem uses type2_cart_kernel with LI+1, LJ+1, LC
        kernel_memory = (
            _estimate_type2_shared_memory(li + 1, lj + 1, lc, precision) // dtype_size
        )

    # Total memory: buf1 + buf_size + kernel_memory (no longer overlapping after fix)
    total_doubles = buf1_size + buf_size + kernel_memory

    return total_doubles * dtype_size


def _compile_ecp_type2_kernel(li: int, lj: int, lc: int, precision: str = "fp64"):
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

    cache_key = (li, lj, lc, "type2", precision)

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
    for filename in [
        "ecp.h",
        "prelude.cuh",
        "gauss_chebyshev.cu",
        "bessel.cu",
        "cart2sph.cu",
        "common.cu",
        "type2_ang_nuc.cu",
        "ecp_type2.cu",
    ]:
        cuda_file = os.path.join(os.path.dirname(__file__), "ecp", filename)
        with open(cuda_file) as f:
            cuda_source += f.read()
            cuda_source += "\n"  # Ensure separation between files

    # Replace DataType typedef and inject constexpr values
    cuda_source = cuda_source.replace(
        "using DataType = double;", ""  # Remove the original declaration
    )

    # Inject constexpr values at the beginning of the source
    cuda_source = const_injection + "\n" + cuda_source

    # Only double precision is supported; no fp32 constant replacements

    # Compile module following JoltQC pattern
    # Inject layout constants via -D to avoid string duplication
    opts = (
        *_get_compile_options(),
        f"-DBASIS_STRIDE={BASIS_STRIDE}",
    )
    mod = cp.RawModule(code=cuda_source, options=opts)
    kernel = mod.get_function("type2_cart")

    # Calculate dynamic shared memory size for type2_cart
    shared_mem_size = _estimate_type2_shared_memory(li, lj, lc, precision)
    kernel.max_dynamic_shared_size_bytes = max(48 * 1024, shared_mem_size)

    # Create wrapper function following JoltQC style
    def kernel_wrapper(*args):
        # Extract nprim values from additional arguments (last 2 arguments)
        # After removing ao_loc, ntasks is now at index 3
        ntasks = args[3]  # Number of tasks
        npi = args[-2]  # npi is second-to-last argument
        npj = args[-1]  # npj is last argument
        block_size = 128
        grid_size = int(ntasks)

        kernel_args = args[:-2] + (int(npi), int(npj))
        kernel((grid_size,), (block_size,), kernel_args, shared_mem=shared_mem_size)

    # Cache the compiled kernel
    _ecp_kernel_cache[cache_key] = kernel_wrapper

    return kernel_wrapper


def _compile_ecp_type1_kernel(li: int, lj: int, precision: str = "fp64"):
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
    for filename in [
        "ecp.h",
        "prelude.cuh",
        "gauss_chebyshev.cu",
        "bessel.cu",
        "cart2sph.cu",
        "common.cu",
        "type1_ang_nuc.cu",
        "ecp_type1.cu",
    ]:
        cuda_file = os.path.join(os.path.dirname(__file__), "ecp", filename)
        with open(cuda_file) as f:
            cuda_source += f.read()
            cuda_source += "\n"  # Ensure separation between files

    # Replace DataType typedef and inject constexpr values
    cuda_source = cuda_source.replace(
        "using DataType = double;", ""  # Remove the original declaration
    )

    # Inject constexpr values at the beginning of the source
    cuda_source = const_injection + "\n" + cuda_source

    # Compile module following JoltQC pattern
    opts = (
        *_get_compile_options(),
        f"-DBASIS_STRIDE={BASIS_STRIDE}",
    )
    mod = cp.RawModule(code=cuda_source, options=opts)
    kernel = mod.get_function("type1_cart")

    # Calculate dynamic shared memory size for type1_cart
    shared_mem_size = _estimate_type1_shared_memory(li, lj, precision)
    kernel.max_dynamic_shared_size_bytes = max(48 * 1024, shared_mem_size)
    if kernel.local_size_bytes > 2048:
        warnings.warn(
            f"High local memory usage detected in type1_cart kernel: {kernel.local_size_bytes} bytes"
        )

    # Create wrapper function following JoltQC style
    def kernel_wrapper(*args):
        # Extract nprim values from additional arguments (last 2 arguments)
        # After removing ao_loc, ntasks is now at index 3
        ntasks = args[3]  # Number of tasks
        npi = args[-2]  # npi is second-to-last argument
        npj = args[-1]  # npj is last argument
        block_size = 128
        grid_size = int(ntasks)

        kernel_args = args[:-2] + (int(npi), int(npj))
        kernel((grid_size,), (block_size,), kernel_args, shared_mem=shared_mem_size)

    # Cache the compiled kernel
    _ecp_kernel_cache[cache_key] = kernel_wrapper

    return kernel_wrapper


def _compile_ecp_type1_ip_kernel(li: int, lj: int, precision: str = "fp64"):
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

    cache_key = (li, lj, "type1_ip", precision)

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
    for filename in [
        "ecp.h",
        "prelude.cuh",
        "gauss_chebyshev.cu",
        "bessel.cu",
        "cart2sph.cu",
        "common.cu",
        "type1_ang_nuc.cu",
        "ecp_type1.cu",
        "ecp_type1_kernel.cu",
        "ecp_type1_ip.cu",
    ]:
        cuda_file = os.path.join(os.path.dirname(__file__), "ecp", filename)
        with open(cuda_file) as f:
            cuda_source += f.read()
            cuda_source += "\n"

    # Replace DataType typedef and inject constexpr values
    cuda_source = cuda_source.replace("using DataType = double;", "")
    cuda_source = const_injection + "\n" + cuda_source

    # Compile module
    opts = (
        *_get_compile_options(),
        f"-DBASIS_STRIDE={BASIS_STRIDE}",
    )
    mod = cp.RawModule(code=cuda_source, options=opts)
    kernel = mod.get_function("type1_cart_ip1")

    # Calculate dynamic shared memory size
    shared_mem_size = _estimate_type1_ip_shared_memory(li, lj, precision)
    kernel.max_dynamic_shared_size_bytes = max(48 * 1024, shared_mem_size)

    if kernel.local_size_bytes > 2048:
        warnings.warn(
            f"High local memory usage detected in type1_cart_ip1 kernel: {kernel.local_size_bytes} bytes"
        )

    def kernel_wrapper(*args):
        # Extract nprim values from additional arguments (last 2 arguments)
        # After removing ao_loc, ntasks is now at index 3
        ntasks = args[3]  # Number of tasks
        npi = args[-2]  # npi is second-to-last argument
        npj = args[-1]  # npj is last argument
        block_size = 128
        grid_size = int(ntasks)

        # Pass original args (excluding npi, npj) plus npi, npj to kernel
        kernel_args = args[:-2] + (int(npi), int(npj))
        kernel((grid_size,), (block_size,), kernel_args, shared_mem=shared_mem_size)

    _ecp_kernel_cache[cache_key] = kernel_wrapper
    return kernel_wrapper


def _compile_ecp_type2_ip_kernel(li: int, lj: int, lk: int, precision: str = "fp64"):
    """
    Compile ECP Type2 IP (first derivative) kernel for specific angular momentum combination

    Args:
        li: Angular momentum of first basis function
        lj: Angular momentum of second basis function
        lk: Angular momentum of ECP center
        precision: Floating point precision ('fp64', 'fp32', 'mixed')

    Returns:
        Compiled IP kernel function
    """

    dtype = get_cp_type(precision)
    dtype_cuda = "float" if dtype == cp.float32 else "double"

    cache_key = (li, lj, lk, "type2_ip", precision)

    if cache_key in _ecp_kernel_cache:
        return _ecp_kernel_cache[cache_key]

    # Create constexpr injection
    const_injection = f"""
using DataType = {dtype_cuda};
constexpr int LI = {li};
constexpr int LJ = {lj};
constexpr int LC = {lk};
"""

    # Read the CUDA source
    cuda_source = ""
    for filename in [
        "ecp.h",
        "prelude.cuh",
        "gauss_chebyshev.cu",
        "bessel.cu",
        "cart2sph.cu",
        "common.cu",
        "type2_ang_nuc.cu",
        "ecp_type2.cu",
        "ecp_type2_kernel.cu",
        "ecp_type2_ip.cu",
    ]:
        cuda_file = os.path.join(os.path.dirname(__file__), "ecp", filename)
        with open(cuda_file) as f:
            cuda_source += f.read()
            cuda_source += "\n"

    # Replace DataType typedef and inject constexpr values
    cuda_source = cuda_source.replace("using DataType = double;", "")
    cuda_source = const_injection + "\n" + cuda_source

    # Only double precision is supported; no fp32 constant replacements

    # Compile module
    opts = (
        *_get_compile_options(),
        f"-DBASIS_STRIDE={BASIS_STRIDE}",
    )
    mod = cp.RawModule(code=cuda_source, options=opts)
    kernel = mod.get_function("type2_cart_ip1")

    # Calculate dynamic shared memory size
    shared_mem_size = _estimate_type2_ip_shared_memory(li, lj, lk, precision)
    kernel.max_dynamic_shared_size_bytes = max(48 * 1024, shared_mem_size)

    if kernel.local_size_bytes > 2048:
        warnings.warn(
            f"High local memory usage detected in type2_cart_ip1 kernel: {kernel.local_size_bytes} bytes"
        )

    def kernel_wrapper(*args):
        # Extract nprim values from additional arguments (last 2 arguments)
        ntasks = args[4]  # Number of tasks
        npi = args[-2]  # npi is second-to-last argument
        npj = args[-1]  # npj is last argument
        block_size = 128
        grid_size = int(ntasks)  # Ensure grid_size is a Python int

        # Ensure proper types for kernel arguments
        kernel_args = args[:-2] + (int(npi), int(npj))

        kernel((grid_size,), (block_size,), kernel_args, shared_mem=shared_mem_size)

    _ecp_kernel_cache[cache_key] = kernel_wrapper
    return kernel_wrapper


def _compile_ecp_type1_ipip_kernel(
    li: int, lj: int, variant: str, precision: str = "fp64"
):
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

    cache_key = (li, lj, variant, "type1_ipip", precision)

    if cache_key in _ecp_kernel_cache:
        return _ecp_kernel_cache[cache_key]

    # Create constexpr injection
    const_injection = f"""
using DataType = {dtype_cuda};
constexpr int LI = {li};
constexpr int LJ = {lj};
"""

    # Read the CUDA source - use the appropriate file for the variant
    cuda_source = ""
    variant_file = f"ecp_type1_{variant}.cu"
    for filename in [
        "ecp.h",
        "prelude.cuh",
        "gauss_chebyshev.cu",
        "bessel.cu",
        "cart2sph.cu",
        "common.cu",
        "type1_ang_nuc.cu",
        "ecp_type1.cu",
        "ecp_type1_kernel.cu",
        variant_file,
    ]:
        cuda_file = os.path.join(os.path.dirname(__file__), "ecp", filename)
        with open(cuda_file) as f:
            cuda_source += f.read()
            cuda_source += "\n"

    # Replace DataType typedef and inject constexpr values
    cuda_source = cuda_source.replace("using DataType = double;", "")
    cuda_source = const_injection + "\n" + cuda_source

    if precision == "fp32":
        cuda_source = cuda_source.replace("M_PI", "M_PI_F")

    # Compile module
    opts = (
        *_get_compile_options(),
        f"-DBASIS_STRIDE={BASIS_STRIDE}",
    )
    mod = cp.RawModule(code=cuda_source, options=opts)
    kernel_name = f"type1_cart_{variant}"
    kernel = mod.get_function(kernel_name)

    # Calculate dynamic shared memory size
    shared_mem_size = _estimate_type1_ipip_shared_memory(li, lj, variant, precision)
    kernel.max_dynamic_shared_size_bytes = max(48 * 1024, shared_mem_size)

    if kernel.local_size_bytes > 2048:
        warnings.warn(
            f"High local memory usage detected in {kernel_name} kernel: {kernel.local_size_bytes} bytes"
        )

    def kernel_wrapper(*args):
        # Extract nprim values from additional arguments (last 2 arguments)
        # After removing ao_loc, ntasks is now at index 3
        ntasks = args[3]  # Number of tasks
        npi = args[-2]  # npi is second-to-last argument
        npj = args[-1]  # npj is last argument
        block_size = 128
        grid_size = int(ntasks)

        # Pass original args (excluding npi, npj) plus npi, npj to kernel
        kernel_args = args[:-2] + (int(npi), int(npj))
        kernel((grid_size,), (block_size,), kernel_args, shared_mem=shared_mem_size)

    _ecp_kernel_cache[cache_key] = kernel_wrapper
    return kernel_wrapper


def _compile_ecp_type2_ipip_kernel(
    li: int, lj: int, lk: int, variant: str, precision: str = "fp64"
):
    """
    Compile ECP Type2 IPIP (second derivative) kernel for specific angular momentum combination

    Args:
        li: Angular momentum of first basis function
        lj: Angular momentum of second basis function
        lk: Angular momentum of ECP center
        variant: IPIP variant ('ipipv' or 'ipvip')
        npi: Number of primitives for shell i
        npj: Number of primitives for shell j
        precision: Floating point precision ('fp64', 'fp32', 'mixed')

    Returns:
        Compiled IPIP kernel function
    """

    dtype = get_cp_type(precision)
    dtype_cuda = "float" if dtype == cp.float32 else "double"

    cache_key = (li, lj, lk, variant, "type2_ipip", precision)

    if cache_key in _ecp_kernel_cache:
        return _ecp_kernel_cache[cache_key]

    # Create constexpr injection
    const_injection = f"""
using DataType = {dtype_cuda};
constexpr int LI = {li};
constexpr int LJ = {lj};
constexpr int LC = {lk};
"""

    # Read the CUDA source - use the appropriate file for the variant
    cuda_source = ""
    variant_file = f"ecp_type2_{variant}.cu"
    for filename in [
        "ecp.h",
        "prelude.cuh",
        "gauss_chebyshev.cu",
        "bessel.cu",
        "cart2sph.cu",
        "common.cu",
        "type2_ang_nuc.cu",
        "ecp_type2.cu",
        "ecp_type2_kernel.cu",
        variant_file,
    ]:
        cuda_file = os.path.join(os.path.dirname(__file__), "ecp", filename)
        with open(cuda_file) as f:
            cuda_source += f.read()
            cuda_source += "\n"

    # Replace DataType typedef and inject constexpr values
    cuda_source = cuda_source.replace("using DataType = double;", "")
    cuda_source = const_injection + "\n" + cuda_source

    if precision == "fp32":
        cuda_source = cuda_source.replace("M_PI", "M_PI_F")

    # Compile module
    opts = (
        *_get_compile_options(),
        f"-DBASIS_STRIDE={BASIS_STRIDE}",
    )
    mod = cp.RawModule(code=cuda_source, options=opts)
    kernel_name = f"type2_cart_{variant}"
    kernel = mod.get_function(kernel_name)

    # Calculate dynamic shared memory size
    shared_mem_size = _estimate_type2_ipip_shared_memory(li, lj, lk, variant, precision)
    kernel.max_dynamic_shared_size_bytes = max(48 * 1024, shared_mem_size)

    if kernel.local_size_bytes > 2048:
        warnings.warn(
            f"High local memory usage detected in {kernel_name} kernel: {kernel.local_size_bytes} bytes"
        )

    def kernel_wrapper(*args):
        # Extract nprim values from additional arguments (last 2 arguments)
        ntasks = args[4]  # Number of tasks
        npi = args[-2]  # npi is second-to-last argument
        npj = args[-1]  # npj is last argument
        block_size = 128
        grid_size = int(ntasks)  # Ensure grid_size is a Python int

        # Ensure proper types for kernel arguments
        kernel_args = args[:-2] + (int(npi), int(npj))
        kernel((grid_size,), (block_size,), kernel_args, shared_mem=shared_mem_size)

    _ecp_kernel_cache[cache_key] = kernel_wrapper
    return kernel_wrapper


def get_ecp_ip(
    mol_or_basis_layout, ip_type="ip", ecp_atoms=None, precision: str = "fp64"
) -> cp.ndarray:
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
    if ip_type != "ip":
        raise ValueError(f"Invalid ip_type: {ip_type}. Only 'ip' is supported.")

    # Handle both mol and basis_layout inputs for backward compatibility
    if hasattr(mol_or_basis_layout, "_mol"):
        basis_layout = mol_or_basis_layout
    else:
        from jqc.pyscf.basis import BasisLayout

        basis_layout = BasisLayout.from_mol(mol_or_basis_layout)

    dtype = get_cp_type(precision)
    mol = basis_layout._mol
    splitted_mol = basis_layout.splitted_mol

    # Spherical-basis fallback: compute per-ECP-atom derivatives on CPU and return GPU array
    # Ensures compatibility with tests expecting per-atom blocks (iprinv-style)
    if not mol.cart:
        nao = mol.nao
        # No ECP on molecule
        if not hasattr(mol, "_ecpbas") or len(mol._ecpbas) == 0:
            if ecp_atoms is None:
                return cp.zeros((0, 3, nao, nao), dtype=dtype)
            else:
                return cp.zeros((len(ecp_atoms), 3, nao, nao), dtype=dtype)

        # Determine the list of ECP atoms to compute
        all_ecp_atoms = sorted(set(mol._ecpbas[:, gto.ATOM_OF]))
        target_atoms = (
            all_ecp_atoms
            if ecp_atoms is None
            else [a for a in ecp_atoms if a in all_ecp_atoms]
        )

        # Compute per-atom derivatives using rinv-at-nucleus trick on CPU
        out_host = np.zeros((len(target_atoms), 3, nao, nao), dtype=np.float64)
        for idx, atm_id in enumerate(target_atoms):
            with mol.with_rinv_at_nucleus(atm_id):
                # iprinv_sph returns (3, nao, nao) for derivatives wrt nucleus atm_id
                h_ip = mol.intor("ECPscalar_iprinv_sph")
            out_host[idx] = h_ip

        return cp.asarray(out_host, dtype=dtype)

    if not hasattr(mol, "_ecpbas") or len(mol._ecpbas) == 0:
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

    # Get packed basis data (coords with ao_loc embedded, plus coefficients/exponents)
    basis_data_dict = basis_layout.basis_data_fp64
    basis_data = basis_data_dict['packed']
    nao = basis_layout._mol.nao_nr()

    ecpbas = cp.asarray(sorted_ecpbas, dtype=cp.int32)
    ecploc = cp.asarray(ecp_loc, dtype=cp.int32)

    # Initialize result matrix in splitted_mol basis
    # IP integrals have 3 components for each ECP atom (x,y,z derivatives)
    # For now, keep backward compatibility with flattened format for kernel interface
    n_ecp_atoms = len(ecp_atoms)
    mat1_flat = cp.zeros((3 * n_ecp_atoms, nao, nao), dtype=dtype)

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

                # Convert tasks to the format expected by the kernel:
                # [ish0, ish1, ..., jsh0, jsh1, ..., ksh0, ksh1, ...]
                ntasks = len(tasks)
                tasks = cp.asarray(
                    tasks, dtype=cp.int32, order="F"
                )  # Shape: (ntasks, 3)

                # Choose appropriate IP kernel based on ECP type
                if lk < 0:
                    _compile_ecp_type1_ip_kernel(li, lj, precision)(
                        mat1_flat,
                        nao,
                        tasks,
                        ntasks,
                        ecpbas,
                        ecploc,
                        basis_data,
                        atm,
                        env,
                        npi,
                        npj,
                    )
                else:
                    # Type2 IP kernel for semi-local channels
                    _compile_ecp_type2_ip_kernel(li, lj, lk, precision)(
                        mat1_flat,
                        nao,
                        tasks,
                        ntasks,
                        ecpbas,
                        ecploc,
                        basis_data,
                        atm,
                        env,
                        npi,
                        npj,
                    )
    result = cp.zeros((n_ecp_atoms, 3, nao_orig, nao_orig), dtype=dtype)

    # Map ECP atoms and transform each component from flattened format
    for i in range(n_ecp_atoms):
        for comp in range(3):
            flat_idx = 3 * i + comp
            result[i, comp] = basis_layout.dm_to_mol(
                mat1_flat[flat_idx : flat_idx + 1].reshape(1, nao, nao)
            )[0]

    return result


def get_ecp_ipip(
    mol_or_basis_layout, ip_type="ipipv", ecp_atoms=None, precision: str = "fp64"
) -> cp.ndarray:
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
    if ip_type not in ["ipipv", "ipvip"]:
        raise ValueError(
            f"Invalid ip_type: {ip_type}. Supported types: 'ipipv', 'ipvip'"
        )

    # Handle both mol and basis_layout inputs for backward compatibility
    if hasattr(mol_or_basis_layout, "_mol"):
        basis_layout = mol_or_basis_layout
    else:
        from jqc.pyscf.basis import BasisLayout

        basis_layout = BasisLayout.from_mol(mol_or_basis_layout)

    dtype = get_cp_type(precision)
    mol = basis_layout._mol
    splitted_mol = basis_layout.splitted_mol

    if not hasattr(mol, "_ecpbas") or len(mol._ecpbas) == 0:
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

    # Get packed basis data (coords with ao_loc embedded, plus coefficients/exponents)
    basis_data_dict = basis_layout.basis_data_fp64
    basis_data = basis_data_dict['packed']
    nao = basis_layout._mol.nao_nr()

    ecpbas = cp.asarray(sorted_ecpbas, dtype=cp.int32)
    ecploc = cp.asarray(ecp_loc, dtype=cp.int32)

    # Initialize result matrix in splitted_mol basis
    # IPIP integrals have 9 components for each ECP atom (xx,xy,xz,yx,yy,yz,zx,zy,zz derivatives)
    # For now, keep backward compatibility with flattened format for kernel interface
    n_ecp_atoms = len(ecp_atoms)
    mat1_flat = cp.zeros((9 * n_ecp_atoms, nao, nao), dtype=dtype)

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
                ntasks = len(tasks)
                tasks = cp.asarray(
                    tasks, dtype=cp.int32, order="F"
                )  # Shape: (ntasks, 3)

                # Choose appropriate IPIP kernel based on ECP type
                if lk < 0:
                    _compile_ecp_type1_ipip_kernel(li, lj, ip_type, precision)(
                        mat1_flat,
                        nao,
                        tasks,
                        ntasks,
                        ecpbas,
                        ecploc,
                        basis_data,
                        atm,
                        env,
                        npi,
                        npj,
                    )
                else:
                    # Type2 IPIP kernel for semi-local channels
                    _compile_ecp_type2_ipip_kernel(li, lj, lk, ip_type, precision)(
                        mat1_flat,
                        nao,
                        tasks,
                        ntasks,
                        ecpbas,
                        ecploc,
                        basis_data,
                        atm,
                        env,
                        npi,
                        npj,
                    )

    result = cp.zeros((n_ecp_atoms, 9, nao_orig, nao_orig), dtype=dtype)

    # Map ECP atoms and transform each component from flattened format
    for i in range(n_ecp_atoms):
        for comp in range(9):
            flat_idx = 9 * i + comp
            result[i, comp] = basis_layout.dm_to_mol(
                mat1_flat[flat_idx : flat_idx + 1].reshape(1, nao, nao)
            )[0]

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
        l_atm, return_inverse=True, return_counts=True, axis=0
    )
    sorted_idx = np.argsort(inv_idx.ravel(), kind="stable").astype(np.int32)

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
                ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i + 1]
                jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j + 1]
                ksh0, ksh1 = lecp_ctr_offsets[k], lecp_ctr_offsets[k + 1]
                grid = np.meshgrid(
                    np.arange(ish0, ish1), np.arange(jsh0, jsh1), np.arange(ksh0, ksh1)
                )
                grid = np.stack(grid, axis=-1).reshape(-1, 3)
                idx = grid[:, 0] <= grid[:, 1]
                tasks[i, j, k] = grid[idx]
    return tasks


def get_ecp(mol_or_basis_layout, precision: str = "fp64") -> cp.ndarray:
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
    if hasattr(mol_or_basis_layout, "_mol"):
        # This is a BasisLayout object
        basis_layout = mol_or_basis_layout
    else:
        # This is a mol object - create BasisLayout
        from jqc.pyscf.basis import BasisLayout

        basis_layout = BasisLayout.from_mol(mol_or_basis_layout)

    dtype = get_cp_type(precision)
    mol = basis_layout._mol
    splitted_mol = basis_layout.splitted_mol

    if not hasattr(mol, "_ecpbas") or len(mol._ecpbas) == 0:
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

    # Get packed basis data (coords with ao_loc embedded, plus coefficients/exponents)
    basis_data_dict = basis_layout.basis_data_fp64
    basis_data = basis_data_dict['packed']
    nao = basis_layout._mol.nao_nr()

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

                # Convert tasks to the format expected by the kernel:
                # [ish0, ish1, ..., jsh0, jsh1, ..., ksh0, ksh1, ...]
                ntasks = len(tasks)
                tasks = cp.asarray(
                    tasks, dtype=cp.int32, order="F"
                )  # Shape: (ntasks, 3)

                # Choose appropriate kernel based on ECP type
                if lk < 0:
                    _compile_ecp_type1_kernel(li, lj, precision)(
                        mat1,
                        nao,
                        tasks,
                        ntasks,
                        ecpbas,
                        ecploc,
                        basis_data,
                        atm,
                        env,
                        npi,
                        npj,
                    )
                else:
                    # Type2 kernel for semi-local channels
                    _compile_ecp_type2_kernel(li, lj, lk, precision)(
                        mat1,
                        nao,
                        tasks,
                        ntasks,
                        ecpbas,
                        ecploc,
                        basis_data,
                        atm,
                        env,
                        npi,
                        npj,
                    )

    # Transform result from splitted_mol basis back to original mol basis
    # Kernels already write symmetric contributions; no host-side symmetrization needed
    result = basis_layout.dm_to_mol(mat1.reshape(1, nao, nao))[0]
    return result


def ecp_generator(mol_or_basis_layout, precision: str = "fp64") -> Dict[str, Any]:
    """
    Generate ECP integral evaluation kernels

    Args:
        mol_or_basis_layout: Either mol (backward compatibility) or BasisLayout object
        precision: Floating point precision

    Returns:
        Dictionary containing ECP evaluation function and metadata
    """
    # Handle both mol and basis_layout inputs for backward compatibility
    if hasattr(mol_or_basis_layout, "_mol"):
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
        "eval_func": _ecp_eval,
        "precision": precision,
        "kernel_type": "ecp_type1",
        "mol_hash": hash(str(mol.atom) + str(mol.basis)),
    }


def precompile_ecp_kernels(precision: str = "fp64"):
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
                        print(
                            f"Compiled ECP Type2 kernel for L=({li},{lj},{lc}) with {precision}"
                        )
                    except Exception as e:
                        print(
                            f"Failed to compile ECP Type2 kernel for L=({li},{lj},{lc}): {e}"
                        )
