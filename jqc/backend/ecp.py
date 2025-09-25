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
from jqc.constants import COORD_STRIDE, PRIM_STRIDE

__all__ = [
    "get_ecp",
    "sort_ecp_basis",
    "make_ecp_tasks",
    "ecp_generator",
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

# Compilation options following JoltQC style
compile_options = ("-std=c++17", "--use_fast_math", "--minimal")

def _compile_ecp_type2_kernel(li: int, lj: int, lc: int, npi=None, npj=None, precision: str = 'fp64'):
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
    # Backward-compatibility shim: allow signature (li, lj, lc, precision)
    # If third positional after lc is a string, it's precision and use npi=npj=1
    if isinstance(npi, str) and npj is None:
        precision = npi  # precision passed in place of npi
        npi = 1
        npj = 1
    if npi is None:
        npi = 1
    if npj is None:
        npj = 1

    dtype = get_cp_type(precision)
    dtype_cuda = "float" if dtype == cp.float32 else "double"

    cache_key = (li, lj, lc, npi, npj, 'type2', precision)

    if cache_key in _ecp_kernel_cache:
        return _ecp_kernel_cache[cache_key]

    # Common macro parameters shared by ECP kernels
    # Note: Some constants are defined in ecp.h, avoid conflicts
    common_macros = f"""
#define THREADS 128
#define NGAUSS 128

// Math constants
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Memory layout constants
#define PRIM_STRIDE {PRIM_STRIDE}
#define COORD_STRIDE {COORD_STRIDE}

// Forward declarations for quadrature arrays
extern __device__ double r128[128];
extern __device__ double w128[128];
"""

    # Create constexpr injection string following 1q1t pattern
    const_injection = f"""
typedef unsigned int uint32_t;
using DataType = {dtype_cuda};
constexpr int LI = {li};
constexpr int LJ = {lj};
constexpr int LC = {lc};
constexpr int NPI = {npi};
constexpr int NPJ = {npj};

{common_macros}

// Pair-stride over (c,e) tuples
constexpr int prim_stride = PRIM_STRIDE / 2;

// Data structure definitions (after macros are defined)
struct __align__(2*sizeof(DataType)) DataType2 {{
    DataType c, e;
}};

struct __align__(COORD_STRIDE*sizeof(DataType)) DataType4 {{
    DataType x, y, z;
#if COORD_STRIDE >= 4
    DataType w;
#endif
#if COORD_STRIDE > 4
    DataType _padding[COORD_STRIDE - 4];
#endif
}};
"""

    # Read the CUDA source
    cuda_source = ""
    for filename in ['ecp.h', 'bessel.cu', 'cart2sph.cu', 'common.cu', 'gauss_chebyshev.cu', 'type2_ang_nuc.cu', 'ecp_type2.cu']:
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
    mod = cp.RawModule(code=cuda_source, options=compile_options)
    kernel = mod.get_function('type2_cart')

    # Create wrapper function following JoltQC style
    def kernel_wrapper(*args):
        ntasks = args[4]  # Number of tasks
        block_size = 128
        grid_size = ntasks
        kernel((grid_size,), (block_size,), args)

    # Cache the compiled kernel
    _ecp_kernel_cache[cache_key] = kernel_wrapper

    return kernel_wrapper


def _compile_ecp_type1_kernel(li: int, lj: int, npi=None, npj=None, precision: str = 'fp64'):
    """
    Compile ECP Type1 kernel for specific angular momentum combination

    Args:
        li: Angular momentum of first basis function
        lj: Angular momentum of second basis function
        precision: Floating point precision ('fp64', 'fp32', 'mixed')

    Returns:
        Compiled kernel function
    """
    # Backward-compatibility shim: allow signature (li, lj, precision)
    if isinstance(npi, str) and npj is None:
        precision = npi
        npi = 1
        npj = 1
    if npi is None:
        npi = 1
    if npj is None:
        npj = 1

    dtype = get_cp_type(precision)
    dtype_cuda = "float" if dtype == cp.float32 else "double"

    cache_key = (li, lj, npi, npj, precision)

    if cache_key in _ecp_kernel_cache:
        return _ecp_kernel_cache[cache_key]

    # Common macro parameters shared by ECP kernels
    # Note: Some constants are defined in ecp.h, avoid conflicts
    common_macros = f"""
#define THREADS 128
#define NGAUSS 128

// Math constants
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// PRIM_STRIDE here matches host scalar stride; device uses prim_stride = PRIM_STRIDE/2
#define PRIM_STRIDE {PRIM_STRIDE}
#define COORD_STRIDE {COORD_STRIDE}

// Forward declarations for quadrature arrays
extern __device__ double r128[128];
extern __device__ double w128[128];
"""

    # Create constexpr injection string following 1q1t pattern
    const_injection = f"""
typedef unsigned int uint32_t;
using DataType = {dtype_cuda};
constexpr int LI = {li};
constexpr int LJ = {lj};
constexpr int NPI = {npi};
constexpr int NPJ = {npj};

{common_macros}

// Data structure definitions (after macros are defined)
struct __align__(2*sizeof(DataType)) DataType2 {{
    DataType c, e;
}};

struct __align__(COORD_STRIDE*sizeof(DataType)) DataType4 {{
    DataType x, y, z;
#if COORD_STRIDE >= 4
    DataType w;
#endif
#if COORD_STRIDE > 4
    DataType _padding[COORD_STRIDE - 4];
#endif
}};
"""

    # Read the CUDA source
    cuda_source = ""
    for filename in ['ecp.h', 'bessel.cu', 'cart2sph.cu', 'common.cu', 'gauss_chebyshev.cu', 'type1_ang_nuc.cu', 'ecp_type1.cu']:
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
    mod = cp.RawModule(code=cuda_source, options=compile_options)
    kernel = mod.get_function('type1_cart')

    # Create wrapper function following JoltQC style
    def kernel_wrapper(*args):
        ntasks = args[4]  # Number of tasks
        block_size = 128
        grid_size = ntasks
        kernel((grid_size,), (block_size,), args)

    # Cache the compiled kernel
    _ecp_kernel_cache[cache_key] = kernel_wrapper

    return kernel_wrapper

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

    # Optional debug toggles to isolate kernel types
    disable_type1 = os.getenv('JQC_DISABLE_ECP_TYPE1') == '1'
    disable_type2 = os.getenv('JQC_DISABLE_ECP_TYPE2') == '1'

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
                    if disable_type1:
                        continue
                    _compile_ecp_type1_kernel(li, lj, npi, npj, precision)(
                        mat1, ao_loc, nao, tasks, ntasks, ecpbas, ecploc,
                        basis_layout.coords, basis_layout.ce,
                        atm, env
                    )
                else:
                    if disable_type2:
                        continue
                    # Type2 kernel for semi-local channels
                    _compile_ecp_type2_kernel(li, lj, lk, npi, npj, precision)(
                        mat1, ao_loc, nao, tasks, ntasks, ecpbas, ecploc,
                        basis_layout.coords, basis_layout.ce, atm, env
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


# Angular momentum limits for kernel generation
MAX_L_ECP = 4  # Maximum angular momentum for ECP kernels

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
