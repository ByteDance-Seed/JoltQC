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

'''
Generate kernels for incremental DFT
'''

import numpy as np
import cupy as cp
from pathlib import Path
code_path = Path(__file__).resolve().parent

__all__ = ['gen_rho_kernel', 'gen_vxc_kernel']

compile_options = ('-std=c++17','--use_fast_math')
nthreads = 256

with open(f'{code_path}/dft_scripts/eval_rho.cu', 'r') as f:
    eval_rho_cuda_code = f.read()

def gen_rho_kernel(ang, nprim, dtype, ndim=1, print_log=False):
    """
    ndim: 
        1: LDA
        4: GGA 
        5: mGGA
    """
    if dtype == np.float64:
        dtype_cuda = 'double'
    elif dtype == np.float32:
        dtype_cuda = 'float'
    else:
        raise RuntimeError('Data type is not supported')
    
    li, lj = ang
    npi, npj = nprim
    macros = f'''
using DataType = {dtype_cuda};
constexpr int li = {li};
constexpr int lj = {lj};
constexpr int npi = {npi};
constexpr int npj = {npj};
constexpr int ndim = {ndim};
constexpr int deriv = {1 if ndim > 1 else 0};
constexpr int nthreads = {nthreads};
'''

    script = macros + eval_rho_cuda_code
    mod = cp.RawModule(code=script, options=compile_options)
    kernel = mod.get_function('eval_rho')
    
    if print_log:
        print(f'Type: ({li}{lj}), \
primitives: ({npi}{npj}), \
registers: {kernel.num_regs:3d}, \
local memory: {kernel.local_size_bytes:4d} Bytes')
     
    def fun(*args):
        ngrids = args[-1]
        blocks = ngrids // nthreads
        if ngrids % nthreads != 0:
            raise RuntimeError(f'The number of grids must be divisible by {nthreads}')
        try:
            kernel(
                (blocks,),
                (nthreads,1),
                args,
            )
        except cp.cuda.runtime.CUDARuntimeError as e:
            print("CUDA Runtime Error in eval_rho kernel:", e)
    
    return script, mod, fun

with open(f'{code_path}/dft_scripts/eval_vxc.cu', 'r') as f:
    eval_vxc_cuda_code = f.read()

def gen_vxc_kernel(ang, nprim, dtype, ndim=1, print_log=False):
    """
    ndim: 
        1: LDA
        4: GGA 
        5: mGGA
    """
    if dtype == np.float64:
        dtype_cuda = 'double'
    elif dtype == np.float32:
        dtype_cuda = 'float'
    else:
        raise RuntimeError('Data type is not supported')
    
    li, lj = ang
    npi, npj = nprim
    macros = f'''
using DataType = {dtype_cuda};
constexpr int li = {li};
constexpr int lj = {lj};
constexpr int npi = {npi};
constexpr int npj = {npj};
constexpr int ndim = {ndim};
constexpr int deriv = {1 if ndim > 1 else 0};
constexpr int nthreads = {nthreads};
'''

    script = macros + eval_vxc_cuda_code
    mod = cp.RawModule(code=script, options=compile_options)
    kernel = mod.get_function('eval_vxc')

    if print_log:
        print(f'Type: ({li}{lj}), \
primitives: ({npi}{npj}), \
registers: {kernel.num_regs:3d}, \
local memory: {kernel.local_size_bytes:4d} Bytes')

    def fun(*args):
        ngrids = args[-1]
        blocks = ngrids // nthreads
        if ngrids % nthreads != 0:
            raise RuntimeError(f'The number of grids must be divisible by {nthreads}')
        try:
            kernel(
                (blocks,),
                (nthreads,1),
                args,
            )
        except cp.cuda.runtime.CUDARuntimeError as e:
            print("CUDA Runtime Error in eval_vxc kernel:", e)
    
    return script, mod, fun    

with open(f'{code_path}/dft_scripts/estimate_log_aovalue.cu', 'r') as f:
    estimate_aovalue_script = f.read()

def estimate_log_aovalue(grid_coords, coords, coeffs, exps, angs, nprims, log_cutoff=-36.8):
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
        Shell coordinates, shape (nbas, 3).
    coeffs : cp.ndarray
        Coefficients of primitive Gaussians.
    exps : cp.ndarray
        Exponents of primitive Gaussians.
    angs : cp.ndarray
        Angular momentum for each shell.
    nprims : cp.ndarray
        Number of primitives for each shell.
    log_cutoff : float, optional
        Logarithm of the cutoff value for screening. Shells with a maximum
        log-value below this on a grid block are ignored. Default is -36.8.

    Returns
    -------
    log_aovalue : cp.ndarray
        A (nblocks, nbas) array containing the maximum log-value of each
        shell on each grid block.
    nnz_indices : cp.ndarray
        A (nblocks, nbas) array. For each block, it contains the indices
        of shells that are considered significant (non-zero).
    nnz_per_block : cp.ndarray
        A (nblocks,) array with the count of significant shells for each
        grid block.
    """
    grid_coords = cp.asarray(grid_coords, dtype=np.float32)
    coords = cp.asarray(coords, dtype=np.float32)
    coeffs = cp.asarray(coeffs, dtype=np.float32)
    exps = cp.asarray(exps, dtype=np.float32)
    ngrids = grid_coords.shape[1]
    assert ngrids % nthreads == 0
    
    nbas = coords.shape[0]
    nblocks = ngrids//256
    log_aovalue = cp.empty((nblocks, nbas), dtype=np.float32)
    #log_aovalue = cp.zeros((nblocks, nbas), dtype=np.float32)

    nnz_indices = cp.empty((nblocks, nbas), dtype=np.int32)
    nnz_per_block = cp.empty((nblocks), dtype=np.int32)
    mod = cp.RawModule(code=estimate_aovalue_script, options=compile_options)
    kernel = mod.get_function('estimate_log_aovalue')
    kernel(
        (ngrids//nthreads,),
        (nthreads,),
        (grid_coords, ngrids, 
         coords, coeffs, exps, angs, nprims, nbas, 
         log_aovalue, nnz_indices, nnz_per_block, 
         np.float32(log_cutoff)),
    )
    return log_aovalue, nnz_indices, nnz_per_block
