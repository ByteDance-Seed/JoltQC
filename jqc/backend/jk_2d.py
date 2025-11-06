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
2D algorithm for JK calculations.

The 2D algorithm computes J/K matrices using a 2D grid where:
- Each block processes one (ij, kl) shell quartet pair
- Threads within a block (16x16) handle multiple pairs from the pair lists
- Supports arbitrary angular momentum combinations (li, lj, lk, ll)

Angular Momentum Support:
- All combinations from (0,0,0,0) to (4,4,4,4) are supported
- The algorithm is most efficient for low angular momentum (s, p, d shells)
- Higher angular momentum (f, g shells) may require more registers

Pair Format:
- ij_pairs and kl_pairs are arrays of flattened shell pair indices
- Each pair index is computed as: pair = ish * nbas + jsh
- Pairs are stored in blocks of 16 for coalesced memory access
- Padding with zeros is used for incomplete blocks
"""

import warnings

import cupy as cp
import numpy as np

from jqc.backend.cuda_scripts import jk_2d_code, rys_roots_code, rys_roots_data
from jqc.backend.util import generate_lookup_table
from jqc.constants import COORD_STRIDE, NPRIM_MAX, PRIM_STRIDE

__all__ = ["gen_kernel"]

THREADS = (16, 16)
compile_options = ("-std=c++17", "--use_fast_math", "--minimal")

_script_cache = {}

def gen_code(keys):
    if keys in _script_cache:
        return _script_cache[keys]
    ang, nprim, dtype, rys_type, n_dm, do_j, do_k = keys
    if dtype == np.float64:
        dtype_cuda = "double"
    elif dtype == np.float32:
        dtype_cuda = "float"
    else:
        raise RuntimeError("Data type is not supported")

    li, lj, lk, ll = ang
    npi, npj, npk, npl = nprim
    nroots = (li + lj + lk + ll) // 2 + 1
    const = f"""
typedef unsigned int uint32_t;
using DataType = {dtype_cuda};
constexpr int li = {li};
constexpr int lj = {lj};
constexpr int lk = {lk};
constexpr int ll = {ll};
constexpr int npi = {npi};
constexpr int npj = {npj};
constexpr int npk = {npk};
constexpr int npl = {npl};
constexpr int n_dm = {n_dm};
constexpr int rys_type = {rys_type};   // 0: omega = 0.0; -1: omega < 0.0; 1 omega > 0.0;
constexpr int do_j = {int(do_j)};
constexpr int do_k = {int(do_k)};
// Inject constants to match host-side layout
#define NPRIM_MAX {NPRIM_MAX}
// PRIM_STRIDE here matches host scalar stride; device uses prim_stride = PRIM_STRIDE/2
#define PRIM_STRIDE {PRIM_STRIDE}
#define COORD_STRIDE {COORD_STRIDE}

// for rys_roots
constexpr int nroots = ((li+lj+lk+ll)/2+1);
"""
    idx_script = generate_lookup_table(li, lj, lk, ll)
    script = (
        const + rys_roots_data[nroots] + rys_roots_code + idx_script + jk_2d_code
    )

    _script_cache[keys] = script
    return script

def gen_kernel(
    ang,
    nprim,
    dtype=np.double,
    n_dm=1,
    do_j=True,
    do_k=True,
    omega=None,
    print_log=False,
    use_cache=True,
):
    """
    Generate a 2D JK kernel for given angular momentum and primitives.

    Args:
        ang: Tuple of 4 integers (li, lj, lk, ll) representing angular momenta.
             Supports 0 (s) through 4 (g) for each shell.
        nprim: Tuple of 4 integers specifying number of primitives per shell
        dtype: Data type (np.float32 or np.float64)
        n_dm: Number of density matrices to process
        do_j: Whether to compute J matrix
        do_k: Whether to compute K matrix
        omega: Range-separation parameter (None for standard integrals)
        print_log: Print kernel info if True
        use_cache: Use cached kernels if available

    Returns:
        Tuple of (script, module, function) where function expects args:
            (nbas, nao, ao_loc, coords, ce_data, dm, vj, vk, omega,
             ij_pairs, n_ij_pairs, kl_pairs, n_kl_pairs)

    Raises:
        RuntimeError: If angular momentum exceeds maximum supported value (4)
                     or if omega value is negative (not supported)
    """
    # Validate angular momentum
    li, lj, lk, ll = ang
    max_l = 4  # Maximum angular momentum (g shell)
    if any(l < 0 or l > max_l for l in ang):
        raise ValueError(
            f"Angular momentum {ang} out of range. Supported: 0 (s) to {max_l} (g)"
        )
    if omega is None:
        rys_type = 0
    elif omega > 0:
        rys_type = 1
    elif omega < 0:
        rys_type = -1
        raise RuntimeError("Omega value is not supported")
    else:
        rys_type = 0

    keys = ang, nprim, dtype, rys_type, n_dm, do_j, do_k
    script = gen_code(keys)

    if not use_cache:
        # Generate a random number to change the cuda code,
        # such that the compiler will recompile the code
        import random

        x = random.random()
        script += f" \n#define RANDOM_NUMBER {x}"

    mod = cp.RawModule(code=script, options=compile_options)
    kernel_vj = mod.get_function("rys_vj_2d") if do_j else None
    kernel_vk = mod.get_function("rys_vk_2d") if do_k else None

    def fun(*args):
        # Args: (nbas, nao, ao_loc, coords, ce_data, dm, vj, vk, omega,
        #        ij_pairs, n_ij_pairs, kl_pairs, n_kl_pairs,
        #        q_cond_ij, q_cond_kl, log_cutoff)
        n_ij_pairs = args[10]
        n_kl_pairs = args[12]
        
        if do_j:
            # VJ: per-pair screening for both ij and kl dimensions
            vj_args = args[:7] + (args[8],) + args[9:]
            grid_vj = (n_ij_pairs * 16, (n_kl_pairs + 15) // 16)
            block_vj = (256,)
            kernel_vj(grid_vj, block_vj, vj_args)
        if do_k:
            # VK: per-pair screening with 2D thread indexing
            vk_args = args[:6] + (args[7], args[8]) + args[9:]
            grid_vk = (n_ij_pairs, n_kl_pairs)
            block_vk = THREADS
            kernel_vk(grid_vk, block_vk, vk_args)

    return script, mod, fun
