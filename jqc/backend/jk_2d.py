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
2D algorithm for JK calculations
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
    kernel = mod.get_function("rys_jk_2d")

    def fun(*args):
        n_ij_pairs = args[-3]
        n_kl_pairs = args[-1]
        grid_x = (n_ij_pairs + THREADS[0] - 1) // THREADS[0]
        grid_y = (n_kl_pairs + THREADS[1] - 1) // THREADS[1]
        kernel((grid_x, grid_y), THREADS, args)

    return script, mod, fun
