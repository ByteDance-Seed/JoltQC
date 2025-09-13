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
1q1t algorithm for JK calculations
'''

import warnings
import cupy as cp
import numpy as np
from jqc.backend.util import generate_lookup_table
from jqc.backend.cuda_scripts import rys_roots_data, rys_roots_code, jk_1q1t_cuda_code

__all__ = ['gen_jk_kernel']

THREADS = 256
compile_options = ('-std=c++17','--use_fast_math', '--minimal')

_script_cache = {}

def gen_code(keys):
    if keys in _script_cache:
        return _script_cache[keys]
    ang, nprim, dtype, rys_type, n_dm, do_j, do_k, nsq_per_block = keys
    if dtype == np.float64:
        dtype_cuda = 'double'
    elif dtype == np.float32:
        dtype_cuda = 'float'
    else:
        raise RuntimeError('Data type is not supported')

    li, lj, lk, ll = ang
    npi, npj, npk, npl = nprim
    nroots = (li+lj+lk+ll)//2 + 1
    const = f'''
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
constexpr int nsq_per_block = {nsq_per_block};

// for rys_roots
constexpr int nroots = ((li+lj+lk+ll)/2+1);
'''
    idx_script = generate_lookup_table(li,lj,lk,ll)
    script = const + rys_roots_data[nroots] \
        + rys_roots_code \
        + idx_script \
        + jk_1q1t_cuda_code
    
    _script_cache[keys] = script
    return script

def gen_kernel(ang, nprim, dtype=np.double, n_dm=1, do_j=True, do_k=True, 
               omega=None, print_log=False, use_cache=True):
    if omega is None:
        rys_type = 0
    elif omega > 0:
        rys_type = 1
    elif omega < 0:
        rys_type = -1
        raise RuntimeError('Omega value is not supported')
    else:
        rys_type = 0

    nsq_per_block = THREADS
    keys = ang, nprim, dtype, rys_type, n_dm, do_j, do_k, nsq_per_block
    script = gen_code(keys)

    if not use_cache:
        # Generate a random number to change the cuda code, 
        # such that the compiler will recompile the code
        import random
        x = random.random()
        script += f" \n#define RANDOM_NUMBER {x}"

    # For counting tasks
    shared_memory = THREADS * 4 

    mod = cp.RawModule(code=script, options=compile_options)
    kernel = mod.get_function('rys_jk')
    if kernel.local_size_bytes > 8192:
        msg = f'Local memory usage is high in 1q1t: {kernel.local_size_bytes} Bytes,'
        msg += f'    ang = {ang}, nprim = {nprim}, dtype = {dtype}, n_dm = {n_dm}'
        warnings.warn(msg)
    if print_log:
        li, lj, lk, ll = ang
        npi, npj, npk, npl = nprim
        fragi = (li+1)*(li+2)//2
        fragj = (lj+1)*(lj+2)//2
        fragk = (lk+1)*(lk+2)//2
        fragl = (ll+1)*(ll+2)//2
        print(f'Type: ({li}{lj}|{lk}{ll}), \
primitives: ({npi}{npj}|{npk}{npl}), \
algorithm: 1q1t, \
threads: ({nsq_per_block:3d},   1), \
frags: ({fragi:2d}, {fragj:2d}, {fragk:2d}, {fragl:2d}), \
shared memory: {shared_memory/1024:5.2f} KB, \
registers: {kernel.num_regs:3d}, \
local memory: {kernel.local_size_bytes:4d} Bytes')
    
    def fun(*args):
        ntasks = args[-1] # the last argument is ntasks
        blocks = (ntasks + nsq_per_block - 1) // nsq_per_block
        kernel(
            (blocks,), 
            (nsq_per_block, 1), 
            args, 
            shared_mem=shared_memory)

    return script, mod, fun
