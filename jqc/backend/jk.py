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
Generate JK kernel, wrap up 1q1t and 1qnt algorithms
'''

import os
import threading
import numpy as np
import json
import cupy as cp
from pathlib import Path
from functools import lru_cache
from jqc.backend.jk_1q1t import gen_kernel as jk_1q1t_kernel
from jqc.backend.jk_1qnt import gen_kernel as jk_1qnt_kernel

device_id = cp.cuda.Device().id
props = cp.cuda.runtime.getDeviceProperties(device_id)
device_name = props['name'].decode()

__all__ = ['gen_jk_kernel']

# Load fragmentation scheme for 1qnt, support FP32 and FP64 only
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, f'data/optimal_scheme_{device_name}_fp32.json')
if not Path(file_path).exists():
    file_path = os.path.join(script_dir, 'data/optimal_scheme_NVIDIA A10_fp32.json')

with open(file_path, 'r') as f:
    frags_fp32 = json.load(f)

file_path = os.path.join(script_dir, f'data/optimal_scheme_{device_name}_fp64.json')
if not Path(file_path).exists():
    file_path = os.path.join(script_dir, 'data/optimal_scheme_NVIDIA A100-SXM4-80GB_fp64.json')
    
with open(file_path, 'r') as f:
    frags_fp64 = json.load(f)

@lru_cache(maxsize=None)
def gen_jk_kernel(angulars, nprimitives, dtype=np.double,
                  n_dm=1, do_j=True, do_k=True, omega=None,
                  frags=None, print_log=False):
    """ Router function for generating JK kernels. 
    If frags = [-1]:      use 1q1t algorithm
    If frags = [x,x,x,x]: use 1qnt algorithm
    """
    
    if frags is None:
        li, lj, lk, ll = angulars
        ijkl_str = key = str(li*1000 + lj*100 + lk*10 + ll)
        if dtype == np.double:
            frags = frags_fp64
        elif dtype == np.float32:
            frags = frags_fp32
        else:
            raise RuntimeError('Data type is not supported')
        if ijkl_str in frags:
            opt_frags = np.array(frags[ijkl_str])
        else:
            raise RuntimeError(f'Optimal scheme for {ijkl_str} is not found')
    else:
        opt_frags = frags
    
    if opt_frags[0] == -1:
        _, _, fun = jk_1q1t_kernel(
            angulars, nprimitives, dtype=dtype, n_dm=n_dm, 
            do_j=do_j, do_k=do_k, omega=omega, print_log=print_log)
    else:
        _, _, fun = jk_1qnt_kernel(
            angulars, nprimitives, frags=opt_frags, dtype=dtype, n_dm=n_dm,
            do_j=do_j, do_k=do_k, omega=omega, print_log=print_log)
    
    return fun

if __name__ == "__main__":
    import time
    from itertools import product
    total_time = 0
    #for li, lj, lk, ll in [[2,2,2,2]]:
    li = lj = lk = ll = 3
    ijkl_str = key = str(li*1000 + lj*100 + lk*10 + ll)
    opt_frags = np.array(frags_fp32[ijkl_str])
    start = time.perf_counter()
    for i in range(10):
        code, kernel, fun = jk_1qnt_kernel(
            (li,lj,lk,ll),
            (1,1,1,1),
            frags=opt_frags,
            dtype=np.float32,
            use_cache=False)
    end = time.perf_counter()
    wall_time = (end - start) * 1000
    print(f'({li},{lj},{lk},{ll})', wall_time)
    total_time += wall_time
    print('total time:', total_time/10)

    with open('tmp.cu', 'w+') as f:
        f.write(code)

    import cupy
    import subprocess
    from cupy.cuda import compiler
    cmd = cupy.cuda.get_nvcc_path().split()
    cmd += ['-arch=sm_80', '-ptx', 'tmp.cu', '-o', 'tmp.ptx'] 
    print(cmd)
    subprocess.run(cmd, capture_output=True, text=True)
    