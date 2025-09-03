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
Generate the task queue for JK calculations
The task is screened with Schwartz inequality and density matrix screening
'''

import cupy as cp
import numpy as np
from jqc.backend.cuda_scripts import screen_jk_tasks_code
from functools import lru_cache

THREADSX = 16
THREADSY = 16
MAX_PAIR_SIZE = 16384
QUEUE_DEPTH = MAX_PAIR_SIZE * MAX_PAIR_SIZE # 2 GB

compile_options = ('-std=c++17','--use_fast_math', '--minimal')

buf = cp.cuda.alloc_pinned_memory(4 * np.uint32().nbytes)
info_init = np.frombuffer(buf, dtype=np.uint32, count=4)
info_init[:] = (0, 0, QUEUE_DEPTH, QUEUE_DEPTH)

@lru_cache(maxsize=None)
def gen_screen_jk_tasks_kernel(do_j=True, do_k=True, omega=None, tile=2):
    if omega is None:
        rys_type = 0
    elif omega > 0:
        rys_type = 1
    elif omega < 0:
        rys_type = -1
    else:
        raise RuntimeError('Omega value is not supported yet')

    const = f'''
constexpr int do_j = {int(do_j)};
constexpr int do_k = {int(do_k)};
constexpr int rys_type = {rys_type};
constexpr int threadsx = {THREADSX};
constexpr int threadsy = {THREADSY};
constexpr int threads = {THREADSX*THREADSY};
constexpr int TILE = {tile};
    '''
    mod = cp.RawModule(code=const + screen_jk_tasks_code, options=compile_options)
    kernel = mod.get_function('screen_jk_tasks')
    print(kernel.num_regs)
    def fun(quartet_idx, info, nbas, 
            tile_ij_mapping, tile_kl_mapping, 
            q_cond, dm_cond, log_cutoff_a, log_cutoff_b):
        nt_ij = tile_ij_mapping.shape[0]
        nt_kl = tile_kl_mapping.shape[0]
        block_size_x = (nt_ij + THREADSX - 1) // THREADSX
        block_size_y = (nt_kl + THREADSY - 1) // THREADSY
        threads = (THREADSX,THREADSY)
        info[:].set(info_init)
        kernel(
            (block_size_x, block_size_y),
            threads,
            (quartet_idx, info, nbas, 
            tile_ij_mapping, tile_kl_mapping, 
            nt_ij, nt_kl,
            q_cond, dm_cond, 
            log_cutoff_a, log_cutoff_b)
        )
        return

    return screen_jk_tasks_code, kernel, fun
