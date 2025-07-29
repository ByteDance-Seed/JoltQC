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


import cupy as cp
from xqc.backend.cuda_scripts import fill_tasks_code
from functools import lru_cache

THREADSX = 16
THREADSY = 16

compile_options = ('-std=c++17','--use_fast_math')

@lru_cache(maxsize=2048)
def generate_fill_tasks_kernel(do_j=True, do_k=True, omega=None, tile=2):
    if omega is None:
        rys_type = 0
    elif omega > 0:
        rys_type = 1
    elif omega < 0:
        rys_type = -1
    else:
        raise RuntimeError('Omega value is not supported yet')

    macros = f'''
constexpr int do_j = {int(do_j)};
constexpr int do_k = {int(do_k)};
constexpr int rys_type = {rys_type};
constexpr int threadsx = {THREADSX};
constexpr int threadsy = {THREADSY};
constexpr int threads = {THREADSX*THREADSY};
constexpr int TILE = {tile};
    '''
    mod = cp.RawModule(code=macros+fill_tasks_code, options=compile_options)
    kernel = mod.get_function('fill_jk_tasks')

    def fun(quartet_idx, info, nbas, 
            tile_ij_mapping, tile_kl_mapping, 
            q_cond, dm_cond, log_cutoff_a, log_cutoff_b):
        nt_ij = tile_ij_mapping.shape[0]
        nt_kl = tile_kl_mapping.shape[0]
        block_size_x = (nt_ij + THREADSX - 1) // THREADSX
        block_size_y = (nt_kl + THREADSY - 1) // THREADSY
        threads = (THREADSX,THREADSY)
        info[:] = 0

        try:
            kernel(
                (block_size_x, block_size_y),
                threads,
                (quartet_idx, info, nbas, 
                tile_ij_mapping, tile_kl_mapping, 
                nt_ij, nt_kl,
                q_cond, dm_cond, 
                log_cutoff_a, log_cutoff_b)
            )
        except cp.cuda.runtime.CUDARuntimeError as e:
            print("CUDA Runtime Error in the task generation kernel:", e)
        return

    return fill_tasks_code, kernel, fun
