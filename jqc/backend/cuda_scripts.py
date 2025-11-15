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

from pathlib import Path

cuda_path = Path(__file__).resolve().parent

cuda_int_wrapper = """
#ifndef CUDA_INT_WRAPPER_H
#define CUDA_INT_WRAPPER_H

#include <cuda_runtime.h>

struct int2 {
    int x, y;
};

struct int3 {
    int x, y, z;
};

struct int4 {
    int x, y, z, w;
};

#endif // CUDA_INT_WRAPPER_H
"""

def _load_kernel(filename):
    with open(cuda_path / filename) as f:
        return f.read()

rys_roots_data = {}
for i in range(1, 10):
    rys_roots_data[i] = _load_kernel(f"rys/rys_root{i}.cu")

rys_roots_code = _load_kernel("rys/rys_roots.cu")
screen_jk_tasks_code = _load_kernel("jk/screen_jk_tasks.cu")
rys_roots_parallel_code = _load_kernel("rys/rys_roots_parallel.cu")

jk_1q1t_code = _load_kernel("jk/1q1t.cu")
jk_1qnt_code = _load_kernel("jk/1qnt.cu")
jk_2d_code = _load_kernel("jk/2d.cu")
jk_2d_vj_code = _load_kernel("jk/2d_vj.cu")
jk_2d_vk_code = _load_kernel("jk/2d_vk.cu")
