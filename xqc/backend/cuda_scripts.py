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
code_path = Path(__file__).resolve().parent

rys_roots_data = {}
for i in range(1,10):
    with open(f'{code_path}/rys/rys_root{i}.cu', 'r') as f:
        rys_roots_data[i] = f.read()

with open(f'{code_path}/rys/rys_roots.cu', 'r') as f:
    rys_roots_code = f.read()

with open(f'{code_path}/cuda/screen_jk_tasks.cu', 'r') as f:
    screen_jk_tasks_code = f.read()

with open(f'{code_path}/rys/rys_roots_parallel.cu', 'r') as f:
    rys_roots_parallel_code = f.read()

with open(f'{code_path}/cuda/1q1t.cu', 'r') as f:
    jk_1q1t_cuda_code = f.read()

with open(f'{code_path}/cuda/1qnt.cu', 'r') as f:
    jk_1qnt_cuda_code = f.read()

