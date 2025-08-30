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


##########################################################
# This example shows how to 
#    - generate a specific JK kernel
#    - dump the complete cuda code for specific kernel
#    - inspect .ptx of the generated kernel 
##########################################################

import time
import numpy as np
from jqc.backend.jk_1qnt import gen_kernel
#from jqc.backend.jk_1q1t import gen_kernel

# angular momentum
li, lj, lk, ll = 0, 0, 0, 0

# number of primitives
npi, npj, npk, npl = 1, 1, 1, 1

# fragmentation for 1QnT algorithm
frags = (10,10,1,1)

total_time = 0
start = time.perf_counter()
code, mod, fun = gen_kernel(
    (li,lj,lk,ll),
    (npi,npj,npk,npl),
    print_log=True,
    #frags=frags,
    dtype=np.float32)
end = time.perf_counter()
wall_time = (end - start) * 1000
total_time += wall_time
print(f'Compile jk_1qnt kernel for ({li},{lj},{lk},{ll}) takes {wall_time:.2f} ms')

# Dump cuda code including data
with open('tmp.cu', 'w+') as f:
    f.write(code)

# Compile the kernel with nvcc, and generate .ptx file
import cupy
import subprocess
cmd = cupy.cuda.get_nvcc_path().split()
cmd += ['-lineinfo', '-arch=sm_80', '-src-in-ptx', '-ptx', 'tmp.cu', '-o', 'tmp.ptx'] 
print(f"running {' '.join(cmd)}")
subprocess.run(cmd, capture_output=True, text=True)

############################ 
# DFT kernels
############################

from jqc.backend.rks import gen_rho_kernel
code, mod, fun = gen_rho_kernel((li,lj), (npi,npj), np.float32)
with open('tmp_rho.cu', 'w+') as f:
    f.write(code)
import cupy
import subprocess
cmd = cupy.cuda.get_nvcc_path().split()
cmd += ['-lineinfo', '-arch=sm_80', '-src-in-ptx', '-ptx', 'tmp_rho.cu', '-o', 'tmp_rho.ptx'] 
print(f"running {' '.join(cmd)}")
subprocess.run(cmd, capture_output=True, text=True)
