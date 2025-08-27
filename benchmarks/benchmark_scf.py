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
import pyscf
from gpu4pyscf.scf import hf
from xqc.pyscf import jk

#atom = 'molecules/h2o.xyz'
#atom = 'molecules/0031-irregular-nitrogenous.xyz'
atom = 'molecules/0112-elongated-nitrogenous.xyz'
basis = 'def2-tzvpp'#'6-31gs'
count = 1

mol = pyscf.M(atom=atom, basis=basis, output=f'gpu4pyscf_{basis}.log', verbose=5, cart=1)
mf = hf.RHF(mol)
mf.verbose = 4
e_pyscf = mf.kernel()
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
for i in range(count):
    mf = hf.RHF(mol)
    mf.verbose = 4
    e_tot = mf.kernel()
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with GPU4PySCF, {elapsed_time_ms/count:.3f} ms")
print(e_tot)

mol = pyscf.M(atom=atom, basis=basis, output=f'xqc-{basis}-fp64.log', verbose=4, cart=1)
#mol = pyscf.M(atom=atom, basis=basis, verbose=0, cart=1)
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
get_jk = jk.generate_get_jk(mol, cutoff_fp64=1e-13, cutoff_fp32=1e-13)
get_j = jk.generate_get_j(mol, cutoff_fp64=1e-13, cutoff_fp32=1e-13)
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print("------- Benchmark FP64 ---------")
print(f"Compilation time, {elapsed_time_ms/count:.3f} ms")
mf_jit = hf.RHF(mol)
mf_jit.get_jk = get_jk # Overwrite PySCF get_jk function
mf_jit.get_j = get_j
e_xqc = mf_jit.kernel()
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
for i in range(count):
    get_jk = jk.generate_get_jk(mol, cutoff_fp64=1e-13, cutoff_fp32=1e-13)
    get_j = jk.generate_get_j(mol, cutoff_fp64=1e-13, cutoff_fp32=1e-13)
    mf_jit = hf.RHF(mol)
    mf_jit.get_jk = get_jk
    mf_jit.get_j = get_j
    e_tot = mf_jit.kernel()
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with xQC, {elapsed_time_ms/count:.3f} ms")
print(e_tot)
print(e_pyscf - e_xqc)

mol = pyscf.M(atom=atom, basis=basis, output=f'xqc-{basis}-fp32.log', verbose=4, cart=1)
#mol = pyscf.M(atom=atom, basis=basis, verbose=0, cart=1)
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
get_jk = jk.generate_get_jk(mol, cutoff_fp64=1e100, cutoff_fp32=1e-13)
get_j = jk.generate_get_j(mol, cutoff_fp64=1e100, cutoff_fp32=1e-13)
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print("------ Benchmark FP32 -------")
print(f"Compilation time, {elapsed_time_ms/count:.3f} ms")
mf_jit = hf.RHF(mol)
mf_jit.get_jk = get_jk # Overwrite PySCF get_jk function
mf_jit.get_j = get_j
e_xqc = mf_jit.kernel()
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
for i in range(count):
    get_jk = jk.generate_get_jk(mol, cutoff_fp64=1e100, cutoff_fp32=1e-13)
    get_j = jk.generate_get_j(mol, cutoff_fp64=1e100, cutoff_fp32=1e-13)
    mf_jit = hf.RHF(mol)
    #mf_jit.verbose = 4
    mf_jit.get_jk = get_jk
    mf_jit.get_j = get_j
    e_tot = mf_jit.kernel()
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with xQC, {elapsed_time_ms/count:.3f} ms")
print(e_tot)
print(e_pyscf - e_xqc)
