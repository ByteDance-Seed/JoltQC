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
from pyscf import lib
from gpu4pyscf.scf import hf
import jqc.pyscf

#atom = 'molecules/h2o.xyz'
atom = 'molecules/0031-irregular-nitrogenous.xyz'
#atom = 'molecules/0084-elongated-halogenated.xyz'
#atom = 'molecules/0112-elongated-nitrogenous.xyz'
#atom = 'molecules/0401-globular-nitrogenous.xyz'
basis = 'def2-tzvpp'#'6-31gs'
count = 1
verbose = 6

lib.num_threads(8)

mol = pyscf.M(atom=atom, basis=basis, output=f'gpu4pyscf_{basis}.log', verbose=verbose, cart=1)
mf = hf.RHF(mol)
mf.verbose = verbose
e_pyscf = mf.kernel()
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
for i in range(count):
    mf = hf.RHF(mol)
    mf.verbose = verbose
    e_tot = mf.kernel()
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with GPU4PySCF, {elapsed_time_ms/count:.3f} ms")
print(f"Total energy GPU4PySCF, {e_tot}")

mol = pyscf.M(atom=atom, basis=basis, output=f'jqc-{basis}-fp64.log', verbose=verbose, cart=1)
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
mf = hf.RHF(mol)
mf_jit = jqc.pyscf.apply(mf)
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print("------- Benchmark FP64 ---------")
print(f"Compilation time, {elapsed_time_ms/count:.3f} ms")
e_jqc = mf_jit.kernel()
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
for i in range(count):
    mf = hf.RHF(mol)
    mf_jit = jqc.pyscf.apply(mf)
    e_tot = mf_jit.kernel()
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with JQC, {elapsed_time_ms/count:.3f} ms")
print(f"Total energy by JQC / FP64, {e_tot}")

mol = pyscf.M(atom=atom, basis=basis, output=f'jqc-{basis}-fp32.log', verbose=verbose, cart=1)
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
mf = hf.RHF(mol)
mf_jit = jqc.pyscf.apply(mf, cutoff_fp32=1e-13, cutoff_fp64=1e100)
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print("------ Benchmark FP32 -------")
print(f"Compilation time, {elapsed_time_ms/count:.3f} ms")
e_jqc = mf_jit.kernel()
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
for i in range(count):
    mf = hf.RHF(mol)
    mf_jit = jqc.pyscf.apply(mf, cutoff_fp32=1e-13, cutoff_fp64=1e100)
    e_tot = mf_jit.kernel()
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with JQC, {elapsed_time_ms/count:.3f} ms")
print(f"Total energy by JQC / FP32, {e_tot}")
print(e_pyscf - e_jqc)
