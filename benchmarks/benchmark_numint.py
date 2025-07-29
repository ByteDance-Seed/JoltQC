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

from types import MethodType
import cupy as cp
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.scf import hf
from xqc.pyscf import jk, rks
from xqc.pyscf.rks import build_grids

#atom = 'molecules/h2o.xyz'
atom = 'molecules/020_Vitamin_C.xyz'
#atom = 'molecules/052_Cetirizine.xyz'
#atom = 'molecules/valinomycin.xyz'
#atom = 'molecules/gly30.xyz'
basis = 'def2-tzvpp'
xc = 'wb97m-v'
xctype = 'mGGA'
count = 1

mol = pyscf.M(atom=atom, basis=basis, output=f'gpu4pyscf-{basis}.log', verbose=5, cart=1)
mf = dft.RKS(mol, xc=xc).density_fit()
mf.verbose = 4
e_tot = mf.kernel()
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
for i in range(count):
    mf = dft.RKS(mol, xc=xc).density_fit()
    mf.verbose = 6
    mf.grids.atom_grid = (99, 590)
    e_tot = mf.kernel()
end.record()
end.synchronize()

elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with GPU4PySCF, {elapsed_time_ms/count} ms")
print(f'Total energy by GPU4PySCF, {e_tot}')
mf = None

mol = pyscf.M(atom=atom, basis=basis, output=f'xqc-{basis}.log', verbose=4, cart=1)
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
nr_rks = rks.generate_rks_kernel(mol, dtype=cp.float64, xc_type=xctype)
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Compilation time, {elapsed_time_ms} ms")

mf_jit = dft.RKS(mol, xc=xc).density_fit()
mf_jit.verbose = 4
mf_jit.grids.atom_grid = (99, 590)
mf_jit.grids.build = MethodType(build_grids, mf_jit.grids)
mf_jit._numint.nr_rks = MethodType(nr_rks, mf_jit._numint)
e_tot = mf_jit.kernel()
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
for i in range(count):
    nr_rks = rks.generate_rks_kernel(mol, dtype=cp.float64, xc_type=xctype)
    mf_jit = dft.RKS(mol, xc=xc).density_fit()
    mf_jit.verbose = 6
    mf_jit.grids.atom_grid = (99, 590)
    mf_jit.grids.build = MethodType(build_grids, mf_jit.grids)
    mf_jit._numint.nr_rks = MethodType(nr_rks, mf_jit._numint)
    e_tot = mf_jit.kernel()
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with xQC, {elapsed_time_ms/count} ms")
print(f'Total energy by xQC, {e_tot}')
