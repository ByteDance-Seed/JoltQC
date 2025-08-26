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
import numpy as np
import pyscf
from pyscf import lib
from gpu4pyscf import dft
from xqc.pyscf import jk, rks
from xqc.pyscf.rks import build_grids

#atom = 'molecules/h2o.xyz'
#atom = 'molecules/0031-irregular-nitrogenous.xyz'
#atom = 'molecules/0051-elongated-halogenated.xyz'
atom = 'molecules/0084-elongated-halogenated.xyz'
basis = 'def2-tzvpp'
#xc = 'wb97m-v'
xc = 'b3lyp'
count = 1

lib.num_threads(8)

##################
# GPU4PySCF
##################

mol = pyscf.M(atom=atom, basis=basis, output=f'gpu4pyscf-{basis}.log', verbose=4)
#mol = pyscf.M(atom=atom, basis=basis, verbose=0)
mf = dft.RKS(mol, xc=xc)
mf.verbose = 4
mf.grids.atom_grid = (99, 590)
e_pyscf = mf.kernel()
dm_pyscf = mf.make_rdm1().get()
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
for i in range(count):
    mf = dft.RKS(mol, xc=xc)
    mf.verbose = 4
    mf.grids.atom_grid = (99, 590)
    e_tot = mf.kernel()
end.record()
end.synchronize()

elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with GPU4PySCF, {elapsed_time_ms/count} ms")
print(f'Total energy by GPU4PySCF, {e_tot}')
mf = None
cp.get_default_memory_pool().free_all_blocks()

#######################
# FP64 precision
#######################

mol = pyscf.M(atom=atom, basis=basis, output=f'xqc-{basis}-fp64.log', verbose=4)
#mol = pyscf.M(atom=atom, basis=basis, verbose=0)
mf_jit = dft.RKS(mol, xc=xc)
mf_jit.verbose = 4
mf_jit.grids.atom_grid = (99, 590)
mf_jit.grids.build = MethodType(build_grids, mf_jit.grids)
mf_jit._numint.get_rho = rks.generate_get_rho(mol)
nr_rks = rks.generate_nr_rks(mol)
mf_jit._numint.nr_rks = MethodType(nr_rks, mf_jit._numint)
mf_jit.get_jk = jk.generate_get_jk(mol)
mf_jit.get_j = jk.generate_get_j(mol)
e_tot = mf_jit.kernel()
dm_fp64 = mf_jit.make_rdm1().get()
print(f'Total energy by xQC (warmup), {e_tot}')

start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
for i in range(count):
    nr_rks = rks.generate_nr_rks(mol)
    mf_jit = dft.RKS(mol, xc=xc)
    mf_jit.verbose = 4
    mf_jit.grids.atom_grid = (99, 590)
    mf_jit.grids.build = MethodType(build_grids, mf_jit.grids)
    mf_jit._numint.nr_rks = MethodType(nr_rks, mf_jit._numint)
    mf_jit._numint.get_rho = rks.generate_get_rho(mol)
    mf_jit.get_jk = jk.generate_get_jk(mol)
    mf_jit.get_j = jk.generate_get_j(mol)
    e_fp64 = mf_jit.kernel()
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with xQC / FP64, {elapsed_time_ms/count} ms")
print(f'Total energy by xQC, {e_tot}')

#######################
# FP32 precision
#######################

mol = pyscf.M(atom=atom, basis=basis, output=f'xqc-{basis}-fp32.log', verbose=4)
#mol = pyscf.M(atom=atom, basis=basis, verbose=0)
mf_jit = dft.RKS(mol, xc=xc)
mf_jit.verbose = 4
mf_jit.grids.atom_grid = (99, 590)
mf_jit.grids.build = MethodType(build_grids, mf_jit.grids)
mf_jit._numint.get_rho = rks.generate_get_rho(mol)
nr_rks = rks.generate_nr_rks(mol, cutoff_fp64=1e-6, cutoff_fp32=1e-13)
mf_jit._numint.nr_rks = MethodType(nr_rks, mf_jit._numint)
mf_jit.get_jk = jk.generate_get_jk(mol, cutoff_fp64=1e100, cutoff_fp32=1e-13)
mf_jit.get_j = jk.generate_get_j(mol, cutoff_fp64=1e100, cutoff_fp32=1e-13)
e_tot = mf_jit.kernel()
dm_fp32 = mf_jit.make_rdm1().get()
print(f'Total energy by xQC (warmup), {e_tot}')

start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
for i in range(count):
    nr_rks = rks.generate_nr_rks(mol, cutoff_fp64=1e-6, cutoff_fp32=1e-13)
    mf_jit = dft.RKS(mol, xc=xc)
    mf_jit.verbose = 4
    mf_jit.grids.atom_grid = (99, 590)
    mf_jit.grids.build = MethodType(build_grids, mf_jit.grids)
    mf_jit._numint.nr_rks = MethodType(nr_rks, mf_jit._numint)
    mf_jit._numint.get_rho = rks.generate_get_rho(mol, cutoff_fp64=1e-6, cutoff_fp32=1e-13)
    mf_jit.get_jk = jk.generate_get_jk(mol, cutoff_fp64=1e100, cutoff_fp32=1e-13)
    mf_jit.get_j = jk.generate_get_j(mol, cutoff_fp64=1e100, cutoff_fp32=1e-13)
    e_fp32 = mf_jit.kernel()
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with xQC / FP32, {elapsed_time_ms/count} ms")
print(f'Total energy by xQC, {e_tot}')

##########################
# Mixed-precision
##########################

cp.get_default_memory_pool().free_all_blocks()
mol = pyscf.M(atom=atom, basis=basis, output=f'xqc-{basis}-fp32+fp64.log', verbose=4)
mf_jit = dft.RKS(mol, xc=xc)
mf_jit.verbose = 4
mf_jit.grids.atom_grid = (99, 590)
mf_jit.grids.build = MethodType(build_grids, mf_jit.grids)
mf_jit._numint.get_rho = rks.generate_get_rho(mol)
nr_rks = rks.generate_nr_rks(mol, cutoff_fp64=1e-6, cutoff_fp32=1e-13)
mf_jit._numint.nr_rks = MethodType(nr_rks, mf_jit._numint)
mf_jit.get_jk = jk.generate_get_jk(mol, cutoff_fp64=1e-6, cutoff_fp32=1e-13)
mf_jit.get_j = jk.generate_get_j(mol, cutoff_fp64=1e-6, cutoff_fp32=1e-13)
e_tot = mf_jit.kernel()
dm_mixed = mf_jit.make_rdm1().get()
print(f'Total energy by xQC (warmup), {e_tot}')

start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
for i in range(count):
    nr_rks = rks.generate_nr_rks(mol, cutoff_fp64=1e-6, cutoff_fp32=1e-13)
    mf_jit = dft.RKS(mol, xc=xc)
    mf_jit.verbose = 4
    mf_jit.grids.atom_grid = (99, 590)
    mf_jit.grids.build = MethodType(build_grids, mf_jit.grids)
    mf_jit._numint.nr_rks = MethodType(nr_rks, mf_jit._numint)
    mf_jit._numint.get_rho = rks.generate_get_rho(mol, cutoff_fp64=1e-6, cutoff_fp32=1e-13)
    mf_jit.get_jk = jk.generate_get_jk(mol, cutoff_fp64=1e-6, cutoff_fp32=1e-13)
    mf_jit.get_j = jk.generate_get_j(mol, cutoff_fp64=1e-6, cutoff_fp32=1e-13)
    e_mixed = mf_jit.kernel()
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with xQC / (FP32 + FP64), {elapsed_time_ms/count} ms")
print(f'Total energy by xQC, {e_tot}')

print('===== Error Summary =====')
print('e_pyscf - e_fp32', abs(e_pyscf - e_fp32))
print('e_pyscf - e_fp64', abs(e_pyscf - e_fp64))
print('e_pyscf - e_mixed', abs(e_pyscf - e_mixed))

print('dm_pyscf - dm_fp32', np.linalg.norm(dm_pyscf - dm_fp32))
print('dm_pyscf - dm_fp64', np.linalg.norm(dm_pyscf - dm_fp64))
print('dm_pyscf - dm_mixed', np.linalg.norm(dm_pyscf - dm_mixed))
