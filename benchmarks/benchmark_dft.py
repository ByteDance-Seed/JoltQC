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
import numpy as np
import pyscf
from pyscf import lib
from gpu4pyscf import dft
import jqc.pyscf

# atom = 'molecules/h2o.xyz'
# atom = "molecules/0031-irregular-nitrogenous.xyz"
# atom = 'molecules/0051-elongated-halogenated.xyz'
# atom = 'molecules/0084-elongated-halogenated.xyz'
atom = 'molecules/0152-elongated-nitrogenous.xyz'
basis = "def2-tzvpp"
#xc = "wb97m-v"
xc = 'b3lyp'
count = 1
grids = (99, 590)

lib.num_threads(8)

##################
# GPU4PySCF
##################

verbose = 6

mol = pyscf.M(atom=atom, basis=basis, output=f"gpu4pyscf-{basis}.log", verbose=verbose)
mf = dft.RKS(mol, xc=xc)
mf.grids.atom_grid = grids
mf.nlcgrids.atom_grid = (50, 194)
e_pyscf = 0.0 #mf.kernel()
#dm_pyscf = mf.make_rdm1().get()
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
for i in range(count):
    mf = dft.RKS(mol, xc=xc)
    mf.verbose = verbose
    mf.grids.atom_grid = grids
    mf.nlcgrids.atom_grid = (50, 194)
    e_tot = 0 #mf.kernel()
end.record()
end.synchronize()

elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with GPU4PySCF, {elapsed_time_ms/count} ms")
print(f"Total energy by GPU4PySCF, {e_tot}")
mf = None

cp.get_default_memory_pool().free_all_blocks()

#######################
# FP64 precision
#######################
print("------- Benchmark FP64 ---------")
mol = pyscf.M(atom=atom, basis=basis, output=f"jqc-{basis}-fp64.log", verbose=verbose)
mf = dft.RKS(mol, xc=xc)
mf_jit = jqc.pyscf.apply(mf)
mf_jit.grids.atom_grid = grids
mf_jit.nlcgrids.atom_grid = (50, 194)
mf_jit.verbose = verbose
e_tot = mf_jit.kernel()
dm_fp64 = mf_jit.make_rdm1().get()
print(f"Total energy by JQC (warmup), {e_tot}")

start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
for i in range(count):
    mf = dft.RKS(mol, xc=xc)
    mf_jit = jqc.pyscf.apply(mf)
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time for compilation, {elapsed_time_ms/count} ms")

start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
for i in range(count):
    mf = dft.RKS(mol, xc=xc)
    mf_jit = jqc.pyscf.apply(mf)
    mf_jit.verbose = verbose
    mf_jit.grids.atom_grid = grids
    mf_jit.nlcgrids.atom_grid = (50, 194)
    e_fp64 = mf_jit.kernel()
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with JQC / FP64, {elapsed_time_ms/count} ms")
print(f"Total energy by JQC, {e_tot}")

#######################
# FP32 precision
#######################
print("------- Benchmark FP32 -----------")
mol = pyscf.M(atom=atom, basis=basis, output=f"jqc-{basis}-fp32.log", verbose=verbose)
mf = dft.RKS(mol, xc=xc)
config_fp32 = {"jk": {"cutoff_fp32": 1e-13, "cutoff_fp64": 1e100}, "dft": {"cutoff_fp32": 1e-13, "cutoff_fp64": 1e100}}
mf_jit = jqc.pyscf.apply(mf, config_fp32)
mf_jit.verbose = verbose
mf_jit.grids.atom_grid = grids
mf_jit.nlcgrids.atom_grid = (50, 194)
e_tot = mf_jit.kernel()
dm_fp32 = mf_jit.make_rdm1().get()
print(f"Total energy by JQC (warmup), {e_tot}")

start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
for i in range(count):
    mf = dft.RKS(mol, xc=xc)
    config_fp32 = {"jk": {"cutoff_fp32": 1e-13, "cutoff_fp64": 1e100}, "dft": {"cutoff_fp32": 1e-13, "cutoff_fp64": 1e100}}
    mf_jit = jqc.pyscf.apply(mf, config_fp32)
    mf_jit.verbose = verbose
    mf_jit.grids.atom_grid = grids
    mf_jit.nlcgrids.atom_grid = (50, 194)
    e_fp32 = mf_jit.kernel()
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with JQC / FP32, {elapsed_time_ms/count} ms")
print(f"Total energy by JQC, {e_tot}")

##########################
# Mixed-precision
##########################
print("------- Benchmark mixed-precision ----------")
cp.get_default_memory_pool().free_all_blocks()
mol = pyscf.M(
    atom=atom, basis=basis, output=f"jqc-{basis}-fp32+fp64.log", verbose=verbose
)
mf = dft.RKS(mol, xc=xc)
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
config_mixed = {"jk": {"cutoff_fp32": 1e-13, "cutoff_fp64": 1e-6}, "dft": {"cutoff_fp32": 1e-13, "cutoff_fp64": 1e-6}}
mf_jit = jqc.pyscf.apply(mf, config_mixed)
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time for compilation, {elapsed_time_ms/count} ms")

mf_jit.verbose = verbose
mf_jit.grids.atom_grid = grids
mf_jit.nlcgrids.atom_grid = (50, 194)
e_tot = mf_jit.kernel()
dm_mixed = mf_jit.make_rdm1().get()
print(f"Total energy by JQC (warmup), {e_tot}")

start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
for i in range(count):
    mf = dft.RKS(mol, xc=xc)
    config_mixed = {"jk": {"cutoff_fp32": 1e-13, "cutoff_fp64": 1e-6}, "dft": {"cutoff_fp32": 1e-13, "cutoff_fp64": 1e-6}}
    mf_jit = jqc.pyscf.apply(mf, config_mixed)
    mf_jit.verbose = verbose
    mf_jit.grids.atom_grid = grids
    mf_jit.nlcgrids.atom_grid = (50, 194)
    e_mixed = mf_jit.kernel()
end.record()
end.synchronize()
elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with JQC / (FP32 + FP64), {elapsed_time_ms/count} ms")
print(f"Total energy by JQC, {e_tot}")

print("===== Error Summary =====")
print("|e_pyscf - e_fp32|  = ", abs(e_pyscf - e_fp32))
print("|e_pyscf - e_fp64|  = ", abs(e_pyscf - e_fp64))
print("|e_pyscf - e_mixed| = ", abs(e_pyscf - e_mixed))

print("||dm_pyscf - dm_fp32||  = ", np.linalg.norm(dm_pyscf - dm_fp32))
print("||dm_pyscf - dm_fp64||  = ", np.linalg.norm(dm_pyscf - dm_fp64))
print("||dm_pyscf - dm_mixed|| = ", np.linalg.norm(dm_pyscf - dm_mixed))
