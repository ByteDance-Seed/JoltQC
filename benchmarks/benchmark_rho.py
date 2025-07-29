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
from pyscf import gto
from gpu4pyscf import scf, dft
from xqc.pyscf import rks
from xqc.pyscf.rks import build_grids
from types import MethodType

basis = gto.basis.parse('''
#O    S
#      0.5484671660E+04       0.1831074430E-02
#      0.8252349460E+03       0.1395017220E-01
#      0.1880469580E+03       0.6844507810E-01
#      0.5296450000E+02       0.2327143360E+00
#      0.1689757040E+02       0.4701928980E+00
#      0.5799635340E+01       0.3585208530E+00
#O    SP
#      0.1553961625E+02      -0.1107775495E+00       0.7087426823E-01
#      0.3599933586E+01      -0.1480262627E+00       0.3397528391E+00
#      0.1013761750E+01       0.1130767015E+01       0.7271585773E+00
O    D
      0.2700058226E+00       0.1000000000E+01
''')
#atom = 'molecules/h2o.xyz'
#atom = 'molecules/gly30.xyz'
#atom = 'molecules/ubiquitin.xyz'
atom = 'molecules/020_Vitamin_C.xyz'
#atom = 'molecules/052_Cetirizine.xyz'

n_warmup = 3
deriv = 1
xctype = 'GGA'
mol = pyscf.M(atom=atom,
              basis='def2-qzvpp', 
              output='pyscf_test.log',
              verbose=4,
              spin=None,
              cart=1)
mf = dft.KS(mol, xc='b3lyp')
mf.grids.level = 2
#mf.grids.atom_grid = (99, 590)

dm = mf.get_init_guess()

dm = cp.ones_like(dm)
mf.grids.build = MethodType(build_grids, mf.grids)
grids = mf.grids
grids.build(sort_grids=True)
ni = mf._numint

# warm up
for i in range(n_warmup):
    ao_gpu = ni.eval_ao(mol, grids.coords, deriv=1, transpose=False)
    rho_pyscf = ni.eval_rho(mol, ao_gpu, dm, xctype=xctype)

start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
mol.verbose = 4
ao_gpu = ni.eval_ao(mol, grids.coords, deriv=1, transpose=False)
rho_pyscf = ni.eval_rho(mol, ao_gpu, dm, xctype=xctype)
mol.verbose = 4
end.record()
end.synchronize()
gpu4pyscf_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with GPU4PySCF, {gpu4pyscf_time_ms}")
#gpu4pyscf_time_ms = 0.0
#rho_pyscf = 0

###### xQC / FP64 #######
rho_kern, vxc_kern = rks.generate_dft_kernel(mol, dtype=cp.float64, xc_type=xctype)
# Warm up
for i in range(n_warmup):
    rho = rho_kern(None, mol, dm, grids)
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
mol.verbose = 4
rho_xqc = rho_kern(None, mol, dm, grids)
mol.verbose = 4
end.record()
end.synchronize()
xqc_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with xQC, {xqc_time_ms}")
print(f"Speedup: {gpu4pyscf_time_ms/xqc_time_ms}")
rho_diff = rho_pyscf - rho_xqc
print('rho[0] diff:', cp.linalg.norm(rho_diff[0]))
print('rho[1:4] diff:', cp.linalg.norm(rho_diff[1:]))

###### xQC / FP32 #######
rho_kern, vxc_kern = rks.generate_dft_kernel(mol, dtype=cp.float32, xc_type=xctype)
# Warm up
for i in range(n_warmup):
    rho_xqc = rho_kern(None, mol, dm, grids)
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
mol.verbose = 4
rho_xqc = rho_kern(None, mol, dm, grids)
mol.verbose = 4
end.record()
end.synchronize()
xqc_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with xQC, {xqc_time_ms}")
print(f"Speedup: {gpu4pyscf_time_ms/xqc_time_ms}")
rho_diff = rho_pyscf - rho_xqc
print('rho[0] diff:', cp.linalg.norm(rho_diff[0]))
print('rho[1] diff:', cp.linalg.norm(rho_diff[1]))
print('rho[2] diff:', cp.linalg.norm(rho_diff[2]))
print('rho[3] diff:', cp.linalg.norm(rho_diff[3]))

