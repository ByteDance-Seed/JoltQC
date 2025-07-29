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
from xqc.pyscf import jk

basis = gto.basis.parse('''
#H   P
#      0.1553961625E+02      -0.1107775495E+00       0.7087426823E-01
#      0.3599933586E+01      -0.1480262627E+00       0.3397528391E+00
#      0.1013761750E+01       0.1130767015E+01       0.7271585773E+00
#    H    P
#        1.8000000              1.0000000
#    H   D
#      0.5484671660E+04       0.1831074430E-02
#      0.8252349460E+03       0.1395017220E-01
#      0.1880469580E+03       0.6844507810E-01
#      0.5296450000E+02       0.2327143360E+00
#      0.1689757040E+02       0.4701928980E+00
#      0.5799635340E+01       0.3585208530E+00
#H    SP
#      0.2700058226E+00       0.1000000000E+01       0.1000000000E+01
#    H    S
#      0.1553961625E+02      -0.1107775495E+00       0.7087426823E-01
#      0.3599933586E+01      -0.1480262627E+00       0.3397528391E+00
#      0.1013761750E+01       0.1130767015E+01       0.7271585773E+00
#    H    P
#      0.1553961625E+02      -0.1107775495E+00       0.7087426823E-01
#      0.3599933586E+01      -0.1480262627E+00       0.3397528391E+00
#      0.1013761750E+01       0.1130767015E+01       0.7271585773E+00
#O    S
#      0.1553961625E+02      0.2
#      0.3599933586E+01      0.2
#      0.1013761750E+01      0.2
#O    P
#      0.1553961625E+02      0.2
#      0.3599933586E+01      0.2
#      0.1013761750E+01      0.2
#O    SP
#      0.1553961625E+02      -0.1107775495E+00       0.7087426823E-01
#      0.3599933586E+01      -0.1480262627E+00       0.3397528391E+00
#      0.1013761750E+01       0.1130767015E+01       0.7271585773E+00
#H    S
#      0.2700058226E+00      1
#      0.2700058226E+00      1
#O    P
#      0.2700058226E+00      1
O    F
      0.2700058226E+00      1
#      0.2700058226E+00      1 
#      0.2700058226E+00      1
#      0.2700058226E+00      1
#      0.2700058226E+00      1
#      0.2700058226E+00      1
''')

#atom = 'molecules/h2o.xyz'
atom = 'molecules/gly30.xyz'
#atom = 'molecules/ubiquitin.xyz'
#atom = 'molecules/020_Vitamin_C.xyz'
#atom = 'molecules/052_Cetirizine.xyz'

n_dm = 1
n_warmup = 3
mol = pyscf.M(atom=atom,
              basis=basis,#'def2-tzvpp',#'6-31g', 
              output='pyscf_test.log',
              verbose=4,
              spin=None)

mf = dft.KS(mol, xc='b3lyp')

dm0 = mf.get_init_guess()
dm = cp.ones_like(dm0)

dm = cp.expand_dims(dm, axis=0)
dm = cp.repeat(dm, repeats=n_dm, axis=0)

vhfopt = scf.jk._VHFOpt(mol).build()
# warm up
for i in range(n_warmup):
    vj, vk = scf.jk.get_jk(mol, dm, hermi=1, vhfopt=vhfopt)

start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
mol.verbose = 4
vj, vk = scf.jk.get_jk(mol, dm, hermi=1, vhfopt=vhfopt)
mol.verbose = 4
end.record()
end.synchronize()
gpu4pyscf_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with GPU4PySCF, {gpu4pyscf_time_ms}")

###### xQC / FP64 #######
get_jk = jk.generate_jk_kernel(dtype=cp.float64)
# Warm up
for i in range(n_warmup):
    vj_jit, vk_jit = get_jk(mol, dm, hermi=1)
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
mol.verbose = 4
vj_jit, vk_jit = get_jk(mol, dm, hermi=1)
mol.verbose = 4
end.record()
end.synchronize()
xqc_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with xQC, {xqc_time_ms}")
print(f"Speedup: {gpu4pyscf_time_ms/xqc_time_ms}")
print('vj diff:', cp.linalg.norm(vj - vj_jit))
print('vk diff:', cp.linalg.norm(vk - vk_jit))

###### xQC / FP32 #######
get_jk = jk.generate_jk_kernel(dtype=cp.float32)
# Warm up
for i in range(n_warmup):
    vj_jit, vk_jit = get_jk(mol, dm, hermi=1)
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
mol.verbose = 4
vj_jit, vk_jit = get_jk(mol, dm, hermi=1)
mol.verbose = 4
end.record()
end.synchronize()
xqc_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with xQC, {xqc_time_ms}")
print(f"Speedup: {gpu4pyscf_time_ms/xqc_time_ms}")
print('vj diff:', cp.linalg.norm(vj - vj_jit))
print('vk diff:', cp.linalg.norm(vk - vk_jit))
