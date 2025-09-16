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

import numpy as np
import cupy as cp
import pyscf
from pyscf import gto
from gpu4pyscf import scf, dft
from jqc.pyscf import rks

basis = gto.basis.parse(
    """
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
"""
)

# atom = 'molecules/h2o.xyz'
atom = "molecules/0031-irregular-nitrogenous.xyz"

n_warmup = 3
deriv = 1
xc = "b3lyp"
xctype = "GGA"
mol = pyscf.M(
    atom=atom, basis="def2-tzvpp", output="pyscf_test.log", verbose=4, spin=None, cart=1
)
mf = dft.KS(mol, xc=xc)
mf.grids.level = 1

dm = mf.get_init_guess()
# dm = cp.empty_like(dm0)
# dm[:] = dm0

dm = cp.ones_like(dm)

from jqc.pyscf.rks import build_grids
from types import MethodType

mf.grids.build = MethodType(build_grids, mf.grids)
grids = mf.grids
grids.build(with_non0tab=False, sort_grids=True)

ngrids = grids.coords.shape[0]
wv = cp.random.rand(4, ngrids)

ni = mf._numint
wv_pyscf = wv.copy()
wv_pyscf[0] *= 0.5
# warm up
for i in range(n_warmup):
    ao_gpu = ni.eval_ao(mol, grids.coords, deriv=1, transpose=False)
    aow = cp.einsum("nip,np->ip", ao_gpu, wv_pyscf)
    vxc_pyscf = ao_gpu[0].dot(aow.T)
    vxc_pyscf += vxc_pyscf.T
    aow = ao_gpu = None

start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
mol.verbose = 4
ao_gpu = ni.eval_ao(mol, grids.coords, deriv=1, transpose=False)
aow = cp.einsum("nip,np->ip", ao_gpu, wv_pyscf)
vxc_pyscf = ao_gpu[0].dot(aow.T)
vxc_pyscf += vxc_pyscf.T
mol.verbose = 4
end.record()
end.synchronize()
gpu4pyscf_time_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Time with GPU4PySCF, {gpu4pyscf_time_ms}")

###### JQC / FP64 #######
cutoff_a = np.log(1e-13)
cutoff_b = np.log(1e6)
_, rho_kern, vxc_kern = rks.generate_rks_kernel(
    mol, cutoff_fp64=1e-13, cutoff_fp32=1e-13
)
# Warm up
for i in range(n_warmup):
    vxc = vxc_kern(mol, grids, xctype, wv)
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
mol.verbose = 4
vxc = vxc_kern(mol, grids, xctype, wv)
mol.verbose = 4
end.record()
end.synchronize()
jqc_time_ms = cp.cuda.get_elapsed_time(start, end)

print("-------------------------------")
print("Benchmark with FP64")
print(f"Time with JQC, {jqc_time_ms}")
print(f"Speedup: {gpu4pyscf_time_ms/jqc_time_ms}")
vxc_diff = vxc_pyscf - vxc
print("vxc diff:", cp.linalg.norm(vxc_diff))

###### JQC / FP32 #######
cutoff_a = np.log(1e-13)
cutoff_b = np.log(1e6)
_, rho_kern, vxc_kern = rks.generate_rks_kernel(
    mol, cutoff_fp64=1e10, cutoff_fp32=1e-13
)
# Warm up
for i in range(n_warmup):
    vxc = vxc_kern(mol, grids, xctype, wv)
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
mol.verbose = 4
vxc = vxc_kern(mol, grids, xctype, wv)
mol.verbose = 4
end.record()
end.synchronize()
jqc_time_ms = cp.cuda.get_elapsed_time(start, end)

print("-------------------------------")
print("Benchmark with FP32")
print(f"Time with JQC, {jqc_time_ms}")
print(f"Speedup: {gpu4pyscf_time_ms/jqc_time_ms}")
vxc_diff = vxc_pyscf - vxc
print("vxc diff:", cp.linalg.norm(vxc_diff))

###### JQC / FP32 + FP64 #######
_, _, vxc_kern = rks.generate_rks_kernel(mol, cutoff_fp64=1e-7, cutoff_fp32=1e-13)
# Warm up
for i in range(n_warmup):
    vxc = vxc_kern(mol, grids, xctype, wv)
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()
mol.verbose = 4
vxc = vxc_kern(mol, grids, xctype, wv)
end.record()
end.synchronize()
jqc_time_ms = cp.cuda.get_elapsed_time(start, end)

print("-------------------------------")
print("Benchmark with FP32 + FP64")
print(f"Time with JQC, {jqc_time_ms}")
print(f"Speedup: {gpu4pyscf_time_ms/jqc_time_ms}")
vxc_diff = vxc_pyscf - vxc
print("vxc diff:", cp.linalg.norm(vxc_diff))
