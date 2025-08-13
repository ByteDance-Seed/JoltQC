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

###########################################################
# This example shows how to use DFT kernels with GPU4PySCF
##########################################################

from functools import partial
import numpy as np
import pyscf
from gpu4pyscf import dft
from xqc.pyscf import rks, jk

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-tzvpp', verbose=4)
mf = dft.RKS(mol, xc='wb97m-v')

mf.grids.atom_grid = (99,590)
mf.conv_tol = 1e-10
mf.max_cycle = 50
e_pyscf = mf.kernel()

mol = pyscf.M(atom=atom, basis='def2-tzvpp', verbose=4)
mf = dft.RKS(mol, xc='wb97m-v')

mf.grids.atom_grid = (99,590)
mf.conv_tol = 1e-10
mf.max_cycle = 50

# Generate a Vxc function, which is compatiable with PySCF
nr_rks = rks.generate_nr_rks(mol)
from types import MethodType
mf._numint.nr_rks = MethodType(nr_rks, mf._numint)

# Generate a JK function, which is compatiable with PySCF
mf.get_jk = jk.generate_jk_kernel(mol)
e_xqc = mf.kernel()

print('total energy with pyscf:', e_pyscf)
print('total energy with xqc  :', e_xqc)
