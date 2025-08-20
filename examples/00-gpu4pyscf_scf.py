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
# This example shows how to use JK kernels with gpu4pyscf
##########################################################

import numpy as np
import pyscf
from gpu4pyscf import scf
from xqc.pyscf import jk

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-tzvpp')
mf = scf.RHF(mol)
mf.verbose = 1
mf.conv_tol = 1e-10
mf.max_cycle = 50

# Overwrite PySCF get_jk function
mf.get_jk = jk.generate_jk_kernel(mol) 
e_tot = mf.kernel()
print('total energy with double precision:', e_tot)

mol = pyscf.M(atom=atom, basis='def2-tzvpp')
mf = scf.RHF(mol)
mf.verbose = 1
mf.conv_tol = 1e-10
mf.max_cycle = 50

mf.get_jk = jk.generate_jk_kernel(mol)
e_tot = mf.kernel()
print('total energy with single precision:', e_tot)
