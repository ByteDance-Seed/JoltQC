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

import pyscf
from gpu4pyscf import scf
import jqc
import jqc.pyscf

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-tzvpp')
mf = scf.RHF(mol)
mf.conv_tol = 1e-10
mf.max_cycle = 50

# Method 1:
# Apply JIT to GPU4PySCF object (recommended)
print("Apply JIT to GPU4PySCF object")
mf = jqc.pyscf.apply(mf)
print("Run JIT GPU4PySCF object")
e_tot = mf.kernel()

# Convert GPU4PySCF object to PySCF object
print("Convert GPU4PySCF object to PySCF object")
mf_cpu = mf.to_cpu()
print("Run PySCF object")
e_tot = mf_cpu.kernel()

# Method 2:
# Apply JIT to PySCF object
print("Apply JIT to PySCF object")
mf = jqc.pyscf.compile(mf_cpu)
print("Run JIT PySCF object")
mf.kernel()
