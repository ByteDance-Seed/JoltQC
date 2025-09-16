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
# This example shows mixed-precision algorithms
##########################################################


import numpy as np
import pyscf
from gpu4pyscf import scf
import jqc

atom = """
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
"""

mol = pyscf.M(atom=atom, basis="def2-tzvpp")
mf = scf.RHF(mol)

# Double precision algorithm (default)
mf = jqc.pyscf.apply(mf)
e_fp64 = mf.kernel()

# Single precision algorithm
mf = scf.RHF(mol)
mf = jqc.pyscf.apply(mf, cutoff_fp64=1e100)
e_fp32 = mf.kernel()

# Mixed precision algorithm
mf = scf.RHF(mol)
mf = jqc.pyscf.apply(mf, cutoff_fp64=1e-6)
e_mixed = mf.kernel()

print("Total energy with different precisions")
print(f"e_fp64 = {e_fp64:.12f}")
print(f"e_fp32 = {e_fp32:.12f}")
print(f"e_mixed = {e_mixed:.12f}")
