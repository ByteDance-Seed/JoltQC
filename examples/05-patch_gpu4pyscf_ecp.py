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

#################################################################
# This example shows how to patch GPU4PySCF ECP functions with
# JoltQC implementations and run SCF, gradients, and Hessian
#
# ⚠️  WARNING: This is a temporary solution for demonstration purposes.
# ⚠️  A more maintainable interface through jqc.pyscf.apply() is not yet
# ⚠️  available for ECP calculations. This manual patching approach may
# ⚠️  break with GPU4PySCF updates and should be used with caution.
#################################################################

import pyscf
import importlib
from jqc.backend.ecp import get_ecp, get_ecp_ip, get_ecp_ipip

ecp_module = importlib.import_module("gpu4pyscf.gto.ecp")

# Replace module-level functions with JoltQC implementations
ecp_module.get_ecp = get_ecp
ecp_module.get_ecp_ip = get_ecp_ip
ecp_module.get_ecp_ipip = get_ecp_ipip

print("✓ Global monkey patch applied - all future imports will use JoltQC ECP implementations")

# Import GPU4PySCF modules
from gpu4pyscf import scf, grad, hessian

# CRITICAL: Patch the cached functions in consuming modules
print("⚠ Patching cached functions in GPU4PySCF modules...")
import gpu4pyscf.scf.hf as hf_module
import gpu4pyscf.grad.rhf as grad_rhf_module
import gpu4pyscf.hessian.rhf as hess_rhf_module

# Patch cached functions in each module
hf_module.get_ecp = get_ecp
grad_rhf_module.get_ecp_ip = get_ecp_ip
hess_rhf_module.get_ecp_ip = get_ecp_ip
hess_rhf_module.get_ecp_ipip = get_ecp_ipip

print("✓ Cached functions patched in all GPU4PySCF modules")

# Define a small molecule with ECP (Cu with crenbl ECP)
atom = """
Cu 0.0 0.0 0.0
H  1.5 0.0 0.0
"""

mol = pyscf.M(atom=atom, basis="sto-3g", ecp="crenbl", verbose=1)

# Run SCF calculation with patched ECP kernels
print("\n1. SCF Calculation:")
print("-" * 30)
mf = scf.RHF(mol)
e_scf = mf.kernel()
print(f"SCF Energy: {e_scf:.8f} Hartree")

# Calculate gradients
print("\n2. Nuclear Gradients:")
print("-" * 30)
g = grad.RHF(mf)
grad_result = g.kernel()
print("Gradients (Hartree/Bohr):")
for i, atom_name in enumerate(['Cu', 'H']):
    print(f"{atom_name:2s}: [{grad_result[i,0]:8.5f}, {grad_result[i,1]:8.5f}, {grad_result[i,2]:8.5f}]")

# Calculate Hessian
print("\n3. Hessian Calculation:")
print("-" * 30)
h = hessian.RHF(mf)
hess_result = h.kernel()
print("Hessian diagonal elements:")
coord_labels = ['Cu-x', 'Cu-y', 'Cu-z', 'H-x', 'H-y', 'H-z']
for i, label in enumerate(coord_labels):
    print(f"{label:4s}: {float(hess_result[i,i]):8.5f}")

print("\n" + "=" * 60)
print("ECP patching example completed successfully!")
print("=" * 60)

