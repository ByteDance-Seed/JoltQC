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

"""
Simple test script for 2D JK algorithm using high-level JQC interface.

Usage:
    python test_2d_simple.py <l> <precision>

Example:
    python test_2d_simple.py 0 fp64   # s shells
    python test_2d_simple.py 1 fp32   # p shells
    python test_2d_simple.py 2 fp64   # d shells
"""

import sys
import cupy as cp
import numpy as np
from pyscf import gto, scf
from gpu4pyscf import scf as gpu_scf

import jqc.pyscf


def generate_basis_for_angular_momentum(l):
    """Generate a custom basis set with specified angular momentum."""
    shell_types = ['S', 'P', 'D', 'F', 'G']
    shell_type = shell_types[l]

    if l == 0:  # s shell
        exponents = [1.27, 0.27, 0.027]
    elif l == 1:  # p shell
        exponents = [2.5, 0.6, 0.15]
    elif l == 2:  # d shell
        exponents = [3.5, 0.8, 0.2]
    elif l == 3:  # f shell
        exponents = [4.5, 1.0, 0.25]
    else:  # g shell (l=4)
        exponents = [5.5, 1.2, 0.3]

    basis_lines = [f"H    {shell_type}"]
    for exp in exponents:
        basis_lines.append(f"      {exp:.10f}      1")

    return "\n".join(basis_lines)


def test_angular_momentum(l, dtype):
    """Test JK computation for given angular momentum."""
    shell_names = ['s', 'p', 'd', 'f', 'g']
    print(f"\nTesting {shell_names[l]}-shell (l={l}) with {dtype.__name__}")
    print("=" * 60)

    # Setup molecule with custom basis
    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 1.1"
    mol.unit = "B"
    basis_str = generate_basis_for_angular_momentum(l)
    mol.basis = gto.basis.parse(basis_str)
    mol.build()

    print(f"  nao: {mol.nao}, nbas: {mol.nbas}")

    # Run standard SCF as reference
    mf_cpu = scf.RHF(mol).run(verbose=0)
    print(f"  CPU SCF energy: {mf_cpu.e_tot:.10f}")

    # Run GPU SCF with JQC (let it choose algorithm)
    mf_gpu = gpu_scf.RHF(mol)
    mf_gpu = jqc.pyscf.apply(mf_gpu)
    mf_gpu.verbose = 0
    mf_gpu.kernel()

    print(f"  GPU SCF energy: {mf_gpu.e_tot:.10f}")

    # Compare energies
    energy_diff = abs(mf_gpu.e_tot - mf_cpu.e_tot)
    print(f"  Energy diff: {energy_diff:.2e}")

    tolerance = 1e-8 if dtype == np.float64 else 1e-5
    if energy_diff < tolerance:
        print(f"  ✓ PASSED (tolerance: {tolerance:.2e})")
        return True
    else:
        print(f"  ✗ FAILED (tolerance: {tolerance:.2e})")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        print("\nError: Expected 2 arguments (l, precision)")
        print("Example: python test_2d_simple.py 0 fp64")
        sys.exit(1)

    try:
        l = int(sys.argv[1])
        if l < 0 or l > 4:
            print(f"Error: Angular momentum {l} out of range [0, 4]")
            sys.exit(1)

        dtype = np.float32 if sys.argv[2] == "fp32" else np.float64

        success = test_angular_momentum(l, dtype)
        sys.exit(0 if success else 1)

    except ValueError as e:
        print(f"Error: Invalid arguments")
        print(__doc__)
        sys.exit(1)
