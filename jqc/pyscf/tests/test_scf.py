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

import unittest

import numpy as np
import pyscf
from gpu4pyscf import scf

import jqc.pyscf

atom = """
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
"""
bas = "def2-tzvpp"


def setUpModule():
    global mol_sph, mol_cart
    mol_sph = pyscf.M(
        atom=atom, basis=bas, max_memory=32000, verbose=1, output="/dev/null"
    )

    mol_cart = pyscf.M(
        atom=atom, basis=bas, max_memory=32000, verbose=1, cart=1, output="/dev/null"
    )


def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart


def run_scf(mol, cutoff_fp32=None, cutoff_fp64=None):
    mf = scf.RHF(mol)

    # Use JoltQC apply to wire in JK paths consistently
    if cutoff_fp32 is not None or cutoff_fp64 is not None:
        config = {
            "jk": {"cutoff_fp32": cutoff_fp32, "cutoff_fp64": cutoff_fp64},
        }
        mf = jqc.pyscf.apply(mf, config)
    else:
        mf = jqc.pyscf.apply(mf)

    e_scf = mf.kernel()
    return e_scf


class KnownValues(unittest.TestCase):
    def test_rhf_spherical(self):
        """Test RHF with spherical basis (default FP64)"""
        e_tot = run_scf(mol_sph)
        e_ref = -76.0624634523
        print("| CPU - GPU | RHF spherical:", e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rhf_cartesian(self):
        """Test RHF with cartesian basis (default FP64)"""
        e_tot = run_scf(mol_cart)
        e_ref = -76.0627443874
        print("| CPU - GPU | RHF cartesian:", e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rhf_apply(self):
        """Test that apply() gives same result as GPU4PySCF"""
        mf = scf.RHF(mol_sph)
        mf = jqc.pyscf.apply(mf)
        e_tot = mf.kernel()
        e_ref = scf.RHF(mol_sph).kernel()
        print("| CPU - GPU | with apply:", e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rhf_qz(self):
        """Test RHF with larger basis set (def2-qzvpp)"""
        mol = pyscf.M(
            atom="""
            H  -0.757    4.   -0.4696
            H   0.757    4.   -0.4696
            """,
            basis="def2-qzvpp",
            unit="B",
            cart=0,
            output="/dev/null",
        )
        mf = scf.RHF(mol)
        mf = jqc.pyscf.apply(mf)
        e_tot = mf.kernel()
        e_ref = scf.RHF(mol).kernel()
        mol.stdout.close()
        print("| CPU - GPU | with qz:", e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-8


class FP32Precision(unittest.TestCase):
    def test_fp32_spherical(self):
        """Test RHF with FP32 precision (spherical basis)"""
        # Force FP32 by setting cutoff_fp64 very high
        cutoff_fp32 = 1e-13
        cutoff_fp64 = 1e100  # Effectively disables FP64
        e_tot = run_scf(mol_sph, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64)
        e_ref = run_scf(mol_sph)
        print(f"| FP64 - FP32 | RHF spherical:", e_tot - e_ref)
        # FP32 should be less accurate than FP64
        assert np.abs(e_tot - e_ref) < 1e-4

    def test_fp32_cartesian(self):
        """Test RHF with FP32 precision (cartesian basis)"""
        cutoff_fp32 = 1e-13
        cutoff_fp64 = 1e100
        e_tot = run_scf(mol_cart, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64)
        e_ref = run_scf(mol_cart)
        print(f"| FP64 - FP32 | RHF cartesian:", e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-4


class MixedPrecision(unittest.TestCase):
    def test_mixed_precision_spherical(self):
        """Test RHF with mixed FP32/FP64 precision (spherical basis)"""
        cutoff_fp32 = 1e-13
        cutoff_fp64 = 1e-7
        e_tot = run_scf(mol_sph, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64)
        e_ref = run_scf(mol_sph)
        print(f"| FP64 - Mixed | RHF spherical:", e_tot - e_ref)
        # Mixed precision should maintain good accuracy
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_mixed_precision_cartesian(self):
        """Test RHF with mixed FP32/FP64 precision (cartesian basis)"""
        cutoff_fp32 = 1e-13
        cutoff_fp64 = 1e-7
        e_tot = run_scf(mol_cart, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64)
        e_ref = run_scf(mol_cart)
        print(f"| FP64 - Mixed | RHF cartesian:", e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_mixed_precision_aggressive(self):
        """Test RHF with aggressive mixed precision (more FP32 usage)"""
        cutoff_fp32 = 1e-15
        cutoff_fp64 = 1e-6
        e_tot = run_scf(mol_sph, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64)
        e_ref = run_scf(mol_sph)
        print(f"| FP64 - Mixed(aggressive) | RHF:", e_tot - e_ref)
        # More aggressive mixed precision may have slightly lower accuracy
        assert np.abs(e_tot - e_ref) < 1e-4

    def test_mixed_precision_conservative(self):
        """Test RHF with conservative mixed precision (more FP64 usage)"""
        cutoff_fp32 = 1e-12
        cutoff_fp64 = 1e-8
        e_tot = run_scf(mol_sph, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64)
        e_ref = run_scf(mol_sph)
        print(f"| FP64 - Mixed(conservative) | RHF:", e_tot - e_ref)
        # Conservative mixed precision should be very close to FP64
        assert np.abs(e_tot - e_ref) < 1e-6


if __name__ == "__main__":
    print("Full Tests for SCF Kernels")
    unittest.main()
