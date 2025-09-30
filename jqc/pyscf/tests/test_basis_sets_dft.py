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
from types import MethodType

import pyscf
from gpu4pyscf import dft

from jqc.pyscf import jk, rks


class BasisSetDFTTests(unittest.TestCase):
    """Test DFT calculations with different basis sets"""

    def test_dft_pbe_basis_6_31g(self):
        """Test PBE DFT calculations with 6-31G basis"""
        self._test_basis_set_pbe("6-31g")

    def test_dft_pbe_basis_6_31gs(self):
        """Test PBE DFT calculations with 6-31G* basis"""
        self._test_basis_set_pbe("6-31g*")

    def test_dft_pbe_basis_sto_3g(self):
        """Test PBE DFT calculations with STO-3G basis"""
        self._test_basis_set_pbe("sto-3g")

    def test_dft_pbe_basis_cc_pvdz(self):
        """Test PBE DFT calculations with cc-pVDZ basis"""
        self._test_basis_set_pbe("cc-pvdz")

    def test_dft_pbe_basis_cc_pvtz(self):
        """Test PBE DFT calculations with cc-pVTZ basis"""
        self._test_basis_set_pbe("cc-pvtz")

    def test_dft_b3lyp_basis_6_31g(self):
        """Test B3LYP DFT calculations with 6-31G basis"""
        self._test_basis_set_b3lyp("6-31g")

    def test_dft_b3lyp_basis_6_31gs(self):
        """Test B3LYP DFT calculations with 6-31G* basis"""
        self._test_basis_set_b3lyp("6-31g*")

    def test_dft_b3lyp_basis_sto_3g(self):
        """Test B3LYP DFT calculations with STO-3G basis"""
        self._test_basis_set_b3lyp("sto-3g")

    def test_dft_b3lyp_basis_cc_pvdz(self):
        """Test B3LYP DFT calculations with cc-pVDZ basis"""
        self._test_basis_set_b3lyp("cc-pvdz")

    def test_dft_b3lyp_basis_cc_pvtz(self):
        """Test B3LYP DFT calculations with cc-pVTZ basis"""
        self._test_basis_set_b3lyp("cc-pvtz")

    def _test_basis_set_pbe(self, basis):
        """Helper method to test PBE with a specific basis set"""
        test_molecule = """
        O       0.0000000000    -0.0000000000     0.1174000000
        H      -0.7570000000    -0.0000000000    -0.4696000000
        H       0.7570000000     0.0000000000    -0.4696000000
        """

        mol_test = pyscf.M(
            atom=test_molecule, basis=basis, output="/dev/null", verbose=0
        )

        # JoltQC calculation
        mf_jolt = dft.RKS(mol_test, xc="PBE")
        mf_jolt.grids.level = 3  # Lower grid level for speed
        mf_jolt.get_jk = jk.generate_jk_kernel()
        nr_rks = rks.generate_nr_rks()
        mf_jolt._numint.nr_rks = MethodType(nr_rks, mf_jolt._numint)
        e_jolt = mf_jolt.kernel()

        # PySCF reference calculation
        mf_ref = dft.RKS(mol_test, xc="PBE")
        mf_ref.grids.level = 3  # Same grid level
        e_ref = mf_ref.kernel()

        energy_diff = abs(e_jolt - e_ref)

        mol_test.stdout.close()

        self.assertLess(
            energy_diff, 1e-7, f"Energy error for PBE/{basis}: {energy_diff:.2e} > 1e-7"
        )

    def _test_basis_set_b3lyp(self, basis):
        """Helper method to test B3LYP with a specific basis set"""
        test_molecule = """
        O       0.0000000000    -0.0000000000     0.1174000000
        H      -0.7570000000    -0.0000000000    -0.4696000000
        H       0.7570000000     0.0000000000    -0.4696000000
        """

        mol_test = pyscf.M(
            atom=test_molecule, basis=basis, output="/dev/null", verbose=0
        )

        # JoltQC calculation
        mf_jolt = dft.RKS(mol_test, xc="B3LYP")
        mf_jolt.grids.level = 3  # Lower grid level for speed
        mf_jolt.get_jk = jk.generate_jk_kernel()
        nr_rks = rks.generate_nr_rks()
        mf_jolt._numint.nr_rks = MethodType(nr_rks, mf_jolt._numint)
        e_jolt = mf_jolt.kernel()

        # PySCF reference calculation
        mf_ref = dft.RKS(mol_test, xc="B3LYP")
        mf_ref.grids.level = 3  # Same grid level
        e_ref = mf_ref.kernel()

        energy_diff = abs(e_jolt - e_ref)

        mol_test.stdout.close()

        self.assertLess(
            energy_diff,
            1e-7,
            f"Energy error for B3LYP/{basis}: {energy_diff:.2e} > 1e-7",
        )

    def test_dft_small_molecule_basis_sets(self):
        """Test DFT calculations with different basis sets on a small molecule"""

        basis_sets = ["sto-3g", "6-31g"]
        functionals = ["PBE", "B3LYP"]

        for basis in basis_sets:
            for xc in functionals:
                with self.subTest(basis=basis, functional=xc):
                    mol_test = pyscf.M(
                        atom="""
        H  0.0  0.0  0.0
        H  0.0  0.0  1.4
        """,
                        basis=basis,
                        output="/dev/null",
                        verbose=0,
                    )

                    # JoltQC calculation
                    mf_jolt = dft.RKS(mol_test, xc=xc)
                    mf_jolt.grids.level = 2  # Lower grid level for speed
                    mf_jolt.get_jk = jk.generate_jk_kernel()
                    nr_rks = rks.generate_nr_rks()
                    mf_jolt._numint.nr_rks = MethodType(nr_rks, mf_jolt._numint)
                    e_jolt = mf_jolt.kernel()

                    # PySCF reference calculation
                    mf_ref = dft.RKS(mol_test, xc=xc)
                    mf_ref.grids.level = 2  # Same grid level
                    e_ref = mf_ref.kernel()

                    energy_diff = abs(e_jolt - e_ref)

                    mol_test.stdout.close()

                    self.assertLess(
                        energy_diff,
                        1e-7,
                        f"Energy error for {xc}/{basis} (H2): {energy_diff:.2e} > 1e-7",
                    )


if __name__ == "__main__":
    print("Basis Set Tests for DFT Calculations")
    unittest.main()
