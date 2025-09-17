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
from pyscf.scf.hf import get_jk

from jqc.constants import TILE
from jqc.pyscf import jk
from jqc.pyscf.basis import BasisLayout


class BasisSetJKTests(unittest.TestCase):
    """Test JK calculations with different basis sets"""

    def test_jk_basis_6_31g(self):
        """Test JK calculations with 6-31G basis"""
        self._test_basis_set("6-31g")

    def test_jk_basis_6_31gs(self):
        """Test JK calculations with 6-31G* basis"""
        self._test_basis_set("6-31g*")

    def test_jk_basis_sto_3g(self):
        """Test JK calculations with STO-3G basis"""
        self._test_basis_set("sto-3g")

    def test_jk_basis_cc_pvdz(self):
        """Test JK calculations with cc-pVDZ basis"""
        self._test_basis_set("cc-pvdz")

    def test_jk_basis_cc_pvtz(self):
        """Test JK calculations with cc-pVTZ basis"""
        self._test_basis_set("cc-pvtz")

    def _test_basis_set(self, basis):
        """Helper method to test a specific basis set"""
        test_molecule = """
        O       0.0000000000    -0.0000000000     0.1174000000
        H      -0.7570000000    -0.0000000000    -0.4696000000
        H       0.7570000000     0.0000000000    -0.4696000000
        """

        mol_test = pyscf.M(
            atom=test_molecule, basis=basis, output="/dev/null", verbose=0
        )

        np.random.seed(42)  # Fixed seed for reproducibility
        nao = mol_test.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        # JoltQC calculation
        basis_layout = BasisLayout.from_mol(mol_test, alignment=TILE)
        get_jk_jit = jk.generate_jk_kernel(basis_layout)
        vj, vk = get_jk_jit(mol_test, dm, hermi=1)
        vj_jolt = vj.get()
        vk_jolt = vk.get()

        # PySCF reference calculation
        vj_ref, vk_ref = get_jk(mol_test, dm, hermi=1)

        # Compare results
        vj_diff = abs(vj_jolt - vj_ref).max()
        vk_diff = abs(vk_jolt - vk_ref).max()

        mol_test.stdout.close()

        self.assertLess(
            vj_diff, 1e-7, f"VJ matrix error for {basis}: {vj_diff:.2e} > 1e-7"
        )
        self.assertLess(
            vk_diff, 1e-7, f"VK matrix error for {basis}: {vk_diff:.2e} > 1e-7"
        )


if __name__ == "__main__":
    print("Basis Set Tests for JK Calculations")
    unittest.main()
