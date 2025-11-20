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
Unit tests for pair-based JK algorithm.

Tests the pair-based 2D algorithm against PySCF reference implementation
across various precision modes, density matrix configurations, and molecular systems.
"""

import unittest

import cupy as cp
import numpy as np
import pyscf
from pyscf import gto
from pyscf.scf.hf import get_jk
from gpu4pyscf import scf

import jqc.pyscf.jk_pair
from jqc.pyscf.basis import BasisLayout


def _to_numpy(x):
    """Convert CuPy array to NumPy array if needed."""
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else x

def setUpModule():
    global mol
    mol = pyscf.M(
        atom="""
        H  -0.757    4.   -0.4696
        H   0.757    4.   -0.4696
        """,
        basis="def2-tzvpp",
        output="/dev/null",
        unit="B",
        cart=1,
    )


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):
    """Test suite for pair-based JK kernels"""

    def test_jk_pair_double(self):
        """Test pair-based JK with double precision"""
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        # Generate pair-based get_jk function
        basis_layout = BasisLayout.from_mol(mol)
        get_jk_pair = jqc.pyscf.jk_pair.generate_get_jk(basis_layout)

        vj, vk = get_jk_pair(mol, dm, hermi=1)
        vj1 = _to_numpy(vj)
        vk1 = _to_numpy(vk)
        ref = get_jk(mol, dm, hermi=1)
        print("vj diff with double precision (pair-based):", abs(vj1 - ref[0]).max())
        print("vk diff with double precision (pair-based):", abs(vk1 - ref[1]).max())
        assert abs(vj1 - ref[0]).max() < 1e-7
        assert abs(vk1 - ref[1]).max() < 1e-7

    def test_jk_pair_single(self):
        """Test pair-based JK with single precision"""
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        # Generate pair-based get_jk function with single precision
        basis_layout = BasisLayout.from_mol(mol)
        get_jk_pair = jqc.pyscf.jk_pair.generate_get_jk(
            basis_layout, cutoff_fp32=1e-13, cutoff_fp64=1e100
        )

        vj, vk = get_jk_pair(mol, dm, hermi=1)
        vj1 = _to_numpy(vj)
        vk1 = _to_numpy(vk)
        ref = get_jk(mol, dm, hermi=1)
        print("vj diff with single precision (pair-based):", abs(vj1 - ref[0]).max())
        print("vk diff with single precision (pair-based):", abs(vk1 - ref[1]).max())
        assert abs(vj1 - ref[0]).max() < 1e-3
        assert abs(vk1 - ref[1]).max() < 1e-3

    @unittest.skip("Multiple density matrices not yet supported in pair-based algorithm")
    def test_jk_pair_multiple_dm(self):
        """Test pair-based JK with multiple density matrices"""
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(3, nao, nao)
        dm = dm + dm.transpose([0, 2, 1])

        basis_layout = BasisLayout.from_mol(mol)
        get_jk_pair = jqc.pyscf.jk_pair.generate_get_jk(basis_layout)

        vj, vk = get_jk_pair(mol, dm, hermi=1)
        vj1 = _to_numpy(vj)
        vk1 = _to_numpy(vk)
        ref = get_jk(mol, dm, hermi=1)
        print("vj diff with multiple DMs (pair-based):", abs(vj1 - ref[0]).max())
        print("vk diff with multiple DMs (pair-based):", abs(vk1 - ref[1]).max())
        assert abs(vj1 - ref[0]).max() < 1e-7
        assert abs(vk1 - ref[1]).max() < 1e-7

    def test_j_pair_only(self):
        """Test pair-based J-only computation"""
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        basis_layout = BasisLayout.from_mol(mol)
        get_j_pair = jqc.pyscf.jk_pair.generate_get_j(basis_layout)

        vj = get_j_pair(mol, dm, hermi=1)
        vj1 = _to_numpy(vj)
        ref = get_jk(mol, dm, hermi=1, with_j=True, with_k=False)
        print("vj diff in pair-based J-only kernel:", abs(vj1 - ref[0]).max())
        assert abs(vj1 - ref[0]).max() < 1e-7

    def test_k_pair_only(self):
        """Test pair-based K-only computation"""
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        basis_layout = BasisLayout.from_mol(mol)
        get_k_pair = jqc.pyscf.jk_pair.generate_get_k(basis_layout)

        vk = get_k_pair(mol, dm, hermi=1)
        vk1 = _to_numpy(vk)
        ref = get_jk(mol, dm, hermi=1)
        print("vk diff in pair-based K-only kernel:", abs(vk1 - ref[1]).max())
        assert abs(vk1 - ref[1]).max() < 1e-7

    def test_k_pair_fp32(self):
        """Test pair-based K-only with single precision"""
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        basis_layout = BasisLayout.from_mol(mol)
        get_jk_pair = jqc.pyscf.jk_pair.generate_get_jk(
            basis_layout, cutoff_fp32=1e-13, cutoff_fp64=1e100
        )

        _, vk = get_jk_pair(mol, dm, hermi=1, with_j=False, with_k=True, omega=0.3)
        vk1 = _to_numpy(vk)
        ref = get_jk(mol, dm, hermi=1, omega=0.3)
        print("vk diff in pair-based K-only kernel (fp32):", abs(vk1 - ref[1]).max())
        assert abs(vk1 - ref[1]).max() < 1e-3

    def test_jk_pair_omega(self):
        """Test pair-based JK with omega parameter (range-separated)"""
        omega = 0.5
        mol_with_omega = pyscf.M(
            atom="""
        H  -0.757    4.   -0.4696
        H   0.757    4.   -0.4696
        """,
            basis="sto-3g",
            unit="B",
            cart=1,
            output="/dev/null",
        )

        mol_with_omega.omega = omega

        np.random.seed(9)
        nao = mol_with_omega.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        basis_layout = BasisLayout.from_mol(mol_with_omega)
        get_jk_pair = jqc.pyscf.jk_pair.generate_get_jk(basis_layout)

        vj, vk = get_jk_pair(mol_with_omega, dm, hermi=1, omega=omega)
        vj1 = _to_numpy(vj)
        vk1 = _to_numpy(vk)
        ref = get_jk(mol_with_omega, dm, hermi=1, omega=omega)
        mol_with_omega.stdout.close()
        print("vj diff with omega (pair-based):", abs(vj1 - ref[0]).max())
        print("vk diff with omega (pair-based):", abs(vk1 - ref[1]).max())
        assert abs(vj1 - ref[0]).max() < 1e-7
        assert abs(vk1 - ref[1]).max() < 1e-7

    @unittest.skip("Mixed precision not yet fully supported in pair-based algorithm")
    def test_jk_pair_mixed_precision(self):
        """Test pair-based JK with mixed FP32/FP64 precision"""
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        basis_layout = BasisLayout.from_mol(mol)
        get_jk_pair = jqc.pyscf.jk_pair.generate_get_jk(
            basis_layout, cutoff_fp32=1e-13, cutoff_fp64=1e-7
        )

        vj, vk = get_jk_pair(mol, dm, hermi=1)
        vj1 = _to_numpy(vj)
        vk1 = _to_numpy(vk)
        ref = get_jk(mol, dm, hermi=1)
        print("vj diff with mixed precision (pair-based):", abs(vj1 - ref[0]).max())
        print("vk diff with mixed precision (pair-based):", abs(vk1 - ref[1]).max())
        assert abs(vj1 - ref[0]).max() < 1e-7
        assert abs(vk1 - ref[1]).max() < 1e-7

    def test_jk_pair_screening(self):
        """Test pair-based JK with well-separated atoms (tests screening)"""
        mol_apart = pyscf.M(
            atom="""
        H  -0.757    4.   -0.4696
        H   100.757    4.   -0.4696
        """,
            basis="sto-3g",
            unit="B",
            cart=1,
            output="/dev/null",
        )

        np.random.seed(9)
        nao = mol_apart.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        basis_layout = BasisLayout.from_mol(mol_apart)
        get_jk_pair = jqc.pyscf.jk_pair.generate_get_jk(basis_layout)

        vj, vk = get_jk_pair(mol_apart, dm, hermi=1)
        vj1 = _to_numpy(vj)
        vk1 = _to_numpy(vk)
        ref = get_jk(mol_apart, dm, hermi=1)
        mol_apart.stdout.close()
        print("vj diff with screening (pair-based):", abs(vj1 - ref[0]).max())
        print("vk diff with screening (pair-based):", abs(vk1 - ref[1]).max())
        assert abs(vj1 - ref[0]).max() < 1e-7
        assert abs(vk1 - ref[1]).max() < 1e-7

    def test_jk_pair_sph(self):
        """Test pair-based JK with spherical basis"""
        mol_sph = pyscf.M(
            atom="""
            H  -0.757    4.   -0.4696
            H   0.757    4.   -0.4696
            """,
            basis="sto-3g",
            unit="B",
            cart=0,
            output="/dev/null",
        )
        np.random.seed(9)
        nao = mol_sph.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        basis_layout = BasisLayout.from_mol(mol_sph)
        get_jk_pair = jqc.pyscf.jk_pair.generate_get_jk(basis_layout)

        vj, vk = get_jk_pair(mol_sph, dm, hermi=1)
        vj1 = _to_numpy(vj)
        vk1 = _to_numpy(vk)
        ref = get_jk(mol_sph, dm, hermi=1)
        mol_sph.stdout.close()
        print("vj diff with spherical basis (pair-based):", abs(vj1 - ref[0]).max())
        print("vk diff with spherical basis (pair-based):", abs(vk1 - ref[1]).max())
        assert abs(vj1 - ref[0]).max() < 1e-7
        assert abs(vk1 - ref[1]).max() < 1e-7

    def test_jk_pair_different_pair_wide(self):
        """Test pair-based VK with different pair widths"""
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        # Test with pair_wide=16
        basis_layout = BasisLayout.from_mol(mol)
        get_jk_pair_16 = jqc.pyscf.jk_pair.generate_get_jk(basis_layout, pair_wide_vk=16)
        vj_16, vk_16 = get_jk_pair_16(mol, dm, hermi=1)

        # Test with pair_wide=64 (default)
        get_jk_pair_64 = jqc.pyscf.jk_pair.generate_get_jk(basis_layout, pair_wide_vk=64)
        vj_64, vk_64 = get_jk_pair_64(mol, dm, hermi=1)

        # Compare results
        vj_16 = _to_numpy(vj_16)
        vk_16 = _to_numpy(vk_16)
        vj_64 = _to_numpy(vj_64)
        vk_64 = _to_numpy(vk_64)

        print("vj diff between pair_wide=16 and pair_wide=64:", abs(vj_16 - vj_64).max())
        print("vk diff between pair_wide=16 and pair_wide=64:", abs(vk_16 - vk_64).max())
        assert abs(vj_16 - vj_64).max() < 1e-10
        assert abs(vk_16 - vk_64).max() < 1e-10


if __name__ == "__main__":
    print("Full Tests for Pair-Based JK Algorithm")
    unittest.main()
