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

import cupy as cp
import numpy as np
import pyscf
from pyscf import gto
from pyscf.scf.hf import get_jk
from gpu4pyscf import scf

import jqc.pyscf


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
        basis="def2-tzvpp",  #'cc-pvdz',
        output="/dev/null",
        unit="B",
        cart=1,
    )


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):
    def test_jk_sph(self):
        mol_sph = pyscf.M(
            atom="""
            H  -0.757    4.   -0.4696
            H   0.757    4.   -0.4696
            """,
            basis="def2-tzvpp",  #'cc-pvdz',
            unit="B",
            cart=0,
            output="/dev/null",
        )
        np.random.seed(9)
        nao = mol_sph.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        # Apply JoltQC to a temporary RHF object to get access to get_jk
        mf_temp = scf.RHF(mol_sph)
        jqc.pyscf.apply(mf_temp)
        vj, vk = mf_temp.get_jk(mol_sph, dm, hermi=1)
        vj1 = _to_numpy(vj)
        vk1 = _to_numpy(vk)
        ref = get_jk(mol_sph, dm, hermi=1)
        mol_sph.stdout.close()
        print("vj diff with double precision:", abs(vj1 - ref[0]).max())
        print("vk diff with double precision:", abs(vk1 - ref[1]).max())
        assert abs(vj1 - ref[0]).max() < 1e-7
        assert abs(vk1 - ref[1]).max() < 1e-7

    def test_jk_double(self):
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        # Apply JoltQC to a temporary RHF object to get access to get_jk
        mf_temp = scf.RHF(mol)
        jqc.pyscf.apply(mf_temp)
        vj, vk = mf_temp.get_jk(mol, dm, hermi=1)
        vj1 = _to_numpy(vj)
        vk1 = _to_numpy(vk)
        ref = get_jk(mol, dm, hermi=1)
        print("vj diff with double precision:", abs(vj1 - ref[0]).max())
        print("vk diff with double precision:", abs(vk1 - ref[1]).max())
        assert abs(vj1 - ref[0]).max() < 1e-7
        assert abs(vk1 - ref[1]).max() < 1e-7

    def test_jk_single(self):
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        # Apply JoltQC with single precision settings
        mf_temp = scf.RHF(mol)
        config = {"jk": {"cutoff_fp32": 1e-13, "cutoff_fp64": 1e100}}
        jqc.pyscf.apply(mf_temp, config=config)
        vj, vk = mf_temp.get_jk(mol, dm, hermi=1)
        vj1 = _to_numpy(vj)
        vk1 = _to_numpy(vk)
        ref = get_jk(mol, dm, hermi=1)
        print("vj diff with single precision:", abs(vj1 - ref[0]).max())
        print("vk diff with single precision:", abs(vk1 - ref[1]).max())
        assert abs(vj1 - ref[0]).max() < 1e-3
        assert abs(vk1 - ref[1]).max() < 1e-3

    def test_jk_multiple_dm(self):
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(3, nao, nao)
        dm = dm + dm.transpose([0, 2, 1])

        # Apply JoltQC to a temporary RHF object to get access to get_jk
        mf_temp = scf.RHF(mol)
        jqc.pyscf.apply(mf_temp)
        vj, vk = mf_temp.get_jk(mol, dm, hermi=1)
        vj1 = _to_numpy(vj)
        vk1 = _to_numpy(vk)
        ref = get_jk(mol, dm, hermi=1)
        print("vj diff with multiple DMs:", abs(vj1 - ref[0]).max())
        print("vk diff with multiple DMs:", abs(vk1 - ref[1]).max())
        assert abs(vj1 - ref[0]).max() < 1e-7
        assert abs(vk1 - ref[1]).max() < 1e-7

    def test_j_dm(self):
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        # Apply JoltQC to a temporary RHF object to get access to get_jk
        mf_temp = scf.RHF(mol)
        jqc.pyscf.apply(mf_temp)
        vj, _ = mf_temp.get_jk(mol, dm, hermi=1, with_j=True, with_k=False)
        vj1 = _to_numpy(vj)
        ref = get_jk(mol, dm, hermi=1, with_j=True, with_k=False)
        print("vj diff in JK kernel:", abs(vj1 - ref[0]).max())
        assert abs(vj1 - ref[0]).max() < 1e-7

    def test_k_dm(self):
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        # Apply JoltQC to a temporary RHF object to get access to get_jk
        mf_temp = scf.RHF(mol)
        jqc.pyscf.apply(mf_temp)
        _, vk = mf_temp.get_jk(mol, dm, hermi=1, with_j=False, with_k=True)
        vk1 = _to_numpy(vk)
        ref = get_jk(mol, dm, hermi=1)
        print("vk diff in JK kernel:", abs(vk1 - ref[1]).max())
        assert abs(vk1 - ref[1]).max() < 1e-7

    def test_k_dm_fp32(self):
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        # Apply JoltQC with single precision settings
        mf_temp = scf.RHF(mol)
        config = {"jk": {"cutoff_fp32": 1e-13, "cutoff_fp64": 1e100}}
        jqc.pyscf.apply(mf_temp, config=config)
        _, vk = mf_temp.get_jk(mol, dm, hermi=1, with_j=False, with_k=True, omega=0.3)
        vk1 = _to_numpy(vk)
        ref = get_jk(mol, dm, hermi=1, omega=0.3)
        print("vk diff in JK kernel:", abs(vk1 - ref[1]).max())
        assert abs(vk1 - ref[1]).max() < 1e-3

    def test_jk_omega(self):
        omega = 0.5
        mol_with_omega = pyscf.M(
            atom="""
        H  -0.757    4.   -0.4696
        H   0.757    4.   -0.4696
        """,
            basis="def2-tzvpp",
            unit="B",
            cart=1,
            output="/dev/null",
        )

        mol_with_omega.omega = omega

        np.random.seed(9)
        nao = mol_with_omega.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        # Apply JoltQC to a temporary RHF object to get access to get_jk
        mf_temp = scf.RHF(mol_with_omega)
        jqc.pyscf.apply(mf_temp)
        vj, vk = mf_temp.get_jk(mol_with_omega, dm, hermi=1, omega=omega)
        vj1 = _to_numpy(vj)
        vk1 = _to_numpy(vk)
        ref = get_jk(mol_with_omega, dm, hermi=1, omega=omega)
        mol_with_omega.stdout.close()
        assert abs(vj1 - ref[0]).max() < 1e-7
        assert abs(vk1 - ref[1]).max() < 1e-7

    def test_jk_mixed_precision(self):
        """Test JK computation with mixed FP32/FP64 precision.

        This test verifies that the mixed precision kernel correctly handles
        both FP32 and FP64 computations based on screening cutoffs, ensuring
        that important integrals use FP64 while less critical ones use FP32.
        """
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        # Apply JoltQC with mixed precision settings
        # cutoff_fp32: 1e-13 means integrals with magnitude > 1e-13 use at least FP32
        # cutoff_fp64: 1e-7 means integrals with magnitude > 1e-7 use FP64
        # This creates three categories:
        #   - integrals > 1e-7: computed with FP64
        #   - integrals between 1e-13 and 1e-7: computed with FP32
        #   - integrals < 1e-13: skipped
        mf_temp = scf.RHF(mol)
        config = {"jk": {"cutoff_fp32": 1e-13, "cutoff_fp64": 1e-7}}
        jqc.pyscf.apply(mf_temp, config=config)
        vj, vk = mf_temp.get_jk(mol, dm, hermi=1)
        vj1 = _to_numpy(vj)
        vk1 = _to_numpy(vk)
        ref = get_jk(mol, dm, hermi=1)
        print("vj diff with mixed precision:", abs(vj1 - ref[0]).max())
        print("vk diff with mixed precision:", abs(vk1 - ref[1]).max())
        # Mixed precision should still maintain good accuracy (better than pure FP32)
        assert abs(vj1 - ref[0]).max() < 1e-7
        assert abs(vk1 - ref[1]).max() < 1e-7

    def test_jk_screening(self):
        mol_apart = pyscf.M(
            atom="""
        H  -0.757    4.   -0.4696
        H   100.757    4.   -0.4696
        """,
            basis="def2-tzvpp",
            unit="B",
            cart=1,
            output="/dev/null",
        )

        np.random.seed(9)
        nao = mol_apart.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        # Apply JoltQC to a temporary RHF object to get access to get_jk
        mf_temp = scf.RHF(mol_apart)
        jqc.pyscf.apply(mf_temp)
        vj, vk = mf_temp.get_jk(mol_apart, dm, hermi=1)
        vj1 = _to_numpy(vj)
        vk1 = _to_numpy(vk)
        ref = get_jk(mol_apart, dm, hermi=1)
        mol_apart.stdout.close()
        assert abs(vj1 - ref[0]).max() < 1e-7
        assert abs(vk1 - ref[1]).max() < 1e-7


if __name__ == "__main__":
    print("Full Tests for SCF JK")
    unittest.main()
