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
import cupy as cp
import pyscf
from pyscf import lib, gto
from pyscf.scf.hf import get_jk
from jqc.pyscf import jk
from jqc.pyscf.basis import BasisLayout
from jqc.constants import TILE


def setUpModule():
    global mol
    basis = gto.basis.parse(
        """
#H    S
#     34.0613410              0.60251978E-02
#      5.1235746              0.45021094E-01
#      1.1646626              0.20189726
#H    S
#      0.32723041             1.0000000
#H    S
#      0.10307241             1.0000000
#H    P
#      1.40700000             1.0000000
#H    P
#      0.38800000             1.0000000
H    D
      1.05700000             1.0000000
#H    F
#      1.05700000             1.0000000
                            """
    )
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

        basis_layout = BasisLayout.from_mol(mol_sph, alignment=TILE)
        get_jk_jit = jk.generate_jk_kernel(basis_layout)
        vj, vk = get_jk_jit(mol_sph, dm, hermi=1)
        vj1 = vj.get()
        vk1 = vk.get()
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

        basis_layout = BasisLayout.from_mol(mol, alignment=TILE)
        get_jk_jit = jk.generate_jk_kernel(basis_layout)
        vj, vk = get_jk_jit(mol, dm, hermi=1)
        vj1 = vj.get()
        vk1 = vk.get()
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

        basis_layout = BasisLayout.from_mol(mol, alignment=TILE)
        get_jk_jit = jk.generate_jk_kernel(
            basis_layout, cutoff_fp32=1e-13, cutoff_fp64=1e100
        )
        vj, vk = get_jk_jit(mol, dm, hermi=1)
        vj1 = vj.get()
        vk1 = vk.get()
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

        basis_layout = BasisLayout.from_mol(mol, alignment=TILE)
        get_jk_jit = jk.generate_jk_kernel(basis_layout)
        vj, vk = get_jk_jit(mol, dm, hermi=1)
        vj1 = vj.get()
        vk1 = vk.get()
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

        basis_layout = BasisLayout.from_mol(mol, alignment=TILE)
        get_jk_jit = jk.generate_jk_kernel(basis_layout)
        vj, _ = get_jk_jit(mol, dm, hermi=1, with_j=True, with_k=False)
        vj1 = vj.get()
        ref = get_jk(mol, dm, hermi=1, with_j=True, with_k=False)
        print("vj diff in JK kernel:", abs(vj1 - ref[0]).max())
        assert abs(vj1 - ref[0]).max() < 1e-7

    def test_k_dm(self):
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        basis_layout = BasisLayout.from_mol(mol, alignment=TILE)
        get_jk_jit = jk.generate_jk_kernel(basis_layout)
        _, vk = get_jk_jit(mol, dm, hermi=1, with_j=False, with_k=True)
        vk1 = vk.get()
        ref = get_jk(mol, dm, hermi=1)
        print("vk diff in JK kernel:", abs(vk1 - ref[1]).max())
        assert abs(vk1 - ref[1]).max() < 1e-7

    def test_k_dm_fp32(self):
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        basis_layout = BasisLayout.from_mol(mol, alignment=TILE)
        get_jk_jit = jk.generate_jk_kernel(
            basis_layout, cutoff_fp32=1e-13, cutoff_fp64=1e100
        )
        _, vk = get_jk_jit(mol, dm, hermi=1, with_j=False, with_k=True, omega=0.3)
        vk1 = vk.get()
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

        basis_layout = BasisLayout.from_mol(mol_with_omega, alignment=TILE)
        get_jk_jit = jk.generate_jk_kernel(basis_layout)
        vj, vk = get_jk_jit(mol_with_omega, dm, hermi=1, omega=omega)
        vj1 = vj.get()
        vk1 = vk.get()
        ref = get_jk(mol_with_omega, dm, hermi=1, omega=omega)
        mol_with_omega.stdout.close()
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

        basis_layout = BasisLayout.from_mol(mol_apart, alignment=TILE)
        get_jk_jit = jk.generate_jk_kernel(basis_layout)
        vj, vk = get_jk_jit(mol_apart, dm, hermi=1)
        vj1 = vj.get()
        vk1 = vk.get()
        ref = get_jk(mol_apart, dm, hermi=1)
        mol_apart.stdout.close()
        assert abs(vj1 - ref[0]).max() < 1e-7
        assert abs(vk1 - ref[1]).max() < 1e-7


if __name__ == "__main__":
    print("Full Tests for SCF JK")
    unittest.main()
