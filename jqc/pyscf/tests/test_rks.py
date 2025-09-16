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
import cupy as cp
from pyscf import lib, gto
from gpu4pyscf.scf.jk import _VHFOpt
from gpu4pyscf import dft
from gpu4pyscf.dft.rks import initialize_grids
from jqc.pyscf import rks

# from jqc.backend.rks import estimate_log_aovalue  # No longer used
from jqc.pyscf.basis import BasisLayout


def setUpModule():
    global mol, grids, ni
    basis = gto.basis.parse(
        """
H    S
     34.0613410              0.60251978E-02
      5.1235746              0.45021094E-01
      1.1646626              0.20189726
H    S
      0.32723041             1.0000000
H    S
      0.10307241             1.0000000
H    P
      1.40700000             1.0000000
H    P
      0.38800000             1.0000000
H    D
      1.05700000             1.0000000
H    F
      1.05700000             1.0000000
                            """
    )
    mol = pyscf.M(
        atom="""
        H  -0.757    4.   -0.4696
        H   0.757    4.   -0.4696
        """,
        basis=basis,  #'def2-tzvpp', #'ccpvdz',
        unit="B",
        cart=1,
        output="/dev/null",
    )
    mol.build()
    mf = dft.KS(mol, xc="b3lyp")
    mf.grids.level = 3
    dm = mf.get_init_guess()
    initialize_grids(mf, mol, dm)
    grids = mf.grids
    ni = mf._numint


def tearDownModule():
    global mol, grids, ni
    mol.stdout.close()
    del mol, grids, ni


class KnownValues(unittest.TestCase):
    def test_dft_double(self):
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)
        xctype = "LDA"

        basis_layout = BasisLayout.from_mol(mol, alignment=1)
        _, rho_kern, vxc_kern = rks.generate_rks_kernel(basis_layout)
        rho = rho_kern(mol, grids, xctype, dm)

        ao_gpu = ni.eval_ao(mol, grids.coords, deriv=0, transpose=False)
        rho_pyscf = ni.eval_rho(mol, ao_gpu, dm, xctype="LDA")

        assert abs(rho - rho_pyscf).max() < 1e-7

        ngrids = grids.coords.shape[0]
        wv = cp.asarray(np.random.rand(ngrids))

        vxc = vxc_kern(mol, grids, xctype, wv)
        aow = ao_gpu * wv
        vxc_pyscf = ao_gpu.dot(aow.T)
        assert abs(vxc - vxc_pyscf).max() < 1e-7

    def test_dft_single(self):
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)
        xctype = "LDA"

        basis_layout = BasisLayout.from_mol(mol, alignment=1)
        _, rho_kern, vxc_kern = rks.generate_rks_kernel(
            basis_layout, cutoff_fp32=1e-13, cutoff_fp64=1e100
        )
        rho = rho_kern(mol, grids, xctype, dm)

        ao_gpu = ni.eval_ao(mol, grids.coords, deriv=0, transpose=False)
        rho_pyscf = ni.eval_rho(mol, ao_gpu, dm, xctype=xctype)
        assert abs(rho - rho_pyscf).max() < 1e-3

        ngrids = grids.coords.shape[0]
        wv = cp.asarray(np.random.rand(ngrids))

        vxc = vxc_kern(mol, grids, xctype, wv)
        aow = ao_gpu * wv
        vxc_pyscf = ao_gpu.dot(aow.T)
        assert abs(vxc - vxc_pyscf).max() < 1e-3

    def test_dft_gga(self):
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)
        xctype = "GGA"

        basis_layout = BasisLayout.from_mol(mol, alignment=1)
        _, rho_kern, vxc_kern = rks.generate_rks_kernel(basis_layout)
        rho = rho_kern(mol, grids, xctype, dm)

        ao_gpu = ni.eval_ao(mol, grids.coords, deriv=1, transpose=False)
        rho_pyscf = ni.eval_rho(mol, ao_gpu, dm, xctype=xctype)
        assert abs(rho - rho_pyscf).max() < 1e-7

        ngrids = grids.coords.shape[0]
        wv = cp.asarray(np.random.rand(4, ngrids))

        vxc = vxc_kern(mol, grids, xctype, wv)

        wv[0] *= 0.5
        aow = cp.einsum("nip,np->ip", ao_gpu, wv)
        vxc_pyscf = ao_gpu[0].dot(aow.T)
        vxc_pyscf += vxc_pyscf.T
        assert abs(vxc - vxc_pyscf).max() < 1e-7

    def test_dft_mgga(self):
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)
        xctype = "MGGA"

        basis_layout = BasisLayout.from_mol(mol, alignment=1)
        _, rho_kern, vxc_kern = rks.generate_rks_kernel(basis_layout)
        rho = rho_kern(mol, grids, xctype, dm)

        ao_gpu = ni.eval_ao(mol, grids.coords, deriv=1, transpose=False)
        rho_pyscf = ni.eval_rho(mol, ao_gpu, dm, xctype="MGGA")
        assert abs(rho - rho_pyscf).max() < 1e-7

        ngrids = grids.coords.shape[0]
        wv = cp.asarray(np.random.rand(5, ngrids))
        vxc = vxc_kern(mol, grids, xctype, wv)

        from gpu4pyscf.dft.numint import _tau_dot, _scale_ao

        wv[[0, 4]] *= 0.5
        vxc_pyscf = _tau_dot(ao_gpu, ao_gpu, wv[4])
        aow = _scale_ao(ao_gpu, wv[:4])
        vxc_pyscf += ao_gpu[0].dot(aow.T)
        vxc_pyscf += vxc_pyscf.T
        assert abs(vxc - vxc_pyscf).max() < 1e-7

    # NOTE: test_estimate_aovalue was removed because it depended on the deprecated
    # format_bas_cache function. The test was not compatible with the new BasisLayout
    # approach and would require a complete rewrite to test estimate_log_aovalue
    # functionality properly.

    def test_nlc_vxc(self):
        nao = mol.nao
        dm = cp.random.rand(nao, nao)
        dm = dm + dm.T

        basis_layout = BasisLayout.from_mol(mol, alignment=1)
        nr_nlc_vxc = rks.generate_nr_nlc_vxc(basis_layout)

        n, e, v = ni.nr_nlc_vxc(mol, grids, "wb97m-v", dm)
        n_jqc, e_jqc, v_jqc = nr_nlc_vxc(ni, mol, grids, "wb97m-v", dm)

        assert np.linalg.norm(n - n_jqc) < 1e-8
        assert np.linalg.norm(e - e_jqc) < 1e-8
        assert np.linalg.norm(v - v_jqc) < 1e-8


if __name__ == "__main__":
    print("Full Tests for rho and Vxc Kernels")
    unittest.main()
