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
from jqc.backend.rks import estimate_log_aovalue
from jqc.pyscf.mol import sort_group_basis

def setUpModule():
    global mol, grids, ni
    basis = gto.basis.parse('''
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
                            ''')
    mol = pyscf.M(
        atom = '''
        H  -0.757    4.   -0.4696
        H   0.757    4.   -0.4696
        ''',
        basis=basis, #'def2-tzvpp', #'ccpvdz',
        unit='B', cart=1, output='/dev/null')
    mol.build()
    mf = dft.KS(mol, xc='b3lyp')
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
        xctype = 'LDA'

        _, rho_kern, vxc_kern = rks.generate_rks_kernel(mol)
        rho = rho_kern(mol, grids, xctype, dm)
        
        ao_gpu = ni.eval_ao(mol, grids.coords, deriv=0, transpose=False)
        rho_pyscf = ni.eval_rho(mol, ao_gpu, dm, xctype='LDA')

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
        xctype = 'LDA'

        _, rho_kern, vxc_kern = rks.generate_rks_kernel(mol, cutoff_fp32=1e-13, cutoff_fp64=1e100)
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
        xctype = 'GGA'

        _, rho_kern, vxc_kern = rks.generate_rks_kernel(mol)
        rho = rho_kern(mol, grids, xctype, dm)

        ao_gpu = ni.eval_ao(mol, grids.coords, deriv=1, transpose=False)
        rho_pyscf = ni.eval_rho(mol, ao_gpu, dm, xctype=xctype)
        assert abs(rho - rho_pyscf).max() < 1e-7

        ngrids = grids.coords.shape[0]
        wv = cp.asarray(np.random.rand(4, ngrids))

        vxc = vxc_kern(mol, grids, xctype, wv)

        wv[0] *= .5
        aow = cp.einsum('nip,np->ip', ao_gpu, wv)
        vxc_pyscf = ao_gpu[0].dot(aow.T)
        vxc_pyscf += vxc_pyscf.T
        assert abs(vxc - vxc_pyscf).max() < 1e-7
    
    def test_dft_mgga(self):
        np.random.seed(9)
        nao = mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)
        xctype = 'MGGA'

        _, rho_kern, vxc_kern = rks.generate_rks_kernel(mol)
        rho = rho_kern(mol, grids, xctype, dm)

        ao_gpu = ni.eval_ao(mol, grids.coords, deriv=1, transpose=False)
        rho_pyscf = ni.eval_rho(mol, ao_gpu, dm, xctype='MGGA')
        assert abs(rho - rho_pyscf).max() < 1e-7

        ngrids = grids.coords.shape[0]
        wv = cp.asarray(np.random.rand(5, ngrids))
        vxc = vxc_kern(mol, grids, xctype, wv)
        
        from gpu4pyscf.dft.numint import _tau_dot, _scale_ao
        wv[[0,4]] *= .5
        vxc_pyscf = _tau_dot(ao_gpu, ao_gpu, wv[4])
        aow = _scale_ao(ao_gpu, wv[:4])
        vxc_pyscf += ao_gpu[0].dot(aow.T)
        vxc_pyscf += vxc_pyscf.T
        assert abs(vxc - vxc_pyscf).max() < 1e-7

    def test_estimate_aovalue(self):
        np.random.seed(9)
        mol = pyscf.M(
        atom = '''
        H  -0.757    4.   -0.4696
        H   10.757    4.   -0.4696
        ''',
        basis='sto3g', #'def2-tzvpp', #'ccpvdz',
        unit='B', cart=1, output='/dev/null')
        mol.build()

        _vhfopt = _VHFOpt(mol)
        _vhfopt.dtype = np.float32
        _vhfopt.tile = 1
        _vhfopt.build()
        sorted_mol = _vhfopt.sorted_mol
        nao = sorted_mol.nao
        dm = np.random.rand(nao, nao)
        dm = dm.dot(dm.T)

        grid_coords = cp.asarray(grids.coords.T, order='C')
        bas_cache, _, _, _ = sort_group_basis(mol, alignment=1, dtype=np.float32)
        coeffs, exps, coords, angs, nprims = bas_cache
        ao_gpu = ni.eval_ao(sorted_mol, grids.coords, deriv=0, transpose=False)
        
        ang = angs[0]
        nprim = nprims[0]
        sparsity = estimate_log_aovalue(grid_coords, coords, coeffs, exps, ang, nprim)
        log_maxval, _, _ = sparsity
        ao_gpu_max = cp.max(ao_gpu.reshape(nao, -1, 256), axis=-1)
        log_ao_gpu = cp.log(cp.abs(ao_gpu_max))
        mol.stdout.close()
        sorted_mol.stdout.close()
        assert (log_maxval.T - log_ao_gpu).min() > -1e-5

if __name__ == "__main__":
    print("Full Tests for rho and Vxc Kernels")
    unittest.main()
