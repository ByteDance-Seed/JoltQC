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

import pyscf
import numpy as np
import unittest
from types import MethodType
import cupy as cp
import xqc.pyscf
from xqc.pyscf import rks, jk
from gpu4pyscf import dft

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''
bas='def2-tzvpp'
grids_level = 5
nlcgrids_level = 2

def setUpModule():
    global mol_sph, mol_cart
    mol_sph = pyscf.M(
        atom=atom,
        basis=bas,
        max_memory=32000,
        verbose=1,
        output='/dev/null'
    )

    mol_cart = pyscf.M(
        atom=atom,
        basis=bas,
        max_memory=32000,
        verbose=1,
        cart=1,
        output = '/dev/null'
    )

def tearDownModule():
    global mol_sph, mol_cart
    mol_sph.stdout.close()
    mol_cart.stdout.close()
    del mol_sph, mol_cart

def run_dft(xc, mol, disp=None):
    mf = dft.RKS(mol, xc=xc)
    mf.disp = disp
    mf.grids.level = grids_level
    mf.nlcgrids.level = nlcgrids_level
    mf.get_jk = jk.generate_jk_kernel(mol)
    nr_rks = rks.generate_nr_rks(mol)
    mf._numint.nr_rks = MethodType(nr_rks, mf._numint)
    e_dft = mf.kernel()
    return e_dft

class KnownValues(unittest.TestCase):
    def test_rks_lda(self):
        e_tot = run_dft('LDA,vwn5', mol_sph)
        e_ref = -75.9046410402
        print('| CPU - GPU | with LDA:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rks_pbe(self):
        e_tot = run_dft('PBE', mol_sph)
        e_ref = -76.3800182418
        print('| CPU - GPU | with PBE:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rks_b3lyp(self):
        e_tot = run_dft('B3LYP', mol_sph)
        e_ref = -76.4666495594
        print('| CPU - GPU | with B3LYP:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rks_m06(self):
        e_tot = run_dft("M06", mol_sph)
        e_ref = -76.4265870634
        print('| CPU - GPU | with M06:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rks_wb97(self):
        e_tot = run_dft("HYB_GGA_XC_WB97", mol_sph)
        e_ref = -76.4486274326
        print('| CPU - GPU | with wB97:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rks_vv10(self):
        e_tot = run_dft('HYB_MGGA_XC_WB97M_V', mol_sph)
        e_ref = -76.4334218842
        print('| CPU - GPU | with wB97m-v:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5

    def test_rks_cart(self):
        e_tot = run_dft('b3lyp', mol_cart)
        e_ref = -76.4672144985
        print('| CPU - GPU | with cart:', e_tot - e_ref)
        assert np.abs(e_tot - e_ref) < 1e-5
    
    def test_rks_compile(self):
        mf = dft.RKS(mol_sph, xc='PBE')
        mf = xqc.pyscf.compile(mf)
        e_tot = mf.kernel() 
        e_ref = dft.RKS(mol_sph, xc='PBE').kernel()
        assert np.abs(e_tot - e_ref) < 1e-5
        
if __name__ == "__main__":
    print("Full Tests for DFT Kernels")
    unittest.main()
