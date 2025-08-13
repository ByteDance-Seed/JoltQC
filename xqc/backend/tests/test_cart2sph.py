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

import numpy as np
import cupy as cp
from pyscf import gto
from xqc.backend.cart2sph import cart2sph, sph2cart

def test_cart2sph():
    basis = '''
O    S
      0.2700058226E+00      1
O    P
      0.2700058226E+00      1
O    D
      0.2700058226E+00      1
O    F
      0.2700058226E+00      1
      '''
    mol_cart = gto.M(atom='O 0 0 0', basis=basis, cart=True)
    mol_sph = gto.M(atom='O 0 0 0', basis=basis, cart=False)

    s_cart = mol_cart.intor('int1e_ovlp_cart')
    s_sph = mol_sph.intor('int1e_ovlp')
    angs = mol_cart._bas[:, gto.ANG_OF]

    cart_offset = mol_cart.ao_loc
    sph_offset = mol_sph.ao_loc
    
    s_cart = cp.asarray(s_cart, order='C')
    s_cart2sph = cart2sph(s_cart, angs, cart_offset, sph_offset, mol_sph.nao)

    assert np.linalg.norm(s_sph - s_cart2sph.get()) < 1e-10

def test_sph2cart():
    basis = '''
O    S
      0.2700058226E+00      1
O    P
      0.2700058226E+00      1
O    D
      0.2700058226E+00      1
O    F
      0.2700058226E+00      1
      '''
    mol_cart = gto.M(atom='O 0 0 0', basis=basis, cart=True)
    mol_sph = gto.M(atom='O 0 0 0', basis=basis, cart=False)

    s_sph = mol_sph.intor('int1e_ovlp')
    angs = mol_cart._bas[:, gto.ANG_OF]

    cart_offset = mol_cart.ao_loc
    sph_offset = mol_sph.ao_loc
    
    s_sph = cp.asarray(s_sph, order='C')

    _c2s = {}
    for l in range(5):
        c2s = gto.mole.cart2sph(l, normalized='sp')
        _c2s[l] = cp.asarray(c2s, order='C')

    eye = cp.eye(mol_sph.nao)
    s0 = sph2cart(eye, angs, cart_offset, sph_offset, mol_cart.nao)
    s1 = cp.empty([mol_cart.nao, mol_cart.nao])
    for p in range(len(cart_offset)-1):
        for q in range(len(sph_offset)-1):
            c2s_left = _c2s[angs[p]]
            c2s_right = _c2s[angs[q]]

            sph_block = eye[sph_offset[p]:sph_offset[p+1], sph_offset[q]:sph_offset[q+1]]
            cart_block = c2s_left @ sph_block @ c2s_right.T
            s1[cart_offset[p]:cart_offset[p+1], cart_offset[q]:cart_offset[q+1]] = cart_block

    assert cp.linalg.norm(s0 - s1) < 1e-10

test_cart2sph()
test_sph2cart()
