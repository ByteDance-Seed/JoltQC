# Adapted from GPU4PySCF gpu4pyscf/gto/tests/test_ecp.py
import unittest
import numpy as np
import cupy as cp
from pyscf import gto

from jqc.backend.ecp import get_ecp


def setUpModule():
    global mol_cart, mol_sph, cu1_basis
    cu1_basis = gto.basis.parse(
        '''
     H    S
           1.8000000              1.0000000
     H    S
           2.8000000              0.0210870             -0.0045400              0.0000000
           1.3190000              0.3461290             -0.1703520              0.0000000
           0.9059000              0.0393780              0.1403820              1.0000000
     H    P
           2.1330000              0.0868660              0.0000000
           1.2000000              0.0000000              0.5000000
           0.3827000              0.5010080              1.0000000
     H    D
           0.3827000              1.0000000
     H    F
           2.1330000              0.1868660              0.0000000
           0.3827000              0.2010080              1.0000000
     H    G
            6.491000E-01           1.0000000
        '''
    )

    mol_cart = gto.M(
        atom="Cu 0 0 0",
        basis="sto-3g",
        ecp="crenbl",
        spin=1,
        charge=0,
        cart=1,
        output="/dev/null",
        verbose=0,
    )

    mol_sph = gto.M(
        atom='''
            Na 0.5 0.5 0.
            Na  0.  1.  1.
        ''',
        basis={'Na': cu1_basis, 'H': cu1_basis},
        ecp={'Na': gto.basis.parse_ecp('''
Na nelec 10
Na ul
2       1.0                   0.5
Na S
2      13.652203             732.2692
2       6.826101              26.484721
Na P
2      10.279868             299.489474
2       5.139934              26.466234
Na D
2       7.349859             124.457595
2       3.674929              14.035995
Na F
2       3.034072              21.531031
Na G
2       4.808857             -21.607597
        ''')},
        output="/dev/null",
        verbose=0,
    )


def tearDownModule():
    global mol_cart, mol_sph
    mol_cart.stdout.close()
    mol_sph.stdout.close()
    del mol_cart, mol_sph


class KnownValues(unittest.TestCase):
    def test_ecp_cart(self):
        h1_cpu = mol_cart.intor('ECPscalar_cart')
        h1_gpu = get_ecp(mol_cart).get()
        assert np.linalg.norm(h1_cpu - h1_gpu) < 2e-10  # Machine precision threshold
    
    def test_ecp_sph(self):
        h1_cpu = mol_sph.intor('ECPscalar_sph')
        h1_gpu = get_ecp(mol_sph).get()
        assert np.linalg.norm(h1_cpu - h1_gpu) < 2e-10  # Machine precision threshold


if __name__ == "__main__":
    unittest.main()

