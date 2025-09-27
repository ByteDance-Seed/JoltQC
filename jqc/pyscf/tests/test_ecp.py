# Adapted from GPU4PySCF gpu4pyscf/gto/tests/test_ecp.py
import unittest
import pytest
import cupy as cp
import numpy as np
from pyscf import gto

from jqc.backend.ecp import get_ecp, get_ecp_ip, get_ecp_ipip

# Skip module if no CUDA device is available
try:
    _ndev = cp.cuda.runtime.getDeviceCount()
except Exception:
    _ndev = 0
if _ndev == 0:
    pytestmark = pytest.mark.skip(reason="No CUDA device available for ECP tests")


def setUpModule():
    global mol_small, mol_cart, mol_sph, cu1_basis
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

    mol_small = gto.M(
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

    mol_cart = gto.M(
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
        cart=1,
        output="/dev/null",
        verbose=0,
    )


def tearDownModule():
    global mol_small, mol_cart, mol_sph
    mol_small.stdout.close()
    mol_cart.stdout.close()
    mol_sph.stdout.close()
    del mol_small, mol_cart, mol_sph


class KnownValues(unittest.TestCase):
    def _tag(self) -> str:
        return f"{self.__class__.__name__}.{self._testMethodName}"

    def _log(self, msg: str):
        # Left-align the tag to a fixed width for tidy, readable output
        print(f"[{self._tag():<40}] {msg}")

    def test_ecp_small_cart(self):
        h1_cpu = mol_small.intor('ECPscalar_cart')
        h1_gpu = get_ecp(mol_small).get()
        self._log(f"norm: {np.linalg.norm(h1_cpu - h1_gpu):.3e}")
        assert np.linalg.norm(h1_cpu - h1_gpu) < 1e-6  # Machine precision threshold

    def test_ecp_cart(self):
        h1_cpu = mol_cart.intor('ECPscalar_cart')
        h1_gpu = get_ecp(mol_cart).get()
        self._log(f"norm: {np.linalg.norm(h1_cpu - h1_gpu):.3e}")
        assert np.linalg.norm(h1_cpu - h1_gpu) < 1e-6  # Machine precision threshold

    def test_ecp_sph(self):
        h1_cpu = mol_sph.intor('ECPscalar_sph')
        h1_gpu = get_ecp(mol_sph).get()
        self._log(f"norm: {np.linalg.norm(h1_cpu - h1_gpu):.3e}")
        assert np.linalg.norm(h1_cpu - h1_gpu) < 1e-6  # Machine precision threshold

    def test_ecp_small_cart_ip1(self):
        # Match gpu4pyscf iprinv style: compare per-ECP-atom contributions
        h1_gpu = get_ecp_ip(mol_small)
        ecp_atoms = set(mol_small._ecpbas[:, gto.ATOM_OF])
        for atm_id in ecp_atoms:
            with mol_small.with_rinv_at_nucleus(atm_id):
                h1_cpu = mol_small.intor('ECPscalar_iprinv_cart')
            self._log(f"atom: {atm_id:2d}  norm: {np.linalg.norm(h1_cpu - h1_gpu[atm_id].get()):.3e}")
            assert np.linalg.norm(h1_cpu - h1_gpu[atm_id].get()) < 1e-6

    def test_ecp_small_cart_ipipv(self):
        h1_cpu = mol_small.intor('ECPscalar_ipipnuc', comp=9)
        h1_gpu = get_ecp_ipip(mol_small, 'ipipv').sum(axis=0).get()
        self._log(f"norm: {np.linalg.norm(h1_cpu - h1_gpu):.3e}")
        assert np.linalg.norm(h1_cpu - h1_gpu) < 1e-6

    def test_ecp_small_cart_ipvip(self):
        h1_cpu = mol_small.intor('ECPscalar_ipnucip', comp=9)
        h1_gpu = get_ecp_ipip(mol_small, 'ipvip').sum(axis=0).get()
        self._log(f"norm: {np.linalg.norm(h1_cpu - h1_gpu):.3e}")
        assert np.linalg.norm(h1_cpu - h1_gpu) < 1e-6

    def test_ecp_cart_ip1(self):
        # Match gpu4pyscf iprinv style: compare per-ECP-atom contributions
        h1_gpu = get_ecp_ip(mol_cart)
        ecp_atoms = set(mol_cart._ecpbas[:, gto.ATOM_OF])
        for atm_id in ecp_atoms:
            with mol_cart.with_rinv_at_nucleus(atm_id):
                h1_cpu = mol_cart.intor('ECPscalar_iprinv_cart')
            self._log(f"atom: {atm_id:2d}  norm: {np.linalg.norm(h1_cpu - h1_gpu[atm_id].get()):.3e}")
            assert np.linalg.norm(h1_cpu - h1_gpu[atm_id].get()) < 1e-6

    def test_ecp_cart_ipnuc(self):
        h1_cpu = mol_cart.intor('ECPscalar_ipnuc_cart')
        h1_gpu = get_ecp_ip(mol_cart).sum(axis=0).get()
        self._log(f"norm: {np.linalg.norm(h1_cpu - h1_gpu):.3e}")
        assert np.linalg.norm(h1_cpu - h1_gpu) < 1e-6

    def test_ecp_cart_ipipv(self):
        h1_cpu = mol_cart.intor('ECPscalar_ipipnuc', comp=9)
        h1_gpu = get_ecp_ipip(mol_cart, 'ipipv').sum(axis=0).get()
        self._log(f"norm: {np.linalg.norm(h1_cpu - h1_gpu):.3e}")
        assert np.linalg.norm(h1_cpu - h1_gpu) < 1e-6

    def test_ecp_cart_ipvip_cart(self):
        h1_cpu = mol_cart.intor('ECPscalar_ipnucip', comp=9)
        h1_gpu = get_ecp_ipip(mol_cart, 'ipvip').sum(axis=0).get()
        self._log(f"norm: {np.linalg.norm(h1_cpu - h1_gpu):.3e}")
        assert np.linalg.norm(h1_cpu - h1_gpu) < 1e-6

    def test_ecp_sph_ip1(self):
        # Match gpu4pyscf iprinv style: compare per-ECP-atom contributions
        h1_gpu = get_ecp_ip(mol_sph)
        ecp_atoms = set(mol_sph._ecpbas[:, gto.ATOM_OF])
        for atm_id in ecp_atoms:
            with mol_sph.with_rinv_at_nucleus(atm_id):
                h1_cpu = mol_sph.intor('ECPscalar_iprinv_sph')
            self._log(f"atom: {atm_id:2d}  norm: {np.linalg.norm(h1_cpu - h1_gpu[atm_id].get()):.3e}")
            assert np.linalg.norm(h1_cpu - h1_gpu[atm_id].get()) < 1e-6

    def test_ecp_sph_ipnuc(self):
        h1_cpu = mol_sph.intor('ECPscalar_ipnuc_sph')
        h1_gpu = get_ecp_ip(mol_sph).sum(axis=0).get()
        self._log(f"norm: {np.linalg.norm(h1_cpu - h1_gpu):.3e}")
        assert np.linalg.norm(h1_cpu - h1_gpu) < 1e-6

    def test_ecp_sph_ipipv(self):
        h1_cpu = mol_sph.intor('ECPscalar_ipipnuc', comp=9)
        h1_gpu = get_ecp_ipip(mol_sph, 'ipipv').sum(axis=0).get()
        self._log(f"norm: {np.linalg.norm(h1_cpu - h1_gpu):.3e}")
        assert np.linalg.norm(h1_cpu - h1_gpu) < 1e-6

    def test_ecp_sph_ipvip(self):
        h1_cpu = mol_sph.intor('ECPscalar_ipnucip', comp=9)
        h1_gpu = get_ecp_ipip(mol_sph, 'ipvip').sum(axis=0).get()
        self._log(f"norm: {np.linalg.norm(h1_cpu - h1_gpu):.3e}")
        assert np.linalg.norm(h1_cpu - h1_gpu) < 1e-6


if __name__ == "__main__":
    unittest.main()
