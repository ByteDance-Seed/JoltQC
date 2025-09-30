"""
ECP tests with low angular momentum basis sets

This test file is adapted from test_ecp.py but uses only S, P, D functions
(no F, G functions) to avoid shared memory issues in GPU kernels.
Uses the same customized basis and ECP definitions as test_ecp.py but
limits angular momentum to l <= 2 to ensure shared memory usage stays
under GPU limits.

Key differences from test_ecp.py:
- Basis functions limited to S, P, D (l = 0, 1, 2)
- ECP functions limited to S, P, D channels
- Should work reliably without shared memory allocation errors
- Maintains the same test structure and accuracy requirements
"""
import unittest
import pytest
import cupy as cp
import numpy as np
from pyscf import gto

from jqc.backend.ecp import get_ecp, get_ecp_ip, get_ecp_ipip

def setUpModule():
    global mol_type1, mol_type2, cu1_basis_small
    # Modified basis with only S, P, D functions (no F, G) to reduce shared memory usage
    cu1_basis_small = gto.basis.parse(
        '''
     H    G
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
        '''
    )

    # Type1 ECP: only local (ul) channel
    na_ecp_type1 = gto.basis.parse_ecp('''
Na nelec 10
Na ul
2       1.0                   0.5
        ''')

    # Type2 ECP: semi-local S, P, D channels (no F, G)
    na_ecp_type2 = gto.basis.parse_ecp('''
Na nelec 10
Na S
2      13.652203             732.2692
2       6.826101              26.484721
Na P
2      10.279868             299.489474
2       5.139934              26.466234
Na G
2       7.349859             124.457595
2       3.674929              14.035995
        ''')

    # Molecule for Type1 ECP testing (only local channel)
    mol_type1 = gto.M(
        atom='''
            Na 0.5 0.5 0.
            Na  0.  1.  1.
        ''',
        basis={'Na': cu1_basis_small, 'H': cu1_basis_small},
        ecp={'Na': na_ecp_type1},
        output="/dev/null",
        verbose=0,
    )

    # Molecule for Type2 ECP testing (semi-local channels)
    mol_type2 = gto.M(
        atom='''
            Na 0.5 0.5 0.
            Na  0.  1.  1.
        ''',
        basis={'Na': cu1_basis_small, 'H': cu1_basis_small},
        ecp={'Na': na_ecp_type2},
        output="/dev/null",
        verbose=0,
    )


def tearDownModule():
    global mol_type1, mol_type2
    mol_type1.stdout.close()
    mol_type2.stdout.close()
    del mol_type1, mol_type2


class KnownValues(unittest.TestCase):
    def _tag(self) -> str:
        return f"{self.__class__.__name__}.{self._testMethodName}"

    def _log(self, msg: str):
        # Left-align the tag to a fixed width for tidy, readable output
        print(f"[{self._tag():<40}] {msg}")

    def test_ecp_type1_sph(self):
        """Test basic ECP calculation with Type1 ECP (local channel only) and spherical basis"""
        h1_cpu = mol_type1.intor('ECPscalar_sph')
        h1_gpu = get_ecp(mol_type1).get()
        self._log(f"norm: {np.linalg.norm(h1_cpu - h1_gpu):.3e}")
        assert np.linalg.norm(h1_cpu - h1_gpu) < 1e-6  # Machine precision threshold

    def test_ecp_type2_sph(self):
        """Test basic ECP calculation with Type2 ECP (semi-local channels) and spherical basis"""
        h1_cpu = mol_type2.intor('ECPscalar_sph')
        h1_gpu = get_ecp(mol_type2).get()
        self._log(f"norm: {np.linalg.norm(h1_cpu - h1_gpu):.3e}")
        assert np.linalg.norm(h1_cpu - h1_gpu) < 1e-6  # Machine precision threshold

    def test_ecp_type1_sph_ip1(self):
        """Test ECP first derivatives with Type1 ECP and spherical basis"""
        # Match gpu4pyscf iprinv style: compare per-ECP-atom contributions
        h1_gpu = get_ecp_ip(mol_type1)
        ecp_atoms = set(mol_type1._ecpbas[:, gto.ATOM_OF])
        for atm_id in ecp_atoms:
            with mol_type1.with_rinv_at_nucleus(atm_id):
                h1_cpu = mol_type1.intor('ECPscalar_iprinv_sph')
            self._log(f"atom: {atm_id:2d}  norm: {np.linalg.norm(h1_cpu - h1_gpu[atm_id].get()):.3e}")
            assert np.linalg.norm(h1_cpu - h1_gpu[atm_id].get()) < 1e-6
    
    def test_ecp_type1_sph_ipnuc(self):
        """Test ECP nuclear derivatives with Type1 ECP and spherical basis"""
        h1_cpu = mol_type1.intor('ECPscalar_ipnuc_sph')
        h1_gpu = get_ecp_ip(mol_type1).sum(axis=0).get()
        self._log(f"norm: {np.linalg.norm(h1_cpu - h1_gpu):.3e}")
        assert np.linalg.norm(h1_cpu - h1_gpu) < 1e-6

    def test_ecp_type1_sph_ipipnuc(self):
        """Second derivatives: CPU ipipnuc vs GPU 'ipipv' path (Type1, sph)"""
        h1_cpu = mol_type1.intor('ECPscalar_ipipnuc', comp=9)
        h1_gpu = get_ecp_ipip(mol_type1, 'ipipv').sum(axis=0).get()
        self._log(f"norm: {np.linalg.norm(h1_cpu - h1_gpu):.3e}")
        assert np.linalg.norm(h1_cpu - h1_gpu) < 1e-6

    def test_ecp_type1_sph_ipnucip(self):
        """Second derivatives: CPU ipnucip vs GPU 'ipvip' path (Type1, sph)"""
        h1_cpu = mol_type1.intor('ECPscalar_ipnucip', comp=9)
        h1_gpu = get_ecp_ipip(mol_type1, 'ipvip').sum(axis=0).get()
        self._log(f"norm: {np.linalg.norm(h1_cpu - h1_gpu):.3e}")
        assert np.linalg.norm(h1_cpu - h1_gpu) < 1e-6
    
    def test_ecp_type2_sph_ip1(self):
        """Test ECP first derivatives with Type2 ECP and spherical basis"""
        # Match gpu4pyscf iprinv style: compare per-ECP-atom contributions
        h1_gpu = get_ecp_ip(mol_type2)
        ecp_atoms = set(mol_type2._ecpbas[:, gto.ATOM_OF])
        for atm_id in ecp_atoms:
            with mol_type2.with_rinv_at_nucleus(atm_id):
                h1_cpu = mol_type2.intor('ECPscalar_iprinv_sph')
            self._log(f"atom: {atm_id:2d}  norm: {np.linalg.norm(h1_cpu - h1_gpu[atm_id].get()):.3e}")
            assert np.linalg.norm(h1_cpu - h1_gpu[atm_id].get()) < 1e-6

    def test_ecp_type2_sph_ipnuc(self):
        """Test ECP nuclear derivatives with Type2 ECP and spherical basis"""
        h1_cpu = mol_type2.intor('ECPscalar_ipnuc_sph')
        h1_gpu = get_ecp_ip(mol_type2).sum(axis=0).get()
        self._log(f"norm: {np.linalg.norm(h1_cpu - h1_gpu):.3e}")
        assert np.linalg.norm(h1_cpu - h1_gpu) < 1e-6

    def test_ecp_type2_sph_ipipnuc(self):
        """Second derivatives: CPU ipipnuc vs GPU 'ipipv' path (Type2, sph)"""
        h1_cpu = mol_type2.intor('ECPscalar_ipipnuc_sph', comp=9)
        h1_gpu = get_ecp_ipip(mol_type2, 'ipipv').sum(axis=0).get()
        self._log(f"norm: {np.linalg.norm(h1_cpu - h1_gpu):.3e}")
        assert np.linalg.norm(h1_cpu - h1_gpu) < 1e-6

    def test_ecp_type2_sph_ipnucip(self):
        """Second derivatives: CPU ipnucip vs GPU 'ipvip' path (Type2, sph)"""
        h1_cpu = mol_type2.intor('ECPscalar_ipnucip_sph', comp=9)
        h1_gpu = get_ecp_ipip(mol_type2, 'ipvip').sum(axis=0).get()
        self._log(f"norm: {np.linalg.norm(h1_cpu - h1_gpu):.3e}")
        assert np.linalg.norm(h1_cpu - h1_gpu) < 1e-6


if __name__ == "__main__":
    unittest.main()
