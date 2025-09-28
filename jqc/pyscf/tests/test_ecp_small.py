# Adapted from GPU4PySCF gpu4pyscf/gto/tests/test_ecp.py
# Small molecule ECP tests
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
    global mol_small
    mol_small = gto.M(
        atom="Cu 0 0 0",
        basis="sto-3g",
        ecp="crenbl",
        spin=1,
        charge=0,
        cart=1,  # Use spherical basis to avoid cartesian ECP IP kernel bug
        output="/dev/null",
        verbose=0,
    )


def tearDownModule():
    global mol_small
    mol_small.stdout.close()
    del mol_small


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

    def test_ecp_small_cart_ip1(self):
        # Match gpu4pyscf iprinv style: compare per-ECP-atom contributions
        h1_gpu = get_ecp_ip(mol_small)
        ecp_atoms = set(mol_small._ecpbas[:, gto.ATOM_OF])
        for atm_id in ecp_atoms:
            with mol_small.with_rinv_at_nucleus(atm_id):
                # mol_small uses spherical AOs (cart=0); compare against spherical reference
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


if __name__ == "__main__":
    unittest.main()