import unittest
import numpy as np
import pyscf

from gpu4pyscf import dft
import jqc.pyscf


# Iodine dimer with def2 ECP/basis (matches GPU4PySCF test)
atom = """
I 0 0 0
I 1 0 0
"""
bas = "def2-tzvpp"
grids_level = 7


def setUpModule():
    global mol
    mol = pyscf.M(atom=atom, basis=bas, ecp=bas, output="/dev/null", verbose=1)
    mol.build()


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


def run_dft(xc):
    mf = dft.RKS(mol, xc=xc)
    # Match GPU4PySCF grid settings
    mf.grids.level = grids_level
    mf.grids.prune = None
    # Apply JoltQC adapters/backends
    mf = jqc.pyscf.apply(mf)
    return mf.kernel()


class KnownValues(unittest.TestCase):
    def test_rks_pbe_ecp(self):
        e_tot = run_dft("PBE")
        # Reference value from GPU4PySCF test_dft_ecp.py
        e_ref = -582.7625143308
        assert np.allclose(e_tot, e_ref, rtol=1e-8, atol=0.0)


if __name__ == "__main__":
    unittest.main()
