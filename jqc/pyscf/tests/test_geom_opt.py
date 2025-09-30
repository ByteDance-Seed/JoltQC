"""
Test geometry optimization with JoltQC using the apply API.

This test demonstrates the use of JoltQC with molecules at different
geometries, comparing JoltQC results with GPU4PySCF reference calculations.
"""

import numpy as np
import pytest
from pyscf import gto
from gpu4pyscf import scf as gpu_scf, dft as gpu_dft

from jqc.pyscf import apply


class TestGeometryOptimization:
    """Test geometry optimization scenarios with JoltQC."""

    def get_water_molecules(self):
        """Get water molecules at different O-H bond lengths."""
        # Water at different O-H bond lengths (simulating geometry optimization steps)
        base_geom = """
        O       0.0000000000    -0.0000000000     0.1174000000
        H      -{0:.3f}000000    -0.0000000000    -0.4696000000
        H       {0:.3f}000000     0.0000000000    -0.4696000000
        """

        # Different O-H distances
        distances = [0.85, 0.90, 0.95]  # in Angstrom
        molecules = []

        for dist in distances:
            atom_str = base_geom.format(dist)
            mol = gto.M(atom=atom_str, basis='sto-3g', verbose=0)
            molecules.append(mol)

        return molecules

    def test_rhf_energy_comparison(self):
        """Test RHF energy calculation comparing JoltQC with GPU4PySCF."""
        molecules = self.get_water_molecules()

        for i, mol in enumerate(molecules):
            # GPU4PySCF reference calculation
            mf_ref = gpu_scf.RHF(mol)
            mf_ref.conv_tol = 1e-8
            e_ref = mf_ref.kernel()

            # JoltQC calculation
            mf_jolt = gpu_scf.RHF(mol)
            mf_jolt.conv_tol = 1e-8
            mf_jolt = apply(mf_jolt)
            e_jolt = mf_jolt.kernel()

            # Results should be very close
            energy_diff = abs(e_ref - e_jolt)
            print(f"Geometry {i+1}: GPU4PySCF = {e_ref:.8f}, JoltQC = {e_jolt:.8f}, diff = {energy_diff:.2e}")
            assert energy_diff < 1e-6, f"Energy difference too large: {energy_diff}"

    def test_rks_energy_comparison(self):
        """Test RKS (DFT) energy calculation comparing JoltQC with GPU4PySCF."""
        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)

        # GPU4PySCF reference calculation
        mf_ref = gpu_dft.RKS(mol)
        mf_ref.xc = 'lda,vwn'
        mf_ref.conv_tol = 1e-8
        e_ref = mf_ref.kernel()

        # JoltQC calculation
        mf_jolt = gpu_dft.RKS(mol)
        mf_jolt.xc = 'lda,vwn'
        mf_jolt.conv_tol = 1e-8
        mf_jolt = apply(mf_jolt)
        e_jolt = mf_jolt.kernel()

        # Results should be very close
        energy_diff = abs(e_ref - e_jolt)
        print(f"RKS: GPU4PySCF = {e_ref:.8f}, JoltQC = {e_jolt:.8f}, diff = {energy_diff:.2e}")
        assert energy_diff < 1e-6, f"Energy difference too large: {energy_diff}"

    def test_geometry_dependent_energies(self):
        """Test that energies change appropriately with geometry."""
        molecules = self.get_water_molecules()
        energies_ref = []
        energies_jolt = []

        for mol in molecules:
            # GPU4PySCF reference
            mf_ref = gpu_scf.RHF(mol)
            mf_ref.conv_tol = 1e-8
            e_ref = mf_ref.kernel()
            energies_ref.append(e_ref)

            # JoltQC calculation
            mf_jolt = gpu_scf.RHF(mol)
            mf_jolt.conv_tol = 1e-8
            mf_jolt = apply(mf_jolt)
            e_jolt = mf_jolt.kernel()
            energies_jolt.append(e_jolt)

        # Both methods should show the same energy trend
        energies_ref = np.array(energies_ref)
        energies_jolt = np.array(energies_jolt)

        # Find minimum energy geometries for both methods
        min_idx_ref = np.argmin(energies_ref)
        min_idx_jolt = np.argmin(energies_jolt)

        # Should find the same minimum geometry
        assert min_idx_ref == min_idx_jolt, "Different minimum geometries found"

        # Energy differences should be consistent
        for i in range(len(molecules)):
            diff = abs(energies_ref[i] - energies_jolt[i])
            assert diff < 1e-6, f"Inconsistent energy at geometry {i}: {diff}"