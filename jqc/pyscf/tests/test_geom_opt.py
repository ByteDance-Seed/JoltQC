"""
Test geometry optimization with JoltQC using the apply API.

This test demonstrates the use of JoltQC with molecules at different
geometries, testing that the new architecture works correctly for
molecules with different coordinates.
"""

import numpy as np
from pyscf import gto

from jqc.pyscf.jk import generate_jk_kernel
from jqc.pyscf.rks import generate_nr_rks


class TestGeometryOptimization:
    """Test geometry optimization scenarios with JoltQC."""

    def get_h2_molecules(self):
        """Get H2 molecules at different bond lengths for geometry optimization test."""
        # H2 at different bond lengths (simulating geometry optimization steps)
        bond_lengths = [0.6, 0.7, 0.8]  # in Angstrom
        molecules = []

        for r in bond_lengths:
            mol = gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis="sto-3g", verbose=0)
            molecules.append(mol)

        return molecules

    def test_jk_kernel_different_geometries(self):
        """Test JK kernel works with molecules at different geometries."""
        molecules = self.get_h2_molecules()

        # Generate JK kernel once
        get_jk = generate_jk_kernel()

        # Test that the kernel works with different geometries
        results = []
        for i, mol in enumerate(molecules):
            # Create simple density matrix
            dm = np.eye(mol.nao)

            # Calculate J/K matrices
            vj, vk = get_jk(mol, dm, hermi=1)

            # Convert to numpy
            vj_np = vj.get() if hasattr(vj, "get") else vj
            vk_np = vk.get() if hasattr(vk, "get") else vk

            results.append((vj_np, vk_np))
            print(f"Geometry {i+1}: VJ shape = {vj_np.shape}, VK shape = {vk_np.shape}")

        # All results should have the same shape
        for i, (vj, vk) in enumerate(results):
            assert vj.shape == (mol.nao, mol.nao), f"Wrong VJ shape for geometry {i+1}"
            assert vk.shape == (mol.nao, mol.nao), f"Wrong VK shape for geometry {i+1}"

        # Results should be different for different geometries
        vj1, vk1 = results[0]
        vj2, vk2 = results[1]
        vj3, vk3 = results[2]

        assert (
            np.linalg.norm(vj1 - vj2) > 1e-8
        ), "VJ matrices should differ for different geometries"
        assert (
            np.linalg.norm(vk1 - vk2) > 1e-8
        ), "VK matrices should differ for different geometries"
        assert (
            np.linalg.norm(vj2 - vj3) > 1e-8
        ), "VJ matrices should differ for different geometries"
        assert (
            np.linalg.norm(vk2 - vk3) > 1e-8
        ), "VK matrices should differ for different geometries"

    def test_rks_kernel_generation(self):
        """Test RKS kernel can be generated successfully."""
        # Just test that the kernel generation works
        rks_kernel = generate_nr_rks()
        assert callable(rks_kernel), "RKS kernel should be callable"

    def test_basis_layout_generation(self):
        """Test that basis layouts are generated correctly for different geometries."""
        from jqc.pyscf.basis import BasisLayout

        molecules = self.get_h2_molecules()

        # Clear cache to start fresh
        BasisLayout.from_mol.cache_clear()

        # Generate basis layouts for each molecule
        layouts = []
        for mol in molecules:
            layout = BasisLayout.from_mol(mol, alignment=1)
            layouts.append(layout)

        # Check that all layouts have the same number of basis functions
        # since they have the same atoms and basis set
        for i, layout in enumerate(layouts):
            assert (
                layout._mol.nao == molecules[0].nao
            ), f"Inconsistent nao for geometry {i+1}"
            assert layout.nbasis > 0, f"No basis functions for geometry {i+1}"

        # Cache should have entries
        cache_info = BasisLayout.from_mol.cache_info()
        assert cache_info.currsize > 0, "Cache should have entries"
