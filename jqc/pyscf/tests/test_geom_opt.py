"""
Test geometry optimization with JoltQC using the apply API.

This test demonstrates the use of JoltQC with molecules at different
geometries, testing that the new architecture works correctly for
molecules with different coordinates.
"""

import unittest
import numpy as np
import pyscf
from pyscf import gto
from gpu4pyscf import dft

from jqc.pyscf.jk import generate_jk_kernel
from jqc.pyscf.rks import generate_nr_rks
import jqc.pyscf


class TestGeometryOptimization(unittest.TestCase):
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

    def test_basis_layout_generation(self):
        """Test that basis layouts are generated correctly for different geometries."""
        from jqc.pyscf.basis import BasisLayout

        molecules = self.get_h2_molecules()

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

    def test_reset_rks_object(self):
        """Test reset function with RKS object"""
        # Original molecule - H2 with distance 1.4 Bohr
        mol1 = pyscf.M(
            atom="""
            H  0.0  0.0  0.0
            H  0.0  0.0  1.4
            """,
            basis="sto-3g",
            output="/dev/null",
            verbose=0,
        )

        # New molecule - H2 with distance 1.6 Bohr
        mol2 = pyscf.M(
            atom="""
            H  0.0  0.0  0.0
            H  0.0  0.0  1.6
            """,
            basis="sto-3g",
            output="/dev/null",
            verbose=0,
        )

        # JoltQC calculation
        mf_jolt = dft.RKS(mol1, xc="PBE")
        mf_jolt.grids.level = 2  # Lower level for speed
        mf_jolt = jqc.pyscf.apply(mf_jolt)
        e1_jolt = mf_jolt.kernel()

        # Reset with new molecule
        mf_reset = mf_jolt.reset(mol2)
        e2_jolt = mf_reset.kernel()

        # Reference PySCF calculations
        mf_ref1 = dft.RKS(mol1, xc="PBE")
        mf_ref1.grids.level = 2
        e1_ref = mf_ref1.kernel()

        mf_ref2 = dft.RKS(mol2, xc="PBE")
        mf_ref2.grids.level = 2
        e2_ref = mf_ref2.kernel()

        # Compare energies
        assert abs(e1_jolt - e1_ref) < 1e-8, f"Energy error for mol1: {abs(e1_jolt - e1_ref):.2e}"
        assert abs(e2_jolt - e2_ref) < 1e-8, f"Energy error for mol2: {abs(e2_jolt - e2_ref):.2e}"

        # Verify that JoltQC kernels are still applied after reset
        assert hasattr(mf_reset, "get_jk")
        assert hasattr(mf_reset._numint, "get_rho")
        assert hasattr(mf_reset._numint, "nr_rks")

        mol1.stdout.close()
        mol2.stdout.close()

    def test_reset_without_recursion(self):
        """Test that reset function doesn't cause RecursionError"""
        mol1 = pyscf.M(
            atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", output="/dev/null", verbose=0
        )
        mol2 = pyscf.M(
            atom="H 0 0 0; H 0 0 1.6", basis="sto-3g", output="/dev/null", verbose=0
        )

        mf = dft.RKS(mol1, xc="PBE")
        mf.grids.level = 2

        # Apply JoltQC
        mf_jolt = jqc.pyscf.apply(mf)

        # This should not cause RecursionError
        mf_reset = mf_jolt.reset(mol2)
        assert mf_reset is not None, "Reset function failed"

        mol1.stdout.close()
        mol2.stdout.close()

    def test_geometry_optimization(self):
        """Test actual geometry optimization with JoltQC and compare with PySCF"""
        # Start with a slightly distorted water molecule
        atom = """
        O       0.0000000000    -0.0000000000     0.1300000000
        H      -0.8000000000    -0.0000000000    -0.5200000000
        H       0.8000000000     0.0000000000    -0.5200000000
        """

        mol_jolt = pyscf.M(atom=atom, basis="sto-3g", output="/dev/null", verbose=0)
        mol_pyscf = pyscf.M(atom=atom, basis="sto-3g", output="/dev/null", verbose=0)

        # Create RKS object with JoltQC
        mf_jolt = dft.RKS(mol_jolt, xc="PBE")
        mf_jolt.grids.level = 2  # Lower level for speed
        mf_jolt.conv_tol = 1e-6
        mf_jolt.max_cycle = 20
        mf_jolt = jqc.pyscf.apply(mf_jolt)

        # Create reference RKS object with PySCF (GPU4PySCF)
        mf_pyscf = dft.RKS(mol_pyscf, xc="PBE").to_gpu()
        mf_pyscf.grids.level = 2
        mf_pyscf.conv_tol = 1e-6
        mf_pyscf.max_cycle = 20

        # Perform geometry optimization with JoltQC (reduced steps for speed)
        from pyscf.geomopt.geometric_solver import optimize
        mol_opt_jolt = optimize(mf_jolt, maxsteps=5)

        # Perform geometry optimization with PySCF reference
        mol_opt_pyscf = optimize(mf_pyscf, maxsteps=5)

        # Get final energies
        mf_final_jolt = dft.RKS(mol_opt_jolt, xc="PBE")
        mf_final_jolt.grids.level = 2
        mf_final_jolt = jqc.pyscf.apply(mf_final_jolt)
        e_final_jolt = mf_final_jolt.kernel()

        mf_final_pyscf = dft.RKS(mol_opt_pyscf, xc="PBE").to_gpu()
        mf_final_pyscf.grids.level = 2
        e_final_pyscf = mf_final_pyscf.kernel()

        # Compare final energies
        energy_diff = abs(e_final_jolt - e_final_pyscf)
        assert energy_diff < 1e-6, f"Final energies should match: JoltQC={e_final_jolt:.8f}, PySCF={e_final_pyscf:.8f}, diff={energy_diff:.2e}"

        # Compare final geometries
        coords_jolt = mol_opt_jolt.atom_coords()
        coords_pyscf = mol_opt_pyscf.atom_coords()
        coord_diff = np.linalg.norm(coords_jolt - coords_pyscf)
        assert coord_diff < 1e-4, f"Final geometries should match: diff={coord_diff:.2e}"

        # Verify that geometry changed during optimization
        initial_coords = mol_jolt.atom_coords()
        geom_change = np.linalg.norm(initial_coords - coords_jolt)
        assert geom_change > 1e-6, f"Geometry should change during optimization: {geom_change:.2e}"

        mol_jolt.stdout.close()
        mol_pyscf.stdout.close()

    def test_pyscf_scanner(self):
        """Test PySCF scanner functionality with JoltQC"""
        # Create a base molecule
        mol1 = pyscf.M(
            atom="""
            H  0.0  0.0  0.0
            H  0.0  0.0  1.4
            """,
            basis="sto-3g",
            output="/dev/null",
            verbose=0,
        )

        # Create RKS object and apply JoltQC
        mf = dft.RKS(mol1, xc="PBE")
        mf.grids.level = 2
        mf.conv_tol = 1e-6
        mf.max_cycle = 20
        mf_jolt = jqc.pyscf.apply(mf)

        # Create scanner from JoltQC-accelerated method
        scanner = mf_jolt.as_scanner()

        # Test scanner with different molecules (different H-H distances)
        distances = [0.8, 1.4, 3.0]  # Very different distances
        energies = []

        for d in distances:
            mol = pyscf.M(
                atom=f"""
                H  0.0  0.0  0.0
                H  0.0  0.0  {d}
                """,
                basis="sto-3g",
                output="/dev/null",
                verbose=0,
            )

            # Use scanner to compute energy
            energy = scanner(mol)
            energies.append(energy)
            mol.stdout.close()

        # Verify that scanner computed different energies for different geometries
        # Check that energies are different (more than 1e-7 difference is sufficient)
        for i in range(len(energies)):
            for j in range(i+1, len(energies)):
                assert abs(energies[i] - energies[j]) > 1e-7, f"Energies should be different: {energies[i]:.8f} vs {energies[j]:.8f}"

        # Verify that scanner attributes are available
        assert hasattr(scanner, 'e_tot'), "Scanner should have e_tot attribute"
        assert hasattr(scanner, 'converged'), "Scanner should have converged attribute"

        # Verify that the last energy matches the scanner's e_tot
        assert abs(energies[-1] - scanner.e_tot) < 1e-10, "Scanner e_tot should match last computed energy"

        mol1.stdout.close()

    def test_gradient_scanner(self):
        """Test PySCF gradient scanner functionality with JoltQC"""
        # Create a base molecule (water)
        mol1 = pyscf.M(
            atom="""
            O       0.0000000000    -0.0000000000     0.1174000000
            H      -0.7570000000    -0.0000000000    -0.4696000000
            H       0.7570000000     0.0000000000    -0.4696000000
            """,
            basis="sto-3g",
            output="/dev/null",
            verbose=0,
        )

        # Create RKS object and apply JoltQC
        mf = dft.RKS(mol1, xc="PBE")
        mf.grids.level = 2
        mf.conv_tol = 1e-6
        mf.max_cycle = 20
        mf_jolt = jqc.pyscf.apply(mf)

        # Create gradient scanner from JoltQC-accelerated method
        grad_scanner = mf_jolt.nuc_grad_method().as_scanner()

        # Test gradient scanner with different molecules
        molecules = [
            mol1,
            pyscf.M(
                atom="H 0 0 0; H 0 0 1.4",
                basis="sto-3g",
                output="/dev/null",
                verbose=0,
            )
        ]

        energies_jolt = []
        gradients_jolt = []
        energies_ref = []
        gradients_ref = []

        for mol in molecules:
            # JoltQC gradient calculation using scanner
            grad_result = grad_scanner(mol)
            # Handle case where grad_scanner returns (energy, gradient) tuple
            if isinstance(grad_result, tuple):
                energy_jolt = grad_result[0]
                grad_jolt = grad_result[1]
            else:
                energy_jolt = grad_scanner.e_tot
                grad_jolt = grad_result

            energies_jolt.append(energy_jolt)
            gradients_jolt.append(grad_jolt)

            # Reference JoltQC calculation for comparison
            mf_ref = dft.RKS(mol, xc="PBE")
            mf_ref.grids.level = 2
            mf_ref.conv_tol = 1e-6
            mf_ref.max_cycle = 20
            mf_ref = jqc.pyscf.apply(mf_ref)
            energy_ref = mf_ref.kernel()
            grad_ref = mf_ref.nuc_grad_method().kernel()

            energies_ref.append(energy_ref)
            gradients_ref.append(grad_ref)

            mol.stdout.close()

        # Compare scanner results with JoltQC reference calculations
        for i, (energy_jolt, energy_ref) in enumerate(zip(energies_jolt, energies_ref)):
            energy_error = abs(energy_jolt - energy_ref)
            assert energy_error < 1e-8, f"Energy error for mol {i+1}: {energy_error:.2e} > 1e-8"

        for i, (grad_jolt, grad_ref) in enumerate(zip(gradients_jolt, gradients_ref)):
            grad_error = np.linalg.norm(grad_jolt - grad_ref)
            assert grad_error < 1e-8, f"Gradient error for mol {i+1}: {grad_error:.2e} > 1e-8"

        # Verify that scanner computed different results for different geometries
        energy_diff = abs(energies_jolt[0] - energies_jolt[1])
        assert energy_diff > 1e-6, f"Energies should be different for different molecules: {energy_diff:.2e}"

        # For gradients, just verify that each is different from zero (since different molecules have different shapes)
        grad_norm1 = np.linalg.norm(gradients_jolt[0])
        grad_norm2 = np.linalg.norm(gradients_jolt[1])
        assert grad_norm1 > 1e-6, f"Gradient norm for mol 1 should be non-zero: {grad_norm1:.2e}"
        assert grad_norm2 > 1e-6, f"Gradient norm for mol 2 should be non-zero: {grad_norm2:.2e}"

        # Verify gradient shapes are correct
        for i, grad in enumerate(gradients_jolt):
            expected_shape = (molecules[i].natm, 3)
            assert grad.shape == expected_shape, f"Gradient {i+1} should have shape {expected_shape}, got {grad.shape}"

        # Verify that gradient scanner has required attributes
        assert hasattr(grad_scanner, 'e_tot'), "Gradient scanner should have e_tot attribute"
        assert hasattr(grad_scanner, 'de'), "Gradient scanner should have de attribute (gradients)"

        # Verify that the last energy and gradient match the scanner's attributes
        energy_diff_attr = abs(energies_jolt[-1] - grad_scanner.e_tot)
        assert energy_diff_attr < 1e-10, f"Scanner e_tot should match last computed energy: {energy_diff_attr:.2e}"

        scanner_grad = grad_scanner.de
        if isinstance(scanner_grad, tuple):
            scanner_grad = scanner_grad[1]  # Extract gradient if it's a tuple
        grad_diff_attr = np.linalg.norm(gradients_jolt[-1] - scanner_grad)
        assert grad_diff_attr < 1e-10, f"Scanner de should match last computed gradient: {grad_diff_attr:.2e}"

        mol1.stdout.close()


if __name__ == "__main__":
    print("Test geometry optimization with JoltQC")
    unittest.main()
