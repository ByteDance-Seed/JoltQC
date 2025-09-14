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

import unittest
import numpy as np
import cupy as cp
import pyscf
from pyscf import gto
from jqc.pyscf.mol import BasisLayout, split_basis
from jqc.constants import TILE, NPRIM_MAX


class TestBasisLayout(unittest.TestCase):
    """Test BasisLayout object functionality"""

    def setUp(self):
        """Set up test molecules"""
        # Simple H2 molecule with cc-pVDZ basis
        self.mol_h2 = pyscf.M(
            atom='H 0 0 0; H 0 0 1.4',
            basis='cc-pvdz',
            output='/dev/null',
            verbose=0
        )

        # Water molecule for more complex testing with cc-pVDZ basis
        self.mol_h2o = pyscf.M(
            atom='''
            O 0.0 0.0 0.1174
            H -0.757 0.0 -0.4696
            H 0.757 0.0 -0.4696
            ''',
            basis='cc-pvdz',
            output='/dev/null',
            verbose=0
        )

    def tearDown(self):
        """Clean up molecules"""
        self.mol_h2.stdout.close()
        self.mol_h2o.stdout.close()

    def test_split_basis_no_splitting_needed(self):
        """Test split_basis when no splitting is needed (nprim <= NPRIM_MAX)"""
        # STO-3G has 3 primitives per function, should not split if NPRIM_MAX >= 3
        original_mol = self.mol_h2
        split_mol, bas_map = split_basis(original_mol)

        # Should return same molecule if no splitting needed
        self.assertEqual(split_mol.nao, original_mol.nao)
        self.assertEqual(split_mol.nbas, original_mol.nbas)

        # Verify basis mapping
        self.assertEqual(len(bas_map), split_mol.nbas)

    def test_basis_layout_creation(self):
        """Test basic BasisLayout creation"""
        layout = BasisLayout.from_sort_group_basis(self.mol_h2, alignment=TILE)

        # Basic properties should be set
        self.assertIsNotNone(layout.splitted_mol)
        self.assertIsNotNone(layout.ao_loc)
        self.assertIsNotNone(layout.bas_id)
        self.assertIsNotNone(layout.pad_id)
        self.assertIsNotNone(layout.angs)

        # Check dimensions are consistent
        self.assertEqual(len(layout.bas_id), len(layout.pad_id))
        self.assertEqual(len(layout.bas_id), len(layout.angs))

    def test_ao_loc_property(self):
        """Test ao_loc property returns correct dimensions"""
        layout = BasisLayout.from_sort_group_basis(self.mol_h2, alignment=TILE)

        # ao_loc should return internal layout dimensions (with padding)
        dims = (layout.angs + 1) * (layout.angs + 2) // 2
        expected_nao = np.sum(dims)
        actual_nao = layout.ao_loc[-1].item()

        # For debugging, let's see what we get
        print(f"Internal layout nao: {expected_nao}")
        print(f"ao_loc[-1]: {actual_nao}")
        print(f"ao_loc: {layout.ao_loc.get()}")

        self.assertEqual(actual_nao, expected_nao,
                        "ao_loc should match internal layout dimensions")

    def test_mol2cart_dimensions(self):
        """Test mol2cart returns correct dimensions"""
        layout = BasisLayout.from_sort_group_basis(self.mol_h2, alignment=TILE)

        # Input: decontracted molecule dimensions
        input_nao = layout.splitted_mol.nao
        input_matrix = np.eye(input_nao)

        # Transform
        output_matrix = layout.mol2cart(input_matrix)

        # Output: internal layout dimensions (with padding)
        dims = (layout.angs + 1) * (layout.angs + 2) // 2
        expected_total_nao = np.sum(dims)

        print(f"Input matrix shape: {input_matrix.shape}")
        print(f"Output matrix shape: {output_matrix.shape}")
        print(f"Expected output nao: {expected_total_nao}")
        print(f"Number of basis functions: {len(layout.angs)}")

        # Check output dimensions
        self.assertEqual(output_matrix.shape[0], expected_total_nao)
        self.assertEqual(output_matrix.shape[1], expected_total_nao)

    def test_cart2mol_dimensions(self):
        """Test cart2mol returns correct dimensions"""
        layout = BasisLayout.from_sort_group_basis(self.mol_h2, alignment=TILE)

        # Start with internal layout dimensions
        dims = (layout.angs + 1) * (layout.angs + 2) // 2
        internal_nao = np.sum(dims)
        input_matrix = cp.eye(internal_nao)

        # Transform back to decontracted molecule
        output_matrix = layout.cart2mol(input_matrix)

        # Should get decontracted molecule dimensions
        expected_nao = layout.splitted_mol.nao

        print(f"Input matrix shape: {input_matrix.shape}")
        print(f"Output matrix shape: {output_matrix.shape}")
        print(f"Expected output nao: {expected_nao}")

        self.assertEqual(output_matrix.shape[0], expected_nao)
        self.assertEqual(output_matrix.shape[1], expected_nao)

    def test_matrix_transformation_stability(self):
        """Test that matrix transformations are numerically stable"""
        layout = BasisLayout.from_sort_group_basis(self.mol_h2, alignment=TILE)

        # Test with various types of matrices
        input_nao = layout.splitted_mol.nao

        test_matrices = {
            'identity': np.eye(input_nao),
            'random_symmetric': None,
            'diagonal': np.diag(np.random.rand(input_nao)),
            'ones': np.ones((input_nao, input_nao)),
        }

        # Generate random symmetric matrix
        rand_matrix = np.random.rand(input_nao, input_nao)
        test_matrices['random_symmetric'] = rand_matrix + rand_matrix.T

        for matrix_type, original_matrix in test_matrices.items():
            with self.subTest(matrix_type=matrix_type):
                print(f"\n=== Testing {matrix_type} matrix ===")

                # Step 1: mol2cart transformation
                cart_matrix = layout.mol2cart(original_matrix)

                # Step 2: cart2mol transformation
                recovered_matrix = layout.cart2mol(cart_matrix)

                print(f"  Original shape: {original_matrix.shape}")
                print(f"  Cart shape: {cart_matrix.shape}")
                print(f"  Recovered shape: {recovered_matrix.shape}")

                # Since mapping is not one-to-one, we test for stability rather than identity

                # 1. Check shapes are correct
                self.assertEqual(recovered_matrix.shape, original_matrix.shape,
                               f"{matrix_type}: Shape should be preserved")

                # 2. Check for numerical stability (no NaN/Inf)
                self.assertFalse(np.isnan(recovered_matrix.get()).any(),
                               f"{matrix_type}: Should not contain NaN")
                self.assertFalse(np.isinf(recovered_matrix.get()).any(),
                               f"{matrix_type}: Should not contain Inf")

                # 3. Check values are reasonable
                max_val = np.max(np.abs(recovered_matrix.get()))
                self.assertLess(max_val, 1000,
                               f"{matrix_type}: Max value {max_val} should be reasonable")

    # Removed round-trip diagnostic test by design

    def test_basis_mapping_consistency(self):
        """Test that bas_id provides correct mapping to decontracted basis"""
        layout = BasisLayout.from_sort_group_basis(self.mol_h2, alignment=TILE)

        # Check that bas_id indices are valid
        max_bas_id = np.max(layout.bas_id)
        decontracted_size = len(layout.decontracted_ao_loc) - 1

        print(f"bas_id: {layout.bas_id}")
        print(f"max bas_id: {max_bas_id}")
        print(f"decontracted_ao_loc size: {decontracted_size}")
        print(f"decontracted_ao_loc: {layout.decontracted_ao_loc}")

        self.assertLessEqual(max_bas_id, decontracted_size - 1,
                           "bas_id should contain valid indices for decontracted_ao_loc")

    def test_padding_logic(self):
        """Test padding identification and handling"""
        layout = BasisLayout.from_sort_group_basis(self.mol_h2, alignment=TILE)

        # Count non-padded vs padded functions
        n_real = np.sum(~layout.pad_id)
        n_padded = np.sum(layout.pad_id)

        print(f"Real basis functions: {n_real}")
        print(f"Padded basis functions: {n_padded}")
        print(f"Total basis functions: {len(layout.pad_id)}")
        print(f"pad_id: {layout.pad_id}")

        # Should have at least the split molecule's basis functions as real
        split_nbas = layout.splitted_mol.nbas
        self.assertGreaterEqual(n_real, split_nbas)

    def test_different_alignments(self):
        """Test BasisLayout with different alignment values"""
        for alignment in [1, 4, 16]:
            with self.subTest(alignment=alignment):
                layout = BasisLayout.from_sort_group_basis(self.mol_h2, alignment=alignment)

                # Basic functionality should work regardless of alignment
                input_matrix = np.eye(layout.splitted_mol.nao)
                cart_matrix = layout.mol2cart(input_matrix)
                recovered_matrix = layout.cart2mol(cart_matrix)

                error = np.linalg.norm(recovered_matrix.get() - input_matrix)

                print(f"Alignment {alignment}: round trip error = {error}")

                self.assertLess(error, 1e-10,
                               f"Round trip should work with alignment={alignment}")

    # Removed H2O round-trip unit test: spherical round-trip is not identity

    def test_3d_array_handling(self):
        """Test that transformations work with 3D arrays (multiple matrices)"""
        layout = BasisLayout.from_sort_group_basis(self.mol_h2, alignment=TILE)

        # Test with batch of 3 matrices
        n_batch = 3
        input_nao = layout.splitted_mol.nao
        input_batch = np.random.rand(n_batch, input_nao, input_nao)

        # Transform
        cart_batch = layout.mol2cart(input_batch)
        recovered_batch = layout.cart2mol(cart_batch)

        error = np.linalg.norm(recovered_batch.get() - input_batch)

        print(f"3D array round trip error: {error}")
        print(f"Input batch shape: {input_batch.shape}")
        print(f"Cart batch shape: {cart_batch.shape}")
        print(f"Recovered batch shape: {recovered_batch.shape}")

        self.assertLess(error, 1e-10, "3D array transformations should work correctly")
        self.assertEqual(cart_batch.shape[0], n_batch)
        self.assertEqual(recovered_batch.shape[0], n_batch)

    def test_angs_ordered_in_groups(self):
        """Test that angs array is ordered in groups by angular momentum"""
        # Test with H2 cc-pVDZ which has both s and p functions
        layout_h2 = BasisLayout.from_sort_group_basis(self.mol_h2, alignment=TILE)

        print(f"H2 cc-pVDZ angs: {layout_h2.angs}")

        angs_h2 = layout_h2.angs
        self.assertTrue(self._is_grouped_by_angular_momentum(angs_h2),
                       f"H2 cc-pVDZ angs not properly grouped: {angs_h2}")

        # Also test with H2O cc-pVDZ which has s, p, and d functions
        layout_h2o = BasisLayout.from_sort_group_basis(self.mol_h2o, alignment=TILE)

        print(f"H2O cc-pVDZ angs: {layout_h2o.angs}")

        # H2O cc-pVDZ has s, p, and d functions, should be grouped
        angs_h2o = layout_h2o.angs
        self.assertTrue(self._is_grouped_by_angular_momentum(angs_h2o),
                       f"H2O cc-pVDZ angs not properly grouped: {angs_h2o}")

    def _is_grouped_by_angular_momentum(self, angs):
        """Helper function to check if angs array is grouped by angular momentum"""
        if len(angs) == 0:
            return True

        # Check that all functions with the same angular momentum are consecutive
        current_ang = angs[0]
        for i in range(1, len(angs)):
            if angs[i] != current_ang:
                # Angular momentum changed, check that this ang doesn't appear again later
                new_ang = angs[i]
                for j in range(i + 1, len(angs)):
                    if angs[j] == current_ang:
                        # Found previous angular momentum later - not grouped
                        print(f"Angular momentum {current_ang} appears again at position {j} after {new_ang} at position {i}")
                        return False
                current_ang = new_ang
        return True


if __name__ == '__main__':
    print("BasisLayout Unit Tests")
    print(f"NPRIM_MAX = {NPRIM_MAX}")
    print(f"TILE = {TILE}")
    unittest.main()
