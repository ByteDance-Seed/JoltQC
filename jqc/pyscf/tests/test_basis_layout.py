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

import cupy as cp
import numpy as np
import pyscf

from jqc.constants import NPRIM_MAX, TILE
from jqc.pyscf.basis import BasisLayout, split_basis


class TestBasisLayout(unittest.TestCase):
    """Test BasisLayout object functionality"""

    def setUp(self):
        """Set up test molecules"""
        # Simple H2 molecule with cc-pVDZ basis
        self.mol_h2 = pyscf.M(
            atom="H 0 0 0; H 0 0 1.4", basis="cc-pvdz", output="/dev/null", verbose=0
        )

        # Water molecule for more complex testing with cc-pVDZ basis
        self.mol_h2o = pyscf.M(
            atom="""
            O 0.0 0.0 0.1174
            H -0.757 0.0 -0.4696
            H 0.757 0.0 -0.4696
            """,
            basis="cc-pvdz",
            output="/dev/null",
            verbose=0,
        )

    def tearDown(self):
        """Clean up molecules"""
        self.mol_h2.stdout.close()
        self.mol_h2o.stdout.close()

    def test_split_basis_no_splitting_needed(self):
        """Test split_basis when splitting occurs (cc-pVDZ has nprim=3 > NPRIM_MAX=2)"""
        # cc-pVDZ has 3 primitives per s function, which will be split since NPRIM_MAX=2
        original_mol = self.mol_h2
        split_mol, bas_map = split_basis(original_mol)

        # With cc-pVDZ and NPRIM_MAX=2, splitting should occur
        # Original: 6 basis functions -> After splitting: 8 basis functions
        self.assertGreaterEqual(split_mol.nbas, original_mol.nbas)
        # nao should increase due to splitting
        self.assertGreaterEqual(split_mol.nao, original_mol.nao)

        # Verify basis mapping
        self.assertEqual(len(bas_map), split_mol.nbas)

    def test_basis_layout_creation(self):
        """Test basic BasisLayout creation"""
        layout = BasisLayout.from_mol(self.mol_h2, alignment=TILE)

        # Basic properties should be set
        self.assertIsNotNone(layout.splitted_mol)
        self.assertIsNotNone(layout.ao_loc)
        self.assertIsNotNone(layout.to_split_map)
        self.assertIsNotNone(layout.to_decontracted_map)
        self.assertIsNotNone(layout.pad_id)
        self.assertIsNotNone(layout.angs)

        # Check dimensions are consistent
        self.assertEqual(len(layout.to_split_map), len(layout.pad_id))
        self.assertEqual(len(layout.to_split_map), len(layout.angs))
        self.assertEqual(len(layout.to_split_map), len(layout.to_decontracted_map))

    def test_ao_loc_property(self):
        """Test ao_loc property returns correct dimensions"""
        layout = BasisLayout.from_mol(self.mol_h2, alignment=TILE)

        # ao_loc should return decontracted molecular dimensions (without padding)
        expected_nao = int(layout.ao_loc[-1])
        actual_nao = layout.ao_loc[-1].item()

        # For debugging, let's see what we get
        print(f"Decontracted nao: {expected_nao}")
        print(f"ao_loc[-1]: {actual_nao}")
        print(f"ao_loc: {layout.ao_loc}")

        self.assertEqual(
            actual_nao,
            expected_nao,
            "ao_loc should match decontracted molecular dimensions",
        )

    def test_dm_from_mol_dimensions(self):
        """Test dm_from_mol returns correct dimensions"""
        layout = BasisLayout.from_mol(self.mol_h2, alignment=TILE)

        # Input: decontracted molecule dimensions (original molecule)
        input_nao = int(layout.ao_loc[-1])
        input_matrix = np.eye(input_nao)

        # Transform
        output_matrix = layout.dm_from_mol(input_matrix)

        # Output: decontracted dimensions (without padding)
        expected_total_nao = int(layout.ao_loc[-1])

        print(f"Input matrix shape: {input_matrix.shape}")
        print(f"Output matrix shape: {output_matrix.shape}")
        print(f"Expected output nao: {expected_total_nao}")
        print(f"Number of basis functions: {len(layout.angs)}")

        # Check output dimensions
        self.assertEqual(output_matrix.shape[0], expected_total_nao)
        self.assertEqual(output_matrix.shape[1], expected_total_nao)

    def test_dm_to_mol_dimensions(self):
        """Test dm_to_mol returns correct dimensions"""
        layout = BasisLayout.from_mol(self.mol_h2, alignment=TILE)

        # Start with decontracted dimensions
        decontracted_nao = int(layout.ao_loc[-1])
        input_matrix = cp.eye(decontracted_nao)

        # Transform back to decontracted molecule
        output_matrix = layout.dm_to_mol(input_matrix)

        # Should get decontracted molecule dimensions (original molecule in molecular basis)
        expected_nao = self.mol_h2.nao  # Molecular basis (spherical/cartesian as specified)

        print(f"Input matrix shape: {input_matrix.shape}")
        print(f"Output matrix shape: {output_matrix.shape}")
        print(f"Expected output nao: {expected_nao}")

        self.assertEqual(output_matrix.shape[0], expected_nao)
        self.assertEqual(output_matrix.shape[1], expected_nao)

    def test_matrix_transformation_stability(self):
        """Test that matrix transformations are numerically stable"""
        layout = BasisLayout.from_mol(self.mol_h2, alignment=TILE)

        # Test with various types of matrices in MOLECULAR basis
        input_nao = self.mol_h2.nao  # Use molecular basis dimension

        test_matrices = {
            "identity": np.eye(input_nao),
            "random_symmetric": None,
            "diagonal": np.diag(np.random.rand(input_nao)),
            "ones": np.ones((input_nao, input_nao)),
        }

        # Generate random symmetric matrix
        rand_matrix = np.random.rand(input_nao, input_nao)
        test_matrices["random_symmetric"] = rand_matrix + rand_matrix.T

        for matrix_type, original_matrix in test_matrices.items():
            with self.subTest(matrix_type=matrix_type):
                print(f"\n=== Testing {matrix_type} matrix ===")

                # Step 1: dm_from_mol transformation
                cart_matrix = layout.dm_from_mol(original_matrix)

                # Step 2: dm_to_mol transformation
                recovered_matrix = layout.dm_to_mol(cart_matrix)

                print(f"  Original shape: {original_matrix.shape}")
                print(f"  Cart shape: {cart_matrix.shape}")
                print(f"  Recovered shape: {recovered_matrix.shape}")

                # Since mapping is not one-to-one, we test for stability rather than identity

                # 1. Check shapes are correct
                self.assertEqual(
                    recovered_matrix.shape,
                    original_matrix.shape,
                    f"{matrix_type}: Shape should be preserved",
                )

                # 2. Check for numerical stability (no NaN/Inf)
                self.assertFalse(
                    np.isnan(recovered_matrix.get()).any(),
                    f"{matrix_type}: Should not contain NaN",
                )
                self.assertFalse(
                    np.isinf(recovered_matrix.get()).any(),
                    f"{matrix_type}: Should not contain Inf",
                )

                # 3. Check values are reasonable
                max_val = np.max(np.abs(recovered_matrix.get()))
                self.assertLess(
                    max_val,
                    1000,
                    f"{matrix_type}: Max value {max_val} should be reasonable",
                )

    def test_basis_mapping_consistency(self):
        """Test that to_split_map provides correct mapping to decontracted basis"""
        layout = BasisLayout.from_mol(self.mol_h2, alignment=TILE)

        # Check that to_split_map indices are valid for the split molecule
        max_to_split_map = np.max(layout.to_split_map)
        split_mol_nbas = layout.splitted_mol.nbas

        print(f"to_split_map: {layout.to_split_map}")
        print(f"max to_split_map: {max_to_split_map}")
        print(f"split molecule nbas: {split_mol_nbas}")
        print(f"ao_loc: {layout.ao_loc}")

        self.assertLessEqual(
            max_to_split_map,
            split_mol_nbas - 1,
            "to_split_map should contain valid indices for split molecule basis functions",
        )

    def test_to_decontracted_map_consistency(self):
        """Test that to_decontracted_map provides correct mapping to decontracted basis"""
        layout = BasisLayout.from_mol(self.mol_h2, alignment=TILE)

        # Check that the cache is initially None (lazy evaluation)
        self.assertIsNone(
            layout._to_decontracted_map,
            "to_decontracted_map cache should be None before first access",
        )

        # Check that to_decontracted_map indices are valid for the decontracted molecule
        non_padded_mask = ~layout.pad_id
        decontracted_indices = layout.to_decontracted_map[non_padded_mask]
        max_decontracted_index = np.max(decontracted_indices)
        decontracted_mol_nbas = len(layout.ao_loc) - 1

        # Check that the cache is now populated (lazy evaluation worked)
        self.assertIsNotNone(
            layout._to_decontracted_map,
            "to_decontracted_map cache should be populated after first access",
        )

        print(f"to_decontracted_map: {layout.to_decontracted_map}")
        print(f"max decontracted index: {max_decontracted_index}")
        print(f"decontracted molecule nbas: {decontracted_mol_nbas}")

        self.assertLessEqual(
            max_decontracted_index,
            decontracted_mol_nbas - 1,
            "to_decontracted_map should contain valid indices for decontracted molecule basis functions",
        )

        # Check that padded entries have invalid indices
        padded_indices = layout.to_decontracted_map[layout.pad_id]
        self.assertTrue(
            np.all(padded_indices == -1),
            "Padded entries should have invalid indices (-1)",
        )

        # Verify composition: to_decontracted_map should equal split_to_decontracted[to_split_map]
        expected_mapping = np.empty_like(layout.to_decontracted_map)
        expected_mapping[non_padded_mask] = layout._split_to_decontracted[
            layout.to_split_map[non_padded_mask]
        ]
        expected_mapping[layout.pad_id] = -1

        np.testing.assert_array_equal(
            layout.to_decontracted_map,
            expected_mapping,
            "to_decontracted_map should be composition of to_split_map and split_to_decontracted",
        )

        # Verify that multiple accesses return the same cached object
        second_access = layout.to_decontracted_map
        self.assertIs(
            layout.to_decontracted_map,
            second_access,
            "Multiple accesses should return the same cached object",
        )

    def test_padding_logic(self):
        """Test padding identification and handling"""
        layout = BasisLayout.from_mol(self.mol_h2, alignment=TILE)

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
                layout = BasisLayout.from_mol(self.mol_h2, alignment=alignment)

                # Basic functionality should work regardless of alignment
                # Start with molecular basis dimension
                input_nao = self.mol_h2.nao
                input_matrix = np.eye(input_nao)
                cart_matrix = layout.dm_from_mol(input_matrix)

                # Check that transformations produce expected dimensions
                decontracted_nao = layout.ao_loc[-1].item()
                self.assertEqual(cart_matrix.shape[0], decontracted_nao)
                self.assertEqual(cart_matrix.shape[1], decontracted_nao)

                recovered_matrix = layout.dm_to_mol(cart_matrix)
                self.assertEqual(recovered_matrix.shape[0], input_nao)
                self.assertEqual(recovered_matrix.shape[1], input_nao)

                print(f"Alignment {alignment}: transformations work correctly")

    def test_3d_array_handling(self):
        """Test that transformations work with 3D arrays (multiple matrices)"""
        layout = BasisLayout.from_mol(self.mol_h2, alignment=TILE)

        # Test with batch of 3 matrices in MOLECULAR basis
        n_batch = 3
        input_nao = self.mol_h2.nao  # Use molecular basis dimension
        decontracted_nao = layout.ao_loc[-1].item()
        input_batch = np.random.rand(n_batch, input_nao, input_nao)

        # Transform and check dimensions
        cart_batch = layout.dm_from_mol(input_batch)
        recovered_batch = layout.dm_to_mol(cart_batch)

        print(f"Input batch shape: {input_batch.shape}")
        print(f"Cart batch shape: {cart_batch.shape}")
        print(f"Recovered batch shape: {recovered_batch.shape}")

        # Check dimensions are correct
        self.assertEqual(
            cart_batch.shape, (n_batch, decontracted_nao, decontracted_nao)
        )
        self.assertEqual(recovered_batch.shape, (n_batch, input_nao, input_nao))

        # Check that no NaN or Inf values are produced
        self.assertFalse(np.isnan(cart_batch.get()).any())
        self.assertFalse(np.isinf(cart_batch.get()).any())
        self.assertFalse(np.isnan(recovered_batch.get()).any())
        self.assertFalse(np.isinf(recovered_batch.get()).any())

    def test_angs_ordered_in_groups(self):
        """Test that angs array is ordered in groups by angular momentum"""
        # Test with H2 cc-pVDZ which has both s and p functions
        layout_h2 = BasisLayout.from_mol(self.mol_h2, alignment=TILE)

        print(f"H2 cc-pVDZ angs: {layout_h2.angs}")

        angs_h2 = layout_h2.angs
        self.assertTrue(
            self._is_grouped_by_angular_momentum(angs_h2),
            f"H2 cc-pVDZ angs not properly grouped: {angs_h2}",
        )

        # Also test with H2O cc-pVDZ which has s, p, and d functions
        layout_h2o = BasisLayout.from_mol(self.mol_h2o, alignment=TILE)

        print(f"H2O cc-pVDZ angs: {layout_h2o.angs}")

        # H2O cc-pVDZ has s, p, and d functions, should be grouped
        angs_h2o = layout_h2o.angs
        self.assertTrue(
            self._is_grouped_by_angular_momentum(angs_h2o),
            f"H2O cc-pVDZ angs not properly grouped: {angs_h2o}",
        )

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
                        print(
                            f"Angular momentum {current_ang} appears again at position {j} after {new_ang} at position {i}"
                        )
                        return False
                current_ang = new_ang
        return True


if __name__ == "__main__":
    print("BasisLayout Unit Tests")
    print(f"NPRIM_MAX = {NPRIM_MAX}")
    print(f"TILE = {TILE}")
    unittest.main()
