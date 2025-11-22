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

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Add the data directory to the path to import generate_fragment
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from generate_fragment import generate_fragments, update_frags


class TestGenerateFragments(unittest.TestCase):
    """Test fragment generation for kernel optimization"""

    def test_generate_fragments_basic(self):
        """Test that generate_fragments produces valid fragment configurations"""
        # Test P-P-P-P case (l=1,1,1,1)
        ang = (1, 1, 1, 1)
        fragments = list(generate_fragments(ang, max_threads=256))

        # Should generate at least one valid fragment
        self.assertGreater(len(fragments), 0)

        # Each fragment should have 4 elements (fi, fj, fk, fl)
        for frag in fragments:
            self.assertEqual(len(frag), 4)
            # All fragment sizes should be positive
            self.assertTrue(all(f > 0 for f in frag))
            # Product should not exceed 256 (max threads per quartet)
            self.assertLessEqual(np.prod(frag), 256)

    def test_generate_fragments_dimensions(self):
        """Test that fragments properly divide the integral dimensions"""
        ang = (2, 0, 1, 1)  # D-S-P-P
        nf = np.array(
            [
                (ang[0] + 1) * (ang[0] + 2) // 2,  # 6 for D
                (ang[1] + 1) * (ang[1] + 2) // 2,  # 1 for S
                (ang[2] + 1) * (ang[2] + 2) // 2,  # 3 for P
                (ang[3] + 1) * (ang[3] + 2) // 2,  # 3 for P
            ]
        )

        fragments = list(generate_fragments(ang, max_threads=256))
        self.assertGreater(len(fragments), 0)

        for frag in fragments:
            # Each fragment dimension must divide the corresponding integral dimension
            for i in range(4):
                self.assertEqual(
                    nf[i] % frag[i],
                    0,
                    f"Fragment {frag[i]} does not divide dimension {nf[i]} for index {i}",
                )

    def test_generate_fragments_thread_constraints(self):
        """Test that fragments satisfy thread count constraints"""
        ang = (1, 1, 1, 1)
        max_threads = 128
        fragments = list(generate_fragments(ang, max_threads=max_threads))

        for frag in fragments:
            nf = np.array([(l + 1) * (l + 2) // 2 for l in ang])
            nthreads = (nf + frag - 1) // frag
            total_threads = np.prod(nthreads)

            # Should not exceed max_threads
            self.assertLessEqual(total_threads, max_threads)
            # Should have at least 3 threads (minimum requirement)
            self.assertGreaterEqual(total_threads, 3)

    def test_generate_fragments_empty_for_large_cases(self):
        """Test that very large angular momentum cases may have no valid fragments"""
        # High angular momentum case that might exceed limits
        ang = (5, 5, 5, 5)
        fragments = list(generate_fragments(ang, max_threads=64))

        # May be empty if no valid fragmentation exists
        # This is expected behavior for constrained cases
        if len(fragments) > 0:
            for frag in fragments:
                self.assertLessEqual(np.prod(frag), 256)


class TestUpdateFrags(unittest.TestCase):
    """Test the update_frags function that benchmarks kernels"""

    @classmethod
    def setUpClass(cls):
        """Create a temporary directory with test molecule file"""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.original_dir = os.getcwd()

        # Create a simple test molecule file
        gly30_path = Path(cls.temp_dir.name) / "gly30.xyz"
        with open(gly30_path, "w") as f:
            f.write(
                """6

C     0.000000     0.000000     0.000000
H     0.000000     0.000000     1.089000
H     1.026719     0.000000    -0.363000
H    -0.513360    -0.889165    -0.363000
H    -0.513360     0.889165    -0.363000
O     2.000000     0.000000     0.000000
"""
            )

        # Change to temp directory
        os.chdir(cls.temp_dir.name)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory"""
        os.chdir(cls.original_dir)
        cls.temp_dir.cleanup()

    def test_update_frags_fp32(self):
        """Test update_frags with fp32 precision for a simple case"""
        # Use a simple angular momentum case
        li, lj, lk, ll = 0, 0, 0, 0  # S-S-S-S
        dtype = "fp32"

        # Should run without errors
        try:
            update_frags(li, lj, lk, ll, dtype)
        except Exception as e:
            self.fail(f"update_frags raised exception: {e}")

        # Check that output file was created
        from jqc.backend.jk import device_name

        output_file = f"optimal_scheme_{device_name}_fp32.json"
        self.assertTrue(
            Path(output_file).exists(), f"Output file {output_file} was not created"
        )

        # Verify JSON structure
        with open(output_file) as f:
            data = json.load(f)
            ang_key = str(1000 * li + 100 * lj + 10 * lk + ll)
            self.assertIn(
                ang_key, data, f"Expected key {ang_key} not found in output JSON"
            )
            # Value should be a list (fragment configuration)
            self.assertIsInstance(data[ang_key], list)

    def test_update_frags_fp64(self):
        """Test update_frags with fp64 precision"""
        li, lj, lk, ll = 0, 0, 0, 0  # S-S-S-S
        dtype = "fp64"

        try:
            update_frags(li, lj, lk, ll, dtype)
        except Exception as e:
            self.fail(f"update_frags raised exception: {e}")

        from jqc.backend.jk import device_name

        output_file = f"optimal_scheme_{device_name}_fp64.json"
        self.assertTrue(Path(output_file).exists())

    def test_update_frags_with_primitives(self):
        """Test update_frags with a case involving primitive gaussians"""
        # P-P-P-P case
        li, lj, lk, ll = 1, 1, 1, 1
        dtype = "fp32"

        try:
            update_frags(li, lj, lk, ll, dtype)
        except Exception as e:
            self.fail(f"update_frags raised exception for P-P-P-P: {e}")

        from jqc.backend.jk import device_name

        output_file = f"optimal_scheme_{device_name}_fp32.json"
        with open(output_file) as f:
            data = json.load(f)
            ang_key = "1111"
            self.assertIn(ang_key, data)

            # Check that the result is either:
            # - [-1] for 1q1t algorithm
            # - A valid fragment configuration [fi, fj, fk, fl]
            result = data[ang_key]
            if result == [-1]:
                # 1q1t algorithm chosen
                pass
            else:
                # 1qnt algorithm with valid fragments
                self.assertEqual(len(result), 4)
                self.assertTrue(all(f > 0 for f in result))

    def test_update_frags_invalid_dtype(self):
        """Test that invalid dtype raises appropriate error"""
        li, lj, lk, ll = 0, 0, 0, 0
        dtype = "invalid"

        with self.assertRaises(RuntimeError) as context:
            update_frags(li, lj, lk, ll, dtype)

        self.assertIn("not supported", str(context.exception))

    def test_update_frags_different_angular_momenta(self):
        """Test with different angular momentum combinations"""
        test_cases = [
            (0, 0, 0, 0),  # S-S-S-S
            (1, 0, 0, 0),  # P-S-S-S
            (1, 1, 0, 0),  # P-P-S-S
        ]

        from jqc.backend.jk import device_name

        for li, lj, lk, ll in test_cases:
            with self.subTest(ang=(li, lj, lk, ll)):
                try:
                    update_frags(li, lj, lk, ll, "fp32")
                    output_file = f"optimal_scheme_{device_name}_fp32.json"
                    with open(output_file) as f:
                        data = json.load(f)
                        ang_key = str(1000 * li + 100 * lj + 10 * lk + ll)
                        self.assertIn(ang_key, data)
                except Exception as e:
                    self.fail(f"Failed for angular momentum {(li, lj, lk, ll)}: {e}")


class TestFragmentIntegration(unittest.TestCase):
    """Integration tests for the fragment generation workflow"""

    def test_fragment_output_consistency(self):
        """Test that fragment generation produces consistent results"""
        ang = (1, 0, 1, 0)  # P-S-P-S
        fragments1 = list(generate_fragments(ang))
        fragments2 = list(generate_fragments(ang))

        # Should produce identical results
        self.assertEqual(len(fragments1), len(fragments2))
        for f1, f2 in zip(fragments1, fragments2):
            np.testing.assert_array_equal(f1, f2)

    def test_fragment_ordering(self):
        """Test that fragments are generated in a deterministic order"""
        # S-S-S-S has no valid fragments (needs at least 3 threads, but would only have 1)
        ang_no_frags = (0, 0, 0, 0)
        fragments_no_frags = list(generate_fragments(ang_no_frags))
        self.assertEqual(len(fragments_no_frags), 0, "S-S-S-S should have no valid fragments")

        # P-P-P-P has many valid fragments - test consistency
        ang = (1, 1, 1, 1)
        fragments1 = list(generate_fragments(ang))
        fragments2 = list(generate_fragments(ang))

        # Should produce identical results in same order
        self.assertGreater(len(fragments1), 0)
        self.assertEqual(len(fragments1), len(fragments2))
        for f1, f2 in zip(fragments1, fragments2):
            np.testing.assert_array_equal(f1, f2)


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
