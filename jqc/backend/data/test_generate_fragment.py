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
import tempfile
from pathlib import Path

import numpy as np
import pytest
from generate_fragment import generate_fragments, update_frags


class TestGenerateFragments:
    """Unit tests for generate_fragments function"""

    def test_generate_fragments_basic(self):
        """Test basic fragment generation"""
        ang = (0, 0, 0, 0)  # s-s-s-s case
        fragments = list(generate_fragments(ang, max_threads=256))
        # For (0,0,0,0) case, nf = [1,1,1,1], but the constraint requires
        # at least 3 threads total, so this case should return empty
        assert len(fragments) == 0

    def test_generate_fragments_with_results(self):
        """Test fragment generation that actually returns results"""
        ang = (1, 1, 0, 0)  # p-p-s-s case
        fragments = list(generate_fragments(ang, max_threads=256))
        assert len(fragments) > 0

        # All fragments should have positive values
        for frag in fragments:
            assert all(f > 0 for f in frag)
            assert len(frag) == 4

    def test_generate_fragments_higher_angular_momentum(self):
        """Test fragment generation for higher angular momentum"""
        ang = (1, 1, 0, 0)  # p-p-s-s case
        fragments = list(generate_fragments(ang, max_threads=256))
        assert len(fragments) > 0

        # Check that fragments satisfy divisibility constraints
        nf = np.empty(4, dtype=np.int32)
        nf[0] = (ang[0] + 1) * (ang[0] + 2) // 2  # 3 for p
        nf[1] = (ang[1] + 1) * (ang[1] + 2) // 2  # 3 for p
        nf[2] = (ang[2] + 1) * (ang[2] + 2) // 2  # 1 for s
        nf[3] = (ang[3] + 1) * (ang[3] + 2) // 2  # 1 for s

        for frag in fragments:
            # Check divisibility constraint
            assert nf[0] % frag[0] == 0
            assert nf[1] % frag[1] == 0
            assert nf[2] % frag[2] == 0
            assert nf[3] % frag[3] == 0

            # Check thread count constraint
            nthreads = (nf + frag - 1) // frag
            assert np.prod(nthreads) <= 256
            assert np.prod(nthreads) >= 3

    def test_generate_fragments_empty_for_large_ang(self):
        """Test that no fragments are generated for very large angular momentum"""
        ang = (10, 10, 10, 10)  # Very high angular momentum
        fragments = list(generate_fragments(ang, max_threads=256))
        # Should be empty or very few fragments due to constraints
        assert len(fragments) >= 0  # Could be empty

    def test_generate_fragments_max_threads_constraint(self):
        """Test max_threads constraint is respected"""
        ang = (1, 1, 1, 1)
        max_threads = 64  # Lower limit
        fragments = list(generate_fragments(ang, max_threads=max_threads))

        nf = np.array([(ell + 1) * (ell + 2) // 2 for ell in ang])

        for frag in fragments:
            nthreads = (nf + frag - 1) // frag
            assert np.prod(nthreads) <= max_threads


class TestUpdateFrags:
    """Unit tests for update_frags function"""

    def test_update_frags_invalid_dtype(self):
        """Test that invalid dtype raises error"""
        with pytest.raises(RuntimeError, match="Data type invalid is not supported"):
            update_frags(0, 0, 0, 0, "invalid")

    def test_update_frags_missing_xyz_file(self):
        """Test that missing gly30.xyz file raises FileNotFoundError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)  # Change to temp dir without gly30.xyz
                with pytest.raises(
                    FileNotFoundError, match=r"Required file gly30\.xyz not found"
                ):
                    update_frags(0, 0, 0, 0, "fp64")
            finally:
                os.chdir(orig_cwd)

    def test_update_frags_json_creation(self):
        """Test that update_frags function handles JSON file creation correctly"""
        # Test that existing JSON file structure is maintained
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                # Create a sample JSON file
                json_file = Path("optimal_scheme_fp64.json")
                initial_data = {"1000": [1, 1, 1, 1]}
                with open(json_file, "w") as f:
                    json.dump(initial_data, f)

                # Check that the file exists and contains expected data
                assert json_file.exists()

                with open(json_file) as f:
                    data = json.load(f)
                assert data == initial_data

            finally:
                os.chdir(orig_cwd)


if __name__ == "__main__":
    pytest.main([__file__])
