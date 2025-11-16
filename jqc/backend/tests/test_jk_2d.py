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

"""
Unit tests for 2D VJ and VK kernels with packed basis_data signature.

Tests verify:
1. Kernel compilation with new packed basis_data signature
2. Packed basis_data format correctness
3. VJ and VK kernel signature acceptance
4. FP32 and FP64 precision handling

NOTE: These tests compile multiple CUDA kernels which can leave CUDA in a state
that affects subsequent tests. Run these tests in isolation for best results:
    pytest jqc/backend/tests/test_jk_2d.py

Higher-level integration testing is done through jqc/pyscf tests.
"""

import unittest

import pytest

import cupy as cp
import numpy as np
import pyscf

from jqc.backend.jk_2d import gen_vj_kernel, gen_vk_kernel, gen_kernel
from jqc.pyscf.basis import BasisLayout


def setUpModule():
    pass


def tearDownModule():
    # CUDA cleanup - errors here are ignored as they don't affect test results
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Device().synchronize()
    except Exception:
        pass  # Ignore CUDA cleanup errors


@pytest.mark.gpu
@pytest.mark.slow
class TestKernelCompilation(unittest.TestCase):
    """Test that 2D kernels compile correctly with the new signature."""

    def test_vj_kernel_compilation_fp64(self):
        """Test VJ kernel compilation with FP64 precision."""
        ang = (0, 0, 0, 0)  # s-s-s-s
        nprim = (1, 1, 1, 1)

        _, _, fun = gen_vj_kernel(ang, nprim, dtype=np.float64, n_dm=1)

        self.assertIsNotNone(fun)

    def test_vk_kernel_compilation_fp64(self):
        """Test VK kernel compilation with FP64 precision."""
        ang = (0, 0, 0, 0)
        nprim = (1, 1, 1, 1)

        _, _, fun = gen_vk_kernel(ang, nprim, dtype=np.float64, n_dm=1)

        self.assertIsNotNone(fun)

    def test_combined_kernel_compilation(self):
        """Test combined JK kernel compilation."""
        ang = (0, 0, 0, 0)
        nprim = (1, 1, 1, 1)

        script, _, fun = gen_kernel(
            ang, nprim, dtype=np.float64, n_dm=1, do_j=True, do_k=True
        )

        self.assertIsNotNone(fun)
        self.assertIn("VJ Kernel", script)
        self.assertIn("VK Kernel", script)
        self.assertIn("BASIS_STRIDE", script)


@pytest.mark.gpu
class TestPackedBasisData(unittest.TestCase):
    """Test that packed basis_data format works correctly."""

    def setUp(self):
        """Set up a simple test molecule."""
        self.mol = pyscf.M(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            output="/dev/null",
            cart=1,
        )
        self.basis_layout = BasisLayout.from_mol(self.mol)

    def tearDown(self):
        if hasattr(self, 'mol'):
            self.mol.stdout.close()

    def test_basis_data_fp32_format(self):
        """Test that basis_data_fp32 has correct format."""
        basis_data = self.basis_layout.basis_data_fp32

        self.assertIn('packed', basis_data)
        self.assertIn('coords', basis_data)
        self.assertIn('ce', basis_data)
        self.assertIn('ao_loc', basis_data)

        packed = basis_data['packed']
        nbasis = self.basis_layout.nbasis

        # Check shape
        self.assertEqual(packed.shape[0], nbasis)
        self.assertEqual(packed.ndim, 2)

        # Check dtype
        self.assertEqual(packed.dtype, np.float32)

    def test_basis_data_fp64_format(self):
        """Test that basis_data_fp64 has correct format."""
        basis_data = self.basis_layout.basis_data_fp64

        self.assertIn('packed', basis_data)
        self.assertIn('coords', basis_data)
        self.assertIn('ce', basis_data)
        self.assertIn('ao_loc', basis_data)

        packed = basis_data['packed']
        nbasis = self.basis_layout.nbasis

        # Check shape
        self.assertEqual(packed.shape[0], nbasis)
        self.assertEqual(packed.ndim, 2)

        # Check dtype
        self.assertEqual(packed.dtype, np.float64)

    def test_ao_loc_stored_in_coords(self):
        """Test that ao_loc values are stored in coords[:, 3]."""
        basis_data_fp32 = self.basis_layout.basis_data_fp32
        basis_data_fp64 = self.basis_layout.basis_data_fp64

        ao_loc = self.basis_layout.ao_loc

        # Check FP32
        coords_fp32 = basis_data_fp32['coords']
        stored_ao_loc_fp32 = coords_fp32[:, 3]
        expected_ao_loc_fp32 = ao_loc[:-1].astype(np.float32)
        self.assertTrue(
            cp.allclose(stored_ao_loc_fp32, expected_ao_loc_fp32, rtol=1e-6)
        )

        # Check FP64
        coords_fp64 = basis_data_fp64['coords']
        stored_ao_loc_fp64 = coords_fp64[:, 3]
        expected_ao_loc_fp64 = ao_loc[:-1].astype(np.float64)
        self.assertTrue(
            cp.allclose(stored_ao_loc_fp64, expected_ao_loc_fp64, rtol=1e-12)
        )


@pytest.mark.gpu
@pytest.mark.slow
class TestKernelSignature(unittest.TestCase):
    """Test that kernels accept the new packed basis_data signature."""

    def setUp(self):
        """Set up test molecule and basis layout."""
        self.mol = pyscf.M(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            output="/dev/null",
            cart=1,
        )
        self.basis_layout = BasisLayout.from_mol(self.mol)

    def tearDown(self):
        if hasattr(self, 'mol'):
            self.mol.stdout.close()
        # NOTE: Don't call free_all_blocks() here as it causes CUDA module deallocation
        # errors when compiled modules are still referenced. Let CuPy manage memory naturally.

    def _make_simple_pairs(self, nbasis):
        """Create simple ij shell pairs (no padding) for VJ tests."""
        pairs = []
        limit = min(nbasis, 2)  # keep arrays tiny for unit tests
        for i in range(limit):
            for j in range(limit):
                pairs.append([i, j])
        if not pairs:
            return cp.empty((0, 2), dtype=cp.int32)
        return cp.asarray(pairs, dtype=cp.int32)

    def _make_tiled_pairs(self, nbasis, tile: int = 16):
        """Create VK-style tiled shell pairs padded to the CUDA tile size."""
        pairs = self._make_simple_pairs(nbasis)
        if pairs.size == 0:
            return pairs, 0

        n_pairs = pairs.shape[0]
        pad = (-n_pairs) % tile
        if pad:
            # Repeat the last valid pair to keep indices in range
            padding = cp.repeat(pairs[-1:], pad, axis=0)
            pairs = cp.concatenate([pairs, padding], axis=0)

        n_tiles = pairs.shape[0] // tile
        return cp.ascontiguousarray(pairs), n_tiles

    def test_vj_kernel_accepts_packed_basis_data(self):
        """Test that VJ kernel accepts packed basis_data parameter."""
        np.random.seed(42)
        nao = int(self.basis_layout.ao_loc[-1])
        dm_gpu = cp.zeros((nao, nao), dtype=np.float64)

        # Get packed basis data
        basis_data = self.basis_layout.basis_data_fp64
        packed_basis = basis_data['packed'].ravel()

        # Verify packed_basis has correct type
        self.assertIsInstance(packed_basis, cp.ndarray)
        self.assertEqual(packed_basis.dtype, np.float64)

        # Prepare output
        vj_gpu = cp.zeros((nao, nao), dtype=np.float64)

        # Create minimal pairs
        nbasis = self.basis_layout.nbasis
        ij_pairs = self._make_simple_pairs(nbasis)
        kl_pairs = self._make_simple_pairs(nbasis)

        n_ij = len(ij_pairs)
        n_kl = len(kl_pairs)
        q_cond_ij = cp.ones(n_ij, dtype=np.float32)
        q_cond_kl = cp.ones(n_kl, dtype=np.float32)
        log_cutoff = np.float32(-30.0)

        # Get kernel
        ang = (0, 0, 0, 0)
        nprim = (1, 1, 1, 1)

        try:
            _, _, fun = gen_vj_kernel(ang, nprim, dtype=np.float64, n_dm=1)

            # Call kernel - this verifies the signature is correct
            omega = None
            fun(
                nao,
                packed_basis,  # New packed signature
                dm_gpu,
                vj_gpu,
                omega,
                ij_pairs,
                n_ij,
                kl_pairs,
                n_kl,
                q_cond_ij,
                q_cond_kl,
                log_cutoff,
            )

            # Verify output shape
            self.assertEqual(vj_gpu.shape, (nao, nao))

        except Exception as e:
            self.fail(f"VJ kernel failed to accept packed basis_data: {e}")

    def test_vk_kernel_accepts_packed_basis_data(self):
        """Test that VK kernel accepts packed basis_data parameter."""
        np.random.seed(42)
        nao = int(self.basis_layout.ao_loc[-1])
        dm_gpu = cp.zeros((nao, nao), dtype=np.float64)

        # Get packed basis data
        basis_data = self.basis_layout.basis_data_fp64
        packed_basis = basis_data['packed'].ravel()

        # Prepare output
        vk_gpu = cp.zeros((nao, nao), dtype=np.float64)

        # Create minimal pairs
        nbasis = self.basis_layout.nbasis
        ij_pairs, n_ij_tiles = self._make_tiled_pairs(nbasis)
        kl_pairs, n_kl_tiles = self._make_tiled_pairs(nbasis)

        q_cond_ij = cp.ones(len(ij_pairs), dtype=np.float32)
        q_cond_kl = cp.ones(len(kl_pairs), dtype=np.float32)
        log_cutoff = np.float32(-30.0)

        # Get kernel
        ang = (0, 0, 0, 0)
        nprim = (1, 1, 1, 1)

        try:
            _, _, fun = gen_vk_kernel(ang, nprim, dtype=np.float64, n_dm=1)

            # Call kernel
            omega = None
            fun(
                nao,
                packed_basis,  # New packed signature
                dm_gpu,
                vk_gpu,
                omega,
                ij_pairs,
                n_ij_tiles,
                kl_pairs,
                n_kl_tiles,
                q_cond_ij,
                q_cond_kl,
                log_cutoff,
            )

            # Verify output shape
            self.assertEqual(vk_gpu.shape, (nao, nao))

        except Exception as e:
            self.fail(f"VK kernel failed to accept packed basis_data: {e}")


if __name__ == "__main__":
    print("Full Tests for 2D VJ/VK Kernels with Packed Basis Data")
    unittest.main()
