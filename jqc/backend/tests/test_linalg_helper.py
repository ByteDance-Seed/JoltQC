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
from jqc.backend.linalg_helper import inplace_add_transpose, max_block_pooling


def setUpModule():
    pass


def tearDownModule():
    pass


class KnownValues(unittest.TestCase):
    def test_inplace_add_transpose(self):
        """Tests the inplace_add_transpose function for correctness."""
        n = 50
        np.random.seed(42)
        A_host = np.random.rand(n, n).astype(np.float64)
        A_device = cp.asarray(A_host)
        expected = A_host + A_host.T
        inplace_add_transpose(A_device)
        result_host = cp.asnumpy(A_device)
        self.assertTrue(np.allclose(result_host, expected))

        n = 50
        np.random.seed(42)
        A_host = np.random.rand(3, n, n).astype(np.float64)
        A_device = cp.asarray(A_host)
        expected = A_host + A_host.transpose([0, 2, 1])
        inplace_add_transpose(A_device)
        result_host = cp.asnumpy(A_device)
        self.assertTrue(np.allclose(result_host, expected))

    def test_max_block_pooling(self):
        """Tests the max_block_pooling function for correctness."""
        np.random.seed(42)

        # Test 2D array
        n = 40
        A_host = np.random.rand(n, n).astype(np.float64)
        offset_host = np.array([0, 10, 25, 40], dtype=np.int32)
        A_device = cp.asarray(A_host)
        offset_device = cp.asarray(offset_host)

        num_blocks = len(offset_host) - 1
        expected = np.zeros((num_blocks, num_blocks), dtype=np.float64)
        for i in range(num_blocks):
            for j in range(num_blocks):
                start_row, end_row = offset_host[i], offset_host[i + 1]
                start_col, end_col = offset_host[j], offset_host[j + 1]
                block = A_host[start_row:end_row, start_col:end_col]
                expected[i, j] = np.max(block)

        result_device = max_block_pooling(A_device, offset_device)
        result_host = cp.asnumpy(result_device)
        self.assertEqual(result_host.shape, expected.shape)
        self.assertTrue(np.allclose(result_host, expected))

        # Test 3D array
        n = 40
        batch_size = 3
        A_host = np.random.rand(batch_size, n, n).astype(np.float64)
        offset_host = np.array([0, 10, 25, 40], dtype=np.int32)
        A_device = cp.asarray(A_host)
        offset_device = cp.asarray(offset_host)

        num_blocks = len(offset_host) - 1
        expected = np.zeros((num_blocks, num_blocks), dtype=np.float64)
        for i in range(num_blocks):
            for j in range(num_blocks):
                start_row, end_row = offset_host[i], offset_host[i + 1]
                start_col, end_col = offset_host[j], offset_host[j + 1]
                block = A_host[:, start_row:end_row, start_col:end_col]
                expected[i, j] = np.max(block)

        result_device = max_block_pooling(A_device, offset_device)
        result_host = cp.asnumpy(result_device)
        self.assertEqual(result_host.shape, expected.shape)
        self.assertTrue(np.allclose(result_host, expected))

    def test_max_block_pooling_fp32(self):
        """Tests the max_block_pooling function with fp32 precision."""
        np.random.seed(42)

        # Test 2D array with fp32
        n = 40
        A_host = np.random.rand(n, n).astype(np.float32)
        offset_host = np.array([0, 10, 25, 40], dtype=np.int32)
        A_device = cp.asarray(A_host)
        offset_device = cp.asarray(offset_host)

        num_blocks = len(offset_host) - 1
        expected = np.zeros((num_blocks, num_blocks), dtype=np.float32)
        for i in range(num_blocks):
            for j in range(num_blocks):
                start_row, end_row = offset_host[i], offset_host[i + 1]
                start_col, end_col = offset_host[j], offset_host[j + 1]
                block = A_host[start_row:end_row, start_col:end_col]
                expected[i, j] = np.max(block)

        result_device = max_block_pooling(A_device, offset_device)
        result_host = cp.asnumpy(result_device)

        # Verify dtype preservation
        self.assertEqual(result_device.dtype, cp.float32)
        self.assertEqual(result_host.dtype, np.float32)
        self.assertEqual(result_host.shape, expected.shape)
        self.assertTrue(np.allclose(result_host, expected, rtol=1e-6, atol=1e-6))

        # Test 3D array with fp32
        batch_size = 3
        A_host_3d = np.random.rand(batch_size, n, n).astype(np.float32)
        A_device_3d = cp.asarray(A_host_3d)

        expected_3d = np.zeros((num_blocks, num_blocks), dtype=np.float32)
        for i in range(num_blocks):
            for j in range(num_blocks):
                start_row, end_row = offset_host[i], offset_host[i + 1]
                start_col, end_col = offset_host[j], offset_host[j + 1]
                block = A_host_3d[:, start_row:end_row, start_col:end_col]
                expected_3d[i, j] = np.max(block)

        result_device_3d = max_block_pooling(A_device_3d, offset_device)
        result_host_3d = cp.asnumpy(result_device_3d)

        # Verify dtype preservation for 3D
        self.assertEqual(result_device_3d.dtype, cp.float32)
        self.assertEqual(result_host_3d.dtype, np.float32)
        self.assertEqual(result_host_3d.shape, expected_3d.shape)
        self.assertTrue(np.allclose(result_host_3d, expected_3d, rtol=1e-6, atol=1e-6))

    def test_max_block_pooling_dtype_consistency(self):
        """Tests that max_block_pooling preserves input dtype for both fp32 and fp64."""
        n = 20
        offset_host = np.array([0, 10, 20], dtype=np.int32)
        offset_device = cp.asarray(offset_host)

        # Use same input data for both precisions to ensure comparable results
        np.random.seed(42)
        A_host = np.random.rand(n, n).astype(np.float64)

        # Test fp64 dtype preservation
        A_fp64 = cp.asarray(A_host, dtype=cp.float64)
        result_fp64 = max_block_pooling(A_fp64, offset_device)
        self.assertEqual(result_fp64.dtype, cp.float64)

        # Test fp32 dtype preservation with same input data
        A_fp32 = cp.asarray(A_host, dtype=cp.float32)
        result_fp32 = max_block_pooling(A_fp32, offset_device)
        self.assertEqual(result_fp32.dtype, cp.float32)

        # Verify results are approximately equal (within fp32 precision)
        result_fp64_as_fp32 = result_fp64.astype(cp.float32)
        self.assertTrue(
            cp.allclose(result_fp32, result_fp64_as_fp32, rtol=1e-5, atol=1e-5)
        )


if __name__ == "__main__":
    print("Full Tests for Linalg Helper")
    unittest.main()
