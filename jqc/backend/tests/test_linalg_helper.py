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

import pytest
import cupy as cp
import numpy as np
from jqc.backend.linalg_helper import inplace_add_transpose, max_block_pooling

def test_inplace_add_transpose():
    """Tests the inplace_add_transpose function for correctness."""
    n = 50
    np.random.seed(42)
    A_host = np.random.rand(n, n).astype(np.float64)
    A_device = cp.asarray(A_host)
    expected = A_host + A_host.T
    inplace_add_transpose(A_device)
    result_host = cp.asnumpy(A_device)
    assert np.allclose(result_host, expected)

    n = 50
    np.random.seed(42)
    A_host = np.random.rand(3, n, n).astype(np.float64)
    A_device = cp.asarray(A_host)
    expected = A_host + A_host.transpose([0,2,1])
    inplace_add_transpose(A_device)
    result_host = cp.asnumpy(A_device)
    assert np.allclose(result_host, expected)

def test_max_block_pooling():
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
            start_row, end_row = offset_host[i], offset_host[i+1]
            start_col, end_col = offset_host[j], offset_host[j+1]
            block = A_host[start_row:end_row, start_col:end_col]
            expected[i, j] = np.max(block)

    result_device = max_block_pooling(A_device, offset_device)
    result_host = cp.asnumpy(result_device)
    assert result_host.shape == expected.shape
    assert np.allclose(result_host, expected)

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
            start_row, end_row = offset_host[i], offset_host[i+1]
            start_col, end_col = offset_host[j], offset_host[j+1]
            block = A_host[:, start_row:end_row, start_col:end_col]
            expected[i, j] = np.max(block)

    result_device = max_block_pooling(A_device, offset_device)
    result_host = cp.asnumpy(result_device)
    assert result_host.shape == expected.shape
    assert np.allclose(result_host, expected)
test_inplace_add_transpose()
test_max_block_pooling()
