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

import cupy as cp
import numpy as np

__all__ = ["inplace_add_transpose", "l2_block_pooling", "max_block_pooling"]

compile_options = ("-std=c++17", "--use_fast_math", "--minimal")


def inplace_add_transpose(mat: cp.ndarray):
    """In-place A <- A + A.T for the last two dimensions of a CuPy array."""
    assert (
        mat.ndim >= 2 and mat.shape[-1] == mat.shape[-2]
    ), "Last two dimensions must be square"
    assert mat.dtype == cp.float64, "Kernel currently only supports float64"
    n = mat.shape[-1]

    _kernel = cp.RawKernel(
        r"""
#define TILE_DIM 32

extern "C" __global__
void add_transpose_inplace(double* __restrict__ A, int batch_size, int n) {
    // Shared memory with padding to avoid bank conflicts
    __shared__ double tile_ij[TILE_DIM][TILE_DIM + 1];
    __shared__ double tile_ji[TILE_DIM][TILE_DIM + 1];

    // No need to check batch_idx bounds - grid is sized exactly
    const size_t matrix_offset = (size_t)blockIdx.z * n * n;
    double* __restrict__ matrix_A = A + matrix_offset;

    const int tile_i = blockIdx.y;
    const int tile_j = blockIdx.x;

    // Only process upper triangular tiles (including diagonal)
    if (tile_j < tile_i) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Precompute global indices
    const int row = tile_i * TILE_DIM + ty;
    const int col = tile_j * TILE_DIM + tx;
    const int row_t = tile_j * TILE_DIM + ty;
    const int col_t = tile_i * TILE_DIM + tx;

    // Precompute masks to reduce branches
    const bool is_diag_tile = (tile_i == tile_j);
    const bool in_bounds_ij = (row < n) & (col < n);
    const bool in_bounds_ji = (row_t < n) & (col_t < n);

    // Load primary tile with coalesced access and __ldg for read-only
    const double val_ij = in_bounds_ij ? __ldg(&matrix_A[row * n + col]) : 0.0;
    const double val_ji = (!is_diag_tile && in_bounds_ji) ? __ldg(&matrix_A[row_t * n + col_t]) : val_ij;

    // Store to shared memory
    tile_ij[ty][tx] = val_ij;
    tile_ji[ty][tx] = val_ji;
    __syncthreads();

    // Transpose access through shared memory
    const double trans_ij = tile_ji[tx][ty];
    const double trans_ji = tile_ij[tx][ty];

    // Compute results with minimal branching
    const bool is_diag_elem = (ty == tx);
    const bool is_upper = (col > row);

    const double sum_ij = val_ij + trans_ij;
    const double sum_ji = val_ji + trans_ji;

    // Select final values using ternary (compiles to predicated instructions)
    const double result_ij = (is_diag_tile & is_diag_elem) ? (2.0 * val_ij) : sum_ij;
    const double result_ji = sum_ji;

    // Write results with minimal branching
    // For diagonal tiles: write upper triangle + diagonal symmetrically
    // For off-diagonal tiles: write both tiles fully
    if (in_bounds_ij) {
        const bool should_write_ij = !is_diag_tile | is_upper | is_diag_elem;
        if (should_write_ij) {
            matrix_A[row * n + col] = result_ij;
            // Symmetric write for diagonal tiles
            if (is_diag_tile & is_upper) {
                matrix_A[col * n + row] = result_ij;
            }
        }
    }

    // Write transpose tile for off-diagonal only
    if (!is_diag_tile && in_bounds_ji) {
        matrix_A[row_t * n + col_t] = result_ji;
    }
}
""",
        "add_transpose_inplace",
        compile_options,
    )

    batch_size = int(np.prod(mat.shape[:-2]))
    threads = (32, 32, 1)
    blocks = (
        (n + 32 - 1) // 32,
        (n + 32 - 1) // 32,
        batch_size,
    )
    _kernel(blocks, threads, (mat, batch_size, n))
    return mat


def max_block_pooling(matrix: cp.ndarray, offsets: cp.ndarray) -> cp.ndarray:
    """
    Blockwise max pooling on square CuPy matrix using a 1D offset array.
    If the input is 3D, it performs block max pooling on the last two dimensions
    and then a max operation along the first dimension.

    Parameters:
    - matrix: CuPy 2D array of shape (N, N) or 3D array of shape (B, N, N). Supports float32 and float64.
    - offsets: 1D array of block boundaries, length K+1

    Returns:
    - output: CuPy 2D array of shape (K, K) with same dtype as input.
    """
    assert matrix.ndim in [2, 3], "Input matrix must be 2D or 3D"
    if matrix.ndim == 2:
        assert matrix.shape[0] == matrix.shape[1], "Input 2D matrix must be square"
    else:  # 3D
        assert (
            matrix.shape[1] == matrix.shape[2]
        ), "Last two dimensions of 3D matrix must be square"

    assert matrix.dtype in [
        cp.float32,
        cp.float64,
    ], "Kernel supports float32 and float64"
    assert offsets.ndim == 1, "Offsets must be a 1D array"

    # Select appropriate kernel based on dtype
    if matrix.dtype == cp.float64:
        data_type = "double"
        kernel_name = "block_max_kernel_fp64"
    else:  # cp.float32
        data_type = "float"
        kernel_name = "block_max_kernel_fp32"

    kernel_code = rf"""
    extern "C" __global__
    void {kernel_name}(const {data_type}* __restrict__ mat,
                       const int* __restrict__ offsets,
                       {data_type}* __restrict__ out,
                       int batch_size,
                       int stride,
                       int k)
    {{
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= k || j >= k) return;

        int r0 = offsets[i];
        int r1 = offsets[i+1];
        int c0 = offsets[j];
        int c1 = offsets[j+1];

        {data_type} maxval = 0.0;
        for (int b = 0; b < batch_size; ++b) {{
            for (int r = r0; r < r1; ++r) {{
                for (int c = c0; c < c1; ++c) {{
                    const {data_type} val = mat[b * stride * stride + r * stride + c];
                    const {data_type} abs_val = abs(val);
                    //if (abs_val > maxval) maxval = abs_val;
                    maxval = max(maxval, abs_val);
                }}
            }}
        }}
        out[i * k + j] = maxval;
    }}
    """

    kernel = cp.RawKernel(kernel_code, kernel_name, options=compile_options)

    n_dim = matrix.shape[-1]
    matrix = matrix.reshape([-1, n_dim, n_dim])
    batch_size = matrix.shape[0]
    k_blocks = offsets.shape[0] - 1

    offsets_dev = cp.asarray(offsets, dtype=cp.int32)
    out = cp.empty((k_blocks * k_blocks,), dtype=matrix.dtype)

    threads = (16, 16)
    blocks = (
        (k_blocks + threads[0] - 1) // threads[0],
        (k_blocks + threads[1] - 1) // threads[1],
    )

    kernel(blocks, threads, (matrix, offsets_dev, out, batch_size, n_dim, k_blocks))

    return out.reshape((k_blocks, k_blocks))


def l2_block_pooling(matrix: cp.ndarray, offsets: cp.ndarray) -> cp.ndarray:
    """
    Blockwise l2 pooling on square CuPy matrix using a 1D offset array.
    If the input is 3D, it performs block max pooling on the last two dimensions
    and then a max operation along the first dimension.

    Parameters:
    - matrix: CuPy 2D array of shape (N, N) or 3D array of shape (B, N, N). Must be of type float64.
    - offsets: 1D array of block boundaries, length K+1

    Returns:
    - output: CuPy 2D array of shape (K, K).
    """
    assert matrix.ndim in [2, 3], "Input matrix must be 2D or 3D"
    if matrix.ndim == 2:
        assert matrix.shape[0] == matrix.shape[1], "Input 2D matrix must be square"
    else:  # 3D
        assert (
            matrix.shape[1] == matrix.shape[2]
        ), "Last two dimensions of 3D matrix must be square"

    assert matrix.dtype == cp.float64, "Kernel currently only supports float64"
    assert offsets.ndim == 1, "Offsets must be a 1D array"

    kernel_code = r"""
    extern "C" __global__
    void block_max_kernel(const double* __restrict__ mat,
                          const int* __restrict__ offsets,
                          double* __restrict__ out,
                          int batch_size,
                          int stride,
                          int k)
    {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= k || j >= k) return;

        int r0 = offsets[i];
        int r1 = offsets[i+1];
        int c0 = offsets[j];
        int c1 = offsets[j+1];

        double l2_norm = 0.0;
        for (int b = 0; b < batch_size; ++b) {
            for (int r = r0; r < r1; ++r) {
                for (int c = c0; c < c1; ++c) {
                    const double val = mat[b * stride * stride + r * stride + c];
                    l2_norm += val * val;
                }
            }
        }
        out[i * k + j] = sqrt(l2_norm);
    }
    """

    kernel = cp.RawKernel(kernel_code, "block_max_kernel", options=compile_options)

    n_dim = matrix.shape[-1]
    matrix = matrix.reshape([-1, n_dim, n_dim])
    batch_size = matrix.shape[0]
    k_blocks = offsets.shape[0] - 1

    offsets_dev = cp.asarray(offsets, dtype=cp.int32)
    out = cp.empty((k_blocks * k_blocks,), dtype=cp.float64)

    threads = (16, 16)
    blocks = (
        (k_blocks + threads[0] - 1) // threads[0],
        (k_blocks + threads[1] - 1) // threads[1],
    )

    kernel(blocks, threads, (matrix, offsets_dev, out, batch_size, n_dim, k_blocks))

    return out.reshape((k_blocks, k_blocks))
