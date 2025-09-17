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


def inplace_add_transpose(A: cp.ndarray):
    """In-place A <- A + A.T for the last two dimensions of a CuPy array."""
    assert (
        A.ndim >= 2 and A.shape[-1] == A.shape[-2]
    ), "Last two dimensions must be square"
    assert A.dtype == cp.float64, "Kernel currently only supports float64"
    n = A.shape[-1]

    _kernel = cp.RawKernel(
        r"""
extern "C" __global__
void add_transpose_inplace(double* A, int batch_size, int n) {
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n || j < i) return;

    size_t matrix_offset = (size_t)batch_idx * n * n;
    double* matrix_A = A + matrix_offset;

    double aij = matrix_A[i * n + j];
    double aji = matrix_A[j * n + i];

    if (i == j) {
        matrix_A[i * n + j] = 2.0 * aij;
    } else {
        double sum = aij + aji;
        matrix_A[i * n + j] = sum;
        matrix_A[j * n + i] = sum;
    }
}
""",
        "add_transpose_inplace",
        compile_options,
    )

    batch_size = int(np.prod(A.shape[:-2]))
    threads = (16, 16, 1)
    blocks = (
        (n + threads[0] - 1) // threads[0],
        (n + threads[1] - 1) // threads[1],
        batch_size,
    )
    _kernel(blocks, threads, (A, batch_size, n))
    return A


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
                    if (abs_val > maxval) maxval = abs_val;
                }}
            }}
        }}
        out[i * k + j] = maxval;
    }}
    """

    kernel = cp.RawKernel(kernel_code, kernel_name, options=compile_options)

    N = matrix.shape[-1]
    matrix = matrix.reshape([-1, N, N])
    batch_size = matrix.shape[0]
    K = offsets.shape[0] - 1

    offsets_dev = cp.asarray(offsets, dtype=cp.int32)
    out = cp.empty((K * K,), dtype=matrix.dtype)

    threads = (16, 16)
    blocks = ((K + threads[0] - 1) // threads[0], (K + threads[1] - 1) // threads[1])

    kernel(blocks, threads, (matrix, offsets_dev, out, batch_size, N, K))

    return out.reshape((K, K))


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

    N = matrix.shape[-1]
    matrix = matrix.reshape([-1, N, N])
    batch_size = matrix.shape[0]
    K = offsets.shape[0] - 1

    offsets_dev = cp.asarray(offsets, dtype=cp.int32)
    out = cp.empty((K * K,), dtype=cp.float64)

    threads = (16, 16)
    blocks = ((K + threads[0] - 1) // threads[0], (K + threads[1] - 1) // threads[1])

    kernel(blocks, threads, (matrix, offsets_dev, out, batch_size, N, K))

    return out.reshape((K, K))
