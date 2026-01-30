/*
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
*/

extern "C" __global__
void cart2cart(double *dst, const double* src,
            const int nao_dst, const int nao_src, const int n_idx,
            const int* __restrict__ dst_indices,
            const int* __restrict__ src_indices){

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n_idx || j >= n_idx) return;

    const int src_i = src_indices[i];
    const int src_j = src_indices[j];
    const int dst_i = dst_indices[i];
    const int dst_j = dst_indices[j];

    const double val = src[src_i * nao_src + src_j];
    atomicAdd(&dst[dst_i * nao_dst + dst_j], val);
}
