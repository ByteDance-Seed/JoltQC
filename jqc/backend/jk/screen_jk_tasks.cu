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

// Portions of this file adapted from GPU4PySCF v1.4 (https://github.com/pyscf/gpu4pyscf)
// Copyright 2025 PySCF developer.
// Licensed under the Apache License, Version 2.0.

// Ensure 64-bit integer width across platforms
typedef unsigned long long uint64_t;

__forceinline__ __device__
int global_offset(int* batch_head, int val){
    // Calculate the cumulative sum of the count array
    constexpr int warp_size = 32;
    constexpr int num_warps = threads / warp_size;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int lane  = tid & (warp_size - 1);    
    const int warp  = tid / warp_size;
    int inclusive = val;
    for (int ofs = 1; ofs < warp_size; ofs <<= 1) {
        int n = __shfl_up_sync(0xffffffff, inclusive, ofs);
        if (lane >= ofs) inclusive += n;                
    }

    __shared__ int warp_tot[num_warps];  
    if (lane == warp_size - 1) warp_tot[warp] = inclusive;  
    __syncthreads(); 

    if (warp == 0) {
        int wval = (lane < num_warps) ? warp_tot[lane] : 0;
#pragma unroll
        for (int ofs = 1; ofs < warp_size; ofs <<= 1) {
            int n = __shfl_up_sync(0xffffffff, wval, ofs);
            if (lane >= ofs) wval += n;
        }
        if (lane < num_warps) warp_tot[lane] = wval;
    }
    __syncthreads();

    // Block-exclusive prefix for this thread
    const int warp_offset      = (warp == 0) ? 0 : warp_tot[warp - 1];
    const int inclusive_block  = warp_offset + inclusive;
    const int exclusive_block  = inclusive_block - val;

    // --- block total is the last warp's inclusive sum
    const int block_total = warp_tot[num_warps - 1];

    if (block_total == 0) return 0;

    // Single atomic to reserve a global range
    __shared__ int base;
    if (tid == 0) base = atomicAdd(batch_head, block_total);
    __syncthreads();

    return base + exclusive_block;
}


extern "C" __global__ 
void screen_jk_tasks(ushort4 *shl_quartet_idx, int *batch_head, const int nbas, 
    const int * __restrict__ tile_ij_mapping, 
    const int * __restrict__ tile_kl_mapping, 
    const int ntiles_ij1, const int ntiles_kl1,
    const float * __restrict__ q_cond,
    const float * __restrict__ dm_cond, 
    const float cutoff, const float cutoff_fp64)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int ij = blockIdx.x * blockDim.x + tx;
    int kl = blockIdx.y * blockDim.y + ty;

    bool active = true;
    if (ij >= ntiles_ij1 || kl >= ntiles_kl1){
        ij = 0;
        kl = 0;
        active = false;
    }

    // Load tile mappings only if active to avoid OOB when mappings are empty
    const int nbas_tiles = nbas / TILE;
    int tile_ij = 0, tile_kl = 0;
    int tile_i = 0, tile_j = 0, tile_k = 0, tile_l = 0;
    int ish0 = 0, jsh0 = 0, ksh0 = 0, lsh0 = 0;
    if (active) {
        tile_ij = tile_ij_mapping[ij];
        tile_kl = tile_kl_mapping[kl];
        // Optimize division and modulo operations
        tile_i = tile_ij / nbas_tiles;
        tile_j = tile_ij - tile_i * nbas_tiles;  // Replace modulo with subtraction
        tile_k = tile_kl / nbas_tiles;
        tile_l = tile_kl - tile_k * nbas_tiles;  // Replace modulo with subtraction
        
        ish0 = tile_i * TILE;
        jsh0 = tile_j * TILE;
        ksh0 = tile_k * TILE;
        lsh0 = tile_l * TILE;
    }
    // Removed unused ish1/jsh1/ksh1/lsh1 to reduce register pressure
    // Number of (i,j,k,l) combinations is TILE^4; we need ceil(TILE^4/64) words
    constexpr int N_BITS = TILE*TILE*TILE*TILE;
    constexpr int mask_size = (N_BITS + 63) / 64;
    uint64_t mask_bits_fp32[mask_size] = {0};
    uint64_t mask_bits_fp64[mask_size] = {0};

    float dm_kl[TILE*TILE];
    float q_kl[TILE*TILE];

    int count_fp32 = 0;
    int count_fp64 = 0;
    if (active){
        // Prefetch KL tile only for active threads
        for (int k = 0; k < TILE; k++){
            const int ksh = ksh0 + k;
            const int ksh_base = ksh * nbas;
            for (int l = 0; l < TILE; l++){
                const int lsh = lsh0 + l;
                const int bas_kl = ksh_base + lsh;
                const int kl_idx = k * TILE + l;
                if constexpr(do_j){
                    dm_kl[kl_idx] = __ldg(&dm_cond[bas_kl]);
                }
                q_kl[kl_idx] = __ldg(&q_cond[bas_kl]);
            }
        }

        for (int i = 0; i < TILE; ++i){
            const int ish = ish0 + i;
            const int ish_base = ish * nbas;
            // j must satisfy jsh <= ish -> j <= ish - jsh0
            int j_end = ish - jsh0 + 1; // exclusive upper bound
            j_end = (j_end < 0) ? 0 : (j_end > TILE ? TILE : j_end);

            for (int j = 0; j < j_end; ++j){
                const int jsh = jsh0 + j;
                const int jsh_base = jsh * nbas;
                const int bas_ij = ish_base + jsh;
                const float q_ij = __ldg(&q_cond[bas_ij]);
                float d_ij = 0.0f;
                if constexpr(do_j){
                    d_ij = __ldg(&dm_cond[bas_ij]);
                }

                float dm_il[TILE], dm_jl[TILE];
                for (int l = 0; l < TILE; l++){
                    const int lsh = lsh0 + l;
                    dm_il[l] = __ldg(&dm_cond[ish_base + lsh]);
                    dm_jl[l] = __ldg(&dm_cond[jsh_base + lsh]);
                }

                // k must satisfy ksh <= ish -> k <= ish - ksh0
                int k_end = ish - ksh0 + 1; // exclusive upper bound
                k_end = (k_end < 0) ? 0 : (k_end > TILE ? TILE : k_end);

                for (int k = 0; k < k_end; ++k){
                    const int ksh = ksh0 + k;
                    const float d_ik = __ldg(&dm_cond[ish_base + ksh]);
                    const float d_jk = __ldg(&dm_cond[jsh_base + ksh]);

                    // l must satisfy lsh <= ksh -> l <= k + (ksh0 - lsh0)
                    int l_end = k + (ksh0 - lsh0) + 1; // exclusive upper bound
                    l_end = (l_end < 0) ? 0 : (l_end > TILE ? TILE : l_end);

                    for (int l = 0; l < l_end; ++l){
                        const int lsh = lsh0 + l;
                        const int bas_kl = ksh * nbas + lsh;
                        if (bas_ij < bas_kl) continue;

                        const float q_ijkl = q_ij + q_kl[k * TILE + l];
                        float d_large = -36.8f;

                        if constexpr(do_k){
                            const float d_il = dm_il[l];
                            const float d_jl = dm_jl[l];
                            d_large = max(d_large, d_ik);
                            d_large = max(d_large, d_jk);
                            d_large = max(d_large, d_il);
                            d_large = max(d_large, d_jl);
                        }
                        if constexpr(do_j){
                            const float d_kl = dm_kl[k * TILE + l];
                            d_large = max(d_large, d_ij);
                            d_large = max(d_large, d_kl);
                        }

                        const float dq = q_ijkl + d_large;
                        if (dq <= cutoff) continue;
                        const bool sel_fp64 = (dq > cutoff_fp64);
                        const bool sel_fp32 = !sel_fp64;

                        const uint64_t idx = ((i * TILE + j) * TILE + k) * TILE + l;
                        const uint64_t word = idx >> 6;
                        const uint64_t bit = idx & 63;
                        const uint64_t bitmask = 1ull << bit;

                        mask_bits_fp32[word] |= bitmask & (-sel_fp32);
                        mask_bits_fp64[word] |= bitmask & (-sel_fp64);
                        count_fp32 += sel_fp32;
                        count_fp64 += sel_fp64;
                    }
                }
            }
        }
    }

    // Check if entire block has no work - all threads must participate
    __shared__ bool has_work;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        has_work = false;
    }
    __syncthreads();

    if (count_fp32 > 0 || count_fp64 > 0) {
        atomicOr((int*)&has_work, 1);
    }
    __syncthreads();

    if (!has_work) {
        return;
    }

    int offset_fp32 = global_offset(batch_head+1, count_fp32);
    int offset_fp64 = global_offset(batch_head+2, -count_fp64) - 1;
    
    if (active){
#pragma unroll
        for (int i = 0; i < TILE; i++){
            for (int j = 0; j < TILE; j++){
                for (int k = 0; k < TILE; k++){
                    for (int l = 0; l < TILE; l++){
                        const uint64_t idx = ((i * TILE + j) * TILE + k) * TILE + l;
                        const uint64_t word = idx >> 6;
                        const uint64_t bit = idx & 63;
                        
                        const bool sel_fp32 = (mask_bits_fp32[word] >> bit) & 1ull;
                        const bool sel_fp64 = (mask_bits_fp64[word] >> bit) & 1ull;
                        
                        ushort4 sq;
                        sq.x = ish0 + i; 
                        sq.y = jsh0 + j; 
                        sq.z = ksh0 + k; 
                        sq.w = lsh0 + l;
                        
                        if (sel_fp32) {
                            shl_quartet_idx[offset_fp32] = sq;
                            ++offset_fp32;
                        }
                        if (sel_fp64) {
                            shl_quartet_idx[offset_fp64] = sq;
                            --offset_fp64;
                        }
                    }
                }
            }
        }
    }
}
