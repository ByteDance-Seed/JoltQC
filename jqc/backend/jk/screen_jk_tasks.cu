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

constexpr float minval = -36.8f; // exp(-36.8) ~ 1e-16

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
#pragma unroll
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
    const float cutoff, const float cutoff_fp64, const float log_max_dm)
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
    int tile_i = 0, tile_j = 0, tile_k = 0, tile_l = 0;
    int ish0 = 0, jsh0 = 0, ksh0 = 0, lsh0 = 0;
    if (active) {
        const int tile_ij = tile_ij_mapping[ij];
        const int tile_kl = tile_kl_mapping[kl];
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
    
    // Number of (i,j,k,l) combinations is TILE^4; we need ceil(TILE^4/64) words
    constexpr int N_BITS = TILE*TILE*TILE*TILE;
    constexpr int mask_size = (N_BITS + 63) / 64;
    uint64_t mask_bits_fp32[mask_size] = {0};
    uint64_t mask_bits_fp64[mask_size] = {0};

    // Cache only frequently accessed read-only data to reduce register pressure
    float q_kl[TILE*TILE];

    int count_fp32 = 0;
    int count_fp64 = 0;

    // Prefetch KL tile and compute max q_kl for early exit
    float q_kl_max = minval;
#pragma unroll
    for (int k = 0; k < TILE; k++){
        const int ksh = ksh0 + k;
#pragma unroll
        for (int l = 0; l < TILE; l++){
            const int lsh = lsh0 + l;
            const int bas_kl = ksh * nbas + lsh;
            const int kl_idx = k * TILE + l;
            q_kl[kl_idx] = active ? __ldg(&q_cond[bas_kl]) : minval;
            q_kl_max = max(q_kl_max, q_kl[kl_idx]);
        }
    }

    // Compute max q_ij for early exit (without loading density matrix yet)
    float q_ij_max = minval;
#pragma unroll
    for (int i = 0; i < TILE; ++i){
        const int ish = ish0 + i;
#pragma unroll
        for (int j = 0; j < TILE; ++j){
            const int jsh = jsh0 + j;
            const int bas_ij = ish * nbas + jsh;
            const float q_ij = active ? __ldg(&q_cond[bas_ij]) : minval;
            q_ij_max = max(q_ij_max, q_ij);
        }
    }

    // Early exit: Check if it's impossible for any quartet to pass screening
    // Screening condition: q_ij + q_kl + d_large > cutoff
    // where d_large = max(dm elements) for this block
    // Use log_max_dm (the global maximum density matrix element) as upper bound for d_large
    // This is conservative: if even with the maximum possible d_large we can't pass, exit early
    // Inactive threads contribute "true" to could_pass so they don't block active threads
    bool could_pass = !active || (q_ij_max + q_kl_max + log_max_dm > cutoff);
    if (!__syncthreads_or(could_pass)) {
        return;
    }
    
    float dm_kl[TILE*TILE];  // Only used if do_j is true
    if constexpr(do_j){
#pragma unroll
        for (int k = 0; k < TILE; k++){
            const int ksh = ksh0 + k;
#pragma unroll
            for (int l = 0; l < TILE; l++){
                const int lsh = lsh0 + l;
                const int bas_kl = ksh * nbas + lsh;
                const int kl_idx = k * TILE + l;
                dm_kl[kl_idx] = active ? __ldg(dm_cond + bas_kl) : minval;
            }
        }
    }

    // Load other dm values on-demand in inner loops to reduce register pressure
    
#pragma unroll
    for (int i = 0; i < TILE; ++i){
        const int ish = ish0 + i;

#pragma unroll
        for (int j = 0; j < TILE; ++j){
            const int jsh = jsh0 + j;
            if (ish < jsh) continue;
            const int ish_base = ish * nbas;
            const int jsh_base = jsh * nbas;
            const int bas_ij = ish_base + jsh;
            const float q_ij = active ? __ldg(&q_cond[bas_ij]) : minval;
            float d_ij = 0.0f;
            if constexpr(do_j){
                d_ij = active ? __ldg(&dm_cond[bas_ij]) : minval;
            }
            float dm_il_prefetch[TILE];
            float dm_jl_prefetch[TILE];
            if constexpr(do_k){
#pragma unroll
                for (int l = 0; l < TILE; ++l){
                    const int lsh = lsh0 + l;
                    dm_il_prefetch[l] = active ? __ldg(&dm_cond[ish_base + lsh]) : minval;
                    dm_jl_prefetch[l] = active ? __ldg(&dm_cond[jsh_base + lsh]) : minval;
                }
            }
            // k must satisfy ksh <= ish -> k <= ish - ksh0
#pragma unroll
            for (int k = 0; k < TILE; ++k){
                const int ksh = ksh0 + k;
                if (ksh > ish) continue;

                // Load density matrix elements on-demand to reduce register pressure
                float d_ik = minval;
                float d_jk = minval;
                if constexpr(do_k){
                    d_ik = active ? __ldg(&dm_cond[ish_base + ksh]) : minval;
                    d_jk = active ? __ldg(&dm_cond[jsh_base + ksh]) : minval;
                }

#pragma unroll
                for (int l = 0; l < TILE; ++l){
                    const int lsh = lsh0 + l;
                    const int bas_kl = ksh * nbas + lsh;
                    if (bas_ij < bas_kl) continue;
                    if (lsh > ksh) continue;

                    const float q_ijkl = q_ij + q_kl[k * TILE + l];
                    float d_large = -36.8f;

                    if constexpr(do_k){
                        const float d_il = dm_il_prefetch[l];
                        const float d_jl = dm_jl_prefetch[l];
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

    // Separate early exit checks for FP32 and FP64
    bool has_fp32 = __syncthreads_or(count_fp32 > 0);
    bool has_fp64 = __syncthreads_or(count_fp64 > 0);

    // Early exit when the entire block produced nothing
    if (!has_fp32 && !has_fp64) {
        return;
    }

    int offset_fp32 = 0;
    int offset_fp64 = 0;

    // Only allocate offsets for active precision levels
    if (has_fp32) {
        offset_fp32 = global_offset(batch_head+1, count_fp32);
    }
    if (has_fp64) {
        offset_fp64 = global_offset(batch_head+2, -count_fp64) - 1;
    }

    // Use __ffsll() to efficiently iterate over set bits only
#pragma unroll
    for (int word_idx = 0; word_idx < mask_size; word_idx++) {
        // Process FP32 bits only if this precision has work
        if (has_fp32) {
            uint64_t word_fp32 = mask_bits_fp32[word_idx];
            while (word_fp32) {
                const int bit_pos = __ffsll(word_fp32) - 1;  // __ffsll returns 1-based index
                const uint64_t idx = (word_idx << 6) + bit_pos;

                // Decode linear index back to (i, j, k, l)
                const int i = idx / (TILE * TILE * TILE);
                const int j = (idx / (TILE * TILE)) % TILE;
                const int k = (idx / TILE) % TILE;
                const int l = idx % TILE;

                ushort4 sq;
                sq.x = ish0 + i;
                sq.y = jsh0 + j;
                sq.z = ksh0 + k;
                sq.w = lsh0 + l;

                shl_quartet_idx[offset_fp32++] = sq;
                word_fp32 &= word_fp32 - 1;  // Clear lowest set bit
            }
        }

        // Process FP64 bits only if this precision has work
        if (has_fp64) {
            uint64_t word_fp64 = mask_bits_fp64[word_idx];
            while (word_fp64) {
                const int bit_pos = __ffsll(word_fp64) - 1;
                const uint64_t idx = (word_idx << 6) + bit_pos;

                const int i = idx / (TILE * TILE * TILE);
                const int j = (idx / (TILE * TILE)) % TILE;
                const int k = (idx / TILE) % TILE;
                const int l = idx % TILE;

                ushort4 sq;
                sq.x = ish0 + i;
                sq.y = jsh0 + j;
                sq.z = ksh0 + k;
                sq.w = lsh0 + l;

                shl_quartet_idx[offset_fp64--] = sq;
                word_fp64 &= word_fp64 - 1;  // Clear lowest set bit
            }
        }
    }
}
