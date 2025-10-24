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
constexpr float minval = -36.8; // exp(-36.8) ~ 1e-16

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
    
    constexpr int align = 2;
    constexpr int align2 = align*align;

    // Fragment-level screening: store fragment indices, not individual items
    constexpr int frag_size = (TILE / align);
    constexpr int num_frags = frag_size * frag_size * frag_size * frag_size;
    constexpr int frag_mask_size = (num_frags + 63) / 64;
    uint64_t frag_mask_fp32[frag_mask_size] = {0};
    uint64_t frag_mask_fp64[frag_mask_size] = {0};

    int count_fp32 = 0;
    int count_fp64 = 0;
    if (active){
        for (int i0 = 0; i0 < TILE; i0 += align){
        for (int j0 = 0; j0 < TILE; j0 += align){
        for (int k0 = 0; k0 < TILE; k0 += align){
        for (int l0 = 0; l0 < TILE; l0 += align){
            const int ish = ish0 + i0;
            const int jsh = jsh0 + j0;
            const int ksh = ksh0 + k0;
            const int lsh = lsh0 + l0;
            const int bas_ij = ish * nbas + jsh;
            const int bas_kl = ksh * nbas + lsh;
            //if (bas_ij < bas_kl) continue;
            if (jsh > ish) continue;
            if (ksh > ish) continue;
            if (lsh > ksh) continue;
            
            float dm_kl[align2];
            float dm_ij[align2];
            float dm_jl[align2];
            float dm_ik[align2];
            float dm_jk[align2];
            float dm_il[align2];

            float q_kl[align2];
            float q_ij[align2];
            for (int ii = 0; ii < align; ++ii){
                const int ish = ish0 + i0 + ii;
                for (int jj = 0; jj < align; ++jj){
                    const int jsh = jsh0 + j0 + jj;
                    const bool mask = ish < nbas && jsh < nbas;
                    const int ij_sh = ish * nbas + jsh;
                    const float dm = mask ? dm_cond[ij_sh] : minval;
                    const float q = mask ? q_cond[ij_sh] : minval;
                    dm_ij[ii*align + jj] = dm;
                    q_ij[ii*align + jj] = q;
                }
                if constexpr(do_k){
                    for (int kk = 0; kk < align; ++kk){
                        const int ksh = ksh0 + k0 + kk;
                        const bool mask = ish < nbas && ksh < nbas;
                        const float dm = mask ? dm_cond[ish * nbas + ksh] : minval;
                        dm_ik[ii*align + kk] = dm;
                    }
                    for (int ll = 0; ll < align; ++ll){
                        const int lsh = lsh0 + l0 + ll;
                        const bool mask = ish < nbas && lsh < nbas;
                        const float dm = mask ? dm_cond[ish * nbas + lsh] : minval;
                        dm_il[ii*align + ll] = dm;
                    }
                }
            }
            if constexpr(do_k){
                for (int jj = 0; jj < align; ++jj){
                    const int jsh = jsh0 + j0 + jj;
                    for (int kk = 0; kk < align; ++kk){
                        const int ksh = ksh0 + k0 + kk;
                        const bool mask = jsh < nbas && ksh < nbas;
                        const float dm = mask ? dm_cond[jsh * nbas + ksh] : minval;
                        dm_jk[jj*align + kk] = dm;
                    }

                    for (int ll = 0; ll < align; ++ll){
                        const int lsh = lsh0 + l0 + ll;
                        const bool mask = jsh < nbas && lsh < nbas;
                        const float dm = mask ? dm_cond[jsh * nbas + lsh] : minval;
                        dm_jl[jj*align + ll] = dm;
                    }
                }
            }

            for (int kk = 0; kk < align; ++kk){
                const int ksh = ksh0 + k0 + kk;
                for (int ll = 0; ll < align; ++ll){
                    const int lsh = lsh0 + l0 + ll;
                    const int kl = kk*align + ll;
                    const int kl_sh = ksh * nbas + lsh;
                    const bool mask = ksh < nbas && lsh < nbas;
                    const float dm = mask ? dm_cond[ksh * nbas + lsh] : minval;
                    const float q = mask ? q_cond[ksh * nbas + lsh] : minval;
                    dm_kl[kl] = dm;
                    q_kl[kl] = q;
                }
            }
            
            bool select_fp32 = false;
            bool select_fp64 = false;
            for (int ii = 0; ii < align; ++ii){
            for (int jj = 0; jj < align; ++jj){
            for (int kk = 0; kk < align; ++kk){
            for (int ll = 0; ll < align; ++ll){
                const float q_ijkl = q_ij[ii*align + jj] + q_kl[kk*align + ll];
                float d_large = minval;

                if constexpr(do_k){
                    const float d_ik = dm_ik[ii*align + kk];
                    const float d_jk = dm_jk[jj*align + kk];
                    const float d_il = dm_il[ii*align + ll];
                    const float d_jl = dm_jl[jj*align + ll];
                    d_large = max(d_large, d_ik);
                    d_large = max(d_large, d_jk);
                    d_large = max(d_large, d_il);
                    d_large = max(d_large, d_jl);
                }
                if constexpr(do_j){
                    const float d_ij = dm_ij[ii*align + jj];
                    const float d_kl = dm_kl[kk*align + ll];
                    d_large = max(d_large, d_ij);
                    d_large = max(d_large, d_kl);
                }

                const float dq = q_ijkl + d_large;
                select_fp32 |= (dq > cutoff) && (dq <= cutoff_fp64);
                select_fp64 |= (dq > cutoff_fp64);
            }}}}

            if (!(select_fp32 || select_fp64)) {
                // Skip entire fragment if no combinations are selected
                continue;
            }

            // Mark the fragment - FP64 takes priority over FP32
            // If any item in fragment needs FP64, entire fragment goes to FP64
            const int frag_i = i0 / align;
            const int frag_j = j0 / align;
            const int frag_k = k0 / align;
            const int frag_l = l0 / align;
            const uint64_t frag_idx = ((frag_i * frag_size + frag_j) * frag_size + frag_k) * frag_size + frag_l;
            const uint64_t frag_word = frag_idx >> 6;
            const uint64_t frag_bit = frag_idx & 63;
            const uint64_t frag_bitmask = 1ull << frag_bit;

            if (select_fp64) {
                // Fragment needs FP64 precision
                frag_mask_fp64[frag_word] |= frag_bitmask;
                count_fp64 += align2*align2;
            } else {
                // Fragment only needs FP32 precision
                frag_mask_fp32[frag_word] |= frag_bitmask;
                count_fp32 += align2*align2;
            }
        }}}}
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
        for (int frag_i = 0; frag_i < frag_size; frag_i++){
        for (int frag_j = 0; frag_j < frag_size; frag_j++){
        for (int frag_k = 0; frag_k < frag_size; frag_k++){
        for (int frag_l = 0; frag_l < frag_size; frag_l++){
            const uint64_t frag_idx = ((frag_i * frag_size + frag_j) * frag_size + frag_k) * frag_size + frag_l;
            const uint64_t frag_word = frag_idx >> 6;
            const uint64_t frag_bit = frag_idx & 63;

            const bool sel_fp32 = (frag_mask_fp32[frag_word] >> frag_bit) & 1ull;
            const bool sel_fp64 = (frag_mask_fp64[frag_word] >> frag_bit) & 1ull;

            if (sel_fp32 || sel_fp64) {
                // Output all items in this fragment
                for (int ii = 0; ii < align; ii++){
                for (int jj = 0; jj < align; jj++){
                for (int kk = 0; kk < align; kk++){
                for (int ll = 0; ll < align; ll++){
                    const int i = frag_i * align + ii;
                    const int j = frag_j * align + jj;
                    const int k = frag_k * align + kk;
                    const int l = frag_l * align + ll;

                    ushort4 sq;
                    sq.x = ish0 + i;
                    sq.y = jsh0 + j;
                    sq.z = ksh0 + k;
                    sq.w = lsh0 + l;

                    if (sel_fp64) {
                        shl_quartet_idx[offset_fp64] = sq;
                        --offset_fp64;
                    } else if (sel_fp32) {
                        shl_quartet_idx[offset_fp32] = sq;
                        ++offset_fp32;
                    }
                }}}}
            }
        }}}}
    }
}
