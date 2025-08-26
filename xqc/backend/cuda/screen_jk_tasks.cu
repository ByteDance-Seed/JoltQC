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

typedef unsigned long uint64_t;

__forceinline__ __device__
int global_offset(int* batch_head, int val){
    // Calculate the cumulative sum of the count array
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int lane  = tid & 31;    
    const int warp  = tid >> 5;
    for (int ofs = 1; ofs < 32; ofs <<= 1) {
        int n = __shfl_up_sync(0xffffffff, val, ofs);
        if (lane >= ofs) val += n;                       
    }
    __syncthreads();
    __shared__ int cum_count[threads];
    
    cum_count[tid] = val;

    __shared__ int warp_tot[threads / 32];  
    if (lane == 31) warp_tot[warp] = val;  
    __syncthreads(); 

    if (warp == 0) {
        int warp_val = warp_tot[lane];
#pragma unroll
        for (int ofs = 1; ofs < 32; ofs <<= 1) {
            int n = __shfl_up_sync(0xffffffff, warp_val, ofs);
            if (lane >= ofs) warp_val += n;
        }
        warp_tot[lane] = warp_val;
    }
    __syncthreads();  
    int warp_offset = (warp == 0) ? 0 : warp_tot[warp-1];
    cum_count[tid] = val + warp_offset;
    __syncthreads();

    const int ntasks = cum_count[threads-1];
    if (ntasks == 0) return; 
    
    // Calculate the global offset
    int offset = 0;
    if (tid == 0){
        offset = atomicAdd(batch_head, ntasks);
        for (int i = 0; i < threads-1; i++){
            cum_count[i] += offset;
        }
    }
    __syncthreads();
    if (tid > 0) {
        offset = cum_count[tid-1];
    }
    return offset;
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

    const int tile_ij = tile_ij_mapping[ij];
    const int tile_kl = tile_kl_mapping[kl];
    
    const int nbas_tiles = nbas / TILE;
    const int tile_i = tile_ij / nbas_tiles;
    const int tile_j = tile_ij % nbas_tiles;
    const int tile_k = tile_kl / nbas_tiles;
    const int tile_l = tile_kl % nbas_tiles;
    
    const int ish0 = tile_i * TILE;
    const int jsh0 = tile_j * TILE;
    const int ksh0 = tile_k * TILE;
    const int lsh0 = tile_l * TILE;
    const int ish1 = ish0 + TILE;
    const int jsh1 = jsh0 + TILE;
    const int ksh1 = ksh0 + TILE;
    const int lsh1 = lsh0 + TILE;

    constexpr int mask_size = (TILE*TILE*TILE*TILE) / 64;
    uint64_t mask_bits_fp32[mask_size] = {0};
    uint64_t mask_bits_fp64[mask_size] = {0};

    int count_fp32 = 0;
    int count_fp64 = 0;
    if (active){
        for (int i = 0; i < TILE; ++i){
            const int ish = ish0 + i;
            for (int j = 0; j < TILE; ++j){
                const int jsh = jsh0 + j;
                if (jsh >= ish+1 || jsh >= jsh1) continue;
                const int bas_ij = ish * nbas + jsh;
                const float q_ij = q_cond [bas_ij];
                const float d_ij = dm_cond[bas_ij];
                for (int k = 0; k < TILE; ++k){
                    const int ksh = ksh0 + k;
                    if (ksh >= ish+1 || ksh >= ksh1) continue;
                    const float d_ik = dm_cond[ish*nbas+ksh];
                    const float d_jk = dm_cond[jsh*nbas+ksh];
                    for (int l = 0; l < TILE; ++l){
                        const int lsh = lsh0 + l;
                        if (lsh >= ksh+1 || lsh >= lsh1) continue;
                        const int bas_kl = ksh * nbas + lsh;
                        if (bas_ij < bas_kl) continue;
                        const float q_ijkl = q_ij + q_cond[bas_kl];
                        float d_large = -36.8f;
                        if constexpr(do_k){
                            const float d_il = dm_cond[ish*nbas+lsh];
                            const float d_jl = dm_cond[jsh*nbas+lsh];
                            d_large = max(d_large, d_ik);
                            d_large = max(d_large, d_jk);
                            d_large = max(d_large, d_il);
                            d_large = max(d_large, d_jl);
                        }
                        if constexpr(do_j){
                            const float d_kl = dm_cond[bas_kl];
                            d_large = max(d_large, d_ij);
                            d_large = max(d_large, d_kl);
                        }
                        float dq = q_ijkl + d_large;
                        bool selected = (dq > cutoff) && (dq <= cutoff_fp64);
                        if (selected){
                            uint64_t idx = i*TILE*TILE*TILE + j*TILE*TILE + k*TILE + l;
                            uint64_t word = idx >> 6; // divide 64
                            uint64_t bit = idx & 63;
                            uint64_t bitmask = 1ull << bit;
                            mask_bits_fp32[word] |= bitmask;
                        }
                        count_fp32 += selected;
                        
                        selected = (dq > cutoff_fp64);
                        if (selected){
                            uint64_t idx = i*TILE*TILE*TILE + j*TILE*TILE + k*TILE + l;
                            uint64_t word = idx >> 6; // divide 64
                            uint64_t bit = idx & 63;
                            uint64_t bitmask = 1ull << bit;
                            mask_bits_fp64[word] |= bitmask;
                        }
                        count_fp64 += selected;
                    }
                }
            }
        }
    }
    int offset_fp32 = global_offset(batch_head+1, count_fp32);
    int offset_fp64 = global_offset(batch_head+2, -count_fp64) - 1;

    if (active){
#pragma unroll
        for (int i = 0; i < TILE; i++){
            for (int j = 0; j < TILE; j++){
                for (int k = 0; k < TILE; k++){
                    for (int l = 0; l < TILE; l++){
                        uint64_t idx = i*TILE*TILE*TILE + j*TILE*TILE + k*TILE + l;
                        uint64_t word = idx >> 6; // divide 64
                        uint64_t bit = idx & 63;
                        bool selected = (mask_bits_fp32[word] >> bit) & 1ull;
                        if (selected){
                            ushort4 sq;
                            sq.x = ish0 + i; 
                            sq.y = jsh0 + j; 
                            sq.z = ksh0 + k; 
                            sq.w = lsh0 + l;
                            shl_quartet_idx[offset_fp32] = sq;
                            ++offset_fp32;
                        }
                        selected = (mask_bits_fp64[word] >> bit) & 1ull;
                        if (selected){
                            ushort4 sq;
                            sq.x = ish0 + i; 
                            sq.y = jsh0 + j; 
                            sq.z = ksh0 + k; 
                            sq.w = lsh0 + l;
                            shl_quartet_idx[offset_fp64] = sq;
                            --offset_fp64;
                        }
                    }
                }
            }
        }
    }
}
