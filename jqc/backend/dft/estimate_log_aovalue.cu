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

constexpr int ng_per_thread = 256;

// BASIS_STRIDE is the total stride: [coords (4), ce (BASIS_STRIDE-4)]
// prim_stride is for ce pairs: (BASIS_STRIDE-4)/2
constexpr int prim_stride = (BASIS_STRIDE - 4) / 2;

// Coords are always 4: [x, y, z, ao_loc/w]
struct __align__(4*sizeof(float)) DataType4 {
    float x, y, z, w;
};
constexpr float exp_cutoff = 36.8;

// Pair of (log_maxval, shell_index) for compact storage
struct __align__(8) LogIdx {
    float log;
    int   idx;
};

extern "C" __global__
void estimate_log_aovalue(
    const double* __restrict__ grid_coords,
    const int ngrids,
    const DataType4* __restrict__ shell_coords,
    const float2* __restrict__ coeff_exp,
    const int nbas,
    const int shell_base,
    LogIdx* __restrict__ logidx,
    int * __restrict__ nnz_per_block,
    float log_cutoff)
{
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;

    __shared__ float gridx_shared[ng_per_thread];
    __shared__ float gridy_shared[ng_per_thread];
    __shared__ float gridz_shared[ng_per_thread];

    const int grid_id = block_id * ng_per_thread + thread_id;
    
    gridx_shared[thread_id] = (float)grid_coords[         grid_id];
    gridy_shared[thread_id] = (float)grid_coords[ngrids  +grid_id];
    gridz_shared[thread_id] = (float)grid_coords[ngrids*2+grid_id];
    
    __shared__ int nnz;
    if (thread_id == 0) nnz = 0;
    __syncthreads();
    
    // Process each shell assigned to this thread
    for (int ish = thread_id; ish < nbas; ish += ng_per_thread){
        DataType4 xi = shell_coords[ish];
        float bas_x = xi.x;
        float bas_y = xi.y;
        float bas_z = xi.z;
        float coeffs_reg[nprim], exps_reg[nprim];
        for (int ip = 0; ip < nprim; ip++){
            const int offset = prim_stride * ish + ip;
            const float2 ce = coeff_exp[offset];
            coeffs_reg[ip] = ce.x;
            exps_reg[ip] = ce.y;
        }
        
        float log_gto_maxval = -1e38f;
        for (int grid_id = 0; grid_id < ng_per_thread; ++grid_id) {
            const float rx = gridx_shared[grid_id] - bas_x;
            const float ry = gridy_shared[grid_id] - bas_y;
            const float rz = gridz_shared[grid_id] - bas_z;
            const float rr = rx * rx + ry * ry + rz * rz + 1e-38f;
            float gto_sup = 0.0f;
#pragma unroll
            for (int ip = 0; ip < nprim; ++ip) {
                const float e = exps_reg[ip] * rr;
                if (e < exp_cutoff){
                    gto_sup += coeffs_reg[ip] * __expf(-e);
                }
            }
            gto_sup = fabs(gto_sup) + 1e-38f; // avoid log(0)
            float log_gto = 0.0f;
            log_gto += __logf(gto_sup);
            log_gto += ang * __logf(rr)/2.0f;
            log_gto_maxval = max(log_gto_maxval, log_gto);
        }
        
        //if (ish < nbas){
        //    log_maxval[block_id * nbas + ish] = log_gto_maxval;
        //}
        if (log_gto_maxval > log_cutoff){
            const int loc = atomicAdd(&nnz, 1);
            const int offset = block_id * nbas + loc;
            logidx[offset].log = log_gto_maxval;
            logidx[offset].idx = shell_base + ish;
        }
    }
    __syncthreads();
    if (thread_id == 0){
        nnz_per_block[block_id] = nnz;
    }
}
