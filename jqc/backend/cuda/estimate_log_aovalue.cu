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
constexpr int prim_stride = PRIM_STRIDE / 2;

// Coordinate stride in floats
static_assert(COORD_STRIDE >= 3, "COORD_STRIDE must be >= 3");
struct __align__(COORD_STRIDE*sizeof(float)) DataType4 {
    float x, y, z;
#if COORD_STRIDE >= 4
    float w;
#endif
#if COORD_STRIDE > 4
    float pad[COORD_STRIDE - 4];
#endif
};
constexpr float exp_cutoff = 36.8;

extern "C" __global__
void estimate_log_aovalue(
    const double* __restrict__ grid_coords,
    const int ngrids,
    const DataType4* __restrict__ shell_coords,
    const float2* __restrict__ coeff_exp,
    const int nbas,
    float* __restrict__ log_maxval,
    int * __restrict__ nnz_idx,
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
            log_maxval[block_id * nbas + loc] = log_gto_maxval;
            nnz_idx[block_id * nbas + loc] = ish;
        }
    }
    __syncthreads();
    if (thread_id == 0){
        nnz_per_block[block_id] = nnz;
    }
}
