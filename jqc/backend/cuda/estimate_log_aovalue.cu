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
constexpr int nprim_max = 16;
constexpr float exp_cutoff = 36.8;

extern "C" __global__
void estimate_log_aovalue(
    const double* __restrict__ grid_coords,
    const int ngrids,
    const float4* __restrict__ shell_coords,
    const float* __restrict__ coeffs,
    const float* __restrict__ exps,
    const int nbas,
    float* __restrict__ log_maxval,
    int * __restrict__ nnz_idx,
    int * __restrict__ nnz_per_block,
    float log_cutoff)
{
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int grid_blocks = ngrids / ng_per_thread;

    __shared__ float gridx_shared[ng_per_thread];
    __shared__ float gridy_shared[ng_per_thread];
    __shared__ float gridz_shared[ng_per_thread];

    const int grid_id = block_id * ng_per_thread + thread_id;
    
    gridx_shared[thread_id] = (float)grid_coords[         grid_id];
    gridy_shared[thread_id] = (float)grid_coords[ngrids  +grid_id];
    gridz_shared[thread_id] = (float)grid_coords[ngrids*2+grid_id];
    
    __shared__ float maxval_smem[ng_per_thread];
    __shared__ int nnz;
    nnz = 0;
    __syncthreads();

    for (int ish = thread_id; ish < nbas; ish += ng_per_thread){
        float4 xi = shell_coords[ish];
        float bas_x = xi.x;// //shell_coords[4*ish];
        float bas_y = xi.y;// //shell_coords[4*ish + 1];
        float bas_z = xi.z;// //shell_coords[4*ish + 2];
        float coeffs_reg[nprim_max], exps_reg[nprim_max];
        for (int ip = 0; ip < nprim; ip++){
            const int offset = nprim_max * ish + ip;
            coeffs_reg[ip] = __ldg(coeffs + offset);
            exps_reg[ip] = __ldg(exps + offset);
        }
        
        float log_gto_maxval = -1e38f;
        for (int grid_id = 0; grid_id < ng_per_thread; ++grid_id) {
            const float rx = gridx_shared[grid_id] - bas_x;
            const float ry = gridy_shared[grid_id] - bas_y;
            const float rz = gridz_shared[grid_id] - bas_z;
            const float rr = rx * rx + ry * ry + rz * rz + 1e-38f;
            float gto_sup = 0.0;
            
            for (int ip = 0; ip < nprim; ++ip) {
                const float e = exps_reg[ip] * rr;
                if (e < exp_cutoff){
                    gto_sup += coeffs_reg[ip] * __expf(-e);
                }
            }
            gto_sup = fabs(gto_sup);
            float log_gto = 0.0f;
            log_gto += (gto_sup > 1e-16f) ? __logf(gto_sup) : -100.0f;
            log_gto += (rr > 1e-32f) ? ang * __logf(rr)/2.0f : -100.0f;
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
