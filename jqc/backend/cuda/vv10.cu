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

// # Portions of this file adapted from GPU4PySCF (https://github.com/pyscf/gpu4pyscf)
// # Copyright 2025 PySCF developer.
// # Licensed under the Apache License, Version 2.0.

// DataType will be defined by the calling code

// NG_PER_BLOCK will be defined by the calling code

extern "C" __global__
void vv10_kernel(double *Fvec, double *Uvec, double *Wvec,
    const double *vvcoords, const double *coords,
    const double *W0p, const double *W0, const double *K,
    const double *Kp, const double *RpW,
    const int vvngrids, const int ngrids)
{
    // grid id - assume 256-aligned grids (guaranteed by padding)
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load grid data (no bounds check needed due to 256-alignment guarantee)
    DataType xi = coords[grid_id];
    DataType yi = coords[ngrids + grid_id];
    DataType zi = coords[2*ngrids + grid_id];
    DataType W0i = W0[grid_id];
    DataType Ki = K[grid_id];

    double F = 0.0;
    double U = 0.0;
    double W = 0.0;

    const double *xj = vvcoords;
    const double *yj = vvcoords + vvngrids;
    const double *zj = vvcoords + 2*vvngrids;

    __shared__ DataType xj_smem[NG_PER_BLOCK];
    __shared__ DataType yj_smem[NG_PER_BLOCK];
    __shared__ DataType zj_smem[NG_PER_BLOCK];
    __shared__ DataType Kp_smem[NG_PER_BLOCK];
    __shared__ DataType W0p_smem[NG_PER_BLOCK];
    __shared__ DataType RpW_smem[NG_PER_BLOCK];

    const int tx = threadIdx.x;

    for (int j = 0; j < vvngrids; j+=blockDim.x) {
        int idx = j + tx;
        
        // Load data directly (no bounds check needed due to 256-alignment guarantee)
        xj_smem[tx] = xj[idx];
        yj_smem[tx] = yj[idx];
        zj_smem[tx] = zj[idx];
        Kp_smem[tx] = Kp[idx];
        W0p_smem[tx] = W0p[idx];
        RpW_smem[tx] = RpW[idx];
        
        __syncthreads();

        // Compute VV10 interaction
#pragma unroll 16
        for (int l = 0; l < NG_PER_BLOCK; ++l){
            DataType DX = xj_smem[l] - xi;
            DataType DY = yj_smem[l] - yi;
            DataType DZ = zj_smem[l] - zi;
            DataType R2 = DX*DX + DY*DY + DZ*DZ;

            DataType gp = R2 * W0p_smem[l] + Kp_smem[l];
            DataType g  = R2*W0i + Ki;
            DataType gt = g + gp;
            DataType ggt = g*gt;
            DataType g_gt = g + gt;
            
            // Add safety check for division by zero
            DataType denominator = gp*ggt*ggt;
            if (denominator > 1e-20) {
                DataType T = RpW_smem[l] / denominator;
                F += T * ggt;
                U += T * g_gt;
                W += T * R2 * g_gt;
            }
        }
        __syncthreads();
    }
    
    // Store results (no bounds check needed due to 256-alignment guarantee)
    Fvec[grid_id] = F * -1.5;
    Uvec[grid_id] = U;
    Wvec[grid_id] = W;

}