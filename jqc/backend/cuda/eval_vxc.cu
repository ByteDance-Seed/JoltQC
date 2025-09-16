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

constexpr DataType max_val = 1e16;
constexpr int nprim_max = NPRIM_MAX;
constexpr int warpsize = 32;
constexpr DataType exp_cutoff = 36.8; // exp(-36.8) ~ 1e-16
constexpr DataType vxc_cutoff = 1e-16;
constexpr DataType vxc_cutoff2 = vxc_cutoff * vxc_cutoff;
constexpr DataType zero = 0.0;
constexpr DataType one = 1.0;
constexpr DataType two = 2.0;
constexpr DataType half = 0.5;

struct __align__(4*sizeof(DataType)) DataType4 {
    DataType x, y, z, w;
};

struct __align__(2*sizeof(DataType)) DataType2 {
    DataType c, e;
};
 
__forceinline__ __device__
DataType warp_reduce(DataType val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__forceinline__ __device__ 
float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__forceinline__ __device__ 
void block_reduce_max(float val, float* maxval) {
    static __shared__ float shared[32]; // One per warp
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warp_reduce_max(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads(); // Wait for all warps
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -1e38;
    if (wid == 0) maxval[0] = warp_reduce_max(val);
    __syncthreads();
}

extern "C" __global__
void eval_vxc(
    const double* __restrict__ grid_coords,
    const DataType4* __restrict__ shell_coords,
    const DataType2* __restrict__ coeff_exp,
    const int nbas,
    double* __restrict__ vxc_mat,
    const int* __restrict__ ao_loc,
    const int nao,
    double* __restrict__ wv_grid,
    const float* log_maxval_i,
    const int* __restrict__ nnz_indices_i,
    const int* __restrict__ nnz_i,
    const int nbas_i,
    const float* log_maxval_j,
    const int* __restrict__ nnz_indices_j,
    const int* __restrict__ nnz_j,
    const int nbas_j,
    float log_cutoff_a, float log_cutoff_b,
    const int ngrids) {
    
    constexpr int nfi = (li+1)*(li+2)/2;
    constexpr int nfj = (lj+1)*(lj+2)/2;
    constexpr int nfij = nfi*nfj;
    const int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_id = blockIdx.x;
    const int grid_blocks = ngrids / nthreads;

    const int nnzi = nnz_i[block_id];
    const int nnzj = nnz_j[block_id];
    if (nnzi == 0 || nnzj == 0) return;

    const int warp_id = threadIdx.x / warpsize;
    const int tid = threadIdx.x;
    const int lane = threadIdx.x % warpsize;

    constexpr int num_warps = nthreads / warpsize;
    DataType gx[3];
    gx[0] = (DataType)grid_coords[grid_id           ];
    gx[1] = (DataType)grid_coords[grid_id +   ngrids];
    gx[2] = (DataType)grid_coords[grid_id + 2*ngrids];

    // ndim = 1 for LDA, 4 for GGA, 5 for mGGA
    DataType wv0, wv_x, wv_y, wv_z, wv_tau;
    wv0 = (DataType)wv_grid[grid_id];
    if constexpr (ndim > 1){
        wv_x = (DataType)wv_grid[grid_id + ngrids];
        wv_y = (DataType)wv_grid[grid_id + 2*ngrids];
        wv_z = (DataType)wv_grid[grid_id + 3*ngrids];
        if constexpr (ndim > 4){
            wv_tau = half * (DataType)wv_grid[grid_id + 4*ngrids];
        }
    }
    float log_wv0 = __logf(fabs(wv0) + 1e-16f);
    __shared__ float log_max_wv0_smem[1];
    block_reduce_max(log_wv0, log_max_wv0_smem);
    float log_max_wv0 = log_max_wv0_smem[0];
    log_cutoff_a -= log_max_wv0;
    log_cutoff_b -= log_max_wv0;
    
    constexpr int smem_size = num_warps * nfij;
    __shared__ DataType vxc_smem[smem_size];

    for (int jsh_nz = 0; jsh_nz < nnzj; jsh_nz++){
        const int offset = jsh_nz + block_id * nbas_j;
        const float log_aoj = log_maxval_j[offset];
        const int jsh = nnz_indices_j[offset];
        const int j0 = ao_loc[jsh];
        
        for (int ish_nz = 0; ish_nz < nnzi; ish_nz++){
            const int offset = ish_nz + block_id * nbas_i;
            const float log_aoi = log_maxval_i[offset];
            const int ish = nnz_indices_i[offset];
            if (ish > jsh) continue;
            if (log_aoi + log_aoj < log_cutoff_a || log_aoi + log_aoj >= log_cutoff_b) continue;

        const DataType4 xj = shell_coords[jsh];
        const DataType gjx = gx[0] - xj.x;
        const DataType gjy = gx[1] - xj.y;
        const DataType gjz = gx[2] - xj.z;
        const DataType rr_gj = gjx*gjx + gjy*gjy + gjz*gjz;
        
        DataType cej = zero;
        DataType cej_2e = zero;
        // Original code:
        // for (int jp = 0; jp < npj; jp++){
        //     const int offset = nprim_max * jsh + jp;
        //     const DataType2 coeff_expj = coeff_exp[offset];
        //     const DataType e = coeff_expj.e;
        //     const DataType e_rr = e * rr_gj;
        //     const DataType c = coeff_expj.c;
        //     const DataType ce = e_rr < exp_cutoff ? c * exp(-e_rr) : zero;
        //     cej += ce;
        //     cej_2e += ce * e;
        // }
        // Optimized version - early continue to avoid exp() calls:
        const int jprim_base = nprim_max * jsh;
        #pragma unroll
        for (int jp = 0; jp < npj; jp++){
            const int jp_off = jprim_base + jp;
            const DataType2 coeff_expj = coeff_exp[jp_off];
            const DataType e = coeff_expj.e;
            const DataType e_rr = e * rr_gj;
            if (e_rr >= exp_cutoff) continue;
            const DataType c = coeff_expj.c;
            const DataType ce = c * exp(-e_rr);
            cej += ce;
            cej_2e += ce * e;
        }
        cej_2e *= -two;

        constexpr int j_pow = lj + deriv;
        DataType xj_pows[j_pow+1], yj_pows[j_pow+1], zj_pows[j_pow+1];
        xj_pows[0] = one; yj_pows[0] = one; zj_pows[0] = one;
        // Original code:
        // for (int i = 0; i < j_pow; i++){
        //     xj_pows[i+1] = xj_pows[i] * gjx;
        //     yj_pows[i+1] = yj_pows[i] * gjy;
        //     zj_pows[i+1] = zj_pows[i] * gjz;
        // }
        // Optimized version - reduces array access overhead:
        DataType gjx_curr = gjx, gjy_curr = gjy, gjz_curr = gjz;
        for (int i = 0; i < j_pow; i++){
            xj_pows[i+1] = gjx_curr;
            yj_pows[i+1] = gjy_curr;
            zj_pows[i+1] = gjz_curr;
            gjx_curr *= gjx;
            gjy_curr *= gjy;
            gjz_curr *= gjz;
        }

        DataType dxj[lj+1], dyj[lj+1], dzj[lj+1];
        if constexpr(deriv > 0){
            // Original code:
            // for (int j = 0; j < lj+1; j++){
            //     dxj[j] = cej_2e * xj_pows[j+1];
            //     dyj[j] = cej_2e * yj_pows[j+1];
            //     dzj[j] = cej_2e * zj_pows[j+1];
            // }
            // for (int j = 1; j < lj+1; j++){
            //     const DataType fac = cej * j;
            //     dxj[j] += fac * xj_pows[j-1];
            //     dyj[j] += fac * yj_pows[j-1];
            //     dzj[j] += fac * zj_pows[j-1];
            // }
            // Optimized version - fused loops to reduce operations:
            dxj[0] = cej_2e * xj_pows[1];
            dyj[0] = cej_2e * yj_pows[1];
            dzj[0] = cej_2e * zj_pows[1];
            for (int j = 1; j < lj+1; j++){
                const DataType fac = cej * j;
                dxj[j] = cej_2e * xj_pows[j+1] + fac * xj_pows[j-1];
                dyj[j] = cej_2e * yj_pows[j+1] + fac * yj_pows[j-1];
                dzj[j] = cej_2e * zj_pows[j+1] + fac * zj_pows[j-1];
            }
        }

        DataType ao_j[nfj], ao_jx[nfj], ao_jy[nfj], ao_jz[nfj], aow_j[nfj];
        // Original code:
        // for (int i = 0, lx = lj; lx >= 0; lx--){
        //     const DataType cx = cej * xj_pows[lx];
        //     for (int ly = lj - lx; ly >= 0; ly--, i++){
        //         const int lz = lj - lx - ly;
        //         ao_j[i] = cx * yj_pows[ly] * zj_pows[lz];
        //         if constexpr(deriv > 0){
        //             ao_jx[i] = dxj[lx] * yj_pows[ly] * zj_pows[lz];
        //             ao_jy[i] = xj_pows[lx] * dyj[ly] * zj_pows[lz];
        //             ao_jz[i] = xj_pows[lx] * yj_pows[ly] * dzj[lz];
        //             aow_j[i] = ao_jx[i] * wv_x + ao_jy[i] * wv_y + ao_jz[i] * wv_z;
        //             if constexpr (ndim > 4){
        //                 ao_jx[i] *= wv_tau;
        //                 ao_jy[i] *= wv_tau;
        //                 ao_jz[i] *= wv_tau;
        //             }
        //         }
        //     }
        // }
        // Optimized version - cached common subexpressions:
#pragma unroll
        for (int i = 0, lx = lj; lx >= 0; lx--){
            const DataType cej_xj_lx = cej * xj_pows[lx];
            const DataType dxj_lx = (deriv > 0) ? dxj[lx] : zero;
            for (int ly = lj - lx; ly >= 0; ly--, i++){
                const int lz = lj - lx - ly;
                const DataType yj_ly_zj_lz = yj_pows[ly] * zj_pows[lz];
                ao_j[i] = cej_xj_lx * yj_ly_zj_lz;
                if constexpr(deriv > 0){                    
                    ao_jx[i] = dxj_lx * yj_ly_zj_lz;
                    ao_jy[i] = xj_pows[lx] * dyj[ly] * zj_pows[lz];
                    ao_jz[i] = xj_pows[lx] * yj_pows[ly] * dzj[lz];
                    aow_j[i] = ao_jx[i] * wv_x + ao_jy[i] * wv_y + ao_jz[i] * wv_z;
                    if constexpr (ndim > 4){
                        ao_jx[i] *= wv_tau;
                        ao_jy[i] *= wv_tau;
                        ao_jz[i] *= wv_tau;
                    }
                }
            }
        }
            const DataType4 xi = shell_coords[ish];
            const DataType gix = gx[0] - xi.x;
            const DataType giy = gx[1] - xi.y;
            const DataType giz = gx[2] - xi.z;
            const DataType rr_gi = gix*gix + giy*giy + giz*giz;

            DataType cei = zero;
            DataType cei_2e = zero;
            // Original code:
            // for (int ip = 0; ip < npi; ip++){
            //     const int ip_offset = ip + ish*nprim_max;
            //     const DataType2 coeff_expi = coeff_exp[ip_offset];
            //     const DataType e = coeff_expi.e;
            //     const DataType e_rr = e * rr_gi;
            //     const DataType c = coeff_expi.c;
            //     const DataType ce = e_rr < exp_cutoff ? c * exp(-e_rr) : zero;
            //     cei += ce;
            //     cei_2e += ce * e;
            // }
            // Optimized version - early continue to avoid exp() calls:
            const int iprim_base = ish * nprim_max;
            #pragma unroll
            for (int ip = 0; ip < npi; ip++){
                const int ip_off = iprim_base + ip;
                const DataType2 coeff_expi = coeff_exp[ip_off];
                const DataType e = coeff_expi.e;
                const DataType e_rr = e * rr_gi;
                if (e_rr >= exp_cutoff) continue;
                const DataType c = coeff_expi.c;
                const DataType ce = c * exp(-e_rr);
                cei += ce;
                cei_2e += ce * e;
            }
            cei_2e *= -two;

            const int i0 = ao_loc[ish];

            constexpr int i_pow = li + deriv;
            DataType xi_pows[i_pow+1], yi_pows[i_pow+1], zi_pows[i_pow+1];
            xi_pows[0] = one; yi_pows[0] = one; zi_pows[0] = one;
            // Original code:
            // for (int i = 0; i < i_pow; i++){
            //     xi_pows[i+1] = xi_pows[i] * gix;
            //     yi_pows[i+1] = yi_pows[i] * giy;
            //     zi_pows[i+1] = zi_pows[i] * giz;
            // }
            // Optimized version - reduces array access overhead:
            DataType gix_curr = gix, giy_curr = giy, giz_curr = giz;
            for (int i = 0; i < i_pow; i++){
                xi_pows[i+1] = gix_curr;
                yi_pows[i+1] = giy_curr;
                zi_pows[i+1] = giz_curr;
                gix_curr *= gix;
                giy_curr *= giy;
                giz_curr *= giz;
            }

            DataType dxi[li+1], dyi[li+1], dzi[li+1];
            if constexpr(deriv > 0){
                // Original code:
                // for (int i = 0; i < li+1; i++){
                //     dxi[i] = cei_2e * xi_pows[i+1];
                //     dyi[i] = cei_2e * yi_pows[i+1];
                //     dzi[i] = cei_2e * zi_pows[i+1];
                // }
                // for (int i = 1; i < li+1; i++){
                //     const DataType fac = cei * i;
                //     dxi[i] += fac * xi_pows[i-1];
                //     dyi[i] += fac * yi_pows[i-1];
                //     dzi[i] += fac * zi_pows[i-1];
                // }
                // Optimized version - fused loops to reduce operations:
                dxi[0] = cei_2e * xi_pows[1];
                dyi[0] = cei_2e * yi_pows[1];
                dzi[0] = cei_2e * zi_pows[1];
                for (int i = 1; i < li+1; i++){
                    const DataType fac = cei * i;
                    dxi[i] = cei_2e * xi_pows[i+1] + fac * xi_pows[i-1];
                    dyi[i] = cei_2e * yi_pows[i+1] + fac * yi_pows[i-1];
                    dzi[i] = cei_2e * zi_pows[i+1] + fac * zi_pows[i-1];
                }
            }
            __syncthreads();
            // Original code:
            // for (int i = 0, lx = li; lx >= 0; lx--){
            //     const DataType cx = cei * xi_pows[lx];
            //     for (int ly = li - lx; ly >= 0; ly--, i++){
            //         const int lz = li - lx - ly;
            //         const DataType ao_i = cx * yi_pows[ly] * zi_pows[lz];
            //         DataType aow0_i, ao_ix, ao_iy, ao_iz, aow_i;
            //         aow0_i = ao_i * wv0;
            //         if constexpr(ndim > 1){
            //             ao_ix = dxi[lx] * yi_pows[ly] * zi_pows[lz];
            //             ao_iy = xi_pows[lx] * dyi[ly] * zi_pows[lz];
            //             ao_iz = xi_pows[lx] * yi_pows[ly] * dzi[lz];
            //             aow_i = ao_ix * wv_x + ao_iy * wv_y + ao_iz * wv_z;
            //         }
            //     }
            // }
            // Optimized version - cached common subexpressions:
#pragma unroll
            for (int i = 0, lx = li; lx >= 0; lx--){
                const DataType cei_xi_lx = cei * xi_pows[lx];
                for (int ly = li - lx; ly >= 0; ly--, i++){
                    const int lz = li - lx - ly;
                    const DataType yi_ly_zi_lz = yi_pows[ly] * zi_pows[lz];
                    const DataType ao_i = cei_xi_lx * yi_ly_zi_lz;
                    DataType aow0_i, ao_ix, ao_iy, ao_iz, aow_i;
                    aow0_i = ao_i * wv0;
                    if constexpr(ndim > 1){
                        ao_ix = dxi[lx] * yi_ly_zi_lz;
                        ao_iy = xi_pows[lx] * dyi[ly] * zi_pows[lz];
                        ao_iz = xi_pows[lx] * yi_pows[ly] * dzi[lz];
                        aow_i = ao_ix * wv_x + ao_iy * wv_y + ao_iz * wv_z;
                    }
#pragma unroll
                    for (int j = 0; j < nfj; j++){
                        DataType vxc_ij = aow0_i * ao_j[j];
                        if constexpr(ndim > 1){
                            vxc_ij += aow_i * ao_j[j];
                            vxc_ij += aow_j[j] * ao_i;
                            if constexpr(ndim > 4){
                                vxc_ij += ao_ix * ao_jx[j];
                                vxc_ij += ao_iy * ao_jy[j];
                                vxc_ij += ao_iz * ao_jz[j];
                            }
                        }
                        vxc_ij = warp_reduce(vxc_ij);
                        if (lane == 0){
                            vxc_smem[(i + j * nfi) * num_warps + warp_id] = vxc_ij;
                        }
                    }
                }
            }
            __syncthreads();

            const DataType fac = (ish == jsh) ? half : one;
            // Block reduction
            const int rank = threadIdx.x % num_warps;
            const int ntasks = (smem_size + 31) / 32 * 32;
            for (int idx = threadIdx.x; idx < ntasks; idx += nthreads){
                DataType vxc_ij = idx < smem_size ? vxc_smem[idx] : 0.0;
                // Warp reduction
                // Assume nthreads = 256
                vxc_ij += __shfl_down_sync(0xffffffff, vxc_ij, 4);
                vxc_ij += __shfl_down_sync(0xffffffff, vxc_ij, 2);
                vxc_ij += __shfl_down_sync(0xffffffff, vxc_ij, 1);

                if (rank == 0){
                    vxc_ij = fac * vxc_ij;
                    const int ij = idx / num_warps;
                    const int i = ij % nfi;
                    const int j = ij / nfi;
                    const int offset = (i0+i)*nao + j0 + j;
                    atomicAdd(vxc_mat + offset, (double)vxc_ij);
                }
            }
        }
    }
}
