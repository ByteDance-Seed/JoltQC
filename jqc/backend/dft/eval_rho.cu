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
constexpr DataType exp_cutoff = 36.8; // exp(-36.8) ~ 1e-16
constexpr DataType rho_cutoff = 1e-16;
constexpr DataType rho_cutoff2 = rho_cutoff * rho_cutoff;
constexpr DataType zero = 0.0;
constexpr DataType one = 1.0;
constexpr DataType two = 2.0;
constexpr DataType half = 0.5;

// BASIS_STRIDE is the total stride: [coords (4), ce (BASIS_STRIDE-4)]
// prim_stride is for ce pairs: (BASIS_STRIDE-4)/2
constexpr int prim_stride = (BASIS_STRIDE - 4) / 2;
constexpr int basis_stride = BASIS_STRIDE;

// Coords are always 4: [x, y, z, ao_loc]
struct __align__(4*sizeof(DataType)) DataType4 {
    DataType x, y, z, w;  // w stores ao_loc
};

struct __align__(2*sizeof(DataType)) DataType2 {
    DataType c, e;
};

// Compact pair storing (log_maxval, shell_index)
struct __align__(8) LogIdx {
    float log;
    int   idx;
};

// Helper to load coords from packed basis_data
__device__ __forceinline__ DataType4 load_coords(const DataType* __restrict__ basis_data, int ish) {
    const DataType* base = basis_data + ish * basis_stride;
    return *reinterpret_cast<const DataType4*>(base);
}

// Helper to get pointer to ce data for a basis
__device__ __forceinline__ const DataType2* load_ce_ptr(const DataType* __restrict__ basis_data, int ish) {
    return reinterpret_cast<const DataType2*>(basis_data + ish * basis_stride + 4);
}

extern "C" __global__
void eval_rho(
    const double* __restrict__ grid_coords,
    const DataType* __restrict__ basis_data,
    const int nbas,
    DataType* __restrict__ dm,
    float* __restrict__ log_dm_shell,
    const int nao,
    double* __restrict__ rho,
    const LogIdx* __restrict__ nz_i,
    const int* __restrict__ nnz_i,
    const int nbas_i,
    const LogIdx* __restrict__ nz_j,
    const int* __restrict__ nnz_j,
    const int nbas_j,
    const float log_cutoff_a, const float log_cutoff_b,
    const int ngrids){
    
    constexpr int nfi = (li+1)*(li+2)/2;
    constexpr int nfj = (lj+1)*(lj+2)/2;
    const int grid_id = blockIdx.x * nthreads + threadIdx.x;
    const int block_id = blockIdx.x;
    const int grid_blocks = ngrids / nthreads;

    const int nnzi = nnz_i[block_id];
    const int nnzj = nnz_j[block_id];
    if (nnzi == 0 || nnzj == 0) return;

    DataType gx[3];
    gx[0] = (DataType)grid_coords[grid_id           ];
    gx[1] = (DataType)grid_coords[grid_id +   ngrids];
    gx[2] = (DataType)grid_coords[grid_id + 2*ngrids];

    // ndim = 1 for LDA, 4 for GGA, 5 for mGGA
    DataType rho_reg[ndim] = {zero};
    for (int jsh_nz = 0; jsh_nz < nnzj; jsh_nz++){
        const int offset = jsh_nz + block_id * nbas_j;
        const int jsh = nz_j[offset].idx;
        const float log_aoj = nz_j[offset].log;
        const DataType4 xj = load_coords(basis_data, jsh);
        const int j0 = (int)xj.w;  // ao_loc stored in w field


        const DataType gjx = gx[0] - xj.x;
        const DataType gjy = gx[1] - xj.y;
        const DataType gjz = gx[2] - xj.z;
        const DataType rr_gj = gjx*gjx + gjy*gjy + gjz*gjz;

        DataType cej = zero;
        DataType cej_2e = zero;
        // Original code:
        // for (int jp = 0; jp < npj; jp++){
        //     const int jp_offset = jp + jsh*prim_stride;
        //     const DataType2 coeff_expj = coeff_exp[jp_offset];
        //     const DataType e = coeff_expj.e;
        //     const DataType e_rr = e * rr_gj;
        //     const DataType c = coeff_expj.c;
        //     const DataType ce = e_rr < exp_cutoff ? c * exp(-e_rr) : zero;
        //     cej += ce;
        //     cej_2e += ce * e;
        // }
        // Optimized version - early continue to avoid exp() calls:
        const DataType2* cej_ptr = load_ce_ptr(basis_data, jsh);
        #pragma unroll
        for (int jp = 0; jp < npj; jp++){
            const DataType2 coeff_expj = cej_ptr[jp];
            const DataType e = coeff_expj.e;
            const DataType e_rr = e * rr_gj;
            //if (e_rr >= exp_cutoff) continue;
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
#pragma unroll
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
#pragma unroll
            for (int j = 1; j < lj+1; j++){
                const DataType fac = cej * j;
                dxj[j] = cej_2e * xj_pows[j+1] + fac * xj_pows[j-1];
                dyj[j] = cej_2e * yj_pows[j+1] + fac * yj_pows[j-1];
                dzj[j] = cej_2e * zj_pows[j+1] + fac * zj_pows[j-1];
            }
        }
        
        DataType ao_j[nfj], ao_jx[nfj], ao_jy[nfj], ao_jz[nfj];
        // Original code:
        // for (int i = 0, lx = lj; lx >= 0; lx--){
        //     for (int ly = lj - lx; ly >= 0; ly--, i++){
        //         const int lz = lj - lx - ly;
        //         ao_j[i] = cej * xj_pows[lx] * yj_pows[ly] * zj_pows[lz];
        //         if constexpr(deriv > 0){
        //             ao_jx[i] = dxj[lx] * yj_pows[ly] * zj_pows[lz];
        //             ao_jy[i] = xj_pows[lx] * dyj[ly] * zj_pows[lz];
        //             ao_jz[i] = xj_pows[lx] * yj_pows[ly] * dzj[lz];
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
                }
            }
        }

        for (int ish_nz = 0; ish_nz < nnzi; ish_nz++){
            const int offset = ish_nz + block_id * nbas_i;
            const int ish = nz_i[offset].idx;
            const float log_aoi = nz_i[offset].log;
            const float log_rho_est = log_aoi + log_aoj + log_dm_shell[ish+jsh*nbas];
            if (ish > jsh || log_rho_est < log_cutoff_a || log_rho_est >= log_cutoff_b) continue;

            const DataType4 xi = load_coords(basis_data, ish);
            const DataType gix = gx[0] - xi.x;
            const DataType giy = gx[1] - xi.y;
            const DataType giz = gx[2] - xi.z;
            const DataType rr_gi = gix*gix + giy*giy + giz*giz;

            DataType cei = zero;
            DataType cei_2e = zero;

            // Original code:
            // for (int ip = 0; ip < npi; ip++){
            //     const int offset = ip + ish*prim_stride;
            //     const DataType2 coeff_expi = coeff_exp[offset];
            //     const DataType e = coeff_expi.e;
            //     const DataType e_rr = e * rr_gi;
            //     const DataType c = coeff_expi.c;
            //     const DataType ce = e_rr < exp_cutoff ? c * exp(-e_rr) : zero;
            //     cei += ce;
            //     cei_2e += ce * e;
            // }
            // Optimized version - early continue to avoid exp() calls:
            const DataType2* cei_ptr = load_ce_ptr(basis_data, ish);
            //const DataType e_rr = cei_ptr[npi - 1].e * rr_gi;
            //if (e_rr >= exp_cutoff) continue; // skip entire shell if last primitive is negligible

            #pragma unroll
            for (int ip = 0; ip < npi; ip++){
                const DataType2 coeff_expi = cei_ptr[ip];
                const DataType e = coeff_expi.e;
                const DataType e_rr = e * rr_gi;
                //if (e_rr >= exp_cutoff) continue;
                const DataType c = coeff_expi.c;
                const DataType ce = c * exp(-e_rr);
                cei += ce;
                cei_2e += ce * e;
            }
            cei_2e *= -two;

            const int i0 = (int)xi.w;  // ao_loc stored in w field

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
#pragma unroll
            for (int i = 0; i < i_pow; i++){
                xi_pows[i+1] = gix_curr;
                yi_pows[i+1] = giy_curr;
                zi_pows[i+1] = giz_curr;
                gix_curr *= gix;
                giy_curr *= giy;
                giz_curr *= giz;
            }

            DataType dxi[li+1], dyi[li+1], dzi[li+1];
            if constexpr (deriv > 0){
                // Original code:
                // for (int i = 0; i < li+1; i++){
                //     dxi[i] = cei_2e * xi_pows[i+1];
                //     dyi[i] = cei_2e * yi_pows[i+1];
                //     dzi[i] = cei_2e * zi_pows[i+1];
                // }
                // for (int i = 1; i < li+1; i++){
                //     DataType fac = cei * i;
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

            DataType rho0 = zero;
            DataType rhox = zero;
            DataType rhoy = zero;
            DataType rhoz = zero;
            DataType tau = zero;

            DataType* dm_ptr = dm + i0 * nao + j0;
            // Original code:
            // for (int i = 0, lx = li; lx >= 0; lx--){
            //     DataType cxi = cei * xi_pows[lx];
            //     for (int ly = li - lx; ly >= 0; ly--, i++){
            //         const int lz = li - lx - ly;
            //         DataType ao_i = cxi * yi_pows[ly] * zi_pows[lz];
            //         if constexpr(ndim > 1){
            //             ao_ix = dxi[lx] * yi_pows[ly] * zi_pows[lz];
            //             ao_iy = xi_pows[lx] * dyi[ly] * zi_pows[lz];
            //             ao_iz = xi_pows[lx] * yi_pows[ly] * dzi[lz];
            //         }
            //     }
            // }
            // Optimized version - cached common subexpressions:
#pragma unroll
            for (int i = 0, lx = li; lx >= 0; lx--){
                const DataType cxi = cei * xi_pows[lx];
                for (int ly = li - lx; ly >= 0; ly--, i++){
                    const int lz = li - lx - ly;
                    const DataType yi_ly_zi_lz = yi_pows[ly] * zi_pows[lz];
                    DataType ao_i = cxi * yi_ly_zi_lz;
                    DataType ao_ix, ao_iy, ao_iz;

                    if constexpr(ndim > 1){
                        ao_ix = dxi[lx] * yi_ly_zi_lz;
                        ao_iy = xi_pows[lx] * dyi[ly] * zi_pows[lz];
                        ao_iz = xi_pows[lx] * yi_pows[ly] * dzi[lz];
                    }

                    DataType s0 = zero, sx = zero, sy = zero, sz = zero;

#pragma unroll
                    for (int j = 0; j < nfj; j++){
                        DataType dm_ij = __ldg(dm_ptr + i*nao + j);
                        s0 += ao_j[j] * dm_ij;
                        if constexpr(ndim > 1){
                            sx += ao_jx[j] * dm_ij;
                            sy += ao_jy[j] * dm_ij;
                            sz += ao_jz[j] * dm_ij;
                        }
                    }

                    rho0 += ao_i * s0;
                    if constexpr(ndim > 1){
                        rhox += ao_ix * s0 + ao_i * sx;
                        rhoy += ao_iy * s0 + ao_i * sy;
                        rhoz += ao_iz * s0 + ao_i * sz;
                        if constexpr(ndim > 4){
                            tau += ao_ix * sx + ao_iy * sy + ao_iz * sz;
                        }
                    }
                }
            }

            const DataType fac = (ish == jsh) ? one : two;
            rho_reg[0] += fac * rho0;
            if constexpr(ndim > 1){
                rho_reg[1] += fac * rhox;
                rho_reg[2] += fac * rhoy;
                rho_reg[3] += fac * rhoz;
                if constexpr(ndim > 4){
                    rho_reg[4] += half * fac * tau;
                }
            }
        }
    }

    for (int i = 0; i < ndim; i++){
        rho[grid_id + ngrids * i] += (double)rho_reg[i];
    }
}
