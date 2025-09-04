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
constexpr int nprim_max = 16;
constexpr DataType exp_cutoff = 36.8; // exp(-36.8) ~ 1e-16
constexpr DataType rho_cutoff = 1e-16;
constexpr DataType rho_cutoff2 = rho_cutoff * rho_cutoff;
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

extern "C" __global__
void eval_rho(
    const double* __restrict__ grid_coords,
    const DataType4* __restrict__ shell_coords,
    const DataType2* __restrict__ coeff_exp,
    const int nbas,
    DataType* __restrict__ dm,
    float* __restrict__ log_dm_shell,
    const int* __restrict__ ao_loc,
    const int nao,
    double* __restrict__ rho,
    const float* log_maxval_i,
    const int* __restrict__ nnz_indices_i,
    const int* __restrict__ nnz_i,
    const int nbas_i,
    const float* log_maxval_j,
    const int* __restrict__ nnz_indices_j,
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
        const int jsh = nnz_indices_j[offset];
        const float log_aoj = log_maxval_j[offset];
        const int j0 = ao_loc[jsh];
        for (int ish_nz = 0; ish_nz < nnzi; ish_nz++){
            const int offset = ish_nz + block_id * nbas_i;
            const int ish = nnz_indices_i[offset];
            const float log_aoi = log_maxval_i[offset];
            const float log_rho_est = log_aoi + log_aoj + log_dm_shell[ish+jsh*nbas];
            if (ish > jsh) continue;
            if (log_rho_est < log_cutoff_a || log_rho_est >= log_cutoff_b) continue;

        const DataType4 xj = shell_coords[jsh];
        const DataType gjx = gx[0] - xj.x;//__ldg(shell_coords + 4*jsh);
        const DataType gjy = gx[1] - xj.y;//__ldg(shell_coords + 4*jsh + 1);
        const DataType gjz = gx[2] - xj.z;//__ldg(shell_coords + 4*jsh + 2);
        const DataType rr_gj = gjx*gjx + gjy*gjy + gjz*gjz;

        DataType cej = zero;
        DataType cej_2e = zero;
        for (int jp = 0; jp < npj; jp++){
            const int jp_offset = jp + jsh*nprim_max;
            const DataType2 coeff_expj = coeff_exp[jp_offset];
            const DataType e = coeff_expj.e;
            const DataType e_rr = e * rr_gj;
            const DataType c = coeff_expj.c;
            const DataType ce = e_rr < exp_cutoff ? c * exp(-e_rr) : zero;
            cej += ce;
            cej_2e += ce * e;
        }
        cej_2e *= -two;

        constexpr int j_pow = lj + deriv;
        DataType xj_pows[j_pow+1], yj_pows[j_pow+1], zj_pows[j_pow+1];
        xj_pows[0] = 1.0; yj_pows[0] = 1.0; zj_pows[0] = 1.0;
#pragma unroll
        for (int i = 0; i < j_pow; i++){
            xj_pows[i+1] = xj_pows[i] * gjx;
            yj_pows[i+1] = yj_pows[i] * gjy;
            zj_pows[i+1] = zj_pows[i] * gjz;
        }

        DataType dxj[lj+1], dyj[lj+1], dzj[lj+1];
        if constexpr(deriv > 0){
#pragma unroll
            for (int j = 0; j < lj+1; j++){
                dxj[j] = cej_2e * xj_pows[j+1];
                dyj[j] = cej_2e * yj_pows[j+1];
                dzj[j] = cej_2e * zj_pows[j+1];
            }
#pragma unroll
            for (int j = 1; j < lj+1; j++){
                const DataType fac = cej * j;
                dxj[j] += fac * xj_pows[j-1];
                dyj[j] += fac * yj_pows[j-1];
                dzj[j] += fac * zj_pows[j-1];
            }
        }
        
        // TODO: cart2sph transformation
        DataType ao_j[nfj], ao_jx[nfj], ao_jy[nfj], ao_jz[nfj];
#pragma unroll
        for (int i = 0, lx = lj; lx >= 0; lx--){
            for (int ly = lj - lx; ly >= 0; ly--, i++){
                const int lz = lj - lx - ly;
                ao_j[i] = cej * xj_pows[lx] * yj_pows[ly] * zj_pows[lz];

                if constexpr(deriv > 0){
                    ao_jx[i] = dxj[lx] * yj_pows[ly] * zj_pows[lz];
                    ao_jy[i] = xj_pows[lx] * dyj[ly] * zj_pows[lz];
                    ao_jz[i] = xj_pows[lx] * yj_pows[ly] * dzj[lz];
                }
            }
        }

            const DataType4 xi = shell_coords[ish];
            const DataType gix = gx[0] - xi.x;//__ldg(shell_coords + 4*ish);
            const DataType giy = gx[1] - xi.y;//__ldg(shell_coords + 4*ish + 1);
            const DataType giz = gx[2] - xi.z;//__ldg(shell_coords + 4*ish + 2);
            const DataType rr_gi = gix*gix + giy*giy + giz*giz;

            DataType cei = zero;
            DataType cei_2e = zero;
            
            for (int ip = 0; ip < npi; ip++){
                const int offset = ip + ish*nprim_max;
                const DataType2 coeff_expi = coeff_exp[offset];
                const DataType e = coeff_expi.e;//(exps + offset);
                const DataType e_rr = e * rr_gi;
                const DataType c = coeff_expi.c;//(coeffs + offset);
                const DataType ce = e_rr < exp_cutoff ? c * exp(-e_rr) : zero;
                cei += ce;
                cei_2e += ce * e;
            }
            cei_2e *= -two;
            
            const int i0 = ao_loc[ish];

            constexpr int i_pow = li + deriv;
            DataType xi_pows[i_pow+1], yi_pows[i_pow+1], zi_pows[i_pow+1];
            xi_pows[0] = one; yi_pows[0] = one; zi_pows[0] = one;
#pragma unroll
            for (int i = 0; i < i_pow; i++){
                xi_pows[i+1] = xi_pows[i] * gix;
                yi_pows[i+1] = yi_pows[i] * giy;
                zi_pows[i+1] = zi_pows[i] * giz;
            }

            DataType dxi[li+1], dyi[li+1], dzi[li+1];
            if constexpr (deriv > 0){
                for (int i = 0; i < li+1; i++){
                    dxi[i] = cei_2e * xi_pows[i+1];
                    dyi[i] = cei_2e * yi_pows[i+1];
                    dzi[i] = cei_2e * zi_pows[i+1];
                }
                for (int i = 1; i < li+1; i++){
                    DataType fac = cei * i;
                    dxi[i] += fac * xi_pows[i-1];
                    dyi[i] += fac * yi_pows[i-1];
                    dzi[i] += fac * zi_pows[i-1];
                }
            }

            DataType rho0 = zero;
            DataType rhox = zero;
            DataType rhoy = zero;
            DataType rhoz = zero;
            DataType tau = zero;

            DataType* dm_ptr = dm + i0 * nao + j0;
#pragma unroll
            for (int i = 0, lx = li; lx >= 0; lx--){
                DataType cxi = cei * xi_pows[lx];
                for (int ly = li - lx; ly >= 0; ly--, i++){
                    const int lz = li - lx - ly;
                    DataType ao_i = cxi * yi_pows[ly] * zi_pows[lz];
                    DataType ao_ix, ao_iy, ao_iz;

                    if constexpr(ndim > 1){
                        ao_ix = dxi[lx] * yi_pows[ly] * zi_pows[lz];
                        ao_iy = xi_pows[lx] * dyi[ly] * zi_pows[lz];
                        ao_iz = xi_pows[lx] * yi_pows[ly] * dzi[lz];
                    }

                    DataType s0 = zero, sx = zero, sy = zero, sz = zero;

#pragma unroll
                    for (int j = 0; j < nfj; j++){
                        const int offset = i * nao + j;
                        DataType dm_ij = __ldg(dm_ptr + offset);
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
