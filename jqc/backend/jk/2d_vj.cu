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

// 2*pi**2.5
constexpr DataType PI_FAC = 34.98683665524972497;
constexpr DataType half = 0.5;
constexpr DataType one = 1.0;
constexpr DataType zero = 0.0;

// BASIS_STRIDE is the total stride: [coords (4), ce (BASIS_STRIDE-4)]
constexpr int basis_stride = BASIS_STRIDE;

constexpr int blocksize = 256;  // Number of threads per block

// Coords are always 4: [x, y, z, ao_loc]
struct __align__(4*sizeof(DataType)) DataType4 {
    DataType x, y, z, w;  // w stores ao_loc
};

struct __align__(2*sizeof(DataType)) DataType2 {
    DataType c, e;
};

// Helper to get pointer to ce data for a basis
__device__ __forceinline__ const DataType2* load_ce_ptr(const DataType* __restrict__ basis_data, int ish) {
    return reinterpret_cast<const DataType2*>(basis_data + ish * basis_stride + 4);
}

extern "C" __global__
void rys_vj_2d(const int nao,
        const DataType* __restrict__ basis_data,
        DataType* __restrict__ dm,
        double* __restrict__ vj,
        const DataType omega,
        const int2* __restrict__ ij_pairs,
        const int n_ij_pairs,
        const int2* __restrict__ kl_pairs,
        const int n_kl_pairs,
        const float* __restrict__ q_cond_ij,
        const float* __restrict__ q_cond_kl,
        const float log_cutoff)
{
    const int ij_idx = blockIdx.x;

    constexpr int stride_i = 1;
    constexpr int stride_j = stride_i * (li+1);
    constexpr int stride_k = stride_j * (lj+1);
    constexpr int stride_l = stride_k * (lk+1);
    constexpr int gsize = (li+1)*(lj+1)*(lk+1)*(ll+1);
    constexpr int gsize2 = 2*gsize;

    constexpr int nfi = (li+1)*(li+2)/2;
    constexpr int nfj = (lj+1)*(lj+2)/2;
    constexpr int nfk = (lk+1)*(lk+2)/2;
    constexpr int nfl = (ll+1)*(ll+2)/2;

    constexpr int gstride_l = 1;
    constexpr int gstride_k = gstride_l * nfl;
    constexpr int gstride_j = gstride_k * nfk;
    constexpr int gstride_i = gstride_j * nfj;
    constexpr int integral_size = nfi*nfj*nfk*nfl;

    // Each thread handles one fixed ij pair
    const int tid = threadIdx.x;
    const int ij0 = ij_idx * blocksize;
    const bool valid_ij = (ij0 + tid) < n_ij_pairs;
    const int my_ij_idx = valid_ij ? ij0 + tid : ij0;

    // Load my ij pair
    const int2 my_ij = ij_pairs[my_ij_idx];
    const int ish = my_ij.x;
    const int jsh = my_ij.y;
    DataType fac_sym_ij = (ish == jsh) ? half * PI_FAC : PI_FAC;
    fac_sym_ij = valid_ij ? fac_sym_ij : zero;

    const DataType* base_i = basis_data + ish * basis_stride;
    const DataType* base_j = basis_data + jsh * basis_stride;
    const DataType4 ri = *reinterpret_cast<const DataType4*>(base_i);
    const DataType4 rj = *reinterpret_cast<const DataType4*>(base_j);
    const int j0 = static_cast<int>(rj.w);

    const DataType rij0 = rj.x - ri.x;
    const DataType rij1 = rj.y - ri.y;
    const DataType rij2 = rj.z - ri.z;
    const DataType rjri[3] = {rij0, rij1, rij2};
    const DataType rr_ij = rjri[0]*rjri[0] + rjri[1]*rjri[1] + rjri[2]*rjri[2];

    // Load and precompute ij primitives
    const DataType2* cei_ptr = load_ce_ptr(basis_data, ish);
    const DataType2* cej_ptr = load_ce_ptr(basis_data, jsh);
    DataType2 reg_cei[npi], reg_cej[npj];
    DataType cicj_inv_aij_arr[npi*npj];  // Cache cicj * inv_aij
    DataType inv_aij_arr[npi*npj];
    DataType aj_aij_arr[npi*npj];  // Cache aj_aij

    for (int ip = 0; ip < npi; ip++){
        reg_cei[ip] = cei_ptr[ip];
    }
    for (int jp = 0; jp < npj; jp++){
        reg_cej[jp] = cej_ptr[jp];
    }
    for (int ip = 0; ip < npi; ip++)
    for (int jp = 0; jp < npj; jp++){
        const DataType ai = reg_cei[ip].e;
        const DataType aj = reg_cej[jp].e;
        const DataType ci = reg_cei[ip].c;
        const DataType cj = reg_cej[jp].c;
        const DataType aij = ai + aj;
        const DataType inv_aij = one / aij;
        const DataType aj_aij = aj * inv_aij;
        const DataType theta_ij = ai * aj_aij;
        const DataType Kab = exp(-theta_ij * rr_ij);
        const DataType cicj = fac_sym_ij * ci * cj * Kab;
        const DataType cicj_inv_aij = cicj * inv_aij;

        const int idx = ip + jp*npi;
        cicj_inv_aij_arr[idx] = cicj_inv_aij;  // Store cicj * inv_aij
        inv_aij_arr[idx] = inv_aij;
        aj_aij_arr[idx] = aj_aij;  // Store precomputed aj_aij
    }

    double reg_vj[nfi*nfj] = {zero};

    // Shared memory for loading kl pairs (declared outside loop)
    __shared__ DataType smem_ckcl[npk*npl*blocksize];
    __shared__ DataType smem_inv_akl[npk*npl*blocksize];
    __shared__ DataType smem_al_akl[npk*npl*blocksize];  // Cache al_akl
    __shared__ DataType smem_rlx[blocksize];
    __shared__ DataType smem_rly[blocksize];
    __shared__ DataType smem_rlz[blocksize];
    __shared__ DataType smem_rkx[blocksize];
    __shared__ DataType smem_rky[blocksize];
    __shared__ DataType smem_rkz[blocksize];
    __shared__ int smem_l_loc[blocksize];
    __shared__ int smem_k_loc[blocksize];

    // Loop over all kl blocks
    const int n_kl_blocks = (n_kl_pairs + blocksize - 1) / blocksize;
    for (int kl_block = 0; kl_block < n_kl_blocks; kl_block++){
        const int kl0_block = kl_block * blocksize;
        const float q_ij = q_cond_ij[ij0];
        const float q_kl = q_cond_kl[kl0_block];
        if (q_ij + q_kl < log_cutoff) {
            // Skip entire kl block
            break;
        }
        
        const bool valid_kl = (kl0_block + tid) < n_kl_pairs;
        const int my_kl_idx = valid_kl ? kl0_block + tid : kl0_block;

        // Load ksh and rk for this kl block
        const int2 my_kl = kl_pairs[my_kl_idx];
        const int ksh = my_kl.x;
        const int lsh = my_kl.y;
        DataType fac_sym_kl = (ksh == lsh) ? half : one;
        fac_sym_kl = valid_kl ? fac_sym_kl : zero;

        const DataType* base_k = basis_data + ksh * basis_stride;
        const DataType* base_l = basis_data + lsh * basis_stride;
        const DataType4 rk = *reinterpret_cast<const DataType4*>(base_k);
        const DataType4 rl = *reinterpret_cast<const DataType4*>(base_l);
        
        // Load kl pairs into shared memory
        smem_rlx[tid] = rl.x;
        smem_rly[tid] = rl.y;
        smem_rlz[tid] = rl.z;
        smem_rkx[tid] = rk.x;
        smem_rky[tid] = rk.y;
        smem_rkz[tid] = rk.z;
        smem_k_loc[tid] = static_cast<int>(rk.w);
        smem_l_loc[tid] = static_cast<int>(rl.w);

        const DataType rkl0 = rl.x - rk.x;
        const DataType rkl1 = rl.y - rk.y;
        const DataType rkl2 = rl.z - rk.z;
        const DataType rlrk[3] = {rkl0, rkl1, rkl2};
        const DataType rr_kl = rlrk[0]*rlrk[0] + rlrk[1]*rlrk[1] + rlrk[2]*rlrk[2];
        
        const DataType2* cek_ptr = load_ce_ptr(basis_data, ksh);
        const DataType2* cel_ptr = load_ce_ptr(basis_data, lsh);
        DataType2 reg_cek[npk], reg_cel[npl];
        for (int kp = 0; kp < npk; kp++){
            reg_cek[kp] = cek_ptr[kp];
        }
        for (int lp = 0; lp < npl; lp++){
            reg_cel[lp] = cel_ptr[lp];
        }
        for (int kp = 0; kp < npk; kp++)
        for (int lp = 0; lp < npl; lp++){
            const DataType ak = reg_cek[kp].e;
            const DataType al = reg_cel[lp].e;
            const DataType akl = ak + al;
            const DataType inv_akl = one / akl;
            const DataType al_akl = al * inv_akl;
            const DataType theta_kl = ak * al_akl;
            const DataType Kcd = exp(-theta_kl * rr_kl);
            const DataType ck = reg_cek[kp].c;
            const DataType cl = reg_cel[lp].c;
            const DataType ckcl = fac_sym_kl * ck * cl * Kcd;

            const int idx = (kp + lp*npk) * blocksize + tid;
            smem_ckcl[idx] = ckcl;
            smem_inv_akl[idx] = inv_akl;
            smem_al_akl[idx] = al_akl;  // Store precomputed al_akl
        }
        __syncthreads();
        
        // Each thread loops over all 256 kl pairs
        for (int kl_tid = 0; kl_tid < blocksize; kl_tid++){
            constexpr int nfkl = nfk*nfl;

            const DataType rk[3] = {smem_rkx[kl_tid], smem_rky[kl_tid], smem_rkz[kl_tid]};
            const DataType rl[3] = {smem_rlx[kl_tid], smem_rly[kl_tid], smem_rlz[kl_tid]};
            const DataType rlrk[3] = {rl[0] - rk[0], rl[1] - rk[1], rl[2] - rk[2]};
            DataType integral[integral_size] = {zero};
            
            for (int kp = 0; kp < npk; kp++)
            for (int lp = 0; lp < npl; lp++){
                const int kl_idx = (kp + lp*npk) * blocksize + kl_tid;
                const DataType inv_akl = smem_inv_akl[kl_idx];
                const DataType ckcl = smem_ckcl[kl_idx];
                const DataType al_akl = smem_al_akl[kl_idx];  // Load precomputed al_akl

                for (int ip = 0; ip < npi; ip++)
                for (int jp = 0; jp < npj; jp++){
                    const int idx = ip + jp*npi;
                    const DataType inv_aij = inv_aij_arr[idx];
                    const DataType cicj_inv_aij = cicj_inv_aij_arr[idx];  // Load cicj * inv_aij
                    const DataType aj_aij = aj_aij_arr[idx];  // Load precomputed aj_aij
                    
                    const DataType xij = rjri[0] * aj_aij + ri.x;
                    const DataType yij = rjri[1] * aj_aij + ri.y;
                    const DataType zij = rjri[2] * aj_aij + ri.z;
                    const DataType xkl = rlrk[0] * al_akl + rk[0];
                    const DataType ykl = rlrk[1] * al_akl + rk[1];
                    const DataType zkl = rlrk[2] * al_akl + rk[2];
                    const DataType Rpq[3] = {xij-xkl, yij-ykl, zij-zkl};
                    
                    const DataType rr = Rpq[0]*Rpq[0] + Rpq[1]*Rpq[1] + Rpq[2]*Rpq[2];
                    const DataType inv_sum = inv_aij + inv_akl;
                    const DataType theta = one / inv_sum;
                    const DataType inv_aijkl = inv_aij * inv_akl * theta;

                    DataType gy0 = cicj_inv_aij * inv_akl * sqrt(inv_aijkl);
                    DataType rw[2*nroots];
                    
                    rys_roots(rr, rw, theta, omega);
                    for (int irys = 0; irys < nroots; irys++){
                        const DataType rt = rw[irys*2];
                        const DataType rt_aa = rt * inv_aijkl;
                        DataType g[3*gsize];
                        g[0] = ckcl;
                        g[gsize] = gy0;
                        g[2*gsize] = rw[(irys*2+1)];
                        
                        // TRR
                        constexpr int lij = li + lj;
                        if constexpr (lij > 0) {
                            const DataType rt_aij = rt * inv_aij * theta;
                            const DataType b10 = half * inv_aij * (one - rt_aij);
                            
            #pragma unroll
                            for (int _ix = 0; _ix < 3; _ix++){
                                DataType *_gix = g + _ix * gsize;
                                const DataType Rpa = rjri[_ix] * aj_aij;
                                const DataType c0x = Rpa - rt_aij * Rpq[_ix];
                                DataType s0x, s1x, s2x;
                                s0x = _gix[0];
                                s1x = c0x * s0x;
                                _gix[stride_i] = s1x;
                                for (int i = 1; i < lij; ++i) {
                                    const DataType i_b10 = i * b10;
                                    s2x = c0x * s1x + i_b10 * s0x;
                                    _gix[i*stride_i + stride_i] = s2x;
                                    s0x = s1x;
                                    s1x = s2x;
                                }
                            }
                        }
                        
                        constexpr int lkl = lk + ll;
                        if constexpr (lkl > 0) {
                            const DataType rt_akl = rt * inv_akl * theta;
                            const DataType b00 = half * rt_aa;
                            const DataType b01 = half * inv_akl * (one - rt_akl);
            #pragma unroll
                            for (int _ix = 0; _ix < 3; _ix++){
                                DataType *_gix = g + _ix * gsize;
                                const DataType Rqc = rlrk[_ix] * al_akl;
                                const DataType cpx = Rqc + rt_akl * Rpq[_ix];
                                
                                DataType s0x, s1x, s2x;
                                s0x = _gix[0];
                                s1x = cpx * s0x;
                                _gix[stride_k] = s1x;
                                
            #pragma unroll
                                for (int k = 1; k < lkl; ++k) {
                                    const DataType k_b01 = k * b01;
                                    s2x = cpx*s1x + k_b01*s0x;
                                    _gix[k*stride_k + stride_k] = s2x;
                                    s0x = s1x;
                                    s1x = s2x;
                                }
            #pragma unroll
                                for (int i = 1; i < lij+1; i++){
                                    const DataType ib00 = i * b00;
                                    const int i_off = i * stride_i;
                                    int i_off_minus = i_off - stride_i;
                                    int i_off_plus_k = i_off + stride_k;
                                    s0x = _gix[i_off];
                                    s1x = cpx * s0x;
                                    s1x += ib00 * _gix[i_off_minus];
                                    _gix[i_off_plus_k] = s1x;
                                    
                                    for (int k = 1; k < lkl; ++k) {
                                        const int k_i_off_minus = i_off_minus + k * stride_k;
                                        const int k_i_off_plus_k = i_off_plus_k + k * stride_k;
                                        const DataType k_b01 = k * b01;
                                        
                                        s2x = cpx * s1x + k_b01 * s0x;
                                        s2x += ib00 * _gix[k_i_off_minus];
                                        _gix[k_i_off_plus_k] = s2x;
                                        s0x = s1x;
                                        s1x = s2x;
                                    }
                                }
                            }
                        }
                        
                        // hrr
                        if constexpr (lj > 0) {
                            constexpr int stride_j_i = stride_j - stride_i;
            #pragma unroll
                            for (int _ix = 0; _ix < 3; _ix++){
                                DataType *_gix = g + _ix * gsize;
                                const DataType rjri_ix = rjri[_ix];
                                for (int kl = 0; kl < lkl+1; kl++){
                                    const int kl_off = kl*stride_k;
                                    const int ijkl0 = kl_off + lij*stride_i;
                                    for (int j = 0; j < lj; ++j) {
                                        DataType s0x, s1x;
                                        const int jkl_off = kl_off + j*stride_j;
                                        int ijkl = ijkl0 + j*stride_j_i;
                                        s1x = _gix[ijkl];
                                        for (ijkl-=stride_i; ijkl >= jkl_off; ijkl-=stride_i) {
                                            s0x = _gix[ijkl];
                                        _gix[(ijkl+stride_j)] = s1x - rjri_ix * s0x;
                                            s1x = s0x;
                                        }
                                    }
                                }
                            }
                        }
                        
                        if constexpr (ll > 0) {
                            constexpr int stride_l_k = stride_l - stride_k;
            #pragma unroll
                            for (int _ix = 0; _ix < 3; _ix++){
                                DataType *_gix = g + _ix * gsize;
                                const DataType rlrk_ix = rlrk[_ix];
                                for (int ij = 0; ij < (li+1)*(lj+1); ij++){
                                    const int ij_off = ij*stride_i;
                                    const int ijl = lkl*stride_k + ij_off;
                                    for (int l = 0; l < ll; ++l) {
                                        const int lstride_l = l*stride_l;
                                        DataType s0x, s1x;
                                        int ijkl = ijl + l*stride_l_k;
                                        s1x = _gix[ijkl];
                                        for (ijkl-=stride_k; ijkl >= lstride_l; ijkl-=stride_k) {
                                            s0x = _gix[ijkl];
                                            _gix[ijkl + stride_l] = s1x - rlrk_ix * s0x;
                                            s1x = s0x;
                                        }
                                    }
                                }
                            }
                        }
                        
                        DataType* gx = g;
                        DataType* gy = g + gsize;
                        DataType* gz = g + gsize2;
            #pragma unroll
                        for (int i = 0; i < nfi; i++){
                            const int base_ij_off_i = i*gstride_i;
                            for (int j = 0; j < nfj; j++){
                                const int addr_ij = i_idx[i] + j_idx[j];
                                const int ij_off = base_ij_off_i + j*gstride_j;
                                for (int k = 0; k < nfk; k++){
                                    const int addr_ijk = addr_ij + k_idx[k];
                                    int integral_off = ij_off + k*gstride_k;
                                    for (int l = 0; l < nfl; l++){
                                        uint32_t addr = addr_ijk + l_idx[l];
                                        uint32_t addrx = addr        & 0x3FF;
                                        uint32_t addry = (addr >> 10) & 0x3FF;
                                        uint32_t addrz = (addr >> 20) & 0x3FF;
                                        integral[integral_off + l*gstride_l] += gx[addrx] * gy[addry] * gz[addrz];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // Contract with DM for this kl pair
            const int k0 = smem_k_loc[kl_tid];
            const int l0 = smem_l_loc[kl_tid];
            for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
                DataType dm_kl_cache[nfkl];
                // Load DM[k][l] for k in [k0, k0+nfk) and l in [l0, l0+nfl)
                for (int k = 0; k < nfk; k++){
                    DataType *dm_ptr = dm + (k0 + k)*nao + l0 + i_dm*nao*nao;
                    for (int l = 0; l < nfl; l++){
                        dm_kl_cache[k + l*nfk] = __ldg(dm_ptr + l);
                    }
                }
                
                for (int i = 0; i < nfi; i++){
                    const int base_off_i = i*gstride_i;
                    for (int j = 0; j < nfj; j++){
                        const int off_j = base_off_i + j*gstride_j;
                        DataType vj_ji = zero;
                        for (int k = 0; k < nfk; k++){
                            int off_k = off_j + k * gstride_k;
                            int idx = off_k;
                            int cache_idx = k;
                            for (int l = 0; l < nfl; l++){
                                vj_ji += integral[idx] * dm_kl_cache[cache_idx];
                                idx += gstride_l;
                                cache_idx += nfk;
                            }
                        }
                        reg_vj[i*nfj + j] += vj_ji;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write results directly to VJ with atomicAdd
    const int i0 = static_cast<int>(ri.w);
    for (int i = 0; i < nfi; i++){
        for (int j = 0; j < nfj; j++){
            // VJ[i0+i][j0+j] contribution
            const int vj_idx = (i0 + i) * nao + (j0 + j);
            atomicAdd(&vj[vj_idx], reg_vj[i*nfj + j]);
        }
    }
}
