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

constexpr int threadx = 16;

template <typename T>
__inline__ __device__ T warpReduceSum(T v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    v += __shfl_down_sync(0xffffffff, v, 4);
    v += __shfl_down_sync(0xffffffff, v, 2);
    v += __shfl_down_sync(0xffffffff, v, 1);
    return v;
}

template <typename T>
__inline__ __device__ T blockReduceSum2D(T v) {
    __syncthreads();
    __shared__ T warpSum[8];  // 256 / 32 = 8 warps
    const int tid  = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp = tid >> 5;   // warp index 0–7
    const int lane = tid & 31;   // lane index 0–31

    // First reduction: inside each warp (registers only)
    v = warpReduceSum(v);

    // Write per-warp sum to shared memory
    if (lane == 0) warpSum[warp] = v;
    __syncthreads();

    // Only first warp handles the 8 partials
    T val = (warp == 0 && lane < 8) ? warpSum[lane] : T(0);

    // Final warp reduction (only 8 valid lanes)
    if (warp == 0) {
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
    }

    return val;
}


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
void rys_pair_vk(const int nao,
        const DataType* __restrict__ basis_data,
        DataType* __restrict__ dm,
        double* __restrict__ vk,
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
    const int kl_idx = blockIdx.y;

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

    constexpr int nfrag = pair_wide / threadx;

    // Load per-pair q_cond values for Schwarz screening
    const int ij0 = ij_idx * pair_wide;
    const int kl0 = kl_idx * pair_wide;
    const float q_ij = __ldg(&q_cond_ij[ij0]);
    const float q_kl = __ldg(&q_cond_kl[kl0]);
    if (q_ij + q_kl < log_cutoff) return;

    const int2 ij = ij_pairs[ij0];
    const int2 kl = kl_pairs[kl0];
    const int ish = ij.x;
    const int ksh = kl.x;
    if (ish < ksh) return;
    DataType fac_sym_kl = one;
    fac_sym_kl *= (ij.x == kl.x) ? half : one;
    
    const DataType* base_i = basis_data + ish * basis_stride;
    const DataType* base_k = basis_data + ksh * basis_stride;
    const DataType4 ri = *reinterpret_cast<const DataType4*>(base_i);
    const DataType4 rk = *reinterpret_cast<const DataType4*>(base_k);

    __shared__ DataType smem_cicj_inv_aij[npi*npj*pair_wide];  // Cache cicj * inv_aij
    __shared__ DataType smem_inv_aij[npi*npj*pair_wide];
    __shared__ DataType smem_aj_aij[npi*npj*pair_wide];  // Cache aj_aij
    __shared__ DataType smem_ckcl[npk*npl*pair_wide];
    __shared__ DataType smem_inv_akl[npk*npl*pair_wide];
    __shared__ DataType smem_al_akl[npk*npl*pair_wide];  // Cache al_akl
    __shared__ DataType smem_rjx[pair_wide];
    __shared__ DataType smem_rjy[pair_wide];
    __shared__ DataType smem_rjz[pair_wide];
    __shared__ int smem_j_loc[pair_wide];
    __shared__ DataType smem_rlx[pair_wide];
    __shared__ DataType smem_rly[pair_wide];
    __shared__ DataType smem_rlz[pair_wide];
    __shared__ int smem_l_loc[pair_wide];

    // preload cicj, inv_aij, rj, and cej for all pairs in the block
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < pair_wide){
        const int ij_offset = ij0 + tid;
        const int2 ij = ij_pairs[ij_offset];
        const int jsh = (ij.y < 0) ? 0 : ij.y;
        const DataType* base_j = basis_data + jsh * basis_stride;
        const DataType4 rj = *reinterpret_cast<const DataType4*>(base_j);

        // Store rj in shared memory
        smem_rjx[tid] = rj.x;
        smem_rjy[tid] = rj.y;
        smem_rjz[tid] = rj.z;
        smem_j_loc[tid] = static_cast<int>(rj.w);

        const DataType fac_sym_ij = (ij.x < 0) ? zero : PI_FAC;
        const DataType rij0 = rj.x - ri.x;
        const DataType rij1 = rj.y - ri.y;
        const DataType rij2 = rj.z - ri.z;
        const DataType rjri[3] = {rij0, rij1, rij2};
        const DataType rr_ij = rjri[0]*rjri[0] + rjri[1]*rjri[1] + rjri[2]*rjri[2];

        const DataType2* cei_ptr = load_ce_ptr(basis_data, ish);
        const DataType2* cej_ptr = load_ce_ptr(basis_data, jsh);
        DataType2 reg_cei[npi], reg_cej[npj];
        for (int ip = 0; ip < npi; ip++){
            reg_cei[ip] = cei_ptr[ip];
        }
        for (int jp = 0; jp < npj; jp++){
            reg_cej[jp] = cej_ptr[jp];
        }
        for (int ip = 0; ip < npi; ip++)
        for (int jp = 0; jp < npj; jp++){
            DataType ai = reg_cei[ip].e;
            DataType aj = reg_cej[jp].e;
            DataType ci = reg_cei[ip].c;
            DataType cj = reg_cej[jp].c;
            const DataType aij = ai + aj;
            const DataType inv_aij = one / aij;
            const DataType aj_aij = aj * inv_aij;
            const DataType theta_ij = ai * aj_aij;
            const DataType Kab = exp(-theta_ij * rr_ij);
            const DataType cicj = fac_sym_ij * ci * cj * Kab;
            const DataType cicj_inv_aij = cicj * inv_aij;

            const int idx = (ip + jp*npi) * pair_wide + tid;
            smem_cicj_inv_aij[idx] = cicj_inv_aij;  // Store cicj * inv_aij
            smem_inv_aij[idx] = inv_aij;
            smem_aj_aij[idx] = aj_aij;  // Store precomputed aj_aij
        }
    }

    // preload ckcl, inv_akl, rl, and cel for all pairs in the block
    if (tid < pair_wide){
        const int kl_offset = kl0 + tid;
        const int2 kl = kl_pairs[kl_offset];
        const int lsh = (kl.y < 0) ? 0 : kl.y;
        const DataType* base_l = basis_data + lsh * basis_stride;
        const DataType4 rl = *reinterpret_cast<const DataType4*>(base_l);

        // Store rl in shared memory
        smem_rlx[tid] = rl.x;
        smem_rly[tid] = rl.y;
        smem_rlz[tid] = rl.z;
        smem_l_loc[tid] = static_cast<int>(rl.w);

        fac_sym_kl = (kl.x < 0) ? zero : fac_sym_kl;
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

            const int idx = (kp + lp*npk) * pair_wide + tid;
            smem_ckcl[idx] = ckcl;
            smem_inv_akl[idx] = inv_akl;
            smem_al_akl[idx] = al_akl;  // Store precomputed al_akl
        }
    }
    __syncthreads();
    
    DataType reg_vk[nfi*nfk] = {zero};
    for (int ij_frag = 0; ij_frag < nfrag; ij_frag++){
        // Load rj and cej from shared memory
        const int smem_ij_idx = threadx*ij_frag + threadIdx.x;
        //const DataType4 rj = smem_rj[smem_ij_idx];
        const DataType rj[3] = {smem_rjx[smem_ij_idx], smem_rjy[smem_ij_idx], smem_rjz[smem_ij_idx]};
        const DataType rjri[3] = {rj[0] - ri.x, rj[1] - ri.y, rj[2] - ri.z};

        for (int kl_frag = 0; kl_frag < nfrag; kl_frag++){
            // Load rl and cel from shared memory
            const int smem_kl_idx = threadx*kl_frag + threadIdx.y;
            //const DataType4 rl = smem_rl[smem_kl_idx];
            const DataType rl[3] = {smem_rlx[smem_kl_idx], smem_rly[smem_kl_idx], smem_rlz[smem_kl_idx]};
            const DataType rlrk[3] = {rl[0] - rk.x, rl[1] - rk.y, rl[2] - rk.z};
            DataType integral[integral_size] = {zero};

        #pragma unroll
            for (int kp = 0; kp < npk; kp++)
            for (int lp = 0; lp < npl; lp++){
                // Load precomputed values from shared memory
                const int kl_idx = (kp + lp*npk) * pair_wide + smem_kl_idx;
                const DataType inv_akl = smem_inv_akl[kl_idx];
                const DataType ckcl = smem_ckcl[kl_idx];
                const DataType al_akl = smem_al_akl[kl_idx];  // Load precomputed al_akl
                for (int ip = 0; ip < npi; ip++)
                for (int jp = 0; jp < npj; jp++){
                    const int idx = (ip + jp*npi) * pair_wide + smem_ij_idx;
                    const DataType inv_aij = smem_inv_aij[idx];
                    const DataType cicj_inv_aij = smem_cicj_inv_aij[idx];  // Load cicj * inv_aij
                    const DataType aj_aij = smem_aj_aij[idx];  // Load precomputed aj_aij

                    const DataType xij = rjri[0] * aj_aij + ri.x;
                    const DataType yij = rjri[1] * aj_aij + ri.y;
                    const DataType zij = rjri[2] * aj_aij + ri.z;
                    const DataType xkl = rlrk[0] * al_akl + rk.x;
                    const DataType ykl = rlrk[1] * al_akl + rk.y;
                    const DataType zkl = rlrk[2] * al_akl + rk.z;
                    const DataType Rpq[3] = {xij-xkl, yij-ykl, zij-zkl};

                    const DataType rr = Rpq[0]*Rpq[0] + Rpq[1]*Rpq[1] + Rpq[2]*Rpq[2];

                    // Optimized computation using precomputed inv_aij and inv_akl
                    // theta = aij * akl / (aij + akl) = 1 / (1/aij + 1/akl) = 1 / (inv_aij + inv_akl)
                    const DataType inv_sum = inv_aij + inv_akl;
                    const DataType theta = one / inv_sum;
                    const DataType inv_aijkl = inv_aij * inv_akl * theta;

                    const DataType gy0 = cicj_inv_aij * inv_akl * sqrt(inv_aijkl);
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
                            // rt_aij = rt_aa * akl = rt * inv_aij * theta
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
                            // rt_akl = rt_aa * aij = rt * inv_akl * theta
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
                                        uint32_t addrx =  addr        & 0x3FF;
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

            const int j0 = smem_j_loc[smem_ij_idx];//static_cast<int>(rj.w);
            const int l0 = smem_l_loc[smem_kl_idx];//static_cast<int>(rl.w);

            constexpr int nfjl = nfj*nfl;

            for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
                DataType dm_jl_cache[nfjl];
                const int dm_offset = j0*nao + l0;
                DataType *dm_ptr = dm + dm_offset;
        #pragma unroll
                for (int j = 0; j < nfj; j++){
                    for (int l = 0; l < nfl; l++){
                        dm_jl_cache[l + j*nfl] = __ldg(dm_ptr + l);
                    }
                    dm_ptr += nao;
                }
        #pragma unroll
                for (int i = 0; i < nfi; i++){
                    const int base_off_i = i*gstride_i;
                    for (int k = 0; k < nfk; k++){
                        DataType vk_ik = zero;
                        int off_k = base_off_i + k * gstride_k;
                        for (int j = 0; j < nfj; j++){
                            const int cache_j = j * nfl;
                            int idx = off_k;
                            for (int l = 0; l < nfl; l++){
                                vk_ik += integral[idx] * dm_jl_cache[cache_j + l];
                                idx += gstride_l;
                            }
                            off_k += gstride_j;
                        }
                        reg_vk[i*nfk + k] += vk_ik;
                    }
                }

                //const int nao2 = nao*nao;
                //dm += nao2;
                //vk += nao2;
            }
        }
    }

    const int i0 = static_cast<int>(ri.w);
    const int k0 = static_cast<int>(rk.w);

    const int vk_offset = i0*nao + k0;
    double *vk_ptr = vk + vk_offset;
    for (int i = 0; i < nfi; i++)
    for (int k = 0; k < nfk; k++){
        DataType vk_tmp = blockReduceSum2D(reg_vk[i*nfk + k]);
        if (threadIdx.x == 0 && threadIdx.y == 0)
            atomicAdd(vk_ptr + i*nao + k, vk_tmp);
    }
}
