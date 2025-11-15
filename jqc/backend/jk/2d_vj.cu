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
constexpr int prim_stride = PRIM_STRIDE / 2;

constexpr int threadx = 1;
constexpr int thready = 256;
constexpr int warp_threads = 32;
constexpr int block_threads = threadx * thready;
constexpr int num_warps = block_threads / warp_threads;

template <typename T>
__device__ __forceinline__ T blockReduceSum2D(T val) {
    // Optimized for (1, 256) thread layout: 1 column × 256 rows = 8 warps
    // Uses bitwise operations for better performance
    const int lane = threadIdx.y & 31;        // threadIdx.y % 32
    const int wid  = threadIdx.y >> 5;        // threadIdx.y / 32 (0-7)

    // Shared memory sized exactly for 8 warps
    __shared__ T shared[8];

    // Step 1: Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Step 2: Warp leaders write partial sums to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Step 3: First warp reduces the 8 partial sums
    if (threadIdx.y < 8) {
        val = shared[threadIdx.y];
        // Reduce 8 values with unrolled shuffle operations
        val += __shfl_down_sync(0xff, val, 4);  // 0xff = first 8 threads active
        val += __shfl_down_sync(0xff, val, 2);
        val += __shfl_down_sync(0xff, val, 1);
    }

    // Final barrier to ensure all threads sync before function returns
    //__syncthreads();

    return val;  // Thread 0 contains the final sum
}

// Make coordinate stride configurable via COORD_STRIDE
static_assert(COORD_STRIDE >= 3, "COORD_STRIDE must be >= 3");
struct __align__(COORD_STRIDE*sizeof(DataType)) DataType4 {
    DataType x, y, z;
#if COORD_STRIDE >= 4
    DataType w;
#endif
#if COORD_STRIDE > 4
    DataType pad[COORD_STRIDE - 4];
#endif
};
static_assert(sizeof(DataType4) == COORD_STRIDE*sizeof(DataType),
              "DataType4 size must equal COORD_STRIDE*sizeof(DataType)");

struct __align__(2*sizeof(DataType)) DataType2 {
    DataType c, e;
};

extern "C" __global__
void rys_vj_2d(const int nbas,
        const int nao,
        const int * __restrict__ ao_loc,
        const DataType4* __restrict__ coords,
        const DataType2* __restrict__ coeff_exp,
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
    constexpr int n_kl = 1;
    const int kl_block_base = blockIdx.y * thready * n_kl;
    constexpr int nfi = (li+1)*(li+2)/2;
    constexpr int nfj = (lj+1)*(lj+2)/2;
    constexpr int nfk = (lk+1)*(lk+2)/2;
    constexpr int nfl = (ll+1)*(ll+2)/2;
    
    if (kl_block_base >= n_kl_pairs) {
        return;
    }

    const float q_ij = __ldg(&q_cond_ij[ij_idx]);
    const float q_kl = __ldg(&q_cond_kl[kl_block_base]);

    // Apply Schwarz screening, skip entire block if inactive
    if (q_ij + q_kl < log_cutoff) { return; }
    
    // Load ij pair index
    const int2 ij = ij_pairs[ij_idx];
    int ish = ij.x;
    int jsh = ij.y;
    // Decode shell indices from flattened pair indices
    //int ish = ij / nbas;
    //int jsh = ij - ish * nbas;

    const int i0 = ao_loc[ish];
    const int j0 = ao_loc[jsh];
    
    // Apply ij symmetry screening
    DataType fac_sym_ij = PI_FAC;
    fac_sym_ij = (ish >= nbas || jsh >= nbas) ? zero : fac_sym_ij;
    fac_sym_ij *= (ish == jsh) ? half : one;

    // Clamped versions for array indexing
    ish = (ish >= nbas) ? 0 : ish;
    jsh = (jsh >= nbas) ? 0 : jsh;
    
    constexpr int stride_i = 1;
    constexpr int stride_j = stride_i * (li+1);
    constexpr int stride_k = stride_j * (lj+1);
    constexpr int stride_l = stride_k * (lk+1);
    constexpr int gsize = (li+1)*(lj+1)*(lk+1)*(ll+1);
    constexpr int gsize2 = 2*gsize;

    constexpr int gstride_l = 1;
    constexpr int gstride_k = gstride_l * nfl;
    constexpr int gstride_j = gstride_k * nfk;
    constexpr int gstride_i = gstride_j * nfj;
    constexpr int integral_size = nfi*nfj*nfk*nfl;

    const DataType4 ri = coords[ish]; 
    const DataType4 rj = coords[jsh];

    const DataType rij0 = rj.x - ri.x;
    const DataType rij1 = rj.y - ri.y;
    const DataType rij2 = rj.z - ri.z;

    const DataType rjri[3] = {rij0, rij1, rij2};
    const DataType rr_ij = rjri[0]*rjri[0] + rjri[1]*rjri[1] + rjri[2]*rjri[2];

    DataType2 reg_cei[npi], reg_cej[npj];
    for (int ip = 0; ip < npi; ip++){
        const int ish_ip = ip + ish*prim_stride;
        reg_cei[ip] = coeff_exp[ish_ip];
    }
    for (int jp = 0; jp < npj; jp++){
        const int jsh_jp = jp + jsh*prim_stride;
        reg_cej[jp] = coeff_exp[jsh_jp];
    }

    // Cache per-(ip,jp) terms in shared memory to avoid repeated expensive exp/div computations
    __shared__ DataType sh_cicj[npi*npj*threadx];
    __shared__ DataType sh_inv_aij[npi*npj*threadx];

    if (threadIdx.y == 0){
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
            const int idx = (ip + jp*npi) * threadx + threadIdx.x;
            sh_cicj[idx] = cicj;
            sh_inv_aij[idx] = inv_aij;
        }
    }
    __syncthreads();
    DataType reg_vj[nfi*nfj] = {zero};

#pragma unroll
    for (int i_kl = 0; i_kl < n_kl; i_kl++){
    int kl_idx = kl_block_base + threadIdx.y * n_kl + i_kl;
    DataType fac_sym_kl = (kl_idx < n_kl_pairs) ? one : zero;
    kl_idx = (kl_idx < n_kl_pairs) ? kl_idx : 0;

    DataType integral[integral_size] = {zero};

    const int2 kl = kl_pairs[kl_idx];
    int ksh = kl.x;
    int lsh = kl.y;

    // Decode kl shell indices and preload coordinates
    //int ksh = kl / nbas;
    //int lsh = kl - ksh * nbas;

    fac_sym_kl = (ksh >= nbas || lsh >= nbas) ? zero : fac_sym_kl;
    fac_sym_kl *= (ksh == lsh) ? half : one;
    
    ksh = (ksh >= nbas) ? 0 : ksh;
    lsh = (lsh >= nbas) ? 0 : lsh;

    const DataType4 rk = coords[ksh]; 
    const DataType4 rl = coords[lsh];

    const DataType rkl0 = rl.x - rk.x;
    const DataType rkl1 = rl.y - rk.y;
    const DataType rkl2 = rl.z - rk.z;

    const DataType rlrk[3] = {rkl0, rkl1, rkl2};
    const DataType rr_kl = rlrk[0]*rlrk[0] + rlrk[1]*rlrk[1] + rlrk[2]*rlrk[2];

#pragma unroll
    for (int kp = 0; kp < npk; kp++)
    for (int lp = 0; lp < npl; lp++){
        const int ksh_kp = kp + ksh*prim_stride;
        const int lsh_lp = lp + lsh*prim_stride;
        const DataType2 cek = coeff_exp[ksh_kp]; 
        const DataType2 cel = coeff_exp[lsh_lp];
        const DataType ak = cek.e;
        const DataType al = cel.e;
        const DataType akl = ak + al;
        const DataType inv_akl = one / akl;
        const DataType al_akl = al * inv_akl;
        const DataType theta_kl = ak * al_akl;
        const DataType Kcd = exp(-theta_kl * rr_kl);
        const DataType ck = cek.c;
        const DataType cl = cel.c;
        const DataType ckcl = fac_sym_kl * ck * cl * Kcd;
        for (int ip = 0; ip < npi; ip++)
        for (int jp = 0; jp < npj; jp++){
            const DataType ai = reg_cei[ip].e;
            const DataType aj = reg_cej[jp].e;
            const DataType aij = ai + aj;

            const int idx = (ip + jp*npi) * threadx + threadIdx.x;
            const DataType inv_aij = sh_inv_aij[idx];
            const DataType cicj = sh_cicj[idx];
            const DataType aj_aij = aj * inv_aij;

            const DataType xij = rjri[0] * aj_aij + ri.x;
            const DataType yij = rjri[1] * aj_aij + ri.y;
            const DataType zij = rjri[2] * aj_aij + ri.z;
            const DataType xkl = rlrk[0] * al_akl + rk.x;
            const DataType ykl = rlrk[1] * al_akl + rk.y;
            const DataType zkl = rlrk[2] * al_akl + rk.z;
            const DataType Rpq[3] = {xij-xkl, yij-ykl, zij-zkl};

            const DataType rr = Rpq[0]*Rpq[0] + Rpq[1]*Rpq[1] + Rpq[2]*Rpq[2];
            const DataType inv_aijkl = one / (aij + akl);
            const DataType theta = aij * akl * inv_aijkl;

            DataType gy0 = cicj * inv_aij * inv_akl * sqrt(inv_aijkl);
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
                    const DataType rt_aij = rt_aa * akl;
                    const DataType b10 = 0.5 * inv_aij * (one - rt_aij);

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
                    const DataType rt_akl = rt_aa * aij;
                    const DataType b00 = 0.5 * rt_aa;
                    const DataType b01 = 0.5 * inv_akl * (one - rt_akl);
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

    const int k0 = ao_loc[ksh];
    const int l0 = ao_loc[lsh];
    constexpr int nfkl = nfk*nfl;

    for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
        DataType dm_kl_cache[nfkl];

        const int dm_offset = k0 + l0*nao;
        DataType *dm_ptr = dm + dm_offset + i_dm*nao*nao;
#pragma unroll
        for (int l = 0; l < nfl; l++){
            for (int k = 0; k < nfk; k++){
                dm_kl_cache[k + l*nfk] = __ldg(dm_ptr + k);
            }
            dm_ptr += nao;
        }

#pragma unroll
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

const int vj_offset = i0 + j0*nao;// + i_dm*nao*nao;
double *vj_ptr = vj + vj_offset;
for (int i = 0; i < nfi; i++){
    for (int j = 0; j < nfj; j++){
        const int offset = j*nao + i;
        DataType vj_ji = reg_vj[i*nfj + j];
        //atomicAdd(vj_ptr + offset, (double)vj_ji);
        DataType vj_tmp = blockReduceSum2D(vj_ji);
        if (threadIdx.x == 0 && threadIdx.y == 0)
            atomicAdd(vj_ptr + offset, (double)vj_tmp);
    }
}

}

