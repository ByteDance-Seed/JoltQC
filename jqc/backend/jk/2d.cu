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
constexpr DataType one = 1.0;
constexpr DataType zero = 0.0;
constexpr int prim_stride = PRIM_STRIDE / 2;


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
void rys_jk_2d(const int nbas,
        const int nao,
        const int * __restrict__ ao_loc,
        const DataType4* __restrict__ coords,
        const DataType2* __restrict__ coeff_exp,
        DataType* __restrict__ dm,
        double* __restrict__ vj,
        double* __restrict__ vk,
        const DataType omega,
        const int* __restrict__ ij_pairs,
        const int n_ij_pairs,
        const int* __restrict__ kl_pairs,
        const int n_kl_pairs)
{
    const int ij_idx = blockIdx.x;
    const int kl_idx = blockIdx.y;

    // Early exit for out-of-bounds blocks
    if (ij_idx >= n_ij_pairs || kl_idx >= n_kl_pairs) {
        return;
    }

    // Extract pair indices with thread indexing
    // Pairs are stored in blocks of 16, with padding for incomplete blocks
    const int ij_offset = ij_idx * 16 + threadIdx.x;
    const int kl_offset = kl_idx * 16 + threadIdx.y;

    // Load pair with bounds checking
    const int ij = (ij_offset < n_ij_pairs * 16) ? ij_pairs[ij_offset] : 0;
    const int kl = (kl_offset < n_kl_pairs * 16) ? kl_pairs[kl_offset] : 0;

    // Decode shell indices from flattened pair indices
    const int ish = ij / nbas;
    const int jsh = ij - ish * nbas;  // Equivalent to ij % nbas but potentially faster
    const int ksh = kl / nbas;
    const int lsh = kl - ksh * nbas;

    // Validate shell indices
    // This handles both padding entries (ij=0, kl=0) and out-of-bounds cases
    if (ish >= nbas || jsh >= nbas || ksh >= nbas || lsh >= nbas) {
        return;
    }

    // Additional validation: ensure pairs represent valid shell combinations
    // For symmetry: require ish >= jsh and ksh >= lsh (or adjust based on your convention)
    if (ij == 0 || kl == 0) {
        return;  // Skip padding entries
    }

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
    
    DataType fac_sym = PI_FAC;
    DataType4 ri, rj, rk, rl;
    ri.x = __ldg(&coords[ish].x);
    ri.y = __ldg(&coords[ish].y);
    ri.z = __ldg(&coords[ish].z);
    rj.x = __ldg(&coords[jsh].x);
    rj.y = __ldg(&coords[jsh].y);
    rj.z = __ldg(&coords[jsh].z);
    rk.x = __ldg(&coords[ksh].x);
    rk.y = __ldg(&coords[ksh].y);
    rk.z = __ldg(&coords[ksh].z);
    rl.x = __ldg(&coords[lsh].x);
    rl.y = __ldg(&coords[lsh].y);
    rl.z = __ldg(&coords[lsh].z);

    const DataType rij0 = rj.x - ri.x;
    const DataType rij1 = rj.y - ri.y;
    const DataType rij2 = rj.z - ri.z;

    const DataType rjri[3] = {rij0, rij1, rij2};
    const DataType rr_ij = rjri[0]*rjri[0] + rjri[1]*rjri[1] + rjri[2]*rjri[2];
    const DataType rkl0 = rl.x - rk.x;
    const DataType rkl1 = rl.y - rk.y;
    const DataType rkl2 = rl.z - rk.z;

    const DataType rlrk[3] = {rkl0, rkl1, rkl2};
    const DataType rr_kl = rlrk[0]*rlrk[0] + rlrk[1]*rlrk[1] + rlrk[2]*rlrk[2];
    DataType integral[integral_size] = {zero};

    // Estimate register usage for caching cei, cej, cicj and inv_aij
    // DataType can be float (1 register) or double (2 registers)
    constexpr int reg_per_datatype = sizeof(DataType) / 4; // 4 bytes per 32-bit register
    constexpr int reg_g = reg_per_datatype * 3 * gsize;
    constexpr int reg_aij_ceij = reg_per_datatype * 2 * npi * npj;
    constexpr int reg_cei_cej = reg_per_datatype * 2 * (npi + npj);
    constexpr int reg_integral = reg_per_datatype * integral_size;
    constexpr int estimated_registers = reg_g + reg_aij_ceij + reg_cei_cej + reg_integral;
    constexpr bool use_cache = (estimated_registers <= 256);

    DataType2 reg_cei[npi], reg_cej[npj];
    for (int ip = 0; ip < npi; ip++){
        const int ish_ip = ip + ish*prim_stride;
        reg_cei[ip].c = __ldg(&coeff_exp[ish_ip].c);
        reg_cei[ip].e = __ldg(&coeff_exp[ish_ip].e);
    }
    for (int jp = 0; jp < npj; jp++){
        const int jsh_jp = jp + jsh*prim_stride;
        reg_cej[jp].c = __ldg(&coeff_exp[jsh_jp].c);
        reg_cej[jp].e = __ldg(&coeff_exp[jsh_jp].e);
    }
    
    // Cache per-(ip,jp) terms to avoid repeated expensive exp/div computations if register usage is reasonable
    DataType reg_cicj[use_cache ? npi*npj : 1];
    DataType reg_inv_aij[use_cache ? npi*npj : 1];
    
    if constexpr (use_cache) {
#pragma unroll
        for (int ip = 0; ip < npi; ip++){
            for (int jp = 0; jp < npj; jp++){
                DataType ai, aj, ci, cj;
                ai = reg_cei[ip].e;
                aj = reg_cej[jp].e;
                ci = reg_cei[ip].c;
                cj = reg_cej[jp].c;

                const DataType aij = ai + aj;
                const DataType inv_aij = one / aij;
                const DataType aj_aij = aj * inv_aij;
                const DataType theta_ij = ai * aj_aij;
                const DataType Kab = exp(-theta_ij * rr_ij);
                const DataType cicj = fac_sym * ci * cj * Kab;
                const int idx = ip + jp*npi;
                reg_cicj[idx] = cicj;
                reg_inv_aij[idx] = inv_aij;
            }
        }
    }

#pragma unroll
    for (int kp = 0; kp < npk; kp++)
    for (int lp = 0; lp < npl; lp++){
        const int ksh_kp = kp + ksh*prim_stride;
        const int lsh_lp = lp + lsh*prim_stride;
        DataType2 cek, cel;
        cek.c = __ldg(&coeff_exp[ksh_kp].c);
        cek.e = __ldg(&coeff_exp[ksh_kp].e);
        cel.c = __ldg(&coeff_exp[lsh_lp].c);
        cel.e = __ldg(&coeff_exp[lsh_lp].e);
        const DataType ak = cek.e;
        const DataType al = cel.e;
        const DataType akl = ak + al;
        const DataType inv_akl = one / akl;
        const DataType al_akl = al * inv_akl;
        const DataType theta_kl = ak * al_akl;
        const DataType Kcd = exp(-theta_kl * rr_kl);
        const DataType ck = cek.c;
        const DataType cl = cel.c;
        const DataType ckcl = ck * cl * Kcd;
        for (int ip = 0; ip < npi; ip++)
        for (int jp = 0; jp < npj; jp++){
            DataType ai, aj, ci, cj;
            if constexpr (use_cache) {
                ai = reg_cei[ip].e;
                aj = reg_cej[jp].e;
                ci = reg_cei[ip].c;
                cj = reg_cej[jp].c;
            } else {
                const int ish_ip = ip + ish*prim_stride;
                const int jsh_jp = jp + jsh*prim_stride;
                DataType2 cei, cej;
                cei.c = __ldg(&coeff_exp[ish_ip].c);
                cei.e = __ldg(&coeff_exp[ish_ip].e);
                cej.c = __ldg(&coeff_exp[jsh_jp].c);
                cej.e = __ldg(&coeff_exp[jsh_jp].e);
                ai = cei.e;
                aj = cej.e;
                ci = cei.c;
                cj = cej.c;
            }
            const DataType aij = ai + aj;

            DataType inv_aij, cicj;
            if constexpr (use_cache) {
                const int idx = ip + jp*npi;
                inv_aij = reg_inv_aij[idx];
                cicj = reg_cicj[idx];
            } else {
                inv_aij = one / aij;
                const DataType aj_aij = aj * inv_aij;
                const DataType theta_ij = ai * aj_aij;
                const DataType Kab = exp(-theta_ij * rr_ij);
                cicj = fac_sym * ci * cj * Kab;
            }
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
                //for i in range(lij):
                //    trr(i+1,0) = c0 * trr(i,0) + i*b10 * trr(i-1,0)
                //for k in range(lkl):
                //    for i in range(lij+1):
                //        trr(i,k+1) = c0p * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                constexpr int lij = li + lj;
                if constexpr (lij > 0) {
                    const DataType rt_aij = rt_aa * akl;
                    const DataType b10 = 0.5 * inv_aij * (one - rt_aij);
                    
#pragma unroll
                    for (int _ix = 0; _ix < 3; _ix++){
                        DataType *_gix = g + _ix * gsize;
                        // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                        const DataType Rpa = rjri[_ix] * aj_aij;
                        const DataType c0x = Rpa - rt_aij * Rpq[_ix];
                        DataType s0x, s1x, s2x;
                        s0x = _gix[0];
                        s1x = c0x * s0x;
                        _gix[stride_i] = s1x;
                        for (int i = 1; i < lij; ++i) {
                            const DataType i_b10 = i * b10;  // Pre-compute to reduce FLOPs
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
                        
                        //  trr(0,1) = c0p * trr(0,0)
                        DataType s0x, s1x, s2x;
                        s0x = _gix[0];
                        s1x = cpx * s0x;
                        _gix[stride_k] = s1x;
                        
                        // trr(0,k+1) = cp * trr(0,k) + k*b01 * trr(0,k-1)
#pragma unroll
                        for (int k = 1; k < lkl; ++k) {
                            const DataType k_b01 = k * b01;  // Pre-compute to reduce FLOPs
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
                            //for i in range(1, lij+1):
                            //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                            s0x = _gix[i_off];
                            s1x = cpx * s0x;
                            s1x += ib00 * _gix[i_off_minus];
                            _gix[i_off_plus_k] = s1x;

                            //for k in range(1, lkl):
                            //    for i in range(lij+1):
                            //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                            for (int k = 1; k < lkl; ++k) {
                                const int k_i_off_minus = i_off_minus + k * stride_k;
                                const int k_i_off_plus_k = i_off_plus_k + k * stride_k;
                                const DataType k_b01 = k * b01;  // Pre-compute to reduce FLOPs

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
                // g(i,j+1) = rirj * g(i,j) +  g(i+1,j)
                // g(...,k,l+1) = rkrl * g(...,k,l) + g(...,k+1,l)
                if constexpr (lj > 0) {
                    constexpr int stride_j_i = stride_j - stride_i;
#pragma unroll
                    for (int _ix = 0; _ix < 3; _ix++){
                        DataType *_gix = g + _ix * gsize;
                        const DataType rjri_ix = rjri[_ix];  // Pre-compute to reduce FLOPs in inner loop
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
                        const DataType rlrk_ix = rlrk[_ix];  // Pre-compute to reduce FLOPs in inner loop
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
                                uint32_t addrx =  addr        & 0x3FF;      // 10 low-order bits
                                uint32_t addry = (addr >> 10) & 0x3FF;      // next 10 bits
                                uint32_t addrz = (addr >> 20) & 0x3FF;      // next 10 bits
                                integral[integral_off + l*gstride_l] += gx[addrx] * gy[addry] * gz[addrz];
                            }
                        }
                    }
                }
            }
        }
    }

    const int i0 = ao_loc[ish];
    const int j0 = ao_loc[jsh];
    const int k0 = ao_loc[ksh];
    const int l0 = ao_loc[lsh];

    constexpr int nfij = nfi*nfj;
    constexpr int nfkl = nfk*nfl;
    constexpr int nfik = nfi*nfk;
    constexpr int nfil = nfi*nfl;
    constexpr int nfjk = nfj*nfk;
    constexpr int nfjl = nfj*nfl;
    
    for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
        if constexpr(do_j){
            // ijkl, kl -> ij
            {
                DataType dm_kl_cache[nfkl];
                const int dm_offset = k0 + l0*nao;
                DataType *dm_ptr = dm + dm_offset;
#pragma unroll
                for (int l = 0; l < nfl; l++){
                    for (int k = 0; k < nfk; k++){
                        dm_kl_cache[k + l*nfk] = __ldg(dm_ptr + k);
                    }
                    dm_ptr += nao;
                }
                const int vj_offset = i0 + j0*nao;
                double *vj_ptr = vj + vj_offset;
#pragma unroll
                for (int i = 0; i < nfi; i++){
                    const int base_off_i = i*gstride_i;
                    for (int j = 0; j < nfj; j++){
                        DataType vj_ji = zero;
                        const int off_j = base_off_i + j*gstride_j;
                        for (int k = 0; k < nfk; k++){
                            int off_k = off_j + k * gstride_k;
                            int idx = off_k;
                            int cache_idx = k; // advances by nfk per l
                            for (int l = 0; l < nfl; l++){
                                vj_ji += integral[idx] * dm_kl_cache[cache_idx];
                                idx += gstride_l;
                                cache_idx += nfk;
                            }
                        }
                        const int offset = j*nao + i;
                        atomicAdd(vj_ptr + offset, (double)vj_ji);
                    }
                }
            }
        }
        
        if constexpr(do_k){
            // ijkl, jl -> ik
            {
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
                const int vk_offset = i0*nao + k0;
                double *vk_ptr = vk + vk_offset;
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
                        const int offset = i*nao + k;
                        atomicAdd(vk_ptr + offset, (double)vk_ik);
                    }
                }
            }
        }
        const int nao2 = nao*nao;
        dm += nao2;
        if constexpr(do_j) vj += nao2;
        if constexpr(do_k) vk += nao2;
    }
}
