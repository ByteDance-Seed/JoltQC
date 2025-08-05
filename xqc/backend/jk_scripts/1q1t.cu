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

// Portions of this file adapted from GPU4PySCF v1.4 (https://github.com/pyscf/gpu4pyscf)
// Copyright 2025 PySCF developer.
// Licensed under the Apache License, Version 2.0.

// 2*pi**2.5
constexpr DataType PI_FAC = 34.98683665524972497;
constexpr DataType half = .5;
constexpr DataType one = 1.0;
constexpr DataType zero = 0.0;
constexpr int nprim_max = 16;

extern "C" __global__
void rys_jk(const int nbas,
        const int * __restrict__ ao_loc, 
        const DataType* __restrict__ coords,
        const DataType* __restrict__ exponents, 
        const DataType* __restrict__ coeffs,
        DataType* dm, 
        double* vj, 
        double* vk, 
        const DataType omega,
        int4* __restrict__ shl_quartet_idx, 
        const int ntasks) // rename
{
    if (ntasks == 0) return;
    const int task_id = blockIdx.x * blockDim.x + threadIdx.x;

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

    const int4 sq = shl_quartet_idx[task_id];
    const bool active = (task_id < ntasks);
    const int ish = active ? sq.x : 0;
    const int jsh = active ? sq.y : 0;
    const int ksh = active ? sq.z : 0;
    const int lsh = active ? sq.w : 0;

    DataType fac_sym = active ? PI_FAC : zero;
    if (ish == jsh) fac_sym *= half;
    if (ksh == lsh) fac_sym *= half;
    if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= half;

    const DataType rix = __ldg(coords + 3*ish);
    const DataType riy = __ldg(coords + 3*ish+1);
    const DataType riz = __ldg(coords + 3*ish+2);
    const DataType rkx = __ldg(coords + 3*ksh);
    const DataType rky = __ldg(coords + 3*ksh+1);
    const DataType rkz = __ldg(coords + 3*ksh+2);

    const DataType rij0 = __ldg(coords + 3*jsh)   - rix;
    const DataType rij1 = __ldg(coords + 3*jsh+1) - riy;
    const DataType rij2 = __ldg(coords + 3*jsh+2) - riz;
    const DataType rjri[3] = {rij0, rij1, rij2};
    const DataType rr_ij = rjri[0]*rjri[0] + rjri[1]*rjri[1] + rjri[2]*rjri[2];
    const DataType rkl0 = __ldg(coords + 3*lsh)   - rkx;
    const DataType rkl1 = __ldg(coords + 3*lsh+1) - rky;
    const DataType rkl2 = __ldg(coords + 3*lsh+2) - rkz;
    const DataType rlrk[3] = {rkl0, rkl1, rkl2};
    const DataType rr_kl = rlrk[0]*rlrk[0] + rlrk[1]*rlrk[1] + rlrk[2]*rlrk[2];
    DataType integral[integral_size] = {zero};

    DataType reg_ai[npi], reg_aj[npj], reg_ci[npi], reg_cj[npj];
    for (int ip = 0; ip < npi; ip++){
        const int ish_ip = ip + ish*nprim_max;
        reg_ai[ip] = __ldg(exponents + ish_ip);
        reg_ci[ip] = __ldg(coeffs + ish_ip);
    }
    for (int jp = 0; jp < npj; jp++){
        const int jsh_jp = jp + jsh*nprim_max;
        reg_aj[jp] = __ldg(exponents + jsh_jp);
        reg_cj[jp] = __ldg(coeffs + jsh_jp);
    }
    DataType reg_cicj[npi*npj];
    for (int ip = 0; ip < npi; ip++){
        for (int jp = 0; jp < npj; jp++){
            const DataType ai = reg_ai[ip];
            const DataType aj = reg_aj[jp];
            const DataType aij = ai + aj;
            const DataType aj_aij = aj / aij;
            const DataType theta_ij = ai * aj_aij;
            const DataType Kab = exp(-theta_ij * rr_ij);
            const DataType ci = reg_ci[ip];
            const DataType cj = reg_cj[jp];
            const DataType cicj = fac_sym * ci * cj * Kab;
            reg_cicj[ip + jp*npi] = cicj;
        }
    }
#pragma unroll
    for (int kp = 0; kp < npk; kp++)
    for (int lp = 0; lp < npl; lp++){
        const int ksh_kp = kp + ksh*nprim_max;
        const int lsh_lp = lp + lsh*nprim_max;
        const DataType ak = __ldg(exponents + ksh_kp);
        const DataType al = __ldg(exponents + lsh_lp);
        const DataType akl = ak + al;
        const DataType al_akl = al / akl;
        const DataType theta_kl = ak * al_akl;
        const DataType Kcd = exp(-theta_kl * rr_kl);
        const DataType ck = __ldg(coeffs + ksh_kp);
        const DataType cl = __ldg(coeffs + lsh_lp);
        const DataType ckcl = ck * cl * Kcd;
        for (int ip = 0; ip < npi; ip++)
        for (int jp = 0; jp < npj; jp++){
            const int ip_offset = ip + ish*nprim_max;
            const int jp_offset = jp + jsh*nprim_max;
            const DataType ai = __ldg(exponents + ip_offset);
            const DataType aj = __ldg(exponents + jp_offset);
            const DataType aij = ai + aj;
            const DataType aj_aij = aj / aij;
            const DataType cicj = reg_cicj[ip + jp*npi];

            const DataType xij = rjri[0] * aj_aij + rix;
            const DataType yij = rjri[1] * aj_aij + riy;
            const DataType zij = rjri[2] * aj_aij + riz;
            const DataType xkl = rlrk[0] * al_akl + rkx;
            const DataType ykl = rlrk[1] * al_akl + rky;
            const DataType zkl = rlrk[2] * al_akl + rkz;
            const DataType Rpq[3] = {xij-xkl, yij-ykl, zij-zkl};

            const DataType rr = Rpq[0]*Rpq[0] + Rpq[1]*Rpq[1] + Rpq[2]*Rpq[2];
            const DataType theta = aij * akl / (aij + akl);

            DataType gy0 = cicj / (aij*akl*sqrt(aij+akl));
            DataType rw[2*nroots];

            rys_roots(rr, rw, theta, omega);
#pragma unroll
            for (int irys = 0; irys < nroots; irys++){
                const DataType rt = rw[irys*2];
                const DataType rt_aa = rt / (aij + akl);
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
                    const DataType b10 = half/aij * (one - rt_aij);
                    
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
                            s2x = c0x * s1x + i * b10 * s0x;
                            _gix[i*stride_i + stride_i] = s2x;
                            s0x = s1x;
                            s1x = s2x;
                        }
                    }
                }

                constexpr int lkl = lk + ll;
                if constexpr (lkl > 0) {
                    const DataType rt_akl = rt_aa * aij;
                    const DataType b00 = half * rt_aa;
                    const DataType b01 = half/akl * (one - rt_akl);
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
                            s2x = cpx*s1x + k*b01*s0x;
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

                            DataType kb01 = zero;
                            //for k in range(1, lkl):
                            //    for i in range(lij+1):
                            //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                            for (int k = 1; k < lkl; ++k) {
                                i_off_minus += stride_k;
                                i_off_plus_k += stride_k;
                                kb01 += b01;

                                s2x = cpx*s1x + kb01*s0x;
                                s2x += ib00 * _gix[i_off_minus];
                                _gix[i_off_plus_k] = s2x;
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
                                    _gix[(ijkl+stride_j)] = s1x - rjri[_ix] * s0x;
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
                                    _gix[ijkl + stride_l] = s1x - rlrk[_ix] * s0x;
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
                    for (int j = 0; j < nfj; j++){
                        const int addr_ij = i_idx[i] + j_idx[j];
                        const int ij_off = i*gstride_i + j*gstride_j;
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
    
    const int nao = ao_loc[nbas];
    
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
            // ijkl, ij -> kl
            {
                const int dm_offset = i0 + j0*nao;
                DataType *dm_ptr = dm + dm_offset;
                DataType vj_lk[nfkl] = {0.0};
#pragma unroll
                for (int i = 0; i < nfi; i++){
                    for (int j = 0; j < nfj; j++){
                        const int dm_offset = j*nao + i;
                        DataType dm_ij = __ldg(dm_ptr + dm_offset);
                        int off = i * gstride_i + j * gstride_j;
                        for (int k = 0; k < nfk; k++){
                            for (int l = 0; l < nfl; l++){
                                vj_lk[l + k*nfl] += integral[l*gstride_l + off] * dm_ij;
                            }
                            off += gstride_k;
                        }
                    }
                }

                const int vj_offset = k0 + l0*nao;
                double *vj_ptr = vj + vj_offset;
                for (int k = 0; k < nfk; k++){
                    for (int l = 0; l < nfl; l++){
                        const int vj_offset = k + l*nao;
                        atomicAdd(vj_ptr + vj_offset, (double)vj_lk[l + k*nfl]);
                    }
                }
            }

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
                    for (int j = 0; j < nfj; j++){
                        DataType vj_ji = zero;
                        int off = i*gstride_i + j*gstride_j;
                        for (int k = 0; k < nfk; k++){
                            for (int l = 0; l < nfl; l++){
                                vj_ji += integral[off + l*gstride_l] * dm_kl_cache[k + l*nfk];
                            }
                            off += gstride_k;
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
                    for (int k = 0; k < nfk; k++){
                        DataType vk_ik = zero;
                        int off = i*gstride_i + k * gstride_k;
                        for (int j = 0; j < nfj; j++){
                            for (int l = 0; l < nfl; l++){
                                vk_ik += integral[off + l*gstride_l] * dm_jl_cache[l + j*nfl];
                            }
                            off += gstride_j;
                        }
                        const int offset = i*nao + k;
                        atomicAdd(vk_ptr + offset, (double)vk_ik);
                    }
                }
            }

            // ijkl, jk -> il
            {
                DataType dm_jk_cache[nfjk];
                const int dm_offset = j0*nao + k0;
                DataType *dm_ptr = dm + dm_offset;
#pragma unroll
                for (int j = 0; j < nfj; j++){
                    for (int k = 0; k < nfk; k++){
                        dm_jk_cache[k + j*nfk] = __ldg(dm_ptr + k);
                    }
                    dm_ptr += nao;
                }

                const int vk_offset = i0*nao + l0;
                double *vk_ptr = vk + vk_offset;
#pragma unroll
                for (int i = 0; i < nfi; i++){
                    for (int l = 0; l < nfl; l++){
                        DataType vk_il = zero;
                        int off = i*gstride_i + l*gstride_l;
                        for (int j = 0; j < nfj; j++){
                            for (int k = 0; k < nfk; k++){
                                vk_il += integral[off + k*gstride_k] * dm_jk_cache[k + j*nfk];
                            }
                            off += gstride_j;
                        }
                        const int offset = i*nao + l;
                        atomicAdd(vk_ptr + offset, (double)vk_il);
                    }
                }
            }

            // ijkl, il -> jk
            {
                DataType dm_il_cache[nfil];
                const int dm_offset = i0*nao + l0;
                DataType *dm_ptr = dm + dm_offset;
                for (int i = 0; i < nfi; i++){
                    for (int l = 0; l < nfl; l++){
                        dm_il_cache[l + i*nfl] = __ldg(dm_ptr + l);
                    }
                    dm_ptr += nao;
                }
                const int vk_offset = j0*nao + k0;
                double *vk_ptr = vk + vk_offset;
                for (int j = 0; j < nfj; j++){
                    for (int k = 0; k < nfk; k++){
                        DataType vk_jk = zero;
                        int off = j * gstride_j + k * gstride_k;
                        for (int i = 0; i < nfi; i++){
                            for (int l = 0; l < nfl; l++){
                                vk_jk += integral[off + l*gstride_l] * dm_il_cache[l + i*nfl];
                            }
                            off += gstride_i;
                        }
                        const int offset = j*nao + k;
                        atomicAdd(vk_ptr + offset, (double)vk_jk);
                    }
                }
            }

            // ijkl, ik -> jl
            {
                DataType vk_jl[nfl*nfj] = {0.0};
                const int dm_offset = i0*nao + k0;
                DataType *dm_ptr = dm + dm_offset;
#pragma unroll
                for (int i = 0; i < nfi; i++){
                    for (int k = 0; k < nfk; k++){
                        const int dm_offset = i*nao + k;
                        DataType dm_ik = __ldg(dm_ptr + dm_offset);
                        int off = i * gstride_i + k * gstride_k;
                        for (int j = 0; j < nfj; j++){
                            for (int l = 0; l < nfl; l++){
                                vk_jl[l + j*nfl] += integral[off + l*gstride_l] * dm_ik;
                            }
                            off += gstride_j;
                        }
                    }
                }

                const int vk_offset = j0*nao + l0;
                double *vk_ptr = vk + vk_offset;
                for (int j = 0; j < nfj; j++){
                    for (int l = 0; l < nfl; l++){
                        const int vk_offset = j*nao + l;
                        atomicAdd(vk_ptr + vk_offset, (double)vk_jl[l + j*nfl]);
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
