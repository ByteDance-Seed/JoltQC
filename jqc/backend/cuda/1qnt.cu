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

struct __align__(4*sizeof(DataType)) DataType4 {
    DataType x, y, z, w;
};

struct __align__(2*sizeof(DataType)) DataType2 {
    DataType c, e;
};

extern "C" __global__
void rys_jk(const int nbas,  
        const int* __restrict__ ao_loc, 
        const DataType4* __restrict__ coords,
        const DataType2* __restrict__ coeff_exp,
        DataType* dm, 
        double* vj, 
        double* vk,
        const DataType omega,
        const ushort4* __restrict__ shl_quartet_idx, 
        const int ntasks)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int task_id = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int nfi = (li+1)*(li+2)/2;
    constexpr int nfj = (lj+1)*(lj+2)/2;
    constexpr int nfk = (lk+1)*(lk+2)/2;
    constexpr int nfl = (ll+1)*(ll+2)/2;

    constexpr int nfij = nfi*nfj;
    constexpr int nfkl = nfk*nfl;
    constexpr int nfik = nfi*nfk;
    constexpr int nfil = nfi*nfl;
    constexpr int nfjk = nfj*nfk;
    constexpr int nfjl = nfj*nfl;

    constexpr int nti = (nfi + fragi - 1) / fragi;
    constexpr int ntj = (nfj + fragj - 1) / fragj;
    constexpr int ntk = (nfk + fragk - 1) / fragk;
    constexpr int ntl = (nfl + fragl - 1) / fragl;
    constexpr int nt_active = nti * ntj * ntk * ntl;

    constexpr int tstride_l = 1;
    constexpr int tstride_k = fragl;
    constexpr int tstride_j = fragl * fragk;
    constexpr int tstride_i = fragl * fragk * fragj;
    constexpr int frag_size = fragi*fragj*fragk*fragl;

    const int t_i = (ty % nti);
    const int t_j = (ty / nti % ntj);
    const int t_k = (ty / (nti*ntj) % ntk);
    const int t_l = (ty / (nti*ntj*ntk));

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    constexpr int gx_stride = smem_stride;
    constexpr int g_stride = 3 * gx_stride;

    // shape of g, (gsize, 3, nsq_per_block)
    constexpr int stride_i = g_stride;
    constexpr int stride_j = stride_i * (li+1);
    constexpr int stride_k = stride_j * (lj+1);
    constexpr int stride_l = stride_k * (lk+1);

    // Dynamic shared memory buffer
    extern __shared__ DataType shared_memory[];

    int kl_idx[fragk*fragl], kl_idy[fragk*fragl], kl_idz[fragk*fragl];
    for (int reg_k = 0; reg_k < fragk; reg_k++){
        const int k = t_k * fragk + reg_k;
        for (int reg_l = 0; reg_l < fragl; reg_l++){
            const int l = t_l * fragl + reg_l;
            if (ty < nt_active) {
                const int kl = reg_l + reg_k * fragl;
                const uint32_t addr = k_idx[k] + l_idx[l];
                const uint32_t kl_x =  addr        & 0x3FF;      // 10 low-order bits
                const uint32_t kl_y = (addr >> 10) & 0x3FF;      // next 10 bits
                const uint32_t kl_z = (addr >> 20) & 0x3FF;      // next 10 bits
                kl_idx[kl] = kl_x * g_stride;
                kl_idy[kl] = kl_y * g_stride + gx_stride;
                kl_idz[kl] = kl_z * g_stride + 2*gx_stride;
            }
        }
    }

    const bool active = (task_id < ntasks);
    ushort4 sq = {0,0,0,0};
    if (active) {
        sq = shl_quartet_idx[task_id];
    }
    
    const int ish = (int)sq.x;
    const int jsh = (int)sq.y;
    const int ksh = (int)sq.z;
    const int lsh = (int)sq.w;
    
    DataType fac_sym = active ? PI_FAC : zero;
    fac_sym *= (ish == jsh) ? half : one;
    fac_sym *= (ksh == lsh) ? half : one;
    fac_sym *= (ish*nbas+jsh == ksh*nbas+lsh) ? half : one;

    const DataType4 ri = coords[ish];
    const DataType4 rj = coords[jsh];
    const DataType4 rk = coords[ksh];
    const DataType4 rl = coords[lsh];

    const DataType rjri0 = rj.x - ri.x;
    const DataType rjri1 = rj.y - ri.y;
    const DataType rjri2 = rj.z - ri.z;
    const DataType rr_ij = rjri0*rjri0 + rjri1*rjri1 + rjri2*rjri2;
    const DataType rlrk0 = rl.x - rk.x;
    const DataType rlrk1 = rl.y - rk.y;
    const DataType rlrk2 = rl.z - rk.z;
    const DataType rr_kl = rlrk0*rlrk0 + rlrk1*rlrk1 + rlrk2*rlrk2;

    // Cache coefficients and precompute cicj values in shared memory for better performance
    DataType2 reg_cei[npi], reg_cej[npj];
    for (int ip = 0; ip < npi; ip++){
        const int ish_ip = ip + ish*nprim_max;
        reg_cei[ip] = coeff_exp[ish_ip];
    }
    for (int jp = 0; jp < npj; jp++){
        const int jsh_jp = jp + jsh*nprim_max;
        reg_cej[jp] = coeff_exp[jsh_jp];
    }
    // Cache per-(ip,jp) terms to avoid repeated expensive exp/div computations
    DataType reg_cicj[npi*npj];
    DataType reg_inv_aij[npi*npj];
#pragma unroll
    for (int ip = 0; ip < npi; ip++){
        for (int jp = 0; jp < npj; jp++){
            const DataType ai = reg_cei[ip].e;
            const DataType aj = reg_cej[jp].e;
            const DataType aij = ai + aj;
            const DataType inv_aij = one / aij;
            const DataType aj_aij = aj * inv_aij;
            const DataType theta_ij = ai * aj_aij;
            const DataType Kab = exp(-theta_ij * rr_ij);
            const DataType ci = reg_cei[ip].c;
            const DataType cj = reg_cej[jp].c;
            const DataType cicj = fac_sym * ci * cj * Kab;
            const int idx = ip + jp*npi;
            reg_cicj[idx] = cicj;
            reg_inv_aij[idx] = inv_aij;
        }
    }

    DataType integral_frag[frag_size] = {zero};
    for (int kp = 0; kp < npk; kp++)
    for (int lp = 0; lp < npl; lp++){
        const int ksh_kp = kp + ksh*nprim_max;
        const int lsh_lp = lp + lsh*nprim_max;
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
        const DataType ckcl = ck * cl * Kcd;

        for (int ip = 0; ip < npi; ip++)
        for (int jp = 0; jp < npj; jp++){
            const DataType ai = reg_cei[ip].e;
            const DataType aj = reg_cej[jp].e;
            const DataType aij = ai + aj;
            const int idx = ip + jp*npi;
            const DataType inv_aij = reg_inv_aij[idx];
            const DataType aj_aij = aj * inv_aij;
            const DataType cicj = reg_cicj[idx];
            
            const DataType xij = rjri0 * aj_aij + ri.x;
            const DataType yij = rjri1 * aj_aij + ri.y;
            const DataType zij = rjri2 * aj_aij + ri.z;
            const DataType xkl = rlrk0 * al_akl + rk.x;
            const DataType ykl = rlrk1 * al_akl + rk.y;
            const DataType zkl = rlrk2 * al_akl + rk.z;
            const DataType Rpq[3] = {xij-xkl, yij-ykl, zij-zkl};

            const DataType rr = Rpq[0]*Rpq[0] + Rpq[1]*Rpq[1] + Rpq[2]*Rpq[2];
            const DataType inv_aijkl = one / (aij + akl);
            const DataType theta = aij * akl * inv_aijkl;
            
            DataType rjri_x = (ty == 0 ? rjri0 : (ty == 1 ? rjri1 : rjri2)); 
            DataType Rpq_x =  (ty == 0 ? (xij-xkl) : (ty == 1 ? (yij-ykl) : (zij-zkl)));
            DataType rlrk_x = (ty == 0 ? rlrk0 : (ty == 1 ? rlrk1 : rlrk2));

            DataType *rw = shared_memory + tx;
            DataType *g = shared_memory + nroots * 2 * gx_stride + tx; 

            rys_roots(rr, rw, ty, gx_stride, theta, omega);
            
            DataType g0xyz;
            if (ty == 0) g0xyz = ckcl; 
            if (ty == 1) g0xyz = cicj * inv_aij * inv_akl * sqrt(inv_aijkl);
            
            __syncthreads();
            for (int irys = 0; irys < nroots; irys++){
                DataType rt_aa;
                g0xyz = (ty == 2) ? rw[(irys*2+1) * gx_stride] : g0xyz;
                if (ty < 3){
                    const DataType rt = rw[(irys*2)*gx_stride];
                    rt_aa = rt * inv_aijkl;
                }
                __syncthreads();
                if (ty < 3) g[ty*gx_stride] = g0xyz;

                // TRR
                //for i in range(lij):
                //    trr(i+1,0) = c0 * trr(i,0) + i*b10 * trr(i-1,0)
                //for k in range(lkl):
                //    for i in range(lij+1):
                //        trr(i,k+1) = c0p * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                constexpr int lij = li + lj;
                if constexpr (lij > 0) {
                    if (ty < 3){
                        const DataType rt_aij = rt_aa * akl;
                        const DataType b10 = half * inv_aij * (one - rt_aij);

                        const int _ix = ty;
                        DataType *gx = g + _ix * gx_stride;

                        // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                        const DataType Rpa = aj_aij * rjri_x;
                        const DataType c0x = Rpa - rt_aij * Rpq_x;
                        DataType s0x, s1x, s2x;
                        s0x = g0xyz;
                        s1x = c0x * s0x;
                        gx[stride_i] = s1x;

                        for (int i = 1; i < lij; ++i) {
                            const DataType i_b10 = i * b10;  // Pre-compute to reduce FLOPs
                            s2x = c0x * s1x + i_b10 * s0x;
                            gx[i*stride_i + stride_i] = s2x;
                            s0x = s1x;
                            s1x = s2x;
                        }
                    }
                }
                                
                constexpr int lkl = lk + ll;
                if constexpr (lkl > 0) {
                    if (ty < 3){
                        const DataType rt_akl = rt_aa * aij;
                        const DataType b00 = half * rt_aa;
                        const DataType b01 = half * inv_akl * (one - rt_akl);

                        const int _ix = ty;
                        DataType *gx = g + _ix * gx_stride;
                        
                        const DataType Rqc = al_akl * rlrk_x;
                        const DataType cpx = Rqc + rt_akl * Rpq_x;
                        
                        //  trr(0,1) = c0p * trr(0,0)
                        DataType s0x, s1x, s2x;
                        s0x = g0xyz;
                        s1x = cpx * s0x;
                        gx[stride_k] = s1x;
                        
                        // trr(0,k+1) = cp * trr(0,k) + k*b01 * trr(0,k-1)
#pragma unroll
                        for (int k = 1; k < lkl; ++k) {
                            const DataType k_b01 = k * b01;  // Pre-compute to reduce FLOPs
                            s2x = cpx*s1x + k_b01*s0x;
                            gx[k*stride_k + stride_k] = s2x;
                            s0x = s1x;
                            s1x = s2x;
                        }
#pragma unroll
                        for (int i = 1; i < lij+1; i++){
                            //for i in range(1, lij+1):
                            //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                            const DataType ib00 = i * b00;
                            const int i_off = i * stride_i;
                            const int i_off_minus = i_off - stride_i;
                            const int i_off_plus_k = i_off + stride_k;
                            s0x = gx[i_off];
                            s1x = cpx * s0x;
                            s1x += ib00 * gx[i_off_minus];
                            gx[i_off_plus_k] = s1x;

                            //for k in range(1, lkl):
                            //    for i in range(lij+1):
                            //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                            
                            for (int k = 1; k < lkl; ++k) {
                                const DataType k_b01 = k * b01;  // Pre-compute to reduce FLOPs
                                s2x = cpx*s1x + k_b01*s0x;
                                s2x += ib00 * gx[i_off_minus + k*stride_k];
                                gx[i_off_plus_k + k*stride_k] = s2x;
                                s0x = s1x;
                                s1x = s2x;
                            }
                        }
                    }
                }
                
                const int _ix = ty;
                DataType *gx = g + _ix * gx_stride;
                // hrr
                // g(i,j+1) = rirj * g(i,j) +  g(i+1,j)
                // g(...,k,l+1) = rkrl * g(...,k,l) + g(...,k+1,l)
                if constexpr (lj > 0) {
                    constexpr int stride_j_i = stride_j - stride_i;
                    if (ty < 3){
#pragma unroll
                        for (int kl = 0; kl < lkl+1; kl++){
                            const int kl_off = kl*stride_k;
                            const int ijkl0 = kl_off + lij*stride_i;
                            for (int j = 0; j < lj; ++j) {
                                DataType s0x, s1x;
                                const int jkl_off = kl_off + j*stride_j;
                                int ijkl = ijkl0 + j*stride_j_i;
                                s1x = gx[ijkl];
                                for (ijkl-=stride_i; ijkl >= jkl_off; ijkl-=stride_i) {
                                    s0x = gx[ijkl];
                                    gx[ijkl + stride_j] = s1x - rjri_x * s0x;  // Reordered for consistency
                                    s1x = s0x;
                                }
                            }
                        }
                    }
                }

                if constexpr (ll > 0) {
                    constexpr int li1xlj1 = (li+1)*(lj+1);
                    constexpr int stride_l_k = stride_l - stride_k;
                    if (ty < 3){
#pragma unroll
                        for (int ij = 0; ij < li1xlj1; ij++){
                            const int ij_off = ij*stride_i;
                            const int ijl = lkl*stride_k + ij_off;
                            for (int l = 0; l < ll; ++l) {
                                const int lstride_l = l*stride_l;
                                int ijkl = ijl + l*stride_l_k;
                                DataType s0x, s1x;
                                s1x = gx[ijkl];
                                for (ijkl-=stride_k; ijkl >= lstride_l; ijkl-=stride_k) {
                                    s0x = gx[ijkl];
                                    gx[ijkl + stride_l] = s1x - rlrk_x * s0x;
                                    s1x = s0x;
                                }
                            }
                        }
                    }
                }
                __syncthreads();
                
                if (ty < nt_active) {
                const int idx_off = t_k * fragk * nfl + t_l * fragl;
#pragma unroll
                for (int reg_i = 0; reg_i < fragi; reg_i++){
                    const int i = t_i * fragi + reg_i;
                    for (int reg_j = 0; reg_j < fragj; reg_j++){
                        const int j = t_j * fragj + reg_j;  
                        const uint32_t addr_ij = j_idx[j] + i_idx[i];
                        const uint32_t ij_x    =  addr_ij        & 0x3FF;      // 10 low-order bits
                        const uint32_t ij_y    = (addr_ij >> 10) & 0x3FF;      // next 10 bits
                        const uint32_t ij_z    = (addr_ij >> 20) & 0x3FF;      // next 10 bits

                        int integral_off = reg_i * tstride_i + reg_j * tstride_j;
                        for (int reg_k = 0; reg_k < fragk; reg_k++){
                            for (int reg_l = 0; reg_l < fragl; reg_l++){
                                const int kl = reg_l + reg_k * fragl;
                                const int addrx = ij_x * g_stride + kl_idx[kl];
                                const int addry = ij_y * g_stride + kl_idy[kl];
                                const int addrz = ij_z * g_stride + kl_idz[kl];
                                integral_frag[integral_off + reg_l*tstride_l] += g[addrx] * g[addry] * g[addrz];
                            }
                            integral_off += tstride_k;
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

    DataType *smem = shared_memory + tx;
    const bool ty_active = (ty < nt_active);
    for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
        // ijkl, ij -> kl
        constexpr int ntij = nti*ntj;
        DataType vj_lk[fragk*fragl] = {zero};
        if (do_j && ty_active){
            const int dm_offset = (i0+t_i*fragi) + (j0+t_j*fragj)*nao;
            DataType *dm_ptr = dm + dm_offset;
#pragma unroll
            for (int i = 0; i < fragi; i++){
                for (int j = 0; j < fragj; j++){
                    const int offset = i + j*nao;
                    DataType dm_ij = __ldg(dm_ptr + offset);
                    int off = i * tstride_i + j * tstride_j;
                    for (int k = 0; k < fragk; k++){
                        for (int l = 0; l < fragl; l++){
                            vj_lk[l + k*fragl] += integral_frag[off + l*tstride_l] * dm_ij;
                        }
                        off += tstride_k;
                    }
                }
            }
        }
        if constexpr(ntij > 1) __syncthreads();
        if (do_j && ty_active){
            const int t_ij = t_i + nti * t_j;
            const int t_kl = t_k * fragk + t_l * fragl * nfk;
            constexpr int smem_kstride = smem_stride;
            constexpr int smem_lstride = smem_stride * nfk;
            DataType* smem_ptr = smem + (t_ij * nfkl + t_kl) * smem_stride;
            const int vj_offset = (l0+t_l*fragl)*nao + (k0+t_k*fragk);
            double *vj_ptr = vj + vj_offset;
#pragma unroll
            for (int k = 0; k < fragk; k++){
                for (int l = 0; l < fragl; l++){
                    if constexpr(ntij > 1){
                        smem_ptr[k*smem_kstride + l*smem_lstride] = vj_lk[l + k*fragl];
                    } else {
                        const int offset = l*nao + k;
                        atomicAdd(vj_ptr + offset, (double)vj_lk[l + k*fragl]);
                    }
                }
            }
        }
        
        if constexpr(do_j && ntij > 1){
            __syncthreads();
            const int vj_offset = l0*nao + k0;
            double *vj_ptr = vj + vj_offset;
            constexpr int stride = nfkl * smem_stride;
            for (int kl = ty; kl < nfkl; kl += nthreads_per_sq){
                DataType vj_tmp = zero;
                const int off = kl * smem_stride;
                for (int m = 0; m < ntij; m++){
                    vj_tmp += smem[off + m*stride];
                }
                const int l = kl / nfk;
                const int k = kl % nfk;
                const int offset = l*nao + k;
                atomicAdd(vj_ptr + offset, (double)vj_tmp);
            }
        }
        
        // ijkl, kl -> ij
        constexpr int ntkl = ntk*ntl;
        DataType dm_kl_cache[fragk*fragl];
        if (do_j && ty_active){
            const int dm_offset = (l0+t_l*fragl)*nao + (k0+t_k*fragk);
            DataType *dm_ptr = dm + dm_offset;
#pragma unroll
            for (int l = 0; l < fragl; l++){
                for (int k = 0; k < fragk; k++){
                    dm_kl_cache[k + l*fragk] = __ldg(dm_ptr + k);
                }
                dm_ptr += nao;
            }
        }
        if constexpr(ntkl > 1) __syncthreads();
        if (do_j && ty_active){
            const int t_kl = t_k + ntk * t_l;
            const int t_ij = t_i * fragi + t_j * fragj * nfi;
            const int smem_off = (t_ij + t_kl * nfij) * smem_stride;
            const int vj_offset = (j0+t_j*fragj)*nao + (i0+t_i*fragi);
            double *vj_ptr = vj + vj_offset;
#pragma unroll
            for (int i = 0; i < fragi; i++){
            for (int j = 0; j < fragj; j++){
                DataType vj_ji = zero;
                int integral_off = i * tstride_i + j * tstride_j;
                for (int l = 0; l < fragl; l++){
                    for (int k = 0; k < fragk; k++){
                        vj_ji += integral_frag[integral_off + k*tstride_k] * dm_kl_cache[k + l*fragk];
                    }
                    integral_off += tstride_l;
                }
                if constexpr(ntkl > 1){
                    const int ij = i + j * nfi;
                    smem[ij * smem_stride + smem_off] = vj_ji;
                } else {
                    const int offset = j*nao + i;
                    atomicAdd(vj_ptr + offset, (double)vj_ji);
                }
            }}
        }
        
        if constexpr(do_j && ntkl > 1){
            __syncthreads();
            const int vj_offset = j0*nao + i0;
            double *vj_ptr = vj + vj_offset;
            constexpr int stride = nfij * smem_stride;
            for (int ij = ty; ij < nfij; ij += nthreads_per_sq){
                DataType vj_tmp = zero;
                const int off = ij * smem_stride;
                for (int m = 0; m < ntkl; m++){
                    vj_tmp += smem[off + m*stride];
                }
                const int j = ij / nfi;
                const int i = ij % nfi;
                const int offset = j*nao + i;
                atomicAdd(vj_ptr + offset, (double)vj_tmp);
            }
        }

        // ijkl, jl -> ik
        constexpr int ntjl = ntj*ntl;
        DataType dm_jl_cache[fragj*fragl];
        if (do_k && ty_active){
            const int dm_offset = (j0+t_j*fragj)*nao + (l0+t_l*fragl);
            DataType *dm_ptr = dm + dm_offset;
#pragma unroll
            for (int j = 0; j < fragj; j++){
                for (int l = 0; l < fragl; l++){
                    dm_jl_cache[l + j*fragl] = __ldg(dm_ptr + l);
                }
                dm_ptr += nao;
            }
        }
        if constexpr(ntjl > 1) __syncthreads();
        if (do_k && ty_active){
            const int vk_offset = (i0+t_i*fragi)*nao + (k0+t_k*fragk);
            double *vk_ptr = vk + vk_offset;
            const int t_jl = t_j + ntj * t_l;
            const int t_ik = t_i * fragi * nfk + t_k * fragk;
            const int smem_off = (t_jl * nfik + t_ik) * smem_stride;
#pragma unroll
            for (int i = 0; i < fragi; i++){
                for (int k = 0; k < fragk; k++){
                    DataType vk_ik = zero;
                    int integral_off = i * tstride_i + k * tstride_k;
                    for (int j = 0; j < fragj; j++){
                        for (int l = 0; l < fragl; l++){
                            vk_ik += integral_frag[integral_off + l*tstride_l] * dm_jl_cache[l + j*fragl];
                        }
                        integral_off += tstride_j;
                    }
                    if constexpr (ntjl > 1){
                        const int ik = i*nfk + k;
                        smem[ik * smem_stride + smem_off] = vk_ik;
                    } else {
                        const int offset = i*nao + k;
                        atomicAdd(vk_ptr + offset, (double)vk_ik);
                    }
                }
            }
        }
        
        if constexpr(do_k && ntjl > 1){
            constexpr int stride = nfik * smem_stride;
            const int vk_offset = i0*nao + k0;
            double *vk_ptr = vk + vk_offset;
            __syncthreads();
            for (int ik = ty; ik < nfik; ik+=nthreads_per_sq){
                DataType vk_tmp = zero;
                const int off = ik * smem_stride;
                for (int m = 0; m < ntjl; m++){
                    vk_tmp += smem[off + m*stride];
                }
                const int k = ik % nfk;
                const int i = ik / nfk;
                const int offset = i*nao + k;
                atomicAdd(vk_ptr + offset, (double)vk_tmp);
            }
        }

        // ijkl, jk -> il
        constexpr int ntjk = ntj*ntk;
        DataType dm_jk_cache[fragj*fragk];
        if (do_k && ty_active){
            const int dm_offset = (j0+t_j*fragj)*nao + (k0+t_k*fragk);
            DataType *dm_ptr = dm + dm_offset;
#pragma unroll
            for (int j = 0; j < fragj; j++){
                for (int k = 0; k < fragk; k++){
                    dm_jk_cache[k + j*fragk] = __ldg(dm_ptr + k);
                }
                dm_ptr += nao;
            }
        }
        if constexpr(ntjk > 1) __syncthreads();
        if (do_k && ty_active){
            const int t_jk = t_j + ntj * t_k;
            const int t_il = t_i * fragi * nfl + t_l * fragl;
            const int smem_off = (t_jk * nfil + t_il) * smem_stride;
            const int vk_offset = (i0+t_i*fragi)*nao + (l0+t_l*fragl);
            double *vk_ptr = vk + vk_offset;
#pragma unroll
            for (int i = 0; i < fragi; i++){
                for (int l = 0; l < fragl; l++){
                    DataType vk_il = zero;
                    int integral_off = i * tstride_i + l * tstride_l;
                    for (int j = 0; j < fragj; j++){
                        for (int k = 0; k < fragk; k++){
                            vk_il += integral_frag[integral_off + k*tstride_k] * dm_jk_cache[k + j*fragk];
                        }
                        integral_off += tstride_j;
                    }
                    if constexpr (ntjk > 1){
                        const int il = i * nfl + l;
                        smem[il * smem_stride + smem_off] = vk_il;
                    } else {
                        const int offset = i*nao + l;
                        atomicAdd(vk_ptr + offset, (double)vk_il);
                    }
                }
            }
        }

        if constexpr(do_k && ntjk > 1){
            __syncthreads();
            const int vk_offset = i0*nao + l0;
            double *vk_ptr = vk + vk_offset;
            for (int il = ty; il < nfil; il += nthreads_per_sq){
                DataType vk_tmp = zero;
                constexpr int stride = nfil * smem_stride;
                const int off = il * smem_stride;
                for (int m = 0; m < ntjk; m++){
                    vk_tmp += smem[off + m*stride];
                }
                const int l = il % nfl;
                const int i = il / nfl;
                const int offset = i*nao + l;
                atomicAdd(vk_ptr + offset, (double)vk_tmp);
            }
        }

        // ijkl, il -> jk
        constexpr int ntil = nti*ntl;
        DataType dm_il_cache[fragi*fragl];
        if (do_k && ty_active){
            const int dm_offset = (i0+t_i*fragi)*nao + (l0+t_l*fragl);
            DataType *dm_ptr = dm + dm_offset;
            for (int i = 0; i < fragi; i++){
                for (int l = 0; l < fragl; l++){
                    dm_il_cache[l + i*fragl] = __ldg(dm_ptr + l);
                }
                dm_ptr += nao;
            }
        }
        
        if constexpr(ntil > 1) __syncthreads();
        if (do_k && ty_active){
            const int t_il = t_l + ntl * t_i;
            const int t_jk = t_j * fragj * nfk + t_k * fragk;
            const int smem_off = (t_jk + t_il * nfjk) * smem_stride;
            const int vk_offset = (j0+t_j*fragj)*nao + (k0+t_k*fragk);
            double *vk_ptr = vk + vk_offset;
#pragma unroll
            for (int j = 0; j < fragj; j++){
                for (int k = 0; k < fragk; k++){
                    DataType vk_jk = zero;
                    int integral_off = j * tstride_j + k * tstride_k;
                    for (int i = 0; i < fragi; i++){
                        for (int l = 0; l < fragl; l++){
                            vk_jk += integral_frag[integral_off + l*tstride_l] * dm_il_cache[l + i*fragl];
                        }
                        integral_off += tstride_i;
                    }
                    if constexpr(ntil > 1){
                        const int jk = j * nfk + k;
                        smem[jk * smem_stride + smem_off] = vk_jk;
                    } else {
                        const int offset = j*nao + k;
                        atomicAdd(vk_ptr + offset, (double)vk_jk);
                    }
                }
            }
        }
        if constexpr(do_k && ntil > 1){
            __syncthreads();
            const int vk_offset = j0*nao + k0;
            double *vk_ptr = vk + vk_offset;
            constexpr int stride = nfjk * smem_stride;
            for (int jk = ty; jk < nfjk; jk += nthreads_per_sq){
                DataType vk_tmp = zero;
                const int off = jk * smem_stride;
                for (int m = 0; m < ntil; m++){
                    vk_tmp += smem[off + m*stride];
                }
                const int k = jk % nfk;
                const int j = jk / nfk;
                const int offset = j*nao + k;
                atomicAdd(vk_ptr + offset, (double)vk_tmp);
            }
        }

        // ijkl, ik -> jl
        constexpr int ntik = nti*ntk;
        DataType vk_jl[fragj*fragl] = {zero};
        if (do_k && ty_active){
            const int dm_offset = (i0+t_i*fragi)*nao + (k0+t_k*fragk);
            DataType *dm_ptr = dm + dm_offset;
#pragma unroll
            for (int i = 0; i < fragi; i++){
                for (int k = 0; k < fragk; k++){
                    const int offset = i*nao + k;
                    DataType dm_ik = __ldg(dm_ptr + offset);
                    int integral_off = i * tstride_i + k * tstride_k;
                    for (int j = 0; j < fragj; j++){
                        for (int l = 0; l < fragl; l++){
                            vk_jl[l + j*fragl] += integral_frag[integral_off + l*tstride_l] * dm_ik;
                        }
                        integral_off += tstride_j;
                    }
                }
            }
        }

        if constexpr(ntik > 1) __syncthreads();
        if (do_k && ty_active){
            const int t_ik = t_i + nti * t_k;
            const int t_jl = t_j * fragj * nfl + t_l * fragl;
            const int smem_off = (t_jl + t_ik * nfjl) * smem_stride;
            const int vk_offset = (j0+t_j*fragj)*nao + (l0+t_l*fragl);
            double *vk_ptr = vk + vk_offset;
            for (int j = 0; j < fragj; j++){
                for (int l = 0; l < fragl; l++){
                    if constexpr(ntik > 1){
                        const int jl = j * nfl + l;
                        smem[jl * smem_stride + smem_off] = vk_jl[l + j*fragl];
                    } else {
                        const int offset = j*nao + l;
                        atomicAdd(vk_ptr + offset, (double)vk_jl[l + j*fragl]);
                    }
                }
            }
        }

        if constexpr(do_k && ntik > 1){
            __syncthreads();
            const int vk_offset = j0*nao + l0;
            double *vk_ptr = vk + vk_offset;
            constexpr int stride = nfjl * smem_stride;
            for (int jl = ty; jl < nfjl; jl+=nthreads_per_sq){
                DataType vk_tmp = zero;
                const int off = jl * smem_stride;
                for (int m = 0; m < ntik; m++){
                    vk_tmp += smem[off + m*stride];
                }
                const int l = jl % nfl;
                const int j = jl / nfl;
                const int offset = j*nao + l;
                atomicAdd(vk_ptr + offset, (double)vk_tmp);
            }
        }
        
        const int nao2 = nao * nao;
        dm += nao2;
        if constexpr(do_j) vj += nao2;
        if constexpr(do_k) vk += nao2;
    }
}
