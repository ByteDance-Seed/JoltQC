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

// BASIS_STRIDE is the total stride: [coords (4), ce (BASIS_STRIDE-4)]
constexpr int basis_stride = BASIS_STRIDE;

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
void rys_1qnt_vjk(const int nao,
        const DataType* __restrict__ basis_data,
        DataType* __restrict__ dm,
        double* __restrict__ vj,
        double* __restrict__ vk,
        const DataType omega,
        const ushort4* __restrict__ shl_quartet_idx,
        const int ntasks)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int task_id = blockIdx.x * blockDim.y + threadIdx.y;

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

    const int t_i = (nti > 0 || tx < nt_active) ? (tx % nti) : 0;
    const int t_j = (ntj > 0 || tx < nt_active) ? (tx / nti % ntj) : 0;
    const int t_k = (ntk > 0 || tx < nt_active) ? (tx / (nti*ntj) % ntk) : 0;
    const int t_l = (ntl > 0 || tx < nt_active) ? (tx / (nti*ntj*ntk)) : 0;

    constexpr int gx_stride = smem_stride;
    constexpr int g_stride = 3 * gx_stride;

    // shape of g, (gsize, 3, nsq_per_block)
    constexpr int stride_i = g_stride;
    constexpr int stride_j = stride_i * (li+1);
    constexpr int stride_k = stride_j * (lj+1);
    constexpr int stride_l = stride_k * (lk+1);

    // Dynamic shared memory buffer
    extern __shared__ DataType shared_memory[];

    // Always cache kl indices for universal performance
    int kl_idx[fragk*fragl];
    int kl_idy[fragk*fragl]; 
    int kl_idz[fragk*fragl];
    
    // Pre-compute kl indices for all fragment combinations
    for (int reg_k = 0; reg_k < fragk; reg_k++){
        const int k = t_k * fragk + reg_k;
        for (int reg_l = 0; reg_l < fragl; reg_l++){
            const int l = t_l * fragl + reg_l;
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

    const bool active = (task_id < ntasks);
    ushort4 sq = {0,0,0,0};
    if (active) {
        sq = shl_quartet_idx[task_id];
    }
    
    const int ish = sq.x;
    const int jsh = sq.y;
    const int ksh = sq.z;
    const int lsh = sq.w;
     
    DataType fac_sym = (active && tx < nt_active) ? PI_FAC : zero;
    fac_sym *= (ish == jsh) ? half : one;
    fac_sym *= (ksh == lsh) ? half : one;
    //fac_sym *= (ish*nbas+jsh == ksh*nbas+lsh) ? half : one;
    fac_sym *= ((ish == ksh) && (jsh == lsh)) ? half : one;

    // Compute base addresses for all shells (allows better instruction-level parallelism)
    const DataType* base_i = basis_data + ish * basis_stride;
    const DataType* base_j = basis_data + jsh * basis_stride;
    const DataType* base_k = basis_data + ksh * basis_stride;
    const DataType* base_l = basis_data + lsh * basis_stride;

    // Load coords from base addresses
    DataType4 ri = *reinterpret_cast<const DataType4*>(base_i);
    DataType4 rj = *reinterpret_cast<const DataType4*>(base_j);
    DataType4 rk = *reinterpret_cast<const DataType4*>(base_k);
    DataType4 rl = *reinterpret_cast<const DataType4*>(base_l);

    const DataType rjri0 = rj.x - ri.x;
    const DataType rjri1 = rj.y - ri.y;
    const DataType rjri2 = rj.z - ri.z;
    const DataType rr_ij = rjri0*rjri0 + rjri1*rjri1 + rjri2*rjri2;
    const DataType rlrk0 = rl.x - rk.x;
    const DataType rlrk1 = rl.y - rk.y;
    const DataType rlrk2 = rl.z - rk.z;
    const DataType rr_kl = rlrk0*rlrk0 + rlrk1*rlrk1 + rlrk2*rlrk2;

    // Estimate register usage for caching cei, cej, cicj and inv_aij
    // Note: g array is in shared memory, not registers, so only count integral_frag + cache arrays
    // DataType can be float (1 register) or double (2 registers)
    constexpr int reg_per_datatype = sizeof(DataType) / 4; // 4 bytes per 32-bit register
    constexpr int reg_g = reg_per_datatype * 3 * frag_size;
    constexpr int reg_aij_ceij = reg_per_datatype * 2 * npi * npj;
    constexpr int reg_cei_cej = reg_per_datatype * 2 * (npi + npj);
    constexpr int reg_integral = reg_per_datatype * frag_size;
    constexpr int estimated_registers = reg_aij_ceij + reg_cei_cej + reg_integral;
    constexpr bool use_cache = (estimated_registers <= 256);

    // Cache cei and cej if register usage is reasonable
    DataType2 reg_cei[npi], reg_cej[npj];
    // Load ce data from packed basis_data
    const DataType2* cei_ptr = load_ce_ptr(basis_data, ish);
    const DataType2* cej_ptr = load_ce_ptr(basis_data, jsh);
    const DataType2* cek_ptr = load_ce_ptr(basis_data, ksh);
    const DataType2* cel_ptr = load_ce_ptr(basis_data, lsh);

    for (int ip = 0; ip < npi; ip++){
        reg_cei[ip] = cei_ptr[ip];
    }
    for (int jp = 0; jp < npj; jp++){
        reg_cej[jp] = cej_ptr[jp];
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

    DataType integral_frag[frag_size] = {zero};
    for (int kp = 0; kp < npk; kp++)
    for (int lp = 0; lp < npl; lp++){
        DataType2 cek = cek_ptr[kp];
        DataType2 cel = cel_ptr[lp];
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
                DataType2 cei = cei_ptr[ip];
                DataType2 cej = cej_ptr[jp];
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
            
            DataType rjri_x = (tx == 0 ? rjri0 : (tx == 1 ? rjri1 : rjri2)); 
            DataType Rpq_x =  (tx == 0 ? (xij-xkl) : (tx == 1 ? (yij-ykl) : (zij-zkl)));
            DataType rlrk_x = (tx == 0 ? rlrk0 : (tx == 1 ? rlrk1 : rlrk2));

            DataType *rw = shared_memory + ty;
            DataType *g = shared_memory + nroots * 2 * gx_stride + ty; 

            rys_roots(rr, rw, tx, gx_stride, theta, omega);
            
            DataType g0xyz;
            if (tx == 0) g0xyz = ckcl; 
            if (tx == 1) g0xyz = cicj * inv_aij * inv_akl * sqrt(inv_aijkl);
            
            __syncthreads();
            for (int irys = 0; irys < nroots; irys++){
                DataType rt_aa;
                g0xyz = (tx == 2) ? rw[(irys*2+1) * gx_stride] : g0xyz;
                if (tx < 3){
                    const DataType rt = rw[(irys*2)*gx_stride];
                    rt_aa = rt * inv_aijkl;
                }
                __syncthreads();
                if (tx < 3) g[tx*gx_stride] = g0xyz;

                // TRR
                //for i in range(lij):
                //    trr(i+1,0) = c0 * trr(i,0) + i*b10 * trr(i-1,0)
                //for k in range(lkl):
                //    for i in range(lij+1):
                //        trr(i,k+1) = c0p * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                constexpr int lij = li + lj;
                if constexpr (lij > 0) {
                    if (tx < 3){
                        const DataType rt_aij = rt_aa * akl;
                        const DataType b10 = half * inv_aij * (one - rt_aij);

                        const int _ix = tx;
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
                    if (tx < 3){
                        const DataType rt_akl = rt_aa * aij;
                        const DataType b00 = half * rt_aa;
                        const DataType b01 = half * inv_akl * (one - rt_akl);

                        const int _ix = tx;
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
                
                const int _ix = tx;
                DataType *gx = g + _ix * gx_stride;
                // hrr
                // g(i,j+1) = rirj * g(i,j) +  g(i+1,j)
                // g(...,k,l+1) = rkrl * g(...,k,l) + g(...,k+1,l)
                if constexpr (lj > 0) {
                    constexpr int stride_j_i = stride_j - stride_i;
                    if (tx < 3){
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
                                    gx[ijkl + stride_j] = s1x - rjri_x * s0x;  // leave form; compiler fuses to FMA
                                    s1x = s0x;
                                }
                            }
                        }
                    }
                }

                if constexpr (ll > 0) {
                    constexpr int li1xlj1 = (li+1)*(lj+1);
                    constexpr int stride_l_k = stride_l - stride_k;
                    if (tx < 3){
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
                
                // Pre-compute base indices to reduce repeated calculations
                const int base_i = t_i * fragi;
                const int base_j = t_j * fragj;

#pragma unroll
                for (int reg_i = 0; reg_i < fragi; reg_i++){
                    const int i = base_i + reg_i;
                    const int base_integral_i = reg_i * tstride_i;
                    for (int reg_j = 0; reg_j < fragj; reg_j++){
                        const int j = base_j + reg_j;
                        const uint32_t addr_ij = j_idx[j] + i_idx[i];
                        const uint32_t ij_x    =  addr_ij        & 0x3FF;      // 10 low-order bits
                        const uint32_t ij_y    = (addr_ij >> 10) & 0x3FF;      // next 10 bits
                        const uint32_t ij_z    = (addr_ij >> 20) & 0x3FF;      // next 10 bits
                        const uint32_t ij_x_stride = ij_x * g_stride;
                        const uint32_t ij_y_stride = ij_y * g_stride;
                        const uint32_t ij_z_stride = ij_z * g_stride;

                        int integral_off = base_integral_i + reg_j * tstride_j;
                        for (int reg_k = 0; reg_k < fragk; reg_k++){
                            for (int reg_l = 0; reg_l < fragl; reg_l++){
                                // Use pre-computed cached kl indices
                                const int kl = reg_l + reg_k * fragl;
                                const int addrx = ij_x_stride + kl_idx[kl];
                                const int addry = ij_y_stride + kl_idy[kl];
                                const int addrz = ij_z_stride + kl_idz[kl];
                                integral_frag[integral_off + reg_l*tstride_l] += g[addrx] * g[addry] * g[addrz];
                            }
                            integral_off += tstride_k;
                        }
                    }
                }
            }
        }
    }

    // ao_loc is stored in the w field of coords
    const int i0 = (int)ri.w;
    const int j0 = (int)rj.w;
    const int k0 = (int)rk.w;
    const int l0 = (int)rl.w;

    DataType *smem = shared_memory + ty;
    //const bool ty_active = (ty < nt_active);
    for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
        // ijkl, ij -> kl
        constexpr int ntij = nti*ntj;
        DataType vj_lk[fragk*fragl] = {zero};
        if (do_j && tx < nt_active){
            const int base_dm_i = i0 + t_i*fragi;
            const int base_dm_j = j0 + t_j*fragj;
            const int dm_offset = base_dm_i + base_dm_j*nao;
            DataType *dm_ptr = dm + dm_offset;
#pragma unroll
            for (int i = 0; i < fragi; i++){
                const int base_off_i = i * tstride_i;
                for (int j = 0; j < fragj; j++){
                    const int offset = i + j*nao;
                    DataType dm_ij = __ldg(dm_ptr + offset);
                    int off = base_off_i + j * tstride_j;
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
        if (do_j && tx < nt_active){
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
            const int vj_offset = l0*nao + k0;
            double *vj_ptr = vj + vj_offset;
            constexpr int stride = nfkl * smem_stride;
            __syncthreads();
            for (int kl = tx; kl < nfkl; kl += nthreads_per_sq){
                DataType vj_tmp = zero;
                const int off = kl * smem_stride;
                for (int m = 0; m < ntij; m++){
                    vj_tmp += smem[off + m*stride];
                }
                // Replace division/modulo with bit operations when nfk is power of 2
                // Otherwise keep original for correctness
                const int l = kl / nfk;
                const int k = kl - l * nfk;  // Replace modulo with subtraction
                const int offset = l*nao + k;
                atomicAdd(vj_ptr + offset, (double)vj_tmp);
            }
        }
        
        // ijkl, kl -> ij
        constexpr int ntkl = ntk*ntl;
        DataType dm_kl_cache[fragk*fragl];
        if (do_j && tx < nt_active){
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
        if (do_j && tx < nt_active){
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
            const int vj_offset = j0*nao + i0;
            double *vj_ptr = vj + vj_offset;
            constexpr int stride = nfij * smem_stride;
            __syncthreads();
            for (int ij = tx; ij < nfij; ij += nthreads_per_sq){
                DataType vj_tmp = zero;
                const int off = ij * smem_stride;
                for (int m = 0; m < ntkl; m++){
                    vj_tmp += smem[off + m*stride];
                }
                const int j = ij / nfi;
                const int i = ij - j * nfi;  // Replace modulo with subtraction
                const int offset = j*nao + i;
                atomicAdd(vj_ptr + offset, (double)vj_tmp);
            }
        }

        // ijkl, jl -> ik
        constexpr int ntjl = ntj*ntl;
        DataType dm_jl_cache[fragj*fragl];
        if (do_k && tx < nt_active){
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
        if (do_k && tx < nt_active){
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
            for (int ik = tx; ik < nfik; ik+=nthreads_per_sq){
                DataType vk_tmp = zero;
                const int off = ik * smem_stride;
                for (int m = 0; m < ntjl; m++){
                    vk_tmp += smem[off + m*stride];
                }
                const int i = ik / nfk;
                const int k = ik - i * nfk;  // Replace modulo with subtraction
                const int offset = i*nao + k;
                atomicAdd(vk_ptr + offset, (double)vk_tmp);
            }
        }

        // ijkl, jk -> il
        constexpr int ntjk = ntj*ntk;
        DataType dm_jk_cache[fragj*fragk];
        if (do_k && tx < nt_active){
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
        if (do_k && tx < nt_active){
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
                        int idx_k = integral_off;
                        const int cache_j = j * fragk;
                        for (int k = 0; k < fragk; k++){
                            vk_il += integral_frag[idx_k] * dm_jk_cache[cache_j + k];
                            idx_k += tstride_k;
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
            const int vk_offset = i0*nao + l0;
            double *vk_ptr = vk + vk_offset;
            __syncthreads();
            for (int il = tx; il < nfil; il += nthreads_per_sq){
                DataType vk_tmp = zero;
                constexpr int stride = nfil * smem_stride;
                const int off = il * smem_stride;
                for (int m = 0; m < ntjk; m++){
                    vk_tmp += smem[off + m*stride];
                }
                const int i = il / nfl;
                const int l = il - i * nfl;  // Replace modulo with subtraction
                const int offset = i*nao + l;
                atomicAdd(vk_ptr + offset, (double)vk_tmp);
            }
        }

        // ijkl, il -> jk
        constexpr int ntil = nti*ntl;
        DataType dm_il_cache[fragi*fragl];
        if (do_k && tx < nt_active){
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
        if (do_k && tx < nt_active){
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
                        int idx_l = integral_off;
                        const int cache_i = i * fragl;
                        for (int l = 0; l < fragl; l++){
                            vk_jk += integral_frag[idx_l] * dm_il_cache[cache_i + l];
                            idx_l += tstride_l;
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
            const int vk_offset = j0*nao + k0;
            double *vk_ptr = vk + vk_offset;
            constexpr int stride = nfjk * smem_stride;
            __syncthreads();
            for (int jk = tx; jk < nfjk; jk += nthreads_per_sq){
                DataType vk_tmp = zero;
                const int off = jk * smem_stride;
                for (int m = 0; m < ntil; m++){
                    vk_tmp += smem[off + m*stride];
                }
                const int j = jk / nfk;
                const int k = jk - j * nfk;  // Replace modulo with subtraction
                const int offset = j*nao + k;
                atomicAdd(vk_ptr + offset, (double)vk_tmp);
            }
        }

        // ijkl, ik -> jl
        constexpr int ntik = nti*ntk;
        DataType vk_jl[fragj*fragl] = {zero};
        if (do_k && tx < nt_active){
            const int dm_offset = (i0+t_i*fragi)*nao + (k0+t_k*fragk);
            DataType *dm_ptr = dm + dm_offset;
#pragma unroll
            for (int i = 0; i < fragi; i++){
                for (int k = 0; k < fragk; k++){
                    const int offset = i*nao + k;
                    const DataType dm_ik = __ldg(dm_ptr + offset);
                    int integral_off = i * tstride_i + k * tstride_k;
                    for (int j = 0; j < fragj; j++){
                        int idx_l = integral_off;
                        const int cache_j = j * fragl;
                        for (int l = 0; l < fragl; l++){
                            vk_jl[cache_j + l] += integral_frag[idx_l] * dm_ik;
                            idx_l += tstride_l;
                        }
                        integral_off += tstride_j;
                    }
                }
            }
        }

        if constexpr(ntik > 1) __syncthreads();
        if (do_k && tx < nt_active){
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
            const int vk_offset = j0*nao + l0;
            double *vk_ptr = vk + vk_offset;
            constexpr int stride = nfjl * smem_stride;
            __syncthreads();
            for (int jl = tx; jl < nfjl; jl+=nthreads_per_sq){
                DataType vk_tmp = zero;
                const int off = jl * smem_stride;
                for (int m = 0; m < ntik; m++){
                    vk_tmp += smem[off + m*stride];
                }
                const int j = jl / nfl;
                const int l = jl - j * nfl;  // Replace modulo with subtraction
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
