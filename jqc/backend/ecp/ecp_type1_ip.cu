/*
 * Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


template <int LIT, int LJT, int orderi, int orderj> __device__
void type1_cart_kernel(double* __restrict__ gctr,
                const int ish, const int jsh, const int ksh,
                const int* __restrict__ ecpbas, const int* __restrict__ ecploc,
                const DataType4* __restrict__ coords,
                const DataType2* __restrict__ coeff_exp,
                const int* __restrict__ atm, const double* __restrict__ env)
{
    // Coordinates from basis layout with explicit COORD_STRIDE handling.
    // Treat coords as scalar array to avoid relying on struct alignment.
    const DataType *coords_scalar = reinterpret_cast<const DataType*>(coords);
    const DataType *ri = coords_scalar + ish * COORD_STRIDE;
    const DataType *rj = coords_scalar + jsh * COORD_STRIDE;

    const int atm_id = ecpbas[ATOM_OF+ecploc[ksh]*BAS_SLOTS];
    const double *rc = env + atm[PTR_COORD+atm_id*ATM_SLOTS];

    double rca[3], rcb[3];
    rca[0] = rc[0] - ri[0];
    rca[1] = rc[1] - ri[1];
    rca[2] = rc[2] - ri[2];
    rcb[0] = rc[0] - rj[0];
    rcb[1] = rc[1] - rj[1];
    rcb[2] = rc[2] - rj[2];
    const double r2ca = rca[0]*rca[0] + rca[1]*rca[1] + rca[2]*rca[2];
    const double r2cb = rcb[0]*rcb[0] + rcb[1]*rcb[1] + rcb[2]*rcb[2];

    double ur = 0.0;
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }

    constexpr int LIJ1 = LIT+LJT+1;
    constexpr int LIJ3 = LIJ1*LIJ1*LIJ1;

    // Use static shared memory like the working type1 kernel
    __shared__ double rad_ang[LIJ3];
    __shared__ double rad_all[LIJ1*LIJ1];

    for (int i = threadIdx.x; i < LIJ3; i+=blockDim.x) {
        rad_ang[i] = 0;
    }
    __syncthreads();
    const double fac = 16.0 * M_PI * M_PI;

    // Coefficient pointers for primitive (c,e) pairs of each shell
    const DataType2* cei = coeff_exp + ish * prim_stride;
    const DataType2* cej = coeff_exp + jsh * prim_stride;

    for (int ip = 0; ip < NPI; ip++){
        for (int jp = 0; jp < NPJ; jp++){
            double rij[3];
            double ai_prim = cei[ip].e;
            double aj_prim = cej[jp].e;
            rij[0] = ai_prim * rca[0] + aj_prim * rcb[0];
            rij[1] = ai_prim * rca[1] + aj_prim * rcb[1];
            rij[2] = ai_prim * rca[2] + aj_prim * rcb[2];
            const double k = 2.0 * norm3d(rij[0], rij[1], rij[2]);
            const double aij = ai_prim + aj_prim;
            type1_rad_part<LIT+LJT>(rad_all, k, aij, ur);

            const double eij = exp(-ai_prim*r2ca - aj_prim*r2cb);
            const double eaij = eij * pow(-2.0*ai_prim, orderi) * pow(-2.0*aj_prim, orderj);
            const double ceij = eaij * cei[ip].c * cej[jp].c;
            type1_rad_ang<LIT+LJT>(rad_ang, rij, rad_all, fac*ceij);
            __syncthreads();
        }
    }

    //constexpr int NFI_MAX = (LIT+orderi+1)*(LIT+orderi+2)/2;
    //constexpr int NFJ_MAX = (LJT+orderj+1)*(LJT+orderj+2)/2;
    constexpr int nfi = (LIT+1) * (LIT+2) / 2;
    constexpr int nfj = (LJT+1) * (LJT+2) / 2;
    double fi[3*nfi];
    double fj[3*nfj];
    cache_fac<LIT>(fi, rca);
    cache_fac<LJT>(fj, rcb);

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int mi = ij%nfi;
        const int mj = ij/nfi;

        const int iy = _cart_pow_y[mi];
        const int iz = _cart_pow_z[mi];
        const int ix = LIT - iy - iz;

        double* fx_i = fi + (ix+1)*ix/2;
        double* fy_i = fi + (iy+1)*iy/2 + nfi;
        double* fz_i = fi + (iz+1)*iz/2 + 2*nfi;

        const int jy = _cart_pow_y[mj];
        const int jz = _cart_pow_z[mj];
        const int jx = LJT - jy - jz;
        double* fx_j = fj + (jx+1)*jx/2;
        double* fy_j = fj + (jy+1)*jy/2 + nfj;
        double* fz_j = fj + (jz+1)*jz/2 + 2*nfj;

        // cache ifac and jfac in register
        double tmp = 0.0;
        for (int i1 = 0; i1 <= ix; i1++){
        for (int i2 = 0; i2 <= iy; i2++){
        for (int i3 = 0; i3 <= iz; i3++){
            double ifac = fx_i[i1] * fy_i[i2] * fz_i[i3];
            for (int j1 = 0; j1 <= jx; j1++){
            for (int j2 = 0; j2 <= jy; j2++){
            for (int j3 = 0; j3 <= jz; j3++){
                double jfac = fx_j[j1] * fy_j[j2] * fz_j[j3];
                const int ijr = (i1+j1)*LIJ1*LIJ1 + (i2+j2)*LIJ1 + (i3+j3);
                tmp += ifac * jfac * rad_ang[ijr];
            }}}
        }}}
        gctr[ij] = tmp;
    }
    return;
}


extern "C" __global__
void type1_cart_ip1(double* __restrict__ gctr,
                const int* __restrict__ ao_loc, const int nao,
                const int* __restrict__ tasks, const int ntasks,
                const int* __restrict__ ecpbas, const int* __restrict__ ecploc,
                const DataType4* __restrict__ coords,
                const DataType2* __restrict__ coeff_exp,
                const int* __restrict__ atm, const double* __restrict__ env)
{
    const int task_id = blockIdx.x;
    if (task_id >= ntasks){
        return;
    }

    const int ish = tasks[task_id];
    const int jsh = tasks[task_id + ntasks];
    const int ksh = tasks[task_id + 2*ntasks];
    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];
    const int ecp_id = ecpbas[ECP_ATOM_ID+ecploc[ksh]*BAS_SLOTS];
    gctr += 3*ecp_id*nao*nao + ioff*nao + joff;
    
    constexpr int nfi = (LI+1) * (LI+2) / 2;
    constexpr int nfj = (LJ+1) * (LJ+2) / 2;
    __shared__ double gctr_smem[nfi*nfj*3];
    for (int ij = threadIdx.x; ij < nfi*nfj*3; ij+=blockDim.x){
        gctr_smem[ij] = 0.0;
    }
    __syncthreads();

    constexpr int nfi_max = (LI+2)*(LI+3)/2;
    constexpr int nfj_max = (LJ+1)*(LJ+2)/2;
    __shared__ double buf[nfi_max*nfj_max];

    // Accumulate derivative contributions with respect to AO i.
    // j-side contributions are accumulated via (j,i) tasks in host tasking (full tasks).
    // Use LI+1 for orderi=1 to match unrolled cache pattern
    type1_cart_kernel<LI+1, LJ, 1, 0>(buf, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env);
    __syncthreads();
    _li_down<LI, LJ>(gctr_smem, buf);

    if constexpr (LI > 0){
        // Use LI-1 for orderi=0 companion
        type1_cart_kernel<LI-1, LJ, 0, 0>(buf, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env);
        __syncthreads();
        _li_up<LI, LJ>(gctr_smem, buf);
    }

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij%nfi;
        const int j = ij/nfi;
        double *gx = gctr;
        double *gy = gctr +   nao*nao;
        double *gz = gctr + 2*nao*nao;
        atomicAdd(gx+i*nao+j, gctr_smem[ij]);
        atomicAdd(gy+i*nao+j, gctr_smem[ij+nfi*nfj]);
        atomicAdd(gz+i*nao+j, gctr_smem[ij+2*nfi*nfj]);
    }
    return;
}

extern "C" __global__
void type1_cart_ipipv(double* __restrict__ gctr,
                const int* __restrict__ ao_loc, const int nao,
                const int* __restrict__ tasks, const int ntasks,
                const int* __restrict__ ecpbas, const int* __restrict__ ecploc,
                const DataType4* __restrict__ coords,
                const DataType2* __restrict__ coeff_exp,
                const int* __restrict__ atm, const double* __restrict__ env)
{
    const int task_id = blockIdx.x;
    if (task_id >= ntasks){
        return;
    }

    const int ish = tasks[task_id];
    const int jsh = tasks[task_id + ntasks];
    const int ksh = tasks[task_id + 2*ntasks];

    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];
    const int ecp_id = ecpbas[ECP_ATOM_ID+ecploc[ksh]*BAS_SLOTS];
    gctr += ioff*nao + joff + 9*ecp_id*nao*nao;

    constexpr int nfi2_max = (LI+3)*(LI+4)/2;
    constexpr int nfj_max = (LJ+1)*(LJ+2)/2;
    __shared__ double buf1[nfi2_max*nfj_max];
    // Use LI+2 for orderi=2 stage
    type1_cart_kernel<LI+2, LJ, 2, 0>(buf1, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env);
    __syncthreads();

    constexpr int nfi1_max = (LI+2)*(LI+3)/2;
    __shared__ double buf[3*nfi1_max*nfj_max];
    for (int i = threadIdx.x; i < 3*nfi1_max*nfj_max; i+=blockDim.x){
        buf[i] = 0.0;
    }
    __syncthreads();
    _li_down<LI+1, LJ>(buf, buf1);

    // Then LI for orderi=1 stage
    type1_cart_kernel<LI, LJ, 1, 0>(buf1, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env);
    __syncthreads();
    _li_up<LI+1, LJ>(buf, buf1);
    _li_down_and_write<LI, LJ>(gctr, buf, nao);

    if constexpr (LI > 0){
        for (int i = threadIdx.x; i < 3*nfi1_max*nfj_max; i+=blockDim.x){
            buf[i] = 0.0;
        }
        __syncthreads();
        _li_down<LI-1, LJ>(buf, buf1);
        if constexpr (LI > 1){
            // Final companion LI-2 for orderi=0
            type1_cart_kernel<LI-2, LJ, 0, 0>(buf1, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env);
            __syncthreads();
            _li_up<LI-1, LJ>(buf, buf1);
        }
        _li_up_and_write<LI, LJ>(gctr, buf, nao);
    }
    return;
}

extern "C" __global__
void type1_cart_ipvip(double* __restrict__ gctr,
                const int* __restrict__ ao_loc, const int nao,
                const int* __restrict__ tasks, const int ntasks,
                const int* __restrict__ ecpbas, const int* __restrict__ ecploc,
                const DataType4* __restrict__ coords,
                const DataType2* __restrict__ coeff_exp,
                const int* __restrict__ atm, const double* __restrict__ env)
{
    const int task_id = blockIdx.x;
    if (task_id >= ntasks){
        return;
    }

    const int ish = tasks[task_id];
    const int jsh = tasks[task_id + ntasks];
    const int ksh = tasks[task_id + 2*ntasks];

    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];
    const int ecp_id = ecpbas[ECP_ATOM_ID+ecploc[ksh]*BAS_SLOTS];
    gctr += ioff*nao + joff + 9*ecp_id*nao*nao;

    constexpr int nfi1_max = (LI+2)*(LI+3)/2;
    constexpr int nfj1_max = (LJ+2)*(LJ+3)/2;
    __shared__ double buf1[nfi1_max*nfj1_max];
    // Start with LI+1, LJ+1 for mixed order
    type1_cart_kernel<LI+1, LJ+1, 1, 1>(buf1, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env);
    __syncthreads();

    constexpr int nfi_max = (LI+1)*(LI+2)/2;
    __shared__ double buf[3*nfi_max*nfj1_max];
    for (int i = threadIdx.x; i < 3*nfi_max*nfj1_max; i+=blockDim.x){
        buf[i] = 0.0;
    }
    __syncthreads();
    _li_down<LI, LJ+1>(buf, buf1);
    if constexpr (LI > 0){
        // Companion LI-1, LJ+1 for orderi=0, orderj=1
        type1_cart_kernel<LI-1, LJ+1, 0, 1>(buf1, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env);
        __syncthreads();
        _li_up<LI, LJ+1>(buf, buf1);
    }
    _lj_down_and_write<LI, LJ>(gctr, buf, nao);

    if constexpr (LJ > 0){
        for (int i = threadIdx.x; i < 3*nfi_max*nfj1_max; i+=blockDim.x){
            buf[i] = 0.0;
        }
        __syncthreads();
        // LI+1, LJ-1 for orderi=1, orderj=0 branch
        type1_cart_kernel<LI+1, LJ-1, 1, 0>(buf1, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env);
        __syncthreads();
        _li_down<LI, LJ-1>(buf, buf1);
        if constexpr (LI > 0){
            // LI-1, LJ-1 for orderi=0, orderj=0 companion
            type1_cart_kernel<LI-1, LJ-1, 0, 0>(buf1, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env);
            __syncthreads();
            _li_up<LI, LJ-1>(buf, buf1);
        }
        _lj_up_and_write<LI, LJ>(gctr, buf, nao);
    }
    return;
}
