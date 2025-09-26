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


template <int LIT, int LJT, int LCT, int orderi, int orderj> __device__
void type2_cart_kernel(double* __restrict__ gctr,
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

    constexpr int LI1 = LIT+1;
    constexpr int LJ1 = LJT+1;
    constexpr int LIC1 = LIT+LCT+1;
    constexpr int LJC1 = LJT+LCT+1;
    constexpr int LCC1 = (2*LCT+1);

    constexpr int BLKI = (LIC1+1)/2 * LCC1;
    constexpr int BLKJ = (LJC1+1)/2 * LCC1;

    // Use static shared memory - sizes must match the template parameters used in function calls
    __shared__ double omegai[LI1*(LI1+1)*(LI1+2)/6 * BLKI];
    __shared__ double omegaj[LJ1*(LJ1+1)*(LJ1+2)/6 * BLKJ];
    __shared__ double rad_all[(LIT+LJT+1) * LIC1 * LJC1];

    type2_facs_omega<LIT>(omegai, rca);
    type2_facs_omega<LJT>(omegaj, rcb);
    __syncthreads();

    set_shared_memory(rad_all, (LIT+LJT+1)*LIC1*LJC1);

    double radi[LIC1];
    double radj[LJC1];
    const double dca = norm3d(rca[0], rca[1], rca[2]);
    const double dcb = norm3d(rcb[0], rcb[1], rcb[2]);

    // Coefficient pointers for primitive (c,e) pairs of each shell
    const DataType2* cei = coeff_exp + ish * prim_stride;
    const DataType2* cej = coeff_exp + jsh * prim_stride;
    type2_facs_rad<orderi, LIT+LCT, NPI>(radi, dca, cei);
    type2_facs_rad<orderj, LJT+LCT, NPJ>(radj, dcb, cej);

    double ur = 0.0;
    // Each ECP shell has multiple powers and primitive basis
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }

    double root = 0.0;
    if (threadIdx.x < NGAUSS){
        root = r128[threadIdx.x];
    }
    for (int p = 0; p <= LIT+LJT; p++){
        double *prad = rad_all + p*LIC1*LJC1;
        for (int i = 0; i <= LIT+LCT; i++){
        for (int j = 0; j <= LJT+LCT; j++){
            block_reduce(radi[i]*radj[j]*ur, prad+i*LJC1+j);
        }}
        ur *= root;
    }
    __syncthreads();

    constexpr int nfi = (LIT+1) * (LIT+2) / 2;
    constexpr int nfj = (LJT+1) * (LJT+2) / 2;

    // Need dedicated shared memory arrays for angi and angj like the working kernel
    __shared__ double angi[LI1*nfi*LIC1];
    __shared__ double angj[LJ1*nfj*LJC1];

    const double fac = 16.0 * M_PI * M_PI;

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        gctr[ij] = 0.0;
    }
    __syncthreads();

    // (k+l)pq,kimp,ljmq->ij
    for (int m = 0; m < LCC1; m++){
        type2_ang<LIT>(angi, rca, omegai+m);
        type2_ang<LJT>(angj, rcb, omegaj+m);
        __syncthreads();
        for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
            const int i = ij%nfi;
            const int j = ij/nfi;
            double s = 0;
            for (int k = 0; k <= LIT; k++){
            for (int l = 0; l <= LJT; l++){
                double *pangi = angi + k*nfi*LIC1 + i*LIC1;
                double *pangj = angj + l*nfj*LJC1 + j*LJC1;
                double *prad = rad_all + (k+l)*LIC1*LJC1;

                double reg_angi[LIC1];
                double reg_angj[LJC1];
                for (int p = 0; p < LIC1; p++){reg_angi[p] = pangi[p];}
                for (int q = 0; q < LJC1; q++){reg_angj[q] = pangj[q];}
                for (int p = 0; p < LIC1; p++){
                for (int q = 0; q < LJC1; q++){
                    s += prad[p*LJC1+q] * reg_angi[p] * reg_angj[q];
                }}
            }}
            gctr[ij] += fac*s;
        }
        __syncthreads();
    }
}

extern "C" __global__
void type2_cart_ip1(double* __restrict__ gctr,
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

    constexpr int orderi = 1;
    constexpr int orderj = 0;
    constexpr int NFI_MAX = (LI+orderi+1)*(LI+orderi+2)/2;
    constexpr int NFJ_MAX = (LJ+orderj+1)*(LJ+orderj+2)/2;
    __shared__ double buf[NFI_MAX*NFJ_MAX];
    // Use LI+1 for orderi=1 stage
    type2_cart_kernel<LI+1, LJ, LC, 1, 0>(buf, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env);
    __syncthreads();
    _li_down<LI, LJ>(gctr_smem, buf);
    if constexpr (LI > 0){
        // Companion LI-1 for orderi=0 stage
        type2_cart_kernel<LI-1, LJ, LC, 0, 0>(buf, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env);
        __syncthreads();
        _li_up<LI, LJ>(gctr_smem, buf);
    }

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij%nfi;
        const int j = ij/nfi;
        double *gx = gctr;
        double *gy = gctr +   nao*nao;
        double *gz = gctr + 2*nao*nao;
        atomicAdd(gx + i*nao + j, gctr_smem[ij]);
        atomicAdd(gy + i*nao + j, gctr_smem[ij+nfi*nfj]);
        atomicAdd(gz + i*nao + j, gctr_smem[ij+2*nfi*nfj]);
    }
    return;
}


extern "C" __global__
void type2_cart_ipipv(double* __restrict__ gctr,
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
    // LI+2 for orderi=2
    type2_cart_kernel<LI+2, LJ, LC, 2, 0>(buf1, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env);
    __syncthreads();

    constexpr int nfi1_max = (LI+2)*(LI+3)/2;
    __shared__ double buf[3*nfi1_max*nfj_max];
    set_shared_memory(buf, 3*nfi1_max*nfj_max);
    _li_down<LI+1, LJ>(buf, buf1);
    _li_down_and_write<LI, LJ>(gctr, buf, nao);

    // LI for orderi=1
    type2_cart_kernel<LI, LJ, LC, 1, 0>(buf1, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env);
    __syncthreads();
    set_shared_memory(buf, 3*nfi1_max*nfj_max);
    _li_up<LI+1, LJ>(buf, buf1);
    _li_down_and_write<LI, LJ>(gctr, buf, nao);

    if constexpr (LI > 0){
        set_shared_memory(buf, 3*nfi1_max*nfj_max);
        _li_down<LI-1, LJ>(buf, buf1);
        _li_up_and_write<LI, LJ>(gctr, buf, nao);
        if constexpr (LI > 1){
            // LI-2 for orderi=0 companion
            type2_cart_kernel<LI-2, LJ, LC, 0, 0>(buf1, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env);
            __syncthreads();
            set_shared_memory(buf, 3*nfi1_max*nfj_max);
            _li_up<LI-1, LJ>(buf, buf1);
            _li_up_and_write<LI, LJ>(gctr, buf, nao);
        }
    }
    return;
}

extern "C" __global__
void type2_cart_ipvip(double* __restrict__ gctr,
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
    // LI+1, LJ+1 for mixed order
    type2_cart_kernel<LI+1, LJ+1, LC, 1, 1>(buf1, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env);
    __syncthreads();

    constexpr int nfi_max = (LI+1)*(LI+2)/2;
    __shared__ double buf[3*nfi_max*nfj1_max];
    set_shared_memory(buf, 3*nfi_max*nfj1_max);
    _li_down<LI, LJ+1>(buf, buf1);
    _lj_down_and_write<LI, LJ>(gctr, buf, nao);

    if constexpr (LI > 0){
        // LI-1, LJ+1 companion
        type2_cart_kernel<LI-1, LJ+1, LC, 0, 1>(buf1, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env);
        __syncthreads();
        set_shared_memory(buf, 3*nfi_max*nfj1_max);
        _li_up<LI, LJ+1>(buf, buf1);
        _lj_down_and_write<LI, LJ>(gctr, buf, nao);
    }

    if constexpr (LJ > 0){
        // LI+1, LJ-1 branch
        type2_cart_kernel<LI+1, LJ-1, LC, 1, 0>(buf1, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env);
        __syncthreads();
        set_shared_memory(buf, 3*nfi_max*nfj1_max);
        _li_down<LI, LJ-1>(buf, buf1);
        _lj_up_and_write<LI, LJ>(gctr, buf, nao);
        if constexpr (LI > 0){
            // LI-1, LJ-1 companion
            type2_cart_kernel<LI-1, LJ-1, LC, 0, 0>(buf1, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env);
            __syncthreads();
            set_shared_memory(buf, 3*nfi_max*nfj1_max);
            _li_up<LI, LJ-1>(buf, buf1);
            _lj_up_and_write<LI, LJ>(gctr, buf, nao);
        }
    }
    return;
}
