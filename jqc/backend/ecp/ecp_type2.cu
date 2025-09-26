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

// Utility functions for 3D vector operations
__device__ __forceinline__
static double norm3d(double x, double y, double z) {
    return sqrt(x*x + y*y + z*z);
}

__device__ __forceinline__
static double rnorm3d(double x, double y, double z) {
    return rsqrt(x*x + y*y + z*z);
}

template <int order, int LIC, int np> __device__
void type2_facs_rad(double* __restrict__ facs, const double rca, const DataType2* __restrict__ ce){
    double root = 0.0;
    if (threadIdx.x < NGAUSS){
        root = r128[threadIdx.x];
    }
    const double r = root - rca;
    const double r2 = r*r;
    for (int j = 0; j <= LIC; j++){
        facs[j] = 0.0;
    }

    for (int ip = 0; ip < np; ip++){
        const double ka = 2.0 * ce[ip].e * rca;
        const double ar2 = ce[ip].e * r2;
        
        double buf[LIC+order+1];
        if (ar2 > EXPCUTOFF + 6.0){
            for (int j = 0; j <= LIC; j++){
                buf[j] = 0.0;
            }
        } else {
            const double t1 = exp(-ar2);
            _ine<LIC>(buf, ka*root);
            for (int j = 0; j <= LIC; j++){
                buf[j] *= t1;
            }
        }
        const double c = pow(-2.0*ce[ip].e, order) * ce[ip].c;
        for (int j = 0; j <= LIC; j++){
            facs[j] += c * buf[j];
        }
    }
}

template <int L> __device__
void type2_facs_omega(double* __restrict__ omega, const double* __restrict__ r){
    double unitr[3];
    if (r[0]*r[0] + r[1]*r[1] + r[2]*r[2] < 1e-16){
        unitr[0] = 0;
        unitr[1] = 0;
        unitr[2] = 0;
    } else {
        // Follow GPU4PySCF convention for unit vector direction
        double norm_r = -rnorm3d(r[0], r[1], r[2]);
        unitr[0] = r[0] * norm_r;
        unitr[1] = r[1] * norm_r;
        unitr[2] = r[2] * norm_r;
    }

    // LC + (i+j+k) + (L + LC) needs to be even
    // When i+j+k + LC is even
    for (int n = threadIdx.x; n < (L+1)*(L+1)*(L+1); n+=blockDim.x){
        const int i = n/(L+1)/(L+1);
        const int j = n/(L+1)%(L+1);
        const int k = n%(L+1);
        if (i+j+k > L || (i+j+k+LC)%2 == 1){
            continue;
        }

        const int L_i = L-i;
        const int ioff = (L_i)*(L_i+1)*(L_i+2)/6;
        const int joff = (L_i-j)*(L_i-j+1)/2;
        constexpr int blk = (L+LC+2)/2 * (LC*2+1);
        double *pomega = omega + (ioff+joff+k)*blk;

        //for (int lmb = need_even; lmb <= L+LC; lmb+=2){
        if constexpr(L+LC >= 0)  {type2_ang_nuc_l<0>(pomega, i, j, k, unitr);  pomega+=(2*LC+1);}
        if constexpr(L+LC >= 2)  {type2_ang_nuc_l<2>(pomega, i, j, k, unitr);  pomega+=(2*LC+1);}
        if constexpr(L+LC >= 4)  {type2_ang_nuc_l<4>(pomega, i, j, k, unitr);  pomega+=(2*LC+1);}
        if constexpr(L+LC >= 6)  {type2_ang_nuc_l<6>(pomega, i, j, k, unitr);  pomega+=(2*LC+1);}
        if constexpr(L+LC >= 8)  {type2_ang_nuc_l<8>(pomega, i, j, k, unitr);  pomega+=(2*LC+1);}
        if constexpr(L+LC >= 10) {type2_ang_nuc_l<10>(pomega, i, j, k, unitr); pomega+=(2*LC+1);}
    }

    // When i+j+k + LC is odd
    for (int n = threadIdx.x; n < (L+1)*(L+1)*(L+1); n+=blockDim.x){
        const int i = n/(L+1)/(L+1);
        const int j = n/(L+1)%(L+1);
        const int k = n%(L+1);
        if (i+j+k > L || (i+j+k+LC)%2 == 0){
            continue;
        }
        const int L_i = L-i;
        const int ioff = (L_i)*(L_i+1)*(L_i+2)/6;
        const int joff = (L_i-j)*(L_i-j+1)/2;
        constexpr int blk = (L+LC+2)/2 * (LC*2+1);
        double *pomega = omega + (ioff+joff+k)*blk;

        //for (int lmb = need_even; lmb <= L+LC; lmb+=2){
        if constexpr(L+LC >= 1)  {type2_ang_nuc_l<1>(pomega, i, j, k, unitr);  pomega+=(2*LC+1);}
        if constexpr(L+LC >= 3)  {type2_ang_nuc_l<3>(pomega, i, j, k, unitr);  pomega+=(2*LC+1);}
        if constexpr(L+LC >= 5)  {type2_ang_nuc_l<5>(pomega, i, j, k, unitr);  pomega+=(2*LC+1);}
        if constexpr(L+LC >= 7)  {type2_ang_nuc_l<7>(pomega, i, j, k, unitr);  pomega+=(2*LC+1);}
        if constexpr(L+LC >= 9)  {type2_ang_nuc_l<9>(pomega, i, j, k, unitr);  pomega+=(2*LC+1);}
    }
}

template <int L> __device__
void type2_ang(double* __restrict__ facs, const double* __restrict__ rca, const double* __restrict__ omega){
    constexpr int L1 = L+1;
    constexpr int nfi = L1*(L1+1)/2;
    constexpr int LCC1 = (2*LC+1);
    constexpr int LC1 = L+LC+1;
    constexpr int BLK = (LC1+1)/2 * LCC1;

    double fi[nfi*3];
    cache_fac<L>(fi, rca);

    // i,j,k,ijkmn->(i+j+k)pmn
    for (int pmn = threadIdx.x; pmn < nfi*LC1; pmn+=blockDim.x){
        const int m = pmn/nfi;
        const int p = pmn%nfi;

        const int iy = _cart_pow_y[p];
        const int iz = _cart_pow_z[p];
        const int ix = L - iy - iz;

        double *fx = fi + (ix+1)*ix/2;
        double *fy = fi + (iy+1)*iy/2 + nfi;
        double *fz = fi + (iz+1)*iz/2 + nfi*2;

        double ang_pmn[L+1];
        for (int i = 0; i < L+1; i++){
            ang_pmn[i] = 0.0;
        }
        
        for (int i = 0; i <= ix; i++){
        for (int j = 0; j <= iy; j++){
        for (int k = 0; k <= iz; k++){
            const int ijk = i+j+k;
            const double fac = fx[i] * fy[j] * fz[k];
            const int L_i = L-i;
            const int ioff = (L_i)*(L_i+1)*(L_i+2)/6;
            const int joff = (L_i-j)*(L_i-j+1)/2;
            const double *pomega = omega + (ioff+joff+k)*BLK;

            if ((LC+ijk)%2 == m%2){
                ang_pmn[ijk] += fac * pomega[m/2*LCC1];
            }
        }}}

        for (int i = 0; i <= L; i++){
            facs[i*nfi*LC1 + p*LC1 + m] = ang_pmn[i];
        }
    }
}

// placeholder for LI, LJ, LC
extern "C" __global__
void type2_cart(double* __restrict__ gctr,
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

    constexpr int LI1 = LI+1;
    constexpr int LJ1 = LJ+1;
    constexpr int LIC1 = LI+LC+1;
    constexpr int LJC1 = LJ+LC+1;
    constexpr int LCC1 = (2*LC+1);

    constexpr int BLKI = (LIC1+1)/2 * LCC1;
    constexpr int BLKJ = (LJC1+1)/2 * LCC1;

    __shared__ double omegai[LI1*(LI1+1)*(LI1+2)/6 * BLKI]; // up to 12600 Bytes
    __shared__ double omegaj[LJ1*(LJ1+1)*(LJ1+2)/6 * BLKJ];

    type2_facs_omega<LI>(omegai, rca);
    type2_facs_omega<LJ>(omegaj, rcb);
    __syncthreads();

    double radi[LIC1];
    double radj[LJC1];
    const double dca = norm3d(rca[0], rca[1], rca[2]);
    const double dcb = norm3d(rcb[0], rcb[1], rcb[2]);

    // Coefficient pointers for primitive (c,e) pairs of each shell
    const DataType2* cei = coeff_exp + ish * prim_stride;
    const DataType2* cej = coeff_exp + jsh * prim_stride;
    type2_facs_rad<0, LI+LC, NPI>(radi, dca, cei);
    type2_facs_rad<0, LJ+LC, NPJ>(radj, dcb, cej);

    __shared__ double rad_all[(LI+LJ+1) * LIC1 * LJC1];
    set_shared_memory(rad_all, (LI+LJ+1)*LIC1*LJC1);

    double ur = 0.0;
    // Each ECP shell has multiple powers and primitive basis
    for (int kbas = ecploc[ksh]; kbas < ecploc[ksh+1]; kbas++){
        ur += rad_part(kbas, ecpbas, env);
    }
    
    double root = 0.0;
    if (threadIdx.x < NGAUSS){
        root = r128[threadIdx.x];
    }
    for (int p = 0; p <= LI+LJ; p++){
        double *prad = rad_all + p*LIC1*LJC1;
        for (int i = 0; i <= LI+LC; i++){
        for (int j = 0; j <= LJ+LC; j++){
            block_reduce(radi[i]*radj[j]*ur, prad+i*LJC1+j);
        }}
        ur *= root;
    }
    __syncthreads();

    constexpr int nfi = (LI+1) * (LI+2) / 2;
    constexpr int nfj = (LJ+1) * (LJ+2) / 2;

    __shared__ double angi[LI1*nfi*LIC1]; // up to 5400 Bytes, further compression
    __shared__ double angj[LJ1*nfj*LJC1];

    // ECP Type2 normalization factor - basis layout already includes normalization
    const double fac = 16.0 * M_PI * M_PI;

    constexpr int nreg = (nfi*nfj + THREADS - 1)/THREADS;
    double reg_gctr[nreg];
    for (int i = 0; i < nreg; i++){
        reg_gctr[i] = 0.0;
    }

    // (k+l)pq,kimp,ljmq->ij
    for (int m = 0; m < LCC1; m++){
        type2_ang<LI>(angi, rca, omegai+m);
        type2_ang<LJ>(angj, rcb, omegaj+m);
        __syncthreads();
        // Accumulate per-thread block partial sums into reg_gctr buckets
        for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
            const int i = ij % nfi;
            const int j = ij / nfi;
            double s = 0.0;
            for (int k = 0; k <= LI; k++){
            for (int l = 0; l <= LJ; l++){
                const double* __restrict__ pangi = angi + k*nfi*LIC1 + i*LIC1;
                const double* __restrict__ pangj = angj + l*nfj*LJC1 + j*LJC1;
                const double* __restrict__ prad  = rad_all + (k+l)*LIC1*LJC1;
                for (int p = 0; p < LIC1; p++){
                    const double ap = pangi[p];
                    const double* __restrict__ prad_row = prad + p*LJC1;
                    for (int q = 0; q < LJC1; q++){
                        s += prad_row[q] * ap * pangj[q];
                    }
                }
            }}
            reg_gctr[ij/THREADS] += fac * s;
        }
        __syncthreads();
    }

    // Write back: each thread writes only its bucket entry to the
    // corresponding (i,j) index it owns to avoid duplicating sums.
    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];
    for (int b = 0; b < nreg; b++){
        const int ij = b * THREADS + threadIdx.x;
        if (ij < nfi*nfj){
            const int i = ij % nfi;
            const int j = ij / nfi;
            const double tmp = reg_gctr[b];
            // Row-major indexing: (row, col) = (i+ioff, j+joff)
            atomicAdd(gctr + (i+ioff)*nao + (j+joff), tmp);
            if (ish != jsh){
                atomicAdd(gctr + (j+joff)*nao + (i+ioff), tmp);
            }
        }
    }
    return;
}
