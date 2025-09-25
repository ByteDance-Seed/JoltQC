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

template <int LIJ> __device__
void type1_rad_part(double* __restrict__ rad_all, double k, double aij, double ur)
{
    const double kaij = k / (2*aij);
    const double fac = kaij * kaij * aij;
    double r = 0.0;
    if (threadIdx.x < NGAUSS){
        r = r128[threadIdx.x];
    }
    double tmp = r - kaij;
    tmp = fac - aij*tmp*tmp;
    constexpr int LIJ1 = LIJ + 1;
    double bval[LIJ1];
    double rur;
    if (ur == 0 || tmp > CUTOFF || tmp < -(EXPCUTOFF+6.+30.)) {
        rur = 0;
        for (int i = 0; i < LIJ1; i++){
            bval[i] = 0;
        }
    } else {
        rur = ur * exp(tmp);
        _ine<LIJ>(bval, k*r);
    }

    for (int i = threadIdx.x; i < LIJ1*LIJ1; i+=blockDim.x){
        rad_all[i] = 0.0;
    }
    __syncthreads();
    for (int lab = 0; lab <= LIJ; lab++){
        if (lab > 0){
            rur *= r;
        }
        for (int i = lab%2; i <= LIJ; i+=2){
            block_reduce(rur*bval[i], rad_all+lab*LIJ1+i);
        }
    }
    __syncthreads();
}


template <int LIJ> __device__
void type1_rad_ang(double *rad_ang, double *r, double *rad_all, const double fac)
{
    double unitr[3];
    if (r[0]*r[0] + r[1]*r[1] + r[2]*r[2] < 1e-16){
        unitr[0] = 0;
        unitr[1] = 0;
        unitr[2] = 0;
    } else {
        double norm_r = -rnorm3d(r[0], r[1], r[2]);
        unitr[0] = r[0] * norm_r;
        unitr[1] = r[1] * norm_r;
        unitr[2] = r[2] * norm_r;
    }

    // loop over i+j+k<=LIJ
    // TODO: find a closed form?
    for (int n = threadIdx.x; n < (LIJ+1)*(LIJ+1)*(LIJ+1); n+=blockDim.x){
        const int i = n/(LIJ+1)/(LIJ+1);
        const int j = n/(LIJ+1)%(LIJ+1);
        const int k = n%(LIJ+1);
        if (i+j+k > LIJ || (i+j+k)%2 == 1){
            continue;
        }
        // need_even to ensure (i+j+k+lmb) is even
        double s = 0.0;
        double *prad = rad_all + (i+j+k)*(LIJ+1);
        if constexpr(LIJ >= 0) s += prad[0] * type1_ang_nuc_l<0>(i, j, k, unitr);
        if constexpr(LIJ >= 2) s += prad[2] * type1_ang_nuc_l<2>(i, j, k, unitr);
        if constexpr(LIJ >= 4) s += prad[4] * type1_ang_nuc_l<4>(i, j, k, unitr);
        if constexpr(LIJ >= 6) s += prad[6] * type1_ang_nuc_l<6>(i, j, k, unitr);
        if constexpr(LIJ >= 8) s += prad[8] * type1_ang_nuc_l<8>(i, j, k, unitr);
        if constexpr(LIJ >= 10)s += prad[10]* type1_ang_nuc_l<10>(i, j, k, unitr);
        //rad_ang[i*(LIJ+1)*(LIJ+1) + j*(LIJ+1) + k] += fac*s;
        atomicAdd(rad_ang + i*(LIJ+1)*(LIJ+1) + j*(LIJ+1) + k, fac*s);
    }

    for (int n = threadIdx.x; n < (LIJ+1)*(LIJ+1)*(LIJ+1); n+=blockDim.x){
        const int i = n/(LIJ+1)/(LIJ+1);
        const int j = n/(LIJ+1)%(LIJ+1);
        const int k = n%(LIJ+1);
        if (i+j+k > LIJ || (i+j+k)%2 == 0){
            continue;
        }
        // need_even to ensure (i+j+k+lmb) is even
        double s = 0.0;
        double *prad = rad_all + (i+j+k)*(LIJ+1);
        if constexpr(LIJ >= 1) s += prad[1] * type1_ang_nuc_l<1>(i, j, k, unitr);
        if constexpr(LIJ >= 3) s += prad[3] * type1_ang_nuc_l<3>(i, j, k, unitr);
        if constexpr(LIJ >= 5) s += prad[5] * type1_ang_nuc_l<5>(i, j, k, unitr);
        if constexpr(LIJ >= 7) s += prad[7] * type1_ang_nuc_l<7>(i, j, k, unitr);
        if constexpr(LIJ >= 9) s += prad[9] * type1_ang_nuc_l<9>(i, j, k, unitr);
        //rad_ang[i*(LIJ+1)*(LIJ+1) + j*(LIJ+1) + k] += fac*s;
        atomicAdd(rad_ang + i*(LIJ+1)*(LIJ+1) + j*(LIJ+1) + k, fac*s);
    }
}


// placeholder for LI, LJ, npi, npj
extern "C" __global__
void type1_cart(double *gctr,
                const int *ao_loc, const int nao,
                const int *tasks, const int ntasks,
                const int *ecpbas, const int *ecploc,
                const DataType4* __restrict__ coords,
                const DataType2* __restrict__ coeff_exp,
                const int *atm, const double *env)
{
    const int task_id = blockIdx.x;
    if (task_id >= ntasks){
        return;
    }

    const int ish = tasks[task_id];
    const int jsh = tasks[task_id + ntasks];
    const int ksh = tasks[task_id + 2*ntasks];

    // Coefficient layout: each shell occupies PRIM_STRIDE scalars -> PRIM_STRIDE/2 (c,e) pairs
    // Use pair-stride indexing to access the (c,e) DataType2 records of a given shell
    constexpr int prim_stride = PRIM_STRIDE / 2;

    // Coordinates from basis layout. Coordinates are stored as a flat
    // array with stride COORD_STRIDE. Use scalar pointer + stride to read
    // x/y/z to avoid relying on struct size/alignment assumptions.
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

    constexpr int LIJ1 = LI+LJ+1;
    __shared__ double rad_ang[LIJ1*LIJ1*LIJ1];
    __shared__ double rad_all[LIJ1*LIJ1];
    set_shared_memory(rad_ang, LIJ1*LIJ1*LIJ1);

    // ECP Type1 normalization factor - basis layout already includes normalization
    const double fac = 16.0 * M_PI * M_PI;
    for (int ip = 0; ip < NPI; ip++){
        const int ish_ip = ip + ish * prim_stride;
        const DataType2 cei = coeff_exp[ish_ip];
        const DataType ai = cei.e;
        const DataType ci = cei.c;
        for (int jp = 0; jp < NPJ; jp++){
            const int jsh_jp = jp + jsh * prim_stride;
            const DataType2 cej = coeff_exp[jsh_jp];
            const DataType aj = cej.e;
            const DataType cj = cej.c;

            double rij[3];
            rij[0] = ai * rca[0] + aj * rcb[0];
            rij[1] = ai * rca[1] + aj * rcb[1];
            rij[2] = ai * rca[2] + aj * rcb[2];
            const double k = 2.0 * norm3d(rij[0], rij[1], rij[2]);
            const double aij = ai + aj;

            type1_rad_part<LI+LJ>(rad_all, k, aij, ur);

            const double eij = exp(-ai*r2ca - aj*r2cb);
            const double ceij = eij * ci * cj;
            type1_rad_ang<LI+LJ>(rad_ang, rij, rad_all, fac*ceij);
            __syncthreads();
        }
    }

    constexpr int nfi = (LI+1) * (LI+2) / 2;
    constexpr int nfj = (LJ+1) * (LJ+2) / 2;
    double fi[3*nfi];
    double fj[3*nfj];
    cache_fac<LI>(fi, rca);
    cache_fac<LJ>(fj, rcb);

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int mi = ij%nfi;
        const int mj = ij/nfi;

        // TODO: Read the same constant memory in each warp
        const int iy = _cart_pow_y[mi];
        const int iz = _cart_pow_z[mi];
        const int ix = LI - iy - iz;

        double* fx_i = fi + (ix+1)*ix/2;
        double* fy_i = fi + (iy+1)*iy/2 + nfi;
        double* fz_i = fi + (iz+1)*iz/2 + 2*nfi;

        const int jy = _cart_pow_y[mj];
        const int jz = _cart_pow_z[mj];
        const int jx = LJ - jy - jz;
        double* fx_j = fj + (jx+1)*jx/2;
        double* fy_j = fj + (jy+1)*jy/2 + nfj;
        double* fz_j = fj + (jz+1)*jz/2 + 2*nfj;

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

        const int ioff = ao_loc[ish];
        const int joff = ao_loc[jsh];
        // Row-major indexing: (row, col) = (mi+ioff, mj+joff)
        // Write both symmetric positions with row-major indexing
        atomicAdd(gctr + (mi+ioff)*nao + (mj+joff), tmp);
        if (ish != jsh){
            atomicAdd(gctr + (mj+joff)*nao + (mi+ioff), tmp);
        }
    }
    return;
}
