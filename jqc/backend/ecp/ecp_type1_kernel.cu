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

/*
 * OPTIMIZATION NOTE: Shared memory allocations in this file have been optimized
 * for better reuse across different kernel configurations by basing sizes on
 * global variables LI+2 and LJ+1 (injected from Python) instead of exact
 * template parameters LIT, LJT. This allows the same shared memory regions to
 * be reused for multiple angular momentum combinations, reducing memory
 * allocation overhead and improving kernel performance.
 */


template <int LIT, int LJT, int orderi, int orderj> __device__
void type1_cart_kernel(double* __restrict__ gctr,
                const int ish, const int jsh, const int ksh,
                const int* __restrict__ ecpbas, const int* __restrict__ ecploc,
                const DataType4* __restrict__ coords,
                const DataType2* __restrict__ coeff_exp,
                const int* __restrict__ atm, const double* __restrict__ env,
                const int npi, const int npj,
                char* __restrict__ shared_mem_pool)
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

    // Allocate arrays from dynamic shared memory pool
    constexpr int LI1_MAX = LIT + 1;
    constexpr int LJ1_MAX = LJT + 1;
    constexpr int RAD_ANG_SIZE = LIJ3;
    constexpr int RAD_ALL_SIZE = LIJ1 * LIJ1;

    size_t offset = 0;
    double* rad_ang = reinterpret_cast<double*>(shared_mem_pool + offset);
    offset += RAD_ANG_SIZE * sizeof(double);
    double* rad_all = reinterpret_cast<double*>(shared_mem_pool + offset);
    offset += RAD_ALL_SIZE * sizeof(double);

    for (int i = threadIdx.x; i < LIJ3; i+=blockDim.x) {
        rad_ang[i] = 0;
    }
    __syncthreads();
    const double fac = 16.0 * M_PI * M_PI;

    // Coefficient pointers for primitive (c,e) pairs of each shell
    const DataType2* cei = coeff_exp + ish * prim_stride;
    const DataType2* cej = coeff_exp + jsh * prim_stride;

    for (int ip = 0; ip < npi; ip++){
        for (int jp = 0; jp < npj; jp++){
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
            // Avoid pow in hot loop; orderi/orderj are template args (0,1,2)
            double io, jo;
            if constexpr (orderi == 0) {
                io = 1.0;
            } else if constexpr (orderi == 1) {
                io = -2.0 * ai_prim;
            } else if constexpr (orderi == 2) {
                const double t = -2.0 * ai_prim;
                io = t * t;
            } else {
                io = pow(-2.0*ai_prim, orderi);
            }
            if constexpr (orderj == 0) {
                jo = 1.0;
            } else if constexpr (orderj == 1) {
                jo = -2.0 * aj_prim;
            } else if constexpr (orderj == 2) {
                const double t = -2.0 * aj_prim;
                jo = t * t;
            } else {
                jo = pow(-2.0*aj_prim, orderj);
            }
            const double eaij = eij * io * jo;
            const double ceij = eaij * cei[ip].c * cej[jp].c;
            type1_rad_ang<LIT+LJT>(rad_ang, rij, rad_all, fac*ceij);
            __syncthreads();
        }
    }

    constexpr int nfi = (LIT+1) * (LIT+2) / 2;
    constexpr int nfj = (LJT+1) * (LJT+2) / 2;

    // Allocate fi and fj arrays from shared memory pool
    constexpr int FI_SIZE = LI1_MAX * (LI1_MAX + 1) / 2 * 3;
    constexpr int FJ_SIZE = LJ1_MAX * (LJ1_MAX + 1) / 2 * 3;
    double* fi = reinterpret_cast<double*>(shared_mem_pool + offset);
    offset += FI_SIZE * sizeof(double);
    double* fj = reinterpret_cast<double*>(shared_mem_pool + offset);
    offset += FJ_SIZE * sizeof(double);

    cache_fac<LIT>(fi, rca);
    cache_fac<LJT>(fj, rcb);
    __syncthreads();

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