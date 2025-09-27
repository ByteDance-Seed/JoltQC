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

template <int order, int LIC> __device__
void type2_facs_rad(double facs[LIC+1], const double rca, const DataType2* __restrict__ ce, const int np){
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
        // Avoid pow in hot loop: order is a template arg (0,1,2 in practice)
        double c;
        if constexpr (order == 0) {
            c = ce[ip].c;
        } else if constexpr (order == 1) {
            c = (-2.0 * ce[ip].e) * ce[ip].c;
        } else if constexpr (order == 2) {
            const double twoe = -2.0 * ce[ip].e;
            c = (twoe * twoe) * ce[ip].c;
        } else {
            c = pow(-2.0*ce[ip].e, order) * ce[ip].c; // Fallback for unexpected order
        }
        for (int j = 0; j <= LIC; j++){
            facs[j] += c * buf[j];
        }
    }
}

template <int L> __device__
void type2_facs_omega(double* __restrict__ omega, const double r[3]){
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
    
    double rx[L+LC+1];
    double ry[L+LC+1];
    double rz[L+LC+1];

    rx[0] = 1.0; ry[0] = 1.0; rz[0] = 1.0;
    for (int i = 1; i <= L+LC; i++){
        rx[i] = rx[i-1] * unitr[0];
        ry[i] = ry[i-1] * unitr[1];
        rz[i] = rz[i-1] * unitr[2];
    }

    constexpr int LLC = L + LC;
    double buf[(LC+1)*(LC+2)/2];
    double c_buf[2*LLC+1];
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

        if constexpr(LLC >= 0)  {
            type2_ang_nuc_l<0>(buf, c_buf, i, j, k, rx, ry, rz);
            cart2sph<LC>(pomega, buf);
            pomega += (2*LC+1);
        }
        if constexpr(LLC >= 2)  {
            type2_ang_nuc_l<2>(buf, c_buf, i, j, k, rx, ry, rz);  
            cart2sph<LC>(pomega, buf);
            pomega += (2*LC+1);
        }
        if constexpr(LLC >= 4)  {
            type2_ang_nuc_l<4>(buf, c_buf, i, j, k, rx, ry, rz);  
            cart2sph<LC>(pomega, buf);
            pomega += (2*LC+1);
        }
        if constexpr(LLC >= 6)  {
            type2_ang_nuc_l<6>(buf, c_buf, i, j, k, rx, ry, rz);  
            cart2sph<LC>(pomega, buf);
            pomega += (2*LC+1);
        }
        if constexpr(LLC >= 8)  {
            type2_ang_nuc_l<8>(buf, c_buf, i, j, k, rx, ry, rz);  
            cart2sph<LC>(pomega, buf);
            pomega += (2*LC+1);
        }
        if constexpr(LLC >= 10) {
            type2_ang_nuc_l<10>(buf, c_buf, i, j, k, rx, ry, rz); 
            cart2sph<LC>(pomega, buf);
            pomega += (2*LC+1);
        }
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
        if constexpr(LLC >= 1)  {
            type2_ang_nuc_l<1>(buf, c_buf, i, j, k, rx, ry, rz);  
            cart2sph<LC>(pomega, buf);
            pomega += (2*LC+1);
        }
        if constexpr(LLC >= 3)  {
            type2_ang_nuc_l<3>(buf, c_buf, i, j, k, rx, ry, rz);
            cart2sph<LC>(pomega, buf);
            pomega += (2*LC+1);
        }
        if constexpr(LLC >= 5)  {
            type2_ang_nuc_l<5>(buf, c_buf, i, j, k, rx, ry, rz);
            cart2sph<LC>(pomega, buf);
            pomega += (2*LC+1);
        }
        if constexpr(LLC >= 7)  {
            type2_ang_nuc_l<7>(buf, c_buf, i, j, k, rx, ry, rz);
            cart2sph<LC>(pomega, buf);
            pomega += (2*LC+1);
        }
        if constexpr(LLC >= 9)  {
            type2_ang_nuc_l<9>(buf, c_buf, i, j, k, rx, ry, rz);
            cart2sph<LC>(pomega, buf); 
            pomega += (2*LC+1);
        }
    }
}

template <int L> __device__
void type2_ang(double* __restrict__ facs, const double rca[3], const double* __restrict__ omega){
    constexpr int L1 = L+1;
    constexpr int NF = L1*(L1+1)/2;
    constexpr int LCC1 = (2*LC+1);
    constexpr int LC1 = L+LC+1;
    constexpr int BLK = (LC1+1)/2 * LCC1;

    // reset shared memory buffer
    for (int i = threadIdx.x; i < L1*LC1*NF; i+=THREADS){
        facs[i] = 0.0;
    }

    __shared__ double fi[NF*3];
    cache_fac<L>(fi, rca);
    __syncthreads();

    // i,j,k,ijkmn->(i+j+k)pmn
    for (int p = 0; p < NF; p++){
        const int iy = _cart_pow_y[p];
        const int iz = _cart_pow_z[p];
        const int ix = L - iy - iz;

        double *fx = fi + (ix+1)*ix/2;
        double *fy = fi + (iy+1)*iy/2 + NF;
        double *fz = fi + (iz+1)*iz/2 + NF*2;
        
        for (int i = 0; i <= ix; i++){
        for (int j = 0; j <= iy; j++){
        for (int k = 0; k <= iz; k++){
            const int ijk = i+j+k;
            const double fac = fx[i] * fy[j] * fz[k];
            const int L_i = L-i;
            const int ioff = (L_i)*(L_i+1)*(L_i+2)/6;
            const int joff = (L_i-j)*(L_i-j+1)/2;
            const double *pomega = omega + (ioff+joff+k)*BLK;
            
            const int parity  = (LC + ijk) & 1;  // required parity (0 = even, 1 = odd)
            // Precompute base pointer for facs
            double* facs_base = &facs[ijk * NF * LC1 + p * LC1];

            // Find this thread’s first valid m that matches parity
            int m = threadIdx.x;
            if ( (m & 1) != parity ) {
                m += 1;  // shift to next parity
            }
            // Assume L + LC + 1 < 128 or 256
            if (m < LC1){
                const int half_index = m >> 1;  // equivalent to m/2 since parity ensures divisibility
                facs_base[m] += fac * pomega[half_index * LCC1];
            }
        }}}
    }
}

// Refactored to take nprim as runtime parameters instead of constexpr injection
extern "C" __global__
void type2_cart(double* __restrict__ gctr,
                const int* __restrict__ ao_loc, const int nao,
                const int* __restrict__ tasks, const int ntasks,
                const int* __restrict__ ecpbas, const int* __restrict__ ecploc,
                const DataType4* __restrict__ coords,
                const DataType2* __restrict__ coeff_exp,
                const int* __restrict__ atm, const double* __restrict__ env,
                const int npi, const int npj)
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

    constexpr int LI1 = LI+1;
    constexpr int LJ1 = LJ+1;
    constexpr int LIC1 = LI+LC+1;
    constexpr int LJC1 = LJ+LC+1;
    constexpr int LCC1 = (2*LC+1);

    constexpr int BLKI = (LIC1+1)/2 * LCC1;
    constexpr int BLKJ = (LJC1+1)/2 * LCC1;

    double rca[3];
    __shared__ double omegai[LI1*(LI1+1)*(LI1+2)/6 * BLKI]; // up to 12600 Bytes
    rca[0] = rc[0] - ri[0];
    rca[1] = rc[1] - ri[1];
    rca[2] = rc[2] - ri[2];
    type2_facs_omega<LI>(omegai, rca);
    const double dca = norm3d(rca[0], rca[1], rca[2]);

    double rcb[3];
    __shared__ double omegaj[LJ1*(LJ1+1)*(LJ1+2)/6 * BLKJ]; // up to 12600 Bytes
    rcb[0] = rc[0] - rj[0];
    rcb[1] = rc[1] - rj[1];
    rcb[2] = rc[2] - rj[2];
    type2_facs_omega<LJ>(omegaj, rcb);
    const double dcb = norm3d(rcb[0], rcb[1], rcb[2]);

    __syncthreads();

    // Coefficient pointers for primitive (c,e) pairs of each shell
    const DataType2* cei = coeff_exp + ish * prim_stride;
    const DataType2* cej = coeff_exp + jsh * prim_stride;
    
    double radi[LIC1];
    type2_facs_rad<0, LI+LC>(radi, dca, cei, npi);
    double radj[LJC1];
    type2_facs_rad<0, LJ+LC>(radj, dcb, cej, npj);

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
#pragma unroll
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
    constexpr double fac = 16.0 * M_PI * M_PI;

    constexpr int nreg = (nfi*nfj + THREADS - 1)/THREADS;
    double reg_gctr[nreg];
    for (int i = 0; i < nreg; i++){
        reg_gctr[i] = 0.0;
    }
    
    // (k+l)pq,kimp,ljmq->ij
#pragma unroll 1
    for (int m = 0; m < LCC1; m++){
        type2_ang<LI>(angi, rca, omegai+m);
        type2_ang<LJ>(angj, rcb, omegaj+m);
        __syncthreads();
        // Accumulate per-thread block partial sums into reg_gctr buckets
        constexpr int PT = 4;  // tile size in p
        constexpr int QT = 4;  // tile size in q
        for (int b = 0; b < nreg; b++){
            const int ij = b * THREADS + threadIdx.x;
            if (ij >= nfi*nfj){
                continue;
            }
            const int i = ij % nfi;
            const int j = ij / nfi;
            // Precompute base offsets to reduce redundant calculations
            const int angi_base = i*LIC1;
            const int angj_base = j*LJC1;
            double s = 0.0;
            for (int k = 0; k <= LI; k++){
            for (int l = 0; l <= LJ; l++){
                // Use precomputed bases to reduce register pressure
                const int angi_offset = k*nfi*LIC1 + angi_base;
                const int angj_offset = l*nfj*LJC1 + angj_base;
                const int rad_offset = (k+l)*LIC1*LJC1;
                // Streaming computation to reduce register pressure (eliminates ap[PT], bq[QT] arrays)
                for (int p0 = 0; p0 < LIC1; p0 += PT){
                    const int pmax = min(PT, LIC1 - p0);
                    for (int q0 = 0; q0 < LJC1; q0 += QT){
                        const int qmax = min(QT, LJC1 - q0);
                        #pragma unroll
                        for (int tp = 0; tp < PT; ++tp){
                            if (tp >= pmax) break;
                            const double ap_val = angi[angi_offset + p0 + tp];
                            #pragma unroll
                            for (int tq = 0; tq < QT; ++tq){
                                if (tq >= qmax) break;
                                const double bq_val = angj[angj_offset + q0 + tq];
                                s += rad_all[rad_offset + (p0 + tp) * LJC1 + q0 + tq] * ap_val * bq_val;
                            }
                        }
                    }
                }
            }}
            reg_gctr[b] += fac * s;
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
