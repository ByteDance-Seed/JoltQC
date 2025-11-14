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

// Portions of this file adapted from GPU4PySCF (https://github.com/pyscf/gpu4pyscf)
// Copyright 2021-2024 PySCF developer.
// Licensed under the Apache License, Version 2.0.

/*
 * OPTIMIZATION NOTE: Shared memory allocations in this file have been optimized
 * for better reuse across different kernel configurations by basing sizes on
 * global variables LI+2 and LJ+1 (injected from Python) instead of exact
 * template parameters LIT, LJT. This allows the same shared memory regions to
 * be reused for multiple angular momentum combinations, reducing memory
 * allocation overhead and improving kernel performance.
 */


template <int LIT, int LJT, int LCT, int orderi, int orderj> __device__
void type2_cart_kernel(double* __restrict__ gctr,
                const int ish, const int jsh, const int ksh,
                const int* __restrict__ ecpbas, const int* __restrict__ ecploc,
                const DataType* __restrict__ basis_data,
                const int* __restrict__ atm, const double* __restrict__ env,
                const int npi, const int npj,
                char* __restrict__ shared_mem_pool)
{
    // Extract coords from packed basis_data
    constexpr int basis_stride = BASIS_STRIDE;
    const DataType* basis_i = basis_data + ish * basis_stride;
    const DataType* basis_j = basis_data + jsh * basis_stride;
    const DataType4 ri_packed = *reinterpret_cast<const DataType4*>(basis_i);
    const DataType4 rj_packed = *reinterpret_cast<const DataType4*>(basis_j);

    const DataType ri[4] = {ri_packed.x, ri_packed.y, ri_packed.z, ri_packed.w};
    const DataType rj[4] = {rj_packed.x, rj_packed.y, rj_packed.z, rj_packed.w};

    const int atm_id = ecpbas[ATOM_OF+ecploc[ksh]*BAS_SLOTS];
    const double *rc = env + atm[PTR_COORD+atm_id*ATM_SLOTS];

    constexpr int LIC1 = LIT+LCT+1;
    constexpr int LJC1 = LJT+LCT+1;
    constexpr int LCC1 = (2*LCT+1);

    // Allocate arrays from dynamic shared memory pool
    constexpr int LI1_MAX = LIT + 1;
    constexpr int LJ1_MAX = LJT + 1;
    constexpr int NFI_MAX = LI1_MAX * (LI1_MAX + 1) / 2;
    constexpr int NFJ_MAX = LJ1_MAX * (LJ1_MAX + 1) / 2;
    constexpr int LIC1_MAX = LI1_MAX + LCT;
    constexpr int LJC1_MAX = LJ1_MAX + LCT;
    constexpr int MAX_BLKI = (LIC1_MAX + 1) / 2 * LCC1;
    constexpr int MAX_BLKJ = (LJC1_MAX + 1) / 2 * LCC1;
    constexpr int OMEGA_I_SIZE = LI1_MAX * (LI1_MAX + 1) * (LI1_MAX + 2) / 6 * MAX_BLKI;
    constexpr int OMEGA_J_SIZE = LJ1_MAX * (LJ1_MAX + 1) * (LJ1_MAX + 2) / 6 * MAX_BLKJ;

    // Partition the shared memory pool
    size_t offset = 0;
    double* omegai = reinterpret_cast<double*>(shared_mem_pool + offset);
    offset += OMEGA_I_SIZE * sizeof(double);
    double* omegaj = reinterpret_cast<double*>(shared_mem_pool + offset);
    offset += OMEGA_J_SIZE * sizeof(double);

    double rca[3];
    rca[0] = rc[0] - ri[0];
    rca[1] = rc[1] - ri[1];
    rca[2] = rc[2] - ri[2];
    type2_facs_omega<LIT>(omegai, rca);
    const double dca = norm3d(rca[0], rca[1], rca[2]);

    double rcb[3];
    rcb[0] = rc[0] - rj[0];
    rcb[1] = rc[1] - rj[1];
    rcb[2] = rc[2] - rj[2];
    type2_facs_omega<LJT>(omegaj, rcb);
    const double dcb = norm3d(rcb[0], rcb[1], rcb[2]);
    __syncthreads();

    // Allocate rad_all from shared memory pool
    constexpr int LIJ1_MAX = LI1_MAX + LJT;
    constexpr int RAD_ALL_SIZE = LIJ1_MAX * LIC1_MAX * LJC1_MAX;
    double* rad_all = reinterpret_cast<double*>(shared_mem_pool + offset);
    offset += RAD_ALL_SIZE * sizeof(double);
    set_shared_memory(rad_all, RAD_ALL_SIZE);

    // Extract coefficient-exponent pointers from packed basis_data
    const DataType2* cei = reinterpret_cast<const DataType2*>(basis_data + ish * basis_stride + 4);
    const DataType2* cej = reinterpret_cast<const DataType2*>(basis_data + jsh * basis_stride + 4);
    double radi[LIC1];
    type2_facs_rad<orderi, LIT+LCT>(radi, dca, cei, npi);
    double radj[LJC1];
    type2_facs_rad<orderj, LJT+LCT>(radj, dcb, cej, npj);

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

    // Allocate angular arrays from shared memory pool
    constexpr int ANGI_SIZE = LI1_MAX * LIC1_MAX * NFI_MAX;
    constexpr int ANGJ_SIZE = LJ1_MAX * LJC1_MAX * NFJ_MAX;
    double* angi = reinterpret_cast<double*>(shared_mem_pool + offset);
    offset += ANGI_SIZE * sizeof(double);
    double* angj = reinterpret_cast<double*>(shared_mem_pool + offset);
    offset += ANGJ_SIZE * sizeof(double);

    constexpr double fac = 16.0 * M_PI * M_PI * (16.0 * M_PI * M_PI); // Additional (4*pi)^2 from angular integration

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        gctr[ij] = 0.0;
    }

    // Allocate fi and fj arrays from shared memory pool
    constexpr int FI_SIZE = LI1_MAX * (LI1_MAX + 1) / 2 * 3;
    constexpr int FJ_SIZE = LJ1_MAX * (LJ1_MAX + 1) / 2 * 3;
    double* fi = reinterpret_cast<double*>(shared_mem_pool + offset);
    offset += FI_SIZE * sizeof(double);
    double* fj = reinterpret_cast<double*>(shared_mem_pool + offset);
    offset += FJ_SIZE * sizeof(double);

    // Calculate fi and fj
    cache_fac<LIT>(fi, rca);
    cache_fac<LJT>(fj, rcb);
    __syncthreads();

    // (k+l)pq,kimp,ljmq->ij
    for (int m = 0; m < LCC1; m++){
        type2_ang<LIT>(angi, fi, omegai+m);
        type2_ang<LJT>(angj, fj, omegaj+m);
        __syncthreads();
        const int PT = 4;  // tile size in p
        const int QT = 4;  // tile size in q
        for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
            const int i = ij%nfi;
            const int j = ij/nfi;
            double s = 0.0;
            for (int k = 0; k <= LIT; k++){
            for (int l = 0; l <= LJT; l++){
                double *pangi = angi + k*nfi*LIC1 + i*LIC1;
                double *pangj = angj + l*nfj*LJC1 + j*LJC1;
                double *prad  = rad_all + (k+l)*LIC1*LJC1;
                for (int p0 = 0; p0 < LIC1; p0 += PT){
                    const int pmax = min(PT, LIC1 - p0);
                    double ap[PT];
                    #pragma unroll
                    for (int tp = 0; tp < PT; ++tp){
                        ap[tp] = (tp < pmax) ? pangi[p0 + tp] : 0.0;
                    }
                    for (int q0 = 0; q0 < LJC1; q0 += QT){
                        const int qmax = min(QT, LJC1 - q0);
                        double bq[QT];
                        #pragma unroll
                        for (int tq = 0; tq < QT; ++tq){
                            bq[tq] = (tq < qmax) ? pangj[q0 + tq] : 0.0;
                        }
                        #pragma unroll
                        for (int tp = 0; tp < PT; ++tp){
                            if (tp >= pmax) break;
                            const double * __restrict__ prad_row = prad + (p0 + tp) * LJC1 + q0;
                            #pragma unroll
                            for (int tq = 0; tq < QT; ++tq){
                                if (tq >= qmax) break;
                                s += prad_row[tq] * ap[tp] * bq[tq];
                            }
                        }
                    }
                }
            }}
            gctr[ij] += fac*s;
        }
        __syncthreads();
    }
}