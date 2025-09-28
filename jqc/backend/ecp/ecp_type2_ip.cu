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



extern "C" __global__
void type2_cart_ip1(double* __restrict__ gctr,
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
    const int ioff = ao_loc[ish];
    const int joff = ao_loc[jsh];
    const int ecp_id = ecpbas[ECP_ATOM_ID+ecploc[ksh]*BAS_SLOTS];
    gctr += 3*ecp_id*nao*nao + ioff*nao + joff;

    constexpr int nfi = (LI+1) * (LI+2) / 2;
    constexpr int nfj = (LJ+1) * (LJ+2) / 2;
    extern __shared__ char shared_mem[];

    // Allocate gctr_smem from shared memory
    double* gctr_smem = reinterpret_cast<double*>(shared_mem);
    size_t gctr_offset = 3 * nfi * nfj * sizeof(double);

    for (int ij = threadIdx.x; ij < nfi*nfj*3; ij+=blockDim.x){
        gctr_smem[ij] = 0.0;
    }
    __syncthreads();

    constexpr int LI1_MAX = LI+2;
    constexpr int LJ1_MAX = LJ+1;
    constexpr int NFI_MAX = LI1_MAX*(LI1_MAX+1)/2;
    constexpr int NFJ_MAX = LJ1_MAX*(LJ1_MAX+1)/2;

    // Allocate buffer and remaining shared memory for kernel
    double* buf = reinterpret_cast<double*>(shared_mem + gctr_offset);
    char* kernel_shared_mem = shared_mem + gctr_offset + NFI_MAX * NFJ_MAX * sizeof(double);

    type2_cart_kernel<LI+1, LJ, LC, 1, 0>(buf, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env, npi, npj, kernel_shared_mem);
    __syncthreads();
    _li_down<LI, LJ>(gctr_smem, buf);
    if constexpr (LI > 0){
        // Companion LI-1 for orderi=0 stage
        type2_cart_kernel<LI-1, LJ, LC, 0, 0>(buf, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env, npi, npj, kernel_shared_mem);
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


