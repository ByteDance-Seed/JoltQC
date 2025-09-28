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
void type1_cart_ipvip(double* __restrict__ gctr,
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
    gctr += ioff*nao + joff + 9*ecp_id*nao*nao;

    constexpr int nfi1_max = (LI+2)*(LI+3)/2;
    constexpr int nfj1_max = (LJ+2)*(LJ+3)/2;
    constexpr int nfi_max = (LI+1)*(LI+2)/2;
    extern __shared__ char shared_mem[];

    // Allocate buffers from dynamic shared memory
    double* buf1 = reinterpret_cast<double*>(shared_mem);
    size_t buf1_offset = nfi1_max * nfj1_max * sizeof(double);
    double* buf = reinterpret_cast<double*>(shared_mem + buf1_offset);
    size_t buf_offset = buf1_offset + 3 * nfi_max * nfj1_max * sizeof(double);
    char* kernel_shared_mem = shared_mem + buf_offset;

    // Start with LI+1, LJ+1 for mixed order
    type1_cart_kernel<LI+1, LJ+1, 1, 1>(buf1, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env, npi, npj, kernel_shared_mem);
    __syncthreads();
    for (int i = threadIdx.x; i < 3*nfi_max*nfj1_max; i+=blockDim.x){
        buf[i] = 0.0;
    }
    __syncthreads();
    _li_down<LI, LJ+1>(buf, buf1);
    if constexpr (LI > 0){
        // Companion LI-1, LJ+1 for orderi=0, orderj=1
        type1_cart_kernel<LI-1, LJ+1, 0, 1>(buf1, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env, npi, npj, kernel_shared_mem);
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
        type1_cart_kernel<LI+1, LJ-1, 1, 0>(buf1, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env, npi, npj, kernel_shared_mem);
        __syncthreads();
        _li_down<LI, LJ-1>(buf, buf1);
        if constexpr (LI > 0){
            // LI-1, LJ-1 for orderi=0, orderj=0 companion
            type1_cart_kernel<LI-1, LJ-1, 0, 0>(buf1, ish, jsh, ksh, ecpbas, ecploc, coords, coeff_exp, atm, env, npi, npj, kernel_shared_mem);
            __syncthreads();
            _li_up<LI, LJ-1>(buf, buf1);
        }
        _lj_up_and_write<LI, LJ>(gctr, buf, nao);
    }
    return;
}