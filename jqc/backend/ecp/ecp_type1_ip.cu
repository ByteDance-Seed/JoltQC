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




extern "C" __global__
void type1_cart_ip1(double* __restrict__ gctr, const int nao,
                const int* __restrict__ tasks, const int ntasks,
                const int* __restrict__ ecpbas, const int* __restrict__ ecploc,
                const DataType* __restrict__ basis_data,
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

    // Extract coords and coeff_exp from packed basis_data
    constexpr int basis_stride = BASIS_STRIDE;
    const DataType* basis_i = basis_data + ish * basis_stride;
    const DataType* basis_j = basis_data + jsh * basis_stride;
    const DataType4 ri = *reinterpret_cast<const DataType4*>(basis_i);
    const DataType4 rj = *reinterpret_cast<const DataType4*>(basis_j);

    const int ioff = static_cast<int>(ri.w);
    const int joff = static_cast<int>(rj.w);

    const int ecp_id = ecpbas[ECP_ATOM_ID+ecploc[ksh]*BAS_SLOTS];
    gctr += ioff*nao + joff + 3*ecp_id*nao*nao;
    
    constexpr int nfi = (LI+1) * (LI+2) / 2;
    constexpr int nfj = (LJ+1) * (LJ+2) / 2;
    extern __shared__ char shared_mem[];

    // Allocate gctr_smem from shared memory
    double* gctr_smem = reinterpret_cast<double*>(shared_mem);
    constexpr size_t gctr_offset = nfi * nfj * 3 * sizeof(double);

    for (int ij = threadIdx.x; ij < nfi*nfj*3; ij+=blockDim.x){
        gctr_smem[ij] = 0.0;
    }
    __syncthreads();

    constexpr int nfi_max = (LI+2)*(LI+3)/2;
    constexpr int nfj_max = (LJ+1)*(LJ+2)/2;

    // Allocate buffer and kernel shared memory
    double* buf = reinterpret_cast<double*>(shared_mem + gctr_offset);
    char* kernel_shared_mem = shared_mem + gctr_offset + 3 * nfi_max * nfj_max * sizeof(double);

    // Accumulate derivative contributions with respect to AO i.
    // j-side contributions are accumulated via (j,i) tasks in host tasking (full tasks).
    // Use LI+1 for orderi=1 to match unrolled cache pattern
    type1_cart_kernel<LI+1, LJ, 1, 0>(buf, ish, jsh, ksh, ecpbas, ecploc,
        basis_data, atm, env, npi, npj, kernel_shared_mem);
    __syncthreads();
    _li_down<LI, LJ>(gctr_smem, buf);
    __syncthreads();

    if constexpr (LI > 0){
        // Use LI-1 for orderi=0 companion
        set_shared_memory(buf, 3 * nfi_max * nfj_max);
        type1_cart_kernel<LI-1, LJ, 0, 0>(buf, ish, jsh, ksh, ecpbas, ecploc,
            basis_data, atm, env, npi, npj, kernel_shared_mem);
        __syncthreads();
        _li_up<LI, LJ>(gctr_smem, buf);
        __syncthreads();
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

