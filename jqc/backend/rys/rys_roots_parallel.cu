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

// Portions of this file adapted from GPU4PySCF v1.4 (https://github.com/pyscf/gpu4pyscf)
// Copyright 2025 PySCF developer.
// Licensed under the Apache License, Version 2.0.

constexpr int DEGREE = 13;
constexpr int DEGREE1 = (DEGREE+1);
constexpr int INTERVALS = 40;

constexpr DataType SQRTPIE4 = .8862269254527580136;
constexpr DataType PIE4     = .7853981633974483096;

__device__ __forceinline__
static void rys_roots(DataType x, DataType *rw, int rt_id, const int stride, DataType theta, DataType omega)
{
    constexpr DataType one = 1.0;
    constexpr DataType two = 2.0;
    constexpr DataType half= .5;
    const int stride2 = stride * 2;

    x *= theta;

    const DataType omega2 = omega*omega;
    const DataType theta_fac = omega2 / (omega2 + theta);
    const DataType sqrt_theta_fac = sqrt(theta_fac);

    if constexpr (rys_type > 0){
        x *= theta_fac;
    }
    
    if (x < 3.e-7){
#pragma unroll
        for (int i = rt_id; i < nroots; i += nthreads_per_sq)  {
            DataType root = ROOT_SMALLX_R0[i] + ROOT_SMALLX_R1[i] * x;
            DataType weight = ROOT_SMALLX_W0[i] + ROOT_SMALLX_W1[i] * x;
            if constexpr(rys_type > 0){
                root *= theta_fac;
                weight *= sqrt_theta_fac;
            }
            rw[i*stride2         ] = root;
            rw[i*stride2 + stride] = weight;
        }
        return;
    }
    
    if (x > 35+nroots*5) {
        const DataType inv_x = one / x; 
        const DataType t = sqrt(PIE4 * inv_x);
#pragma unroll
        for (int i = rt_id; i < nroots; i += nthreads_per_sq)  {
            DataType root = ROOT_LARGEX_R_DATA[i] * inv_x;
            DataType weight = ROOT_LARGEX_W_DATA[i] * t;
            if constexpr(rys_type > 0){
                root *= theta_fac;
                weight *= sqrt_theta_fac;
            }
            rw[i*stride2         ] = root;
            rw[i*stride2 + stride] = weight;
        }
        return;
    }

    if constexpr(nroots == 1) {
        const DataType tt = sqrt(x);                       // 1 sqrt
        const DataType erf_tt = erf(tt);                   // 1 erf
        const DataType e = exp(-x);                        // 1 exp

        const DataType inv_tt = SQRTPIE4 / tt;             // 1 div
        const DataType fmt0 = inv_tt * erf_tt;

        DataType weight = fmt0;
        const DataType fmt1 = (half / x) * (fmt0 - e);     // 1 div
        DataType root = fmt1 / fmt0;                               // 1 div
        
        if constexpr(rys_type > 0){
            root *= theta_fac;
            weight *= sqrt_theta_fac;
        }
        rw[0] = root;
        rw[stride] = weight;
        return;
    }

    const int it = (int)(x * .4f);
    const DataType u = (x - it * DataType(2.5)) * DataType(0.8) - DataType(1.);
    const DataType u2 = u * two;
    const DataType *datax = ROOT_RW_DATA + it;
#pragma unroll
    for (int i = rt_id; i < nroots; i += nthreads_per_sq) {
        {
            const DataType *c = datax + (2*i) * DEGREE1 * INTERVALS;
            DataType c0 = c[DEGREE*INTERVALS];
            DataType c1 = c[DEGREE*INTERVALS - INTERVALS];
    #pragma unroll
            for (int n = DEGREE-2; n > 0; n-=2) {
                const DataType c2 = c[n*INTERVALS] - c1;
                const DataType c3 = c0 + c1*u2;
                c1 = c2 + c3*u2;
                c0 = c[n*INTERVALS - INTERVALS] - c3;
            }
            DataType root = c0 + c1*u;
            if constexpr(rys_type > 0){
                root *= theta_fac;
            }
            rw[i*stride2] = root;
        }

        {
            const DataType *c = datax + (2*i+1) * DEGREE1 * INTERVALS;
            DataType c0 = c[DEGREE*INTERVALS];
            DataType c1 = c[DEGREE*INTERVALS - INTERVALS];
    #pragma unroll
            for (int n = DEGREE-2; n > 0; n-=2) {
                const DataType c2 = c[n*INTERVALS] - c1;
                const DataType c3 = c0 + c1*u2;
                c1 = c2 + c3*u2;
                c0 = c[n*INTERVALS - INTERVALS] - c3;
            }
            DataType weight = c0 + c1*u;
            if constexpr(rys_type > 0){
                weight *= sqrt_theta_fac;
            }
            rw[i*stride2 + stride] = weight;
        }
    }
}
