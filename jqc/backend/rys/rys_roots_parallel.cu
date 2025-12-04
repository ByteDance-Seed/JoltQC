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
constexpr int DEGREE_FP64 = 5;
constexpr DataType SQRTPIE4 = .8862269254527580136;
constexpr DataType PIE4     = .7853981633974483096;

__device__ __forceinline__
static void rys_roots(DataType x, DataType *rw, int rt_id, const int stride, DataType theta, DataType omega)
{
    constexpr DataType one = 1.0;
    constexpr DataType two = 2.0;
    constexpr DataType half= .5;
    constexpr DataType small_x = 3.e-7;
    constexpr DataType large_x = nroots*5 + 35;
    const int stride2 = stride * 2;

    x *= theta;

    DataType theta_fac;
    DataType sqrt_theta_fac = one;
    if constexpr (rys_type > 0){
        const DataType omega2 = omega*omega;
        theta_fac = omega2 / (omega2 + theta);
        x *= theta_fac;
        sqrt_theta_fac = sqrt(theta_fac);
    }
    
    if (x < small_x) {
        const DataType *smallx_data = ROOT_SMALLX_DATA;
#pragma unroll
        for (int i = rt_id; i < nroots; i += nthreads_per_sq)  {
            const int base = i * 4;
            DataType root = smallx_data[base] + smallx_data[base+1] * x;       // R0 + R1 * x
            DataType weight = smallx_data[base+2] + smallx_data[base+3] * x;   // W0 + W1 * x
            if constexpr(rys_type > 0){
                root *= theta_fac;
                weight *= sqrt_theta_fac;
            }
            rw[i*stride2         ] = root;
            rw[i*stride2 + stride] = weight;
        }
        return;
    }
    
    if (x > large_x) {
        // Optimize using rsqrt: rsqrt(x) = 1/sqrt(x)
        const DataType inv_sqrt_x = rsqrt(x);
        const DataType inv_x = inv_sqrt_x * inv_sqrt_x;  // (1/sqrt(x))^2 = 1/x
        const DataType t = SQRTPIE4 * inv_sqrt_x;
        const DataType *largex_data = ROOT_LARGEX_DATA;
#pragma unroll
        for (int i = rt_id; i < nroots; i += nthreads_per_sq)  {
            const int base = i * 2;
            DataType root = largex_data[base] * inv_x;     // R * inv_x
            DataType weight = largex_data[base+1] * t;     // W * t
            if constexpr(rys_type > 0){
                root *= theta_fac;
                weight *= sqrt_theta_fac;
            }
            rw[i*stride2         ] = root;
            rw[i*stride2 + stride] = weight;
        }
        return;
    }
    /*
    if constexpr(nroots == 1) {
        // Optimize using rsqrt: rsqrt(x) = 1/sqrt(x)
        const DataType inv_sqrt_x = rsqrt(x);
        const DataType tt = x * inv_sqrt_x;  // x * (1/sqrt(x)) = sqrt(x)
        const DataType erf_tt = erf(tt);
        const DataType e = exp(-x);

        // Derive inv_x from inv_sqrt_x to avoid division
        const DataType inv_x = inv_sqrt_x * inv_sqrt_x;  // (1/sqrt(x))^2 = 1/x
        const DataType fmt0 = SQRTPIE4 * inv_sqrt_x * erf_tt;
        DataType weight = fmt0;

        // Optimize the final division: fmt1/fmt0 = (half*inv_x*(fmt0-e))/fmt0 = half*inv_x*(1-e/fmt0)
        const DataType fmt1 = half * inv_x * (fmt0 - e);
        DataType root = fmt1 / fmt0;

        if constexpr(rys_type > 0){
            root *= theta_fac;
            weight *= sqrt_theta_fac;
        }
        rw[0] = root;
        rw[stride] = weight;
        return;
    }
    */
    const int it = (int)(x * .4f);
    //const DataType u = (x - it * DataType(2.5)) * DataType(0.8) - DataType(1.);
    const DataType u = DataType(0.8) * x - DataType(2 * it + 1);
    const DataType u2 = u * two;

    // New layout: [NROOTS, INTERVALS, DEGREE1, 2 (interleaved root/weight)]

#pragma unroll
    for (int i = rt_id; i < nroots; i += nthreads_per_sq) {
        // Base address for interleaved root/weight data
        const int base = i * INTERVALS * DEGREE1 * 2 + it * DEGREE1 * 2;
        const DataType *c_data = ROOT_RW_DATA + base;

        // Evaluate Chebyshev series using Clenshaw's algorithm
        // Use float for n >= DEGREE_FP64 for performance
        float y2_r_f = 0;
        float y1_r_f = 0;
        float y2_w_f = 0;
        float y1_w_f = 0;
        float u2_f = u2;
#pragma unroll
        for (int n = DEGREE; n >= DEGREE_FP64; n--) {
            const float y0_r_f = (float)c_data[n * 2] + u2_f * y1_r_f - y2_r_f;
            const float y0_w_f = (float)c_data[n * 2 + 1] + u2_f * y1_w_f - y2_w_f;
            y2_r_f = y1_r_f;
            y1_r_f = y0_r_f;
            y2_w_f = y1_w_f;
            y1_w_f = y0_w_f;
        }

        // Use DataType for n < DEGREE_FP64 for precision
        DataType y2_r = y2_r_f;
        DataType y1_r = y1_r_f;
        DataType y2_w = y2_w_f;
        DataType y1_w = y1_w_f;
#pragma unroll
        for (int n = DEGREE_FP64-1; n >= 1; n--) {
            const DataType y0_r = c_data[n * 2] + u2 * y1_r - y2_r;
            const DataType y0_w = c_data[n * 2 + 1] + u2 * y1_w - y2_w;
            y2_r = y1_r;
            y1_r = y0_r;
            y2_w = y1_w;
            y1_w = y0_w;
        }

        // Final polynomial evaluation: C_0 + u*y1 - y2
        DataType root_val = c_data[0] + u * y1_r - y2_r;
        DataType weight_val = c_data[1] + u * y1_w - y2_w;

        if constexpr(rys_type > 0){
            root_val *= theta_fac;
            weight_val *= sqrt_theta_fac;
        }

        rw[i*stride2] = root_val;
        rw[i*stride2 + stride] = weight_val;
    }
}
