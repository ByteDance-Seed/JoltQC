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

// Type traits for vectorized loads
template<typename T> struct VecType2 { };
template<> struct VecType2<float> { using type = float2; };
template<> struct VecType2<double> { using type = double2; };
using DataTypeVec2 = typename VecType2<DataType>::type;

template<typename T> struct VecType4 { };
template<> struct VecType4<float> { using type = float4; };
template<> struct VecType4<double> { using type = double4; };
using DataTypeVec4 = typename VecType4<DataType>::type;

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
        // Use DataTypeVec4 to load [R0, R1, W0, W1] for each root
        const DataTypeVec4 *smallx_data = reinterpret_cast<const DataTypeVec4*>(ROOT_SMALLX_DATA);
#pragma unroll
        for (int i = rt_id; i < nroots; i += nthreads_per_sq)  {
            const DataTypeVec4 vec4 = smallx_data[i];
            DataType root = vec4.x + vec4.y * x;   // R0 + R1 * x
            DataType weight = vec4.z + vec4.w * x; // W0 + W1 * x
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
        const DataType inv_x = one / x;
        const DataType t = SQRTPIE4 * sqrt(inv_x);
        // Use DataTypeVec2 to load [R, W] for each root
        const DataTypeVec2 *largex_data = reinterpret_cast<const DataTypeVec2*>(ROOT_LARGEX_DATA);
#pragma unroll
        for (int i = rt_id; i < nroots; i += nthreads_per_sq)  {
            const DataTypeVec2 vec2 = largex_data[i];
            DataType root = vec2.x * inv_x;   // R * inv_x
            DataType weight = vec2.y * t;     // W * t
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

        // Combine inv_x and inv_tt calculations to reduce FLOPs
        const DataType inv_x = one / x;
        const DataType fmt0 = (SQRTPIE4 / tt) * erf_tt;  // Combined division and multiplication

        DataType weight = fmt0;
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

    const int it = (int)(x * .4f);
    const DataType u = (x - it * DataType(2.5)) * DataType(0.8) - DataType(1.);
    const DataType u2 = u * two;

    // New layout: [NROOTS, INTERVALS, DEGREE1, 2 (interleaved root/weight)]
    // Use DataTypeVec4 for vectorized loads of 4 consecutive values

#pragma unroll
    for (int i = rt_id; i < nroots; i += nthreads_per_sq) {
        // Base address for interleaved root/weight data
        const int base = i * INTERVALS * DEGREE1 * 2 + it * DEGREE1 * 2;
        const DataTypeVec4 *c_data = reinterpret_cast<const DataTypeVec4*>(ROOT_RW_DATA + base);

        // Initial load of coefficients DEGREE and DEGREE-1
        // vec4 loads [root_{DEGREE-1}, weight_{DEGREE-1}, root_DEGREE, weight_DEGREE]
        DataTypeVec4 vec4 = c_data[DEGREE/2];
        DataType c0_r = vec4.z;  // root_DEGREE
        DataType c1_r = vec4.x;  // root_{DEGREE-1}
        DataType c0_w = vec4.w;  // weight_DEGREE
        DataType c1_w = vec4.y;  // weight_{DEGREE-1}

#pragma unroll
        for (int n = DEGREE-2; n > 0; n-=2) {
            // Load 4 consecutive values: [root_{n-1}, weight_{n-1}, root_n, weight_n]
            const DataTypeVec4 vec4 = c_data[(n-1)/2];

            // Process root polynomial
            const DataType c2_r = vec4.z - c1_r;     // root_n - c1_r
            const DataType c3_r = c0_r + c1_r * u2;
            c1_r = c2_r + c3_r * u2;
            c0_r = vec4.x - c3_r;                    // root_{n-1} - c3_r

            // Process weight polynomial
            const DataType c2_w = vec4.w - c1_w;     // weight_n - c1_w
            const DataType c3_w = c0_w + c1_w * u2;
            c1_w = c2_w + c3_w * u2;
            c0_w = vec4.y - c3_w;                    // weight_{n-1} - c3_w
        }

        // Final polynomial evaluation and optional scaling
        DataType root_val = c0_r + c1_r*u;
        DataType weight_val = c0_w + c1_w*u;

        if constexpr(rys_type > 0){
            root_val *= theta_fac;
            weight_val *= sqrt_theta_fac;
        }

        rw[i*stride2] = root_val;
        rw[i*stride2 + stride] = weight_val;
    }
}
