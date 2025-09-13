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
static void rys_roots(DataType x, DataType *rw, DataType theta, DataType omega)
{
    constexpr DataType one = 1.0;
    constexpr DataType two = 2.0;
    constexpr DataType half= .5;
    
    x *= theta;
    
    DataType theta_fac;
    DataType sqrt_theta_fac = one;
    if constexpr(rys_type > 0){
        const DataType omega2 = omega*omega;
        theta_fac = omega2 / (omega2 + theta);
        x *= theta_fac;
        sqrt_theta_fac = sqrt(theta_fac);
    }

    if (x < 3.e-7){
#pragma unroll
        for (int i = 0; i < nroots; i++)  {
            rw[2*i] = ROOT_SMALLX_R0[i] + ROOT_SMALLX_R1[i] * x;
            rw[2*i+1] = ROOT_SMALLX_W0[i] + ROOT_SMALLX_W1[i] * x;
            if constexpr(rys_type > 0){
                rw[2*i] *= theta_fac;
                rw[2*i+1] *= sqrt_theta_fac;
            }
        }
        return;
    }

    if (x > 35+nroots*5) {
        const DataType inv_x = one / x; 
        const DataType t = SQRTPIE4 * sqrt(inv_x);
#pragma unroll
        for (int i = 0; i < nroots; i++)  {
            rw[2*i] = ROOT_LARGEX_R_DATA[i] * inv_x;
            rw[2*i+1] = ROOT_LARGEX_W_DATA[i] * t;
            if constexpr(rys_type > 0){
                rw[2*i] *= theta_fac;
                rw[2*i+1] *= sqrt_theta_fac;
            }
        }
        return;
    }
    
    if constexpr(nroots == 1) {
        const DataType tt = sqrt(x);
        const DataType erf_tt = erf(tt);
        const DataType e = exp(-x);

        // Combine inv_x and inv_tt calculations to reduce FLOPs
        const DataType inv_x = one / x;
        const DataType fmt0 = (SQRTPIE4 / tt) * erf_tt;  // Combined division and multiplication
        rw[1] = fmt0;

        // Optimize the final division: fmt1/fmt0 = (half*inv_x*(fmt0-e))/fmt0 = half*inv_x*(1-e/fmt0)
        const DataType fmt1 = half * inv_x * (fmt0 - e);
        rw[0] = fmt1 / fmt0;
        if constexpr(rys_type > 0){
            rw[0] *= theta_fac;
            rw[1] *= sqrt_theta_fac;
        }
        return;
    }
    
    const int it = (int)(x * .4f);
    const DataType u = (x - it * DataType(2.5)) * DataType(0.8) - DataType(1.);
    const DataType u2 = u * two;
    const DataType *datax = ROOT_RW_DATA + it;
    
    // Pre-compute common values to reduce FLOPs
    const int addr_stride = DEGREE1 * INTERVALS;
    const int degree_intervals = DEGREE * INTERVALS;
    const int degree_intervals_minus = degree_intervals - INTERVALS;
    
#pragma unroll
    for (int i = 0; i < nroots; i++) {
        // Pre-compute base addresses to avoid repeated multiplication
        const int base_addr = (2*i) * addr_stride;
        const DataType *c_root = datax + base_addr;
        const DataType *c_weight = datax + base_addr + addr_stride;
        
        // Root and weight calculation - optimized polynomial evaluation
        DataType c0_r = c_root[degree_intervals];
        DataType c1_r = c_root[degree_intervals_minus];
        DataType c0_w = c_weight[degree_intervals];
        DataType c1_w = c_weight[degree_intervals_minus];
        
#pragma unroll
        for (int n = DEGREE-2; n > 0; n-=2) {
            // Process both root and weight polynomials in parallel
            const int n_intervals = n*INTERVALS;
            const int n_intervals_minus = n_intervals - INTERVALS;
            
            // Pre-compute shared multiplications
            const DataType c1_r_u2 = c1_r * u2;
            const DataType c1_w_u2 = c1_w * u2;
            
            // Root polynomial step
            const DataType c2_r = c_root[n_intervals] - c1_r;
            const DataType c3_r = c0_r + c1_r_u2;
            c1_r = c2_r + c3_r * u2;
            c0_r = c_root[n_intervals_minus] - c3_r;
            
            // Weight polynomial step  
            const DataType c2_w = c_weight[n_intervals] - c1_w;
            const DataType c3_w = c0_w + c1_w_u2;
            c1_w = c2_w + c3_w * u2;
            c0_w = c_weight[n_intervals_minus] - c3_w;
        }
        
        // Final polynomial evaluation and optional scaling
        DataType root_val = c0_r + c1_r*u;
        DataType weight_val = c0_w + c1_w*u;
        
        if constexpr(rys_type > 0){
            root_val *= theta_fac;
            weight_val *= sqrt_theta_fac;
        }
        
        rw[2*i] = root_val;
        rw[2*i+1] = weight_val;
    }
}
