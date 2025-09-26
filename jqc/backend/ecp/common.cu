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
template <int l> __device__ __host__
Cartesian<(l+1)*(l+2)/2> ang_nuc_l(double rx, double ry, double rz){
    double rxPow[l+1], ryPow[l+1], rzPow[l+1];
    rxPow[0] = ryPow[0] = rzPow[0] = 1.0;
    for (int i = 1; i <= l; i++) {
        rxPow[i] = rxPow[i - 1] * rx;
        ryPow[i] = ryPow[i - 1] * ry;
        rzPow[i] = rzPow[i - 1] * rz;
    }

    double g[(l+1)*(l+2)/2];
    int index = 0;
    for (int i = l; i >= 0; i--) {
        for (int j = l - i; j >= 0; j--) {
            int k = l - i - j;
            g[index++] = rxPow[i] * ryPow[j] * rzPow[k];
        }
    }

    double c[2*l+1];
    cart2sph(c, l, g);
    Cartesian<(l+1)*(l+2)/2> omega;
    sph2cart(omega.data, l, c);
    return omega;
}
*/

__device__
double rad_part(const int ish, const int* __restrict__ ecpbas, const double* __restrict__ env){
    const int npk = ecpbas[ish*BAS_SLOTS+NPRIM_OF];
    const int r_order = ecpbas[ish*BAS_SLOTS+RADI_POWER];
    const int exp_ptr = ecpbas[ish*BAS_SLOTS+PTR_EXP];
    const int coeff_ptr = ecpbas[ish*BAS_SLOTS+PTR_COEFF];

    double u1 = 0.0;
    double r = 0.0;
    if (threadIdx.x < NGAUSS){
        r = r128[threadIdx.x];
    }
    for (int kp = 0; kp < npk; kp++){
        const double ak = env[exp_ptr+kp];
        const double ck = env[coeff_ptr+kp];
        u1 += ck * exp(-ak * r * r);
    }
    double w = 0.0;
    if (threadIdx.x < NGAUSS){
        w = w128[threadIdx.x];
    }
    return u1 * pow(r, r_order) * w;
}

template <int L> __device__
void cache_fac(double *fx, const double ri[3]){
    const int LI1 = L + 1;
    const int nfi = (LI1+1)*LI1/2;
    for (int i = threadIdx.x; i <= L; i+=blockDim.x){
        const int xoffset = i*(i+1)/2;
        const int yoffset = xoffset + nfi;
        for (int j = 0; j <= i; j++){
            const double bfac = _binom[xoffset+j]; // binom(i,j)
            fx[xoffset+j        ] = bfac * pow(ri[0], i-j);
            fx[yoffset+j        ] = bfac * pow(ri[1], i-j);
            fx[yoffset+j +   nfi] = bfac * pow(ri[2], i-j);
        }
    }
}

__device__
void block_reduce(double val, double* __restrict__ d_out) {
    __shared__ double warp_sums[THREADS / 32];
    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;

    // Step 1: Reduce within each warp using shuffle
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Step 2: Store warp results in shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // Step 3: Reduce across warps (only first warp participates)
    if (warp_id == 0) {
        val = (tid < (THREADS / 32)) ? warp_sums[tid] : 0.0;
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }

    // Step 4: First thread writes final result to global memory
    if (tid == 0) {
        d_out[0] += val;
    }
    __syncthreads();
}

__device__ __forceinline__
void set_shared_memory(double *smem, const int size) {
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        smem[i] = 0.0;
    }
    __syncthreads();
}

template <int LI_PARAM, int LJ_PARAM> __device__
void _li_up(double* __restrict__ out, double* __restrict__ buf){
    constexpr int nfj = (LJ_PARAM+1) * (LJ_PARAM+2) / 2;
    constexpr int nfi = (LI_PARAM+1) * (LI_PARAM+2) / 2;
    constexpr int nfi0 = LI_PARAM * (LI_PARAM+1) / 2;
    double *outx = out;
    double *outy = outx + nfi*nfj;
    double *outz = outy + nfi*nfj;
    for (int ij = threadIdx.x; ij < nfi0*nfj; ij+=blockDim.x){
        const int i = ij % nfi0;
        const int j = ij / nfi0;
        const double yfac = (_cart_pow_y[i] + 1);
        const double zfac = (_cart_pow_z[i] + 1);
        const double xfac = (LI_PARAM-1 - _cart_pow_y[i] - _cart_pow_z[i] + 1);

        atomicAdd(outx + j*nfi +          i, xfac * buf[j*nfi0 + i]);
        atomicAdd(outy + j*nfi + _y_addr[i], yfac * buf[j*nfi0 + i]);
        atomicAdd(outz + j*nfi + _z_addr[i], zfac * buf[j*nfi0 + i]);
    }
    __syncthreads();
}

template <int LI_PARAM, int LJ_PARAM> __device__
void _li_up_and_write(double* __restrict__ out, double* __restrict__ buf, const int nao){
    constexpr int nfi0 = LI_PARAM * (LI_PARAM+1) / 2;
    constexpr int nfj = (LJ_PARAM+1) * (LJ_PARAM+2) / 2;
    double *outxx = out ;
    double *outxy = out + nao*nao;
    double *outxz = out + 2*nao*nao;
    double *outyx = out + 3*nao*nao;
    double *outyy = out + 4*nao*nao;
    double *outyz = out + 5*nao*nao;
    double *outzx = out + 6*nao*nao;
    double *outzy = out + 7*nao*nao;
    double *outzz = out + 8*nao*nao;

    for (int ij = threadIdx.x; ij < nfi0*nfj; ij+=blockDim.x){
        const int i = ij % nfi0;
        const int j = ij / nfi0;
        const double yfac = (_cart_pow_y[i] + 1);
        const double zfac = (_cart_pow_z[i] + 1);
        const double xfac = (LI_PARAM-1 - _cart_pow_y[i] - _cart_pow_z[i] + 1);

        const int i_addr[3] = {i, _y_addr[i], _z_addr[i]};
        atomicAdd(outxx + j + i_addr[0]*nao, xfac * buf[j*nfi0 + i]);
        atomicAdd(outxy + j + i_addr[1]*nao, yfac * buf[j*nfi0 + i]);
        atomicAdd(outxz + j + i_addr[2]*nao, zfac * buf[j*nfi0 + i]);

        atomicAdd(outyx + j + i_addr[0]*nao, xfac * buf[j*nfi0 + i + nfi0*nfj]);
        atomicAdd(outyy + j + i_addr[1]*nao, yfac * buf[j*nfi0 + i + nfi0*nfj]);
        atomicAdd(outyz + j + i_addr[2]*nao, zfac * buf[j*nfi0 + i + nfi0*nfj]);

        atomicAdd(outzx + j + i_addr[0]*nao, xfac * buf[j*nfi0 + i + 2*nfi0*nfj]);
        atomicAdd(outzy + j + i_addr[1]*nao, yfac * buf[j*nfi0 + i + 2*nfi0*nfj]);
        atomicAdd(outzz + j + i_addr[2]*nao, zfac * buf[j*nfi0 + i + 2*nfi0*nfj]);
    }
    __syncthreads();
}


template <int LI_PARAM, int LJ_PARAM> __device__
void _li_down(double* __restrict__ out, double* __restrict__ buf){
    constexpr int nfi = (LI_PARAM+1) * (LI_PARAM+2) / 2;
    constexpr int nfj = (LJ_PARAM+1) * (LJ_PARAM+2) / 2;
    constexpr int nfi1= (LI_PARAM+2) * (LI_PARAM+3) / 2;
    double *outx = out;
    double *outy = outx + nfi*nfj;
    double *outz = outy + nfi*nfj;

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij % nfi;
        const int j = ij / nfi;
        atomicAdd(outx + j*nfi+i, buf[j*nfi1+i]);
        atomicAdd(outy + j*nfi+i, buf[j*nfi1+_y_addr[i]]);
        atomicAdd(outz + j*nfi+i, buf[j*nfi1+_z_addr[i]]);
    }
    __syncthreads();
}

template <int LI_PARAM, int LJ_PARAM> __device__
void _lj_down(double* __restrict__ out, double* __restrict__ buf){
    constexpr int nfi = (LI_PARAM+1) * (LI_PARAM+2) / 2;
    constexpr int nfj = (LJ_PARAM+1) * (LJ_PARAM+2) / 2;
    constexpr int nfj1= (LJ_PARAM+2) * (LJ_PARAM+3) / 2;
    double *outx = out;
    double *outy = outx + nfi*nfj;
    double *outz = outy + nfi*nfj;

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij % nfi;
        const int j = ij / nfi;
        const int j_addr[3] = {j, _y_addr[j], _z_addr[j]};
        atomicAdd(outx + j*nfi + i, buf[j_addr[0]*nfi + i]);
        atomicAdd(outy + j*nfi + i, buf[j_addr[1]*nfi + i]);
        atomicAdd(outz + j*nfi + i, buf[j_addr[2]*nfi + i]);
    }
    __syncthreads();
}


template <int LI_PARAM, int LJ_PARAM> __device__
void _li_down_and_write(double* __restrict__ out, double* __restrict__ buf, const int nao){
    constexpr int nfi = (LI_PARAM+1) * (LI_PARAM+2) / 2;
    constexpr int nfj = (LJ_PARAM+1) * (LJ_PARAM+2) / 2;
    constexpr int nfi1= (LI_PARAM+2) * (LI_PARAM+3) / 2;
    double *outxx = out ;
    double *outxy = out + nao*nao;
    double *outxz = out + 2*nao*nao;
    double *outyx = out + 3*nao*nao;
    double *outyy = out + 4*nao*nao;
    double *outyz = out + 5*nao*nao;
    double *outzx = out + 6*nao*nao;
    double *outzy = out + 7*nao*nao;
    double *outzz = out + 8*nao*nao;

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij % nfi;
        const int j = ij / nfi;
        const int i_addr[3] = {i, _y_addr[i], _z_addr[i]};

        atomicAdd(outxx + j + i*nao, buf[j*nfi1 + i_addr[0]]);
        atomicAdd(outxy + j + i*nao, buf[j*nfi1 + i_addr[1]]);
        atomicAdd(outxz + j + i*nao, buf[j*nfi1 + i_addr[2]]);

        atomicAdd(outyx + j + i*nao, buf[j*nfi1 + i_addr[0] + nfi1*nfj]);
        atomicAdd(outyy + j + i*nao, buf[j*nfi1 + i_addr[1] + nfi1*nfj]);
        atomicAdd(outyz + j + i*nao, buf[j*nfi1 + i_addr[2] + nfi1*nfj]);

        atomicAdd(outzx + j + i*nao, buf[j*nfi1 + i_addr[0] + 2*nfi1*nfj]);
        atomicAdd(outzy + j + i*nao, buf[j*nfi1 + i_addr[1] + 2*nfi1*nfj]);
        atomicAdd(outzz + j + i*nao, buf[j*nfi1 + i_addr[2] + 2*nfi1*nfj]);
    }
    __syncthreads();
}


template <int LI_PARAM, int LJ_PARAM> __device__
void _lj_up_and_write(double* __restrict__ out, double* __restrict__ buf, const int nao){
    constexpr int nfi = (LI_PARAM+1)*(LI_PARAM+2)/2;
    constexpr int nfj0 = LJ_PARAM * (LJ_PARAM+1) / 2;
    double *outxx = out;
    double *outxy = out + nao*nao;
    double *outxz = out + 2*nao*nao;
    double *outyx = out + 3*nao*nao;
    double *outyy = out + 4*nao*nao;
    double *outyz = out + 5*nao*nao;
    double *outzx = out + 6*nao*nao;
    double *outzy = out + 7*nao*nao;
    double *outzz = out + 8*nao*nao;
    for (int ij = threadIdx.x; ij < nfi*nfj0; ij+=blockDim.x){
        const int i = ij % nfi;
        const int j = ij / nfi;
        const double yfac = (_cart_pow_y[j] + 1);
        const double zfac = (_cart_pow_z[j] + 1);
        const double xfac = (LJ_PARAM-1 - _cart_pow_y[j] - _cart_pow_z[j] + 1);
        const int j_addr[3] = {j, _y_addr[j], _z_addr[j]};

        atomicAdd(outxx + j_addr[0] + nao*i, xfac * buf[j*nfi + i]);
        atomicAdd(outxy + j_addr[1] + nao*i, yfac * buf[j*nfi + i]);
        atomicAdd(outxz + j_addr[2] + nao*i, zfac * buf[j*nfi + i]);

        atomicAdd(outyx + j_addr[0] + nao*i, xfac * buf[j*nfi + i + nfi*nfj0]);
        atomicAdd(outyy + j_addr[1] + nao*i, yfac * buf[j*nfi + i + nfi*nfj0]);
        atomicAdd(outyz + j_addr[2] + nao*i, zfac * buf[j*nfi + i + nfi*nfj0]);

        atomicAdd(outzx + j_addr[0] + nao*i, xfac * buf[j*nfi + i + 2*nfi*nfj0]);
        atomicAdd(outzy + j_addr[1] + nao*i, yfac * buf[j*nfi + i + 2*nfi*nfj0]);
        atomicAdd(outzz + j_addr[2] + nao*i, zfac * buf[j*nfi + i + 2*nfi*nfj0]);
    }
    __syncthreads();
}

template <int LI_PARAM, int LJ_PARAM> __device__
void _lj_down_and_write(double* __restrict__ out, double* __restrict__ buf, const int nao){
    constexpr int nfi = (LI_PARAM+1) * (LI_PARAM+2) / 2;
    constexpr int nfj = (LJ_PARAM+1) * (LJ_PARAM+2) / 2;
    constexpr int nfj1 = (LJ_PARAM+2) * (LJ_PARAM+3) / 2;
    double *outxx = out ;
    double *outxy = out + nao*nao;
    double *outxz = out + 2*nao*nao;
    double *outyx = out + 3*nao*nao;
    double *outyy = out + 4*nao*nao;
    double *outyz = out + 5*nao*nao;
    double *outzx = out + 6*nao*nao;
    double *outzy = out + 7*nao*nao;
    double *outzz = out + 8*nao*nao;
    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij % nfi;
        const int j = ij / nfi;
        const int j_addr[3] = {j, _y_addr[j], _z_addr[j]};

        atomicAdd(outxx + j + i*nao, buf[j_addr[0]*nfi + i]);
        atomicAdd(outxy + j + i*nao, buf[j_addr[1]*nfi + i]);
        atomicAdd(outxz + j + i*nao, buf[j_addr[2]*nfi + i]);

        atomicAdd(outyx + j + i*nao, buf[j_addr[0]*nfi + i + nfi*nfj1]);
        atomicAdd(outyy + j + i*nao, buf[j_addr[1]*nfi + i + nfi*nfj1]);
        atomicAdd(outyz + j + i*nao, buf[j_addr[2]*nfi + i + nfi*nfj1]);

        atomicAdd(outzx + j + i*nao, buf[j_addr[0]*nfi + i + 2*nfi*nfj1]);
        atomicAdd(outzy + j + i*nao, buf[j_addr[1]*nfi + i + 2*nfi*nfj1]);
        atomicAdd(outzz + j + i*nao, buf[j_addr[2]*nfi + i + 2*nfi*nfj1]);
    }
    __syncthreads();
}
