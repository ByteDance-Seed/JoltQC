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

// JoltQC ECP CUDA prelude
// Common declarations shared across ECP kernels

#ifndef JQC_ECP_PRELUDE_CUH
#define JQC_ECP_PRELUDE_CUH

// DataType must be defined by the including translation unit
// e.g., using DataType = double; or float

// PRIM_STRIDE and COORD_STRIDE must be defined via NVRTC -D options
#ifndef PRIM_STRIDE
#error "PRIM_STRIDE must be defined (e.g., -DPRIM_STRIDE=...)"
#endif
#ifndef COORD_STRIDE
#error "COORD_STRIDE must be defined (e.g., -DCOORD_STRIDE=...)"
#endif

typedef unsigned int uint32_t;

// Math constants fallback
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Pair-stride over (c,e) tuples
constexpr int prim_stride = PRIM_STRIDE / 2;

// Data structure definitions (after macros are defined)
struct __align__(2*sizeof(DataType)) DataType2 {
    DataType c, e;
};

struct __align__(COORD_STRIDE*sizeof(DataType)) DataType4 {
    DataType x, y, z;
#if COORD_STRIDE >= 4
    DataType w;
#endif
#if COORD_STRIDE > 4
    DataType _padding[COORD_STRIDE - 4];
#endif
};

// Forward declarations for quadrature arrays
extern __device__ double r128[128];
extern __device__ double w128[128];

#endif // JQC_ECP_PRELUDE_CUH
