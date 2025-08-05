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

import numpy as np
import cupyx

LMAX = 4

def iter_cart_xyz(n):
    """Generates Cartesian exponents (lx, ly, lz) for a given angular momentum.

    Args:
        n (int): The total angular momentum.

    Returns:
        np.ndarray: An array of shape ((n+1)*(n+2)//2, 3) with all
                    (lx, ly, lz) combinations such that lx + ly + lz = n.
    """
    xyz = [(x, y, n-x-y)
            for x in reversed(range(n+1))
            for y in reversed(range(n+1-x))]
    return np.array(xyz)

def pack3int(idx):
    """Packs three 10-bit integers into a single 32-bit integer.

    This is used to store Cartesian exponents (x, y, z) in a compact format
    for efficient lookup in CUDA kernels. Each of x, y, and z must be in
    the range [0, 1023].

    Args:
        idx (np.ndarray): A (3, N) array of integers to be packed.

    Returns:
        int or np.ndarray: The packed integer(s).
    """
    return (idx[0] & 0x3FF) | ((idx[1] & 0x3FF) << 10) | ((idx[2] & 0x3FF) << 20)

def unpack3int(idx):
    """Unpacks a 32-bit integer into three 10-bit integers.

    This is the inverse operation of pack3int.

    Args:
        idx (int): The packed integer.

    Returns:
        tuple[int, int, int]: The unpacked (x, y, z) integers.
    """
    return idx & 0x3FF, (idx >> 10) & 0x3FF, (idx >> 20) & 0x3FF

shell_idx = {}
for li in range(LMAX+1):
    ixyz = iter_cart_xyz(li)
    ixyz = pack3int(ixyz.T)
    shell_idx[li] = ixyz
    
def generate_lookup_table(li, lj, lk, ll):
    """Generates C++ code for basis function index lookup tables.

    These tables are intended to be used as `__device__` constants in CUDA
    kernels for mapping multi-dimensional indices to a flat layout, which is
    useful for ERI (electron repulsion integral) calculations.

    Args:
        li, lj, lk, ll (int): Angular momenta for the four shells (i, j, k, l).

    Returns:
        str: A string containing the C++ code for the lookup tables.
    """
        
    nfi = (li+1)*(li+2)//2
    nfj = (lj+1)*(lj+2)//2
    nfk = (lk+1)*(lk+2)//2
    nfl = (ll+1)*(ll+2)//2

    i_idx = shell_idx[li]
    j_idx = shell_idx[lj] * (li+1)
    k_idx = shell_idx[lk] * (li+1) * (lj+1)
    l_idx = shell_idx[ll] * (li+1) * (lj+1) * (lk+1)

    i_idx_str = ', '.join(f'{x}' for x in i_idx)
    j_idx_str = ', '.join(f'{x}' for x in j_idx)
    k_idx_str = ', '.join(f'{x}' for x in k_idx)
    l_idx_str = ', '.join(f'{x}' for x in l_idx)

    idx_code = f'''
constexpr __device__ uint32_t i_idx[{nfi}] = {{ {i_idx_str} }};
constexpr __device__ uint32_t j_idx[{nfj}] = {{ {j_idx_str} }};
constexpr __device__ uint32_t k_idx[{nfk}] = {{ {k_idx_str} }};
constexpr __device__ uint32_t l_idx[{nfl}] = {{ {l_idx_str} }};
    '''
    return idx_code
