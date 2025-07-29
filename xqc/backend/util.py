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
    xyz = [(x, y, n-x-y)
            for x in reversed(range(n+1))
            for y in reversed(range(n+1-x))]
    return np.array(xyz)

def pack3int(idx):
    ''' Store in (x,y,z) as ONE integer
    '''
    return (idx[0] & 0x3FF) | ((idx[1] & 0x3FF) << 10) | ((idx[2] & 0x3FF) << 20)

def unpack3int(idx):
    return idx & 0x3FF, (idx >> 10) & 0x3FF, (idx >> 20) & 0x3FF

def g_pair_idx(ij_inc=None):
    dat = []
    xyz = [iter_cart_xyz(li) for li in range(LMAX+1)]
    for li in range(LMAX+1):
        for lj in range(LMAX+1):
            li1 = li + 1
            idx = (xyz[lj][:,None] * li1 + xyz[li]).transpose(2,0,1)
            idx = pack3int(idx)
            dat.append(idx.ravel())
    g_idx = np.hstack(dat).astype(np.int32)
    offsets = np.cumsum([0] + [x.size for x in dat]).astype(np.int32)
    offsets = offsets[:-1]
    g_idx_pinned = cupyx.empty_like_pinned(g_idx)
    g_idx_pinned[:] = g_idx
    offsets_pinned = cupyx.empty_like_pinned(offsets)
    offsets_pinned[:] = offsets
    return g_idx_pinned, offsets_pinned

g_idx, offsets = g_pair_idx()

def get_ij_pair(li,lj):
    li1 = li + 1
    xyz = [iter_cart_xyz(li) for li in range(LMAX+1)]
    idx = (xyz[lj][:,None] * li1 + xyz[li]).transpose(2,0,1)
    idx = pack3int(idx)
    return idx

shell_idx = {}
for li in range(LMAX+1):
    ixyz = iter_cart_xyz(li)
    ixyz = pack3int(ixyz.T)
    shell_idx[li] = ixyz
    
def generate_lookup_table(li, lj, lk, ll):
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
