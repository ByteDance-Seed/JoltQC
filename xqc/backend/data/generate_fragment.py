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

import json
import numpy as np
import cupy as cp
from pyscf import gto, lib
from xqc.backend import jk_1qnt as jk_algo1
from xqc.backend import jk_1q1t as jk_algo0

'''
Script for greedy search the optimal fragmentation for the kernel.
'''

'''
How to run ?

> python3 generate_fragment.py {li} {lj} {lk} {ll} fp64 > logs/{li}{lj}{lk}{ll}_fp64.log

e.g.
> python3 generate_fragment.py 1 1 1 1 fp64 > logs/1111_fp64.log
or 
> python3 generate_fragment.py 1 1 1 1 fp32 > logs/1111_fp32.log
'''

def generate_fragments(ang, max_threads = 256):
    ''' Create a tile scheme satisfy three conditions
    1. Cover (nfi, nfj, nfk, nfl), # of threads < max_threads, 
       (Note: possibly use a lot of local memory)
    2. Shared memory < max_shmem_per_block
    3. At least 4 threads for each quartet
    '''
    li, lj, lk, ll = ang
    nf = np.empty(4, dtype=np.int32)
    nf[0] = (li+1)*(li+2)//2
    nf[1] = (lj+1)*(lj+2)//2
    nf[2] = (lk+1)*(lk+2)//2
    nf[3] = (ll+1)*(ll+2)//2

    for fi in range(1,nf[0]+1):
        for fj in range(1,nf[1]+1):
            for fk in range(1,nf[2]+1):
                for fl in range(1,nf[3]+1):
                    # early exit if the fragment size is greater than 256,
                    # as it will use too many registers
                    if fi * fj * fk * fl > 256:
                        continue
                    fragments = np.array([fi,fj,fk,fl], dtype=np.int32)
                    # integral dimension must be a multiple of fragment size
                    if nf[0] % fi != 0 or nf[1] % fj != 0 or nf[2] % fk != 0 or nf[3] % fl != 0:
                        continue
                    nthreads = (nf + fragments - 1) // fragments
                    if np.prod(nthreads) > max_threads:
                        continue
                    if np.prod(nthreads) < 4:
                        continue
                    yield fragments

from xqc.pyscf import jk
from xqc.pyscf.mol import format_bas_cache
from gpu4pyscf.scf.jk import _VHFOpt

def update_frags(i,j,k,l,dtype_str):
    if dtype_str=='fp32':
        dtype = np.float32
    elif dtype_str=='fp64':
        dtype = np.float64
    else:
        raise RuntimeError(f'Data type {dtype_str} is not supported')

    basis = gto.basis.parse('''
    H    S
        0.10307241             1.0000000
        0.10307241             1.0000000
        0.10307241             1.0000000
    H    P
        0.40700000             1.0000000
        0.10307241             1.0000000
        0.10307241             1.0000000
    H    D
        0.05700000             1.0000000
    H    F
        0.05700000             1.0000000
    H    G
        0.05700000             1.0000000
                                ''')
    xyz = 'gly30.xyz'
    mol = gto.M(atom=xyz, basis=basis, unit='Angstrom')
    opt = _VHFOpt(mol).build()
    mol = opt.sorted_mol

    nbas = mol.nbas
    nfragments = nbas//jk.TILE
    dm_cond = cp.ones([nbas, nbas], dtype=np.float32)
    q_cond = cp.ones([nbas, nbas], dtype=np.float32)
    tile_q_cond = cp.ones([nfragments, nfragments], dtype=np.float32)
    log_cutoff = -12
    uniq_l_ctr = opt.uniq_l_ctr
    l_ctr_bas_loc = opt.l_ctr_offsets
    cutoff = np.asarray([-30, 100])
    tile_mappings = jk._make_tril_tile_mappings(l_ctr_bas_loc, tile_q_cond, cutoff)
    nao = mol.nao
    dms = cp.empty([nao,nao])

    vj = cp.empty_like(dms)
    vk = cp.empty_like(dms)

    bas_cache = format_bas_cache(opt.sorted_mol, dtype=dtype)
    coords, coeffs, exponents, ao_loc, _, _ = bas_cache

    tile_ij_mapping = tile_mappings[i,j][:256]
    tile_kl_mapping = tile_mappings[k,l][:256]
    
    li, ip = uniq_l_ctr[i]
    lj, jp = uniq_l_ctr[j]
    lk, kp = uniq_l_ctr[k] 
    ll, lp = uniq_l_ctr[l]
    ang = (li, lj, lk, ll)
    nprim = (ip, jp, kp, lp)
    best_time = 1e100
    best_frag = None

    from xqc.backend.jk_tasks import generate_fill_tasks_kernel
    script, kernel, gen_tasks_fun = generate_fill_tasks_kernel(tile=jk.TILE)
    QUEUE_DEPTH = jk.QUEUE_DEPTH
    cp.get_default_memory_pool().free_all_blocks()
    pool = cp.empty((QUEUE_DEPTH), dtype=jk.int4_dtype)
    info = cp.zeros(2, dtype=np.int32)

    gen_tasks_fun(
        pool, info, np.int32(mol.nbas), 
        tile_ij_mapping, tile_kl_mapping,
        q_cond, dm_cond,
        np.float32(log_cutoff), np.float32(100)
    )

    max_shm = 48*1024 # 48 KB for compatibility
    # Measure GPU time of algorithm 1 with different fragments
    for frag in generate_fragments(ang):
        try:
            script, kernel, fun = jk_algo1.gen_kernel(
                ang, nprim, 
                frags=frag, dtype=dtype, 
                max_shm=max_shm)
        except:
            print(f"failed to generate kernel {ang}/{nprim} with frag {frag}")
            continue
        args = (np.int32(mol.nbas), ao_loc, coords, exponents, coeffs,
                dms, vj, vk, np.double(0.0), pool, info[1].get())
        fun(*args)
        
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        fun(*args)
        end.record()
        end.synchronize()
        elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
        if elapsed_time_ms < best_time:
            best_time = elapsed_time_ms
            best_frag = frag
        print(f'{ang} : algorithm 1qnt with frag {frag} takes \
{elapsed_time_ms:.3f}ms, best time: {best_time:.3f}ms')
    
    # No need to check 1q1t algorithm for low-angular momentum
    if (li+lj+lk+ll)//2 + 1 < 5:
        # Measure time of algorithm 0
        script, kernel, fun = jk_algo0.gen_kernel(ang, nprim, dtype=dtype)
        kernel.compile()
        args = (np.int32(mol.nbas), ao_loc, coords, exponents, coeffs,
                dms, vj, vk, np.double(0.0), pool, info[1].get())
        fun(*args)
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        fun(*args)
        end.record()
        end.synchronize()
        elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
        if elapsed_time_ms < best_time:
            best_time = elapsed_time_ms
            best_frag = np.array([-1])
        print(f'{ang} : algorithm 1q1t takes {elapsed_time_ms:.3f}:ms, best time: {best_time:.3f}ms')
    print('Optimal frag:', best_frag)
    with open(f'optimal_scheme_{dtype_str}.json', 'r') as f:
        data = json.load(f)
    ang_num = 1000*li + 100*lj + 10*lk + ll
    data.update({int(ang_num): best_frag.tolist()})
    with open(f'optimal_scheme_{dtype_str}.json', 'w') as f: 
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    import sys
    li = int(sys.argv[1])
    lj = int(sys.argv[2])
    lk = int(sys.argv[3])
    ll = int(sys.argv[4])
    dtype = str(sys.argv[5])
    update_frags(li, lj, lk, ll, dtype)
