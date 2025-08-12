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

# Portions of this file adapted from GPU4PySCF (https://github.com/pyscf/gpu4pyscf)
# Copyright 2025 PySCF developer.
# Licensed under the Apache License, Version 2.0.

'''
Generate JK kernels for PySCF
'''

import time
import math
import numpy as np
import cupy as cp
import logging
from collections import Counter
from pyscf import lib
from xqc.backend.linalg_helper import inplace_add_transpose, max_block_pooling
from xqc.backend.jk_tasks import generate_fill_tasks_kernel
from xqc.backend.jk import gen_jk_kernel
from xqc.backend.cart2sph import mol2cart, cart2mol
from xqc.pyscf.mol import compute_q_matrix, create_sorted_basis

__all__ = [
    'get_jk', 'get_j',
]

int4_dtype = np.dtype([('x', 'i4'), ('y', 'i4'), ('z', 'i4'), ('w', 'i4')])

LMAX = 4
TILE = 4
MAX_PAIR_SIZE = 16384
QUEUE_DEPTH = MAX_PAIR_SIZE * MAX_PAIR_SIZE # 4 GB
PTR_BAS_COORD = 7
GROUP_SIZE = 256
NPRIM_MAX = 16

logger = logging.getLogger(__name__)

def generate_jk_kernel(mol, dtype=np.float64):
    # TODO: mixed-precision
    cutoff_fp32 = 1e100
    direct_scf_tol = 1e-13  # TODO: use the parameter in mf object
    bas_cache, bas_mapping, padding_mask, group_info = create_sorted_basis(mol, alignment=TILE, dtype=dtype)
    
    # TODO: Q matrix for short-range
    q_matrix = compute_q_matrix(mol)
    nbas = bas_mapping.shape[0]
    nao_orig = mol.nao
    q_matrix = q_matrix[bas_mapping[:,None], bas_mapping]
    q_matrix[padding_mask, :] = -100
    q_matrix[:, padding_mask] = -100  # set the Q matrix for padded basis to -100
    q_matrix = cp.asarray(q_matrix, dtype=np.float32)
    def get_jk(mol_ref, dm, hermi=0, vhfopt=None,
            with_j=True, with_k=True, omega=None, verbose=None):
        '''
        Compute J, K matrices
        
        Args:
            mol_ref: pyscf Mole object
            dm: density matrix
            hermi: hermiticity of the density matrix
            vhfopt: vhfopt object
            with_j: whether to compute J matrix
            with_k: whether to compute K matrix
            omega: short ranged J/K parameter
            verbose: verbose level, for compatibility only
        '''
        assert with_j or with_k
        if omega is not None:
            assert omega >= 0.0, "short ranged J/K not supported"

        assert mol_ref == mol, "mol_ref must be the same as mol"
        cputime_start  = time.perf_counter()
        
        _, _, coords, angs, _ = bas_cache
        group_key, group_offset = group_info
        uniq_l = group_key[:,0]
        l_ctr_bas_loc = group_offset
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)

        if np.any(uniq_l > LMAX):
            raise RuntimeError('LMAX > 4 is not supported')

        log_cutoff_fp64 = np.float32(math.log(direct_scf_tol))
        log_cutoff_fp32 = np.float32(math.log(cutoff_fp32))
        logger.debug(f"Compute J/K matrices with do_j={with_j} and do_k={with_k}, omega={omega}")
        
        dm = cp.asarray(dm, order='C')
        dms = dm.reshape(-1,nao_orig,nao_orig)

        ao_loc = np.concatenate(([0], np.cumsum((angs+1)*(angs+2)//2)))
        nao = ao_loc[-1]
        ao_loc = cp.asarray(ao_loc, dtype=np.int32)

        # Convert the density matrix to cartesian basis
        dms = mol2cart(dms, angs, ao_loc, bas_mapping, mol)
        dms = dms.reshape(-1, nao, nao)
        dm_cond = max_block_pooling(dms, ao_loc)
        dms = cp.asarray(dms, dtype=dtype) # transfer to current device
        if hermi == 0:
            # Wrap the triu contribution to tril
            dm_cond = dm_cond + dm_cond.T
        log_dm_cond = cp.log(dm_cond + 1e-300).astype(np.float32)
        log_max_dm = log_dm_cond.max().item()
        log_dm_cond = cp.asarray(log_dm_cond, dtype=np.float32)

        if hermi == 0:
            # Contract the tril and triu parts separately
            dms = cp.vstack([dms, dms.transpose(0,2,1)])
        n_dm = dms.shape[0]

        vj = vk = 0
        if with_k:
            vk = cp.zeros(dms.shape)
        if with_j:
            vj = cp.zeros(dms.shape)

        cutoff_a = log_cutoff_fp64 - log_max_dm
        cutoff_b = log_cutoff_fp32 - log_max_dm
        cutoff_range = [cutoff_a, cutoff_b]
        tile_pairs = make_tile_pairs(l_ctr_bas_loc, q_matrix, cutoff_range)
        info = cp.empty(2, dtype=np.uint32)
        logger.debug(f'q_cond and dm_cond')
        
        _, _, gen_tasks_fun = generate_fill_tasks_kernel(do_j=with_j, do_k=with_k, tile=TILE)
        logger.debug(f'Generate tasks kernel')
        timing_counter = Counter()
        kern_counts = 0
        quartet_list = cp.empty((QUEUE_DEPTH), dtype=int4_dtype)
        tasks = [(i,j,k,l) for i in range(n_groups)
                            for j in range(i+1)
                            for k in range(i+1)
                            for l in range(k+1)]
        
        stream = cp.cuda.stream.get_current_stream()
        with stream:
            for task in tasks[::-1]:
                i, j, k, l = task
                li, ip = group_key[i]
                lj, jp = group_key[j]
                lk, kp = group_key[k]
                ll, lp = group_key[l]
                ang = (li, lj, lk, ll)
                nprim = (ip, jp, kp, lp)

                tile_ij = tile_pairs[i,j]
                tile_kl = tile_pairs[k,l]
                ntile_ij = tile_ij.size
                ntile_kl = tile_kl.size
                if ntile_kl == 0 or ntile_ij == 0:
                    continue
                ntasks = 0

                jk_kernel = gen_jk_kernel(ang, nprim, do_j=with_j, do_k=with_k, 
                                        dtype=dtype, n_dm=n_dm, omega=omega)
                
                # Setup events for timing the critical kernels
                start = end = None
                if logger.level > logging.INFO:
                    start = cp.cuda.Event()
                    end = cp.cuda.Event()
                    start.record()

                # Breakdown the tile pairs into smaller chunks
                PAIR_TILE_SIZE = MAX_PAIR_SIZE//(TILE*TILE)
                for t_ij0, t_ij1 in lib.prange(0, ntile_ij, PAIR_TILE_SIZE):
                    for t_kl0, t_kl1 in lib.prange(0, ntile_kl, PAIR_TILE_SIZE):
                        # TODO: early exit if the chunk is negligible?
                        # Create a task list containing significant quartets
                        gen_tasks_fun(
                            quartet_list, info, nbas, 
                            tile_ij[t_ij0:t_ij1], tile_kl[t_kl0:t_kl1],
                            q_matrix, log_dm_cond,
                            log_cutoff_fp64, log_cutoff_fp32)
                        kern_counts += 1
                        n_quartets = info[1].item()
                        if n_quartets <= 0:
                            continue

                        # Run the tasks
                        coeffs, exponents, coords, _, _ = bas_cache
                        jk_kernel(nbas, ao_loc, coords, exponents, coeffs,
                                dms, vj, vk, omega, quartet_list, n_quartets)
                        kern_counts += 1
                        ntasks += n_quartets
                        
                if logger.level > logging.INFO:
                    stream.synchronize()
                    end.record()
                    end.synchronize()
                    elasped_time = cp.cuda.Event.elapsed_time(start, end)

                    llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                    msg = f'proc {llll}/({ip}{jp}{kp}{lp}), tasks = {ntasks}, time = {elasped_time:.2f} ms'
                    logger.debug(msg)
                    timing_counter[llll] += elasped_time
        
        stream.synchronize()

        # Remove the padding basis
        ao_loc = ao_loc[:-1][~padding_mask]
        angs0 = angs[~padding_mask]
        bmap = bas_mapping[~padding_mask]

        if with_j:
            if hermi == 1:
                vj *= 2.
            else:
                vj, vjT = vj[:n_dm//2], vj[n_dm//2:]
                vj += vjT.transpose(0,2,1)
            vj = inplace_add_transpose(vj)
            vj = cart2mol(vj, angs0, ao_loc, bmap, mol)
            vj = vj.reshape(dm.shape)

        if with_k:
            if hermi == 1:
                vk = inplace_add_transpose(vk)
            else:
                vk, vkT = vk[:n_dm//2], vk[n_dm//2:]
                vk += vkT.transpose(0,2,1)
            vk = cart2mol(vk, angs0, ao_loc, bmap, mol)
            vk = vk.reshape(dm.shape)
        
        if logger.level >= logging.DEBUG:
            logger.debug('kernel launches %d', kern_counts)
            for llll, t in timing_counter.items():
                logger.debug('%s wall time %.2f', llll, t)

        cputime_end  = time.perf_counter()
        logger.debug(f'vj and vk take {cputime_end - cputime_start:.2f} s')
        return vj, vk
    return get_jk

def make_tile_pairs(l_ctr_bas_loc, q_matrix, cutoff, tile=TILE):
    ''' Make tile pairs with GTO pairing screening and symmetry
        Filter out the tiles that are not in the cutoff range [cutoff[0], cutoff[1]]
    '''
    assert len(cutoff) == 2
    assert q_matrix.shape[0] % tile == 0
    tile_pairs = {}
    n_groups = len(l_ctr_bas_loc) - 1
    ntiles = q_matrix.shape[0] // tile
    tile_loc = l_ctr_bas_loc // tile
    tiled_q_matrix = q_matrix.reshape([ntiles, tile, ntiles, tile]).max(axis=(1,3))
    for i in range(n_groups):
        i0, i1 = tile_loc[i], tile_loc[i+1]
        for j in range(i+1):
            j0, j1 = tile_loc[j], tile_loc[j+1]
            sub_tile_q = tiled_q_matrix[i0:i1,j0:j1]
            mask = sub_tile_q > cutoff[0]
            mask &= sub_tile_q <= cutoff[1]
            if i == j:
                mask = cp.tril(mask)
            t_ij = (cp.arange(i0, i1, dtype=np.int32)[:,None] * ntiles +
                    cp.arange(j0, j1, dtype=np.int32))
            idx = cp.argsort(sub_tile_q[mask])[::-1]
            tile_pairs[i,j] = t_ij[mask][idx]
    return tile_pairs

