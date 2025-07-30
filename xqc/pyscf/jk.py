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

'''
Generate JK kernels for PySCF
'''

import math
import numpy as np
import cupy as cp
from collections import Counter
from pyscf import lib
from gpu4pyscf.lib.cupy_helper import condense, transpose_sum
from gpu4pyscf.__config__ import num_devices
from gpu4pyscf.lib import logger
from gpu4pyscf.scf import jk
from xqc.backend.jk_tasks import generate_fill_tasks_kernel
from xqc.backend.jk import gen_jk_kernel
from xqc.pyscf.mol import format_bas_cache

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

def generate_jk_kernel(dtype=np.float64):
    cutoff_fp32 = 1e100
    def get_jk(mol, dm, hermi=0, vhfopt=None,
            with_j=True, with_k=True, omega=None, verbose=None):
        '''
        Compute J, K matrices
        '''
        assert with_j or with_k
        if omega is not None:
            assert omega >= 0.0, "short ranged J/K not supported"
        
        if vhfopt is None:
            _vhfopt = jk._VHFOpt(mol)
            _vhfopt.tile = TILE
            _vhfopt.build()
        else:
            assert vhfopt.tile == TILE

        bas_cache = format_bas_cache(_vhfopt.sorted_mol, dtype=dtype)
        coords, coeffs, exponents, ao_loc, _, _ = bas_cache

        uniq_l_ctr = _vhfopt.uniq_l_ctr
        uniq_l = uniq_l_ctr[:,0]
        l_ctr_bas_loc = _vhfopt.l_ctr_offsets
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = np.count_nonzero(uniq_l <= LMAX)
        if np.any(uniq_l > LMAX):
            raise RuntimeError('LMAX > 4 is not supported')

        log_cutoff_fp64 = np.float32(math.log(_vhfopt.direct_scf_tol))
        log_cutoff_fp32 = np.float32(math.log(cutoff_fp32))

        log = logger.new_logger(mol, verbose)
        log.debug(f"Compute J/K matrices with do_j={with_j} and do_k={with_k}, omega={omega}")
        t0 = log.init_timer()
        
        sorted_mol = _vhfopt.sorted_mol
        nbas = np.int32(sorted_mol.nbas)
        nao_orig = _vhfopt.mol.nao
        
        dm = cp.asarray(dm, order='C')
        dms = dm.reshape(-1,nao_orig,nao_orig)

        #:dms = cp.einsum('pi,nij,qj->npq', vhfopt.coeff, dms, vhfopt.coeff)
        dms = _vhfopt.apply_coeff_C_mat_CT(dms)

        device_id = cp.cuda.device.get_device_id()
        stream = cp.cuda.stream.get_current_stream()
        
        dm_cond = condense('absmax', dms, ao_loc)
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
        tile_q_cond = _vhfopt.tile_q_cond

        vj = vk = 0
        if with_k:
            vk = cp.zeros(dms.shape)
        if with_j:
            vj = cp.zeros(dms.shape)

        cutoff_a = log_cutoff_fp64 - log_max_dm
        cutoff_b = log_cutoff_fp32 - log_max_dm
        cutoff_range = [cutoff_a, cutoff_b]
        tile_mappings = _make_tril_tile_mappings(l_ctr_bas_loc, tile_q_cond, cutoff_range)
        info = cp.empty(2, dtype=np.uint32)
        t1 = log.timer_debug1(f'q_cond and dm_cond on Device {device_id}', *t0)
        
        _, _, gen_tasks_fun = generate_fill_tasks_kernel(do_j=with_j, do_k=with_k, tile=TILE)
        t1 = log.timer_debug1(f'Generate tasks kernel', *t1)
        timing_counter = Counter()
        kern_counts = 0
        quartet_list = cp.empty((QUEUE_DEPTH), dtype=int4_dtype)
        tasks = [(i,j,k,l)
                    for i in range(n_groups)
                    for j in range(i+1)
                    for k in range(i+1)
                    for l in range(k+1)]

        with stream:
            for task in tasks[::-1]:
                i, j, k, l = task
                tile_ij_mapping = tile_mappings[i,j]
                tile_kl_mapping = tile_mappings[k,l]

                li, ip = uniq_l_ctr[i]
                lj, jp = uniq_l_ctr[j]
                lk, kp = uniq_l_ctr[k]
                ll, lp = uniq_l_ctr[l]
                angs = (li, lj, lk, ll)
                nprim = (ip, jp, kp, lp)

                ntile_ij = tile_ij_mapping.size
                ntile_kl = tile_kl_mapping.size
                if ntile_kl == 0 or ntile_ij == 0:
                    continue
                ntasks = 0

                jk_kernel = gen_jk_kernel(angs, nprim, do_j=with_j, do_k=with_k, 
                                        dtype=dtype, n_dm=n_dm, omega=omega)
                
                PAIR_TILE_SIZE = MAX_PAIR_SIZE//(TILE*TILE)
                for t_ij0, t_ij1 in lib.prange(0, ntile_ij, PAIR_TILE_SIZE):
                    for t_kl0, t_kl1 in lib.prange(0, ntile_kl, PAIR_TILE_SIZE):
                        # TODO: early exit?
                        
                        # Create a task list containing significant quartets
                        gen_tasks_fun(
                            quartet_list, info, nbas, 
                            tile_ij_mapping[t_ij0:t_ij1], tile_kl_mapping[t_kl0:t_kl1],
                            _vhfopt.q_cond, log_dm_cond,
                            log_cutoff_fp64, log_cutoff_fp32)
                        kern_counts += 1
                        n_quartets = info[1].item()
                        if n_quartets <= 0:
                            continue

                        # Run the tasks
                        jk_kernel(nbas, ao_loc, coords, exponents, coeffs,
                                dms, vj, vk, omega, quartet_list, n_quartets)
                        kern_counts += 1
                        ntasks += n_quartets
                        
                if log.verbose >= logger.DEBUG2:
                    stream.synchronize()
                    llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                    msg = f'proc {llll}/({ip}{jp}{kp}{lp}), tasks = {ntasks} on Dev{device_id}'
                    t1, t1p = log.timer_debug1(msg, *t1), t1
                    timing_counter[llll] += t1[1] - t1p[1]
                if num_devices > 1:
                    stream.synchronize()
        
        stream.synchronize()
        if with_j:
            if hermi == 1:
                vj *= 2.
            else:
                vj, vjT = vj[:n_dm//2], vj[n_dm//2:]
                vj += vjT.transpose(0,2,1)
        if with_k:
            if hermi == 1:
                vk = transpose_sum(vk)
            else:
                vk, vkT = vk[:n_dm//2], vk[n_dm//2:]
                vk += vkT.transpose(0,2,1)
            
        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', kern_counts)
            for llll, t in timing_counter.items():
                log.debug1('%s wall time %.2f', llll, t)

        if with_j:
            vj = transpose_sum(vj)

        h_shls = _vhfopt.h_shls
        if h_shls:
            raise RuntimeError('h orbitals are not supported')

        if with_k:
            vk = _vhfopt.apply_coeff_CT_mat_C(vk)
            vk = vk.reshape(dm.shape)

        if with_j:
            vj = _vhfopt.apply_coeff_CT_mat_C(vj)
            vj = vj.reshape(dm.shape)
        log.timer('vj and vk', *t0)
        return vj, vk
    return get_jk

def _make_tril_tile_mappings(l_ctr_bas_loc, tile_q_cond, cutoff, tile=TILE):
    ''' Make tile mappings for the lower triangle of the J/K matrix
        Filter out the tiles that are not in the cutoff range [cutoff[0], cutoff[1]]
    '''
    assert len(cutoff) == 2
    n_groups = len(l_ctr_bas_loc) - 1
    ntiles = tile_q_cond.shape[0]
    tile_mappings = {}
    for i in range(n_groups):
        ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i+1]
        i0 = ish0 // tile
        i1 = ish1 // tile
        for j in range(i+1):    
            jsh0, jsh1 = l_ctr_bas_loc[j], l_ctr_bas_loc[j+1]
            j0 = jsh0 // tile
            j1 = jsh1 // tile
            sub_tile_q = tile_q_cond[i0:i1,j0:j1]
            mask = sub_tile_q > cutoff[0]
            mask &= sub_tile_q <= cutoff[1]
            if i == j:
                mask = cp.tril(mask)
            t_ij = (cp.arange(i0, i1, dtype=np.int32)[:,None] * ntiles +
                    cp.arange(j0, j1, dtype=np.int32))
            idx = cp.argsort(sub_tile_q[mask])[::-1]
            tile_mappings[i,j] = t_ij[mask][idx]
    return tile_mappings
