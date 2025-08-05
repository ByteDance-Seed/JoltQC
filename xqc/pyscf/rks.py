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
Generate DFT kernels for PySCF
'''

import numpy as np
import math
import cupy as cp
from gpu4pyscf.scf.jk import _VHFOpt
from gpu4pyscf.lib import logger
from gpu4pyscf.dft.gen_grid import arg_group_grids
from xqc.backend.linalg_helper import max_block_pooling, inplace_add_transpose
from xqc.pyscf.mol import format_bas_cache
from xqc.backend.rks import gen_rho_kernel, gen_vxc_kernel, estimate_log_aovalue

__all__ = ['build_grids', 'generate_rks_kernel']

rho_vxc_cutoff = 1e-13
LMAX = 4

def build_grids(grids, mol=None, with_non0tab=False, sort_grids=True, **kwargs):
    if mol is None: mol = grids.mol
    if grids.verbose >= logger.WARN:
        grids.check_sanity()
    log = logger.new_logger(grids)
    t0 = log.init_timer()
    atom_grids_tab = grids.gen_atomic_grids(
        mol, grids.atom_grid, grids.radi_method, grids.level, grids.prune, **kwargs)
    grids.coords, grids.weights = grids.get_partition(
        mol, atom_grids_tab, grids.radii_adjust, grids.atomic_radii, grids.becke_scheme)

    atm_idx = cp.empty(grids.coords.shape[0], dtype=np.int32)
    quadrature_weights = cp.empty(grids.coords.shape[0])
    p0 = p1 = 0
    for ia in range(mol.natm):
        r, vol = atom_grids_tab[mol.atom_symbol(ia)]
        p0, p1 = p1, p1 + vol.size
        atm_idx[p0:p1] = ia
        quadrature_weights[p0:p1] = vol
    grids.atm_idx = atm_idx
    grids.quadrature_weights = quadrature_weights

    t0 = log.timer_debug1('generating atomic grids', *t0)
    ngrids = grids.coords.shape[0]
    alignment = grids.alignment
    if alignment > 1:
        padding = (ngrids + alignment - 1) // alignment * alignment - ngrids
        log.debug('Padding %d grids', padding)
        if padding > 0:
            grids.coords = cp.vstack([grids.coords, cp.full((padding, 3), 1e-4)])
            grids.weights = cp.hstack([grids.weights, cp.zeros(padding)])
            grids.quadrature_weights = cp.hstack([grids.quadrature_weights, cp.zeros(padding)])
            grids.atm_idx = cp.hstack([grids.atm_idx, cp.full(padding, -1, dtype=np.int32)])

    if sort_grids:
        idx = arg_group_grids(mol, grids.coords, box_size=1.0)
        grids.coords = grids.coords[idx]
        grids.weights = grids.weights[idx]
        grids.quadrature_weights = grids.quadrature_weights[idx]
        grids.atm_idx = grids.atm_idx[idx]
        t0 = log.timer_debug1('sorting grids', *t0)

    if with_non0tab:
        raise RuntimeError('with_non0tab is not supported yet')
    else:
        grids.screen_index = grids.non0tab = None
    log.info('tot grids = %d', len(grids.weights))

    grids._non0ao_idx = None
    return grids

def generate_rks_kernel(mol, dtype=np.float64, xc_type='LDA'):
    rho_fun, vxc_fun = generate_dft_kernel(mol, dtype, xc_type)
    _cache = {}
    def rks_fun(ni, mol, grids, xc_code, dm, max_memory=2000, verbose=None):
        """ rks kernel for PySCF, with incremental DFT implementation
        
        Args:
            ni: PySCF numint object, a placeholder for compatibility only
            mol: PySCF molecule object
            grids: PySCF grid object
            xc_code: xc code
            dm: density matrix
            max_memory: max memory in MB, placeholder for compatibility only
            verbose: verbose level
        
        Returns:
            nelec: number of electrons
            excsum: exchange correlation energy
            vxcmat: vxc matrix
        """
        if 'mol' not in _cache or _cache['mol'] != mol:
            _cache['mol'] = mol
            _cache['dm_prev'] = 0
            _cache['rho_prev'] = 0
            _cache['wv_prev'] = 0
            _cache['vxcmat_prev'] = 0
            
        dm_prev = _cache['dm_prev']
        rho_prev = _cache['rho_prev']
        wv_prev = _cache['wv_prev']
        vxcmat_prev = _cache['vxcmat_prev']

        # Evaluate rho on grids for given density matrix
        xctype = ni._xc_type(xc_code)
        weights = grids.weights
        dm_diff = dm - dm_prev
        rho_diff = rho_fun(None, mol, dm_diff, grids)
        rho = rho_prev + rho_diff

        # Evaluate vxc on grids via libxc
        exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[:2]

        # Integrate vxc on grids
        den = rho[0] * weights
        vxc = cp.asarray(vxc, order='C')
        exc = cp.asarray(exc, order='C')
        excsum = float(cp.dot(den, exc[:,0]))
        nelec = float(den.sum())
        wv = vxc * weights
        wv_diff = wv - wv_prev
        vxcmat_diff = vxc_fun(None, mol, wv_diff, grids)
        vxcmat = vxcmat_prev + vxcmat_diff

        _cache['dm_prev'] = dm.copy()
        _cache['rho_prev'] = rho
        _cache['wv_prev'] = wv
        _cache['vxcmat_prev'] = vxcmat.copy()
        return nelec, excsum, vxcmat[0]
    return rks_fun

def generate_dft_kernel(mol, dtype=np.float64, xc_type='LDA', cutoff=rho_vxc_cutoff):
    xc_type = xc_type.upper()
    if xc_type == 'LDA':
        ndim = 1
    elif xc_type == 'GGA':
        ndim = 4
    elif xc_type == 'MGGA':
        ndim = 5
    else:
        raise ValueError(f'Unknown xc_type: {xc_type}')
    
    # TODO: Remove the data structure
    _vhfopt = _VHFOpt(mol)
    _vhfopt.dtype = dtype
    _vhfopt.tile = 1
    _vhfopt.build()
    sorted_mol = _vhfopt.sorted_mol
    uniq_l_ctr = _vhfopt.uniq_l_ctr
    l_ctr_bas_loc = _vhfopt.l_ctr_offsets
    bas_cache = format_bas_cache(_vhfopt.sorted_mol, dtype=dtype)

    nao = sorted_mol.nao
    coords, coeffs, exps, ao_loc, nprims, angs = bas_cache
    nbas = nprims.shape[0]
    log_cutoff = math.log(cutoff)
    
    def rho_fun(ni, mol, dm, grids, max_memory=2000, verbose=None):
        ngrids = grids.coords.shape[0]
        grid_coords = cp.asarray(grids.coords.T, dtype=dtype, order='C')        
        rho = cp.zeros((ndim, ngrids), dtype=np.float64)
        dm = _vhfopt.apply_coeff_C_mat_CT(dm)
        dm_cond = max_block_pooling(dm, ao_loc)
        dm_cond = cp.asarray(dm_cond, dtype=np.float32)
        log_dm_shell = cp.log(cp.abs(dm_cond) + 1e-200)
        log_dm = cp.max(log_dm_shell).item()
        log_ao_cutoff = log_cutoff - log_dm # ao.T * dm * ao < cutoff
        n_groups = len(l_ctr_bas_loc) - 1
        
        # Estimate the value of AO on grids, construct the sparsity
        ao_sparsity = {}
        for i in range(n_groups):
            ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i+1]
            x = coords[ish0:ish1]
            c = coeffs[ish0:ish1]
            e = exps[ish0:ish1]
            a = angs[ish0:ish1]
            n = nprims[ish0:ish1]
            s = estimate_log_aovalue(grid_coords, x, c, e, a, n, log_cutoff=log_ao_cutoff)
            log_maxval, indices, nnz = s
            indices += ish0
            ao_sparsity[i] = (log_maxval, indices, nnz)

        dm = cp.asarray(dm, dtype=dtype)
        for i in range(n_groups):
            for j in range(i, n_groups):
                li, ip = uniq_l_ctr[i]
                lj, jp = uniq_l_ctr[j]
                ang = (li,lj)
                nprim = (ip, jp)
                script, mod, fun = gen_rho_kernel(ang, nprim, dtype, ndim)
                log_maxval_i, indices_i, nnz_i = ao_sparsity[i]
                log_maxval_j, indices_j, nnz_j = ao_sparsity[j]
                nbas_i = indices_i.shape[1]
                nbas_j = indices_j.shape[1]
                fun(grid_coords, 
                    coords, coeffs, exps, nbas,
                    dm, log_dm_shell, ao_loc, nao, 
                    rho,
                    log_maxval_i, indices_i, nnz_i, nbas_i,
                    log_maxval_j, indices_j, nnz_j, nbas_j,
                    np.float32(log_cutoff), ngrids
                )
        return rho

    def vxc_fun(ni, mol, wv, grids, max_memory=None, verbose=None):
        ngrids = grids.coords.shape[0]
        grid_coords = cp.asarray(grids.coords.T, dtype=dtype, order='C')
        vxc = cp.zeros([1, nao, nao], dtype=np.float64)
        ngrids_per_atom = ngrids / mol.natm
        wv_max = cp.max(cp.abs(wv)).item()
        wv_max = wv_max * ngrids_per_atom # roughly integrate(wv * ao.T * ao) < cutoff
        log_wv_max = math.log(wv_max)
        log_ao_cutoff = log_cutoff - log_wv_max
        n_groups = len(l_ctr_bas_loc) - 1
        
        # Estimate the value of AO on grids, construct the sparsity
        ao_sparsity = {}
        for i in range(n_groups):
            ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i+1]
            x = coords[ish0:ish1]
            c = coeffs[ish0:ish1]
            e = exps[ish0:ish1]
            a = angs[ish0:ish1]
            n = nprims[ish0:ish1]
            s = estimate_log_aovalue(grid_coords, x, c, e, a, n, log_cutoff=log_ao_cutoff)
            log_maxval, indices, nnz = s
            indices += ish0
            ao_sparsity[i] = (log_maxval, indices, nnz)
        
        wv = cp.asarray(wv, dtype=dtype)
        for i in range(n_groups):
            for j in range(i, n_groups):
                li, ip = uniq_l_ctr[i]
                lj, jp = uniq_l_ctr[j]
                ang = (li,lj)
                nprim = (ip, jp)
                script, mod, fun = gen_vxc_kernel(ang, nprim, dtype, ndim)
                log_maxval_i, indices_i, nnz_i = ao_sparsity[i]
                log_maxval_j, indices_j, nnz_j = ao_sparsity[j]
                nbas_i = indices_i.shape[1]
                nbas_j = indices_j.shape[1]
                fun(grid_coords, 
                    coords, coeffs, exps, nbas,
                    vxc, ao_loc, nao,
                    wv, 
                    log_maxval_i, indices_i, nnz_i, nbas_i,
                    log_maxval_j, indices_j, nnz_j, nbas_j,
                    np.float32(log_cutoff), ngrids
                )

        vxc = _vhfopt.apply_coeff_CT_mat_C(vxc)
        vxc = inplace_add_transpose(vxc)
        return vxc
    return rho_fun, vxc_fun

def create_tasks(l_ctr_bas_loc, ovlp, cutoff=rho_vxc_cutoff):
    n_groups = len(l_ctr_bas_loc) - 1
    tasks = {}
    for i in range(n_groups):
        for j in range(i+1):
            ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i+1]
            jsh0, jsh1 = l_ctr_bas_loc[j], l_ctr_bas_loc[j+1]
            mask = ovlp[ish0:ish1, jsh0:jsh1] > cp.log(cutoff)
            if i == j:
                mask = cp.tril(mask)
            pairs = cp.argwhere(mask)
            pairs[:,0] += ish0
            pairs[:,1] += jsh0
            tasks[i,j] = cp.asarray(pairs, dtype=np.int32)
    return tasks
