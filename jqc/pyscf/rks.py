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

"""
Generate DFT kernels for PySCF
"""

import math
import time
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft import libxc
from pyscf.dft.gen_grid import GROUP_BOUNDARY_PENALTY
from jqc.backend.linalg_helper import max_block_pooling, inplace_add_transpose
from jqc.pyscf.basis import BasisLayout
from jqc.backend.rks import (
    gen_rho_kernel,
    gen_vxc_kernel,
    estimate_log_aovalue,
    vv10nlc,
)

# dm_to_mol and dm_from_mol are now methods of BasisLayout class
from jqc.constants import LMAX

__all__ = ["build_grids", "generate_rks_kernel"]

ao_cutoff = 1e-13
GROUP_BOX_SIZE = 3.0
DIM_BY_XC = {"LDA": 1, "GGA": 4, "MGGA": 5}

# Fused kernel using CuPy's ElementwiseKernel for better performance
_nelec_excsum_elementwise = cp.ElementwiseKernel(
    "T rho, T weights, T exc",
    "T nelec_contrib, T excsum_contrib",
    """
    nelec_contrib = rho * weights;
    excsum_contrib = rho * weights * exc;
    """,
    "nelec_excsum_fused",
)


def arg_group_grids(mol, coords, box_size=GROUP_BOX_SIZE):
    """
    Parition the entire space into small boxes according to the input box_size.
    Group the grids against these boxes.
    """
    atom_coords = mol.atom_coords()
    boundary = [
        atom_coords.min(axis=0) - GROUP_BOUNDARY_PENALTY,
        atom_coords.max(axis=0) + GROUP_BOUNDARY_PENALTY,
    ]
    # how many boxes inside the boundary
    boxes = ((boundary[1] - boundary[0]) * (1.0 / box_size)).round().astype(int)

    # box_size is the length of each edge of the box
    box_size = cp.asarray((boundary[1] - boundary[0]) / boxes)
    frac_coords = (coords - cp.asarray(boundary[0])) * (1.0 / box_size)
    box_ids = cp.floor(frac_coords).astype(int)
    box_ids[box_ids < -1] = -1
    box_ids[box_ids[:, 0] > boxes[0], 0] = boxes[0]
    box_ids[box_ids[:, 1] > boxes[1], 1] = boxes[1]
    box_ids[box_ids[:, 2] > boxes[2], 2] = boxes[2]

    boxes *= 2  # for safety
    box_id = (
        box_ids[:, 0] + box_ids[:, 1] * boxes[0] + box_ids[:, 2] * boxes[0] * boxes[1]
    )
    return cp.argsort(box_id)


def build_grids(grids, mol=None, with_non0tab=False, sort_grids=True, **kwargs):
    """
    Build grids for DFT. Copied from GPU4PySCF with different sorting algorithm.

    Parameters
    ----------
    grids : Grid
        PySCF Grid object.
    mol : Mole, optional
        PySCF Mole object.
    with_non0tab : bool, optional
        Whether to generate non-zero tab.
    sort_grids : bool, optional
        Whether to sort grids.

    Returns
    -------
    grids : Grid
        PySCF Grid object.
    """

    if mol is None:
        mol = grids.mol
    if grids.verbose >= 5:
        grids.check_sanity()
    logger = lib.logger.new_logger(mol, grids.verbose)

    cputime_start = time.perf_counter()
    atom_grids_tab = grids.gen_atomic_grids(
        mol, grids.atom_grid, grids.radi_method, grids.level, grids.prune, **kwargs
    )
    grids.coords, grids.weights = grids.get_partition(
        mol, atom_grids_tab, grids.radii_adjust, grids.atomic_radii, grids.becke_scheme
    )

    atm_idx = cp.empty(grids.coords.shape[0], dtype=np.int32)
    quadrature_weights = cp.empty(grids.coords.shape[0])
    p0 = p1 = 0
    for ia in range(mol.natm):
        r, vol = atom_grids_tab[mol.atom_symbol(ia)]
        p0, p1 = p1, p1 + vol.size
        atm_idx[p0:p1] = ia
        quadrature_weights[p0:p1] = cp.asarray(vol)
    grids.atm_idx = atm_idx
    grids.quadrature_weights = quadrature_weights

    logger.debug(f"generating atomic grids {time.perf_counter() - cputime_start}")
    ngrids = grids.coords.shape[0]
    alignment = grids.alignment
    if alignment > 1:
        padding = (ngrids + alignment - 1) // alignment * alignment - ngrids
        logger.debug(f"Padding {padding} grids")
        if padding > 0:
            grids.coords = cp.vstack([grids.coords, cp.full((padding, 3), 1e-4)])
            grids.weights = cp.hstack([grids.weights, cp.zeros(padding)])
            grids.quadrature_weights = cp.hstack(
                [grids.quadrature_weights, cp.zeros(padding)]
            )
            grids.atm_idx = cp.hstack(
                [grids.atm_idx, cp.full(padding, -1, dtype=np.int32)]
            )

    if sort_grids:
        idx = arg_group_grids(mol, grids.coords, box_size=1.0)
        grids.coords = grids.coords[idx]
        grids.weights = grids.weights[idx]
        grids.quadrature_weights = grids.quadrature_weights[idx]
        grids.atm_idx = grids.atm_idx[idx]
        logger.debug(f"sorting grids {time.perf_counter() - cputime_start}")

    if with_non0tab:
        raise RuntimeError("with_non0tab is not supported yet")
    else:
        grids.screen_index = grids.non0tab = None
    logger.info(f"tot grids = {len(grids.weights)}")

    grids._non0ao_idx = None
    return grids


def generate_get_veff():
    from gpu4pyscf.dft.rks import initialize_grids
    from gpu4pyscf.lib.cupy_helper import tag_array

    def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None:
            mol = ks.mol
        if dm is None:
            dm = ks.make_rdm1()
        t0 = (logger.process_clock(), logger.perf_counter())
        initialize_grids(ks, mol, dm)

        ground_state = getattr(dm, "ndim", 0) == 2

        ni = ks._numint
        if hermi == 2:  # because rho = 0
            n, exc, vxc = 0, 0, 0
        else:
            n, exc, vxc = ni.nr_rks(mol, ks.grids, ks.xc, dm)
            if ks.do_nlc():
                if ni.libxc.is_nlc(ks.xc):
                    xc = ks.xc
                else:
                    assert ni.libxc.is_nlc(ks.nlc)
                    xc = ks.nlc
                n, enlc, vnlc = ni.nr_nlc_vxc(mol, ks.nlcgrids, xc, dm)

                exc += enlc
                vxc += vnlc
            logger.debug(ks, "nelec by numeric integration = %s", n)
        t0 = logger.timer_debug1(ks, "vxc tot", *t0)

        if not ni.libxc.is_hybrid_xc(ks.xc):
            vk = None
            if (
                ks._eri is None
                and ks.direct_scf
                and getattr(vhf_last, "vj", None) is not None
            ):
                ddm = cp.asarray(dm) - cp.asarray(dm_last)
                vj = ks.get_j(mol, ddm, hermi)
                vj += vhf_last.vj
            else:
                vj = ks.get_j(mol, dm, hermi)

            vxc += vj
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
            if (
                ks._eri is None
                and ks.direct_scf
                and getattr(vhf_last, "vk", None) is not None
            ):
                ddm = cp.asarray(dm) - cp.asarray(dm_last)
                vj, vk = ks.get_jk(mol, ddm, hermi)
                vk *= hyb
                if abs(omega) > 1e-10:  # For range separated Coulomb operator
                    vklr = ks.get_k(mol, ddm, hermi, omega=omega)
                    vklr *= alpha - hyb
                    vk += vklr
                vj += vhf_last.vj
                vk += vhf_last.vk
            else:
                vj, vk = ks.get_jk(mol, dm, hermi)
                vk *= hyb
                if abs(omega) > 1e-10:
                    vklr = ks.get_k(mol, dm, hermi, omega=omega)
                    vklr *= alpha - hyb
                    vk += vklr
            vxc += vj
            vxc -= vk * 0.5
            if ground_state:
                exc -= cp.einsum("ij,ji", dm, vk).real * 0.5 * 0.5

        if ground_state:
            ecoul = cp.einsum("ij,ji", dm, vj).real * 0.5
        else:
            ecoul = None
        t0 = logger.timer_debug1(ks, "veff", *t0)
        vxc = tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
        return vxc

    return get_veff


def generate_nr_rks(basis_layout, cutoff_fp64=1e-13, cutoff_fp32=1e-13):
    rks_fun, _, _ = generate_rks_kernel(
        basis_layout, cutoff_fp64=cutoff_fp64, cutoff_fp32=cutoff_fp32
    )
    return rks_fun


def generate_get_rho(basis_layout, cutoff_fp64=1e-13, cutoff_fp32=1e-13):
    _, rho_fun, _ = generate_rks_kernel(
        basis_layout, cutoff_fp64=cutoff_fp64, cutoff_fp32=cutoff_fp32
    )

    def get_rho(mol, dm, grids, *args, **kwargs):
        dm = cp.asarray(dm)
        rho = rho_fun(mol, grids, "LDA", dm)
        return rho[0]

    return get_rho


def generate_rks_kernel(basis_layout, cutoff_fp64=1e-13, cutoff_fp32=1e-13):
    # Cache frequently accessed properties to reduce overhead
    mol = basis_layout._mol
    _, _, angs, nprims = basis_layout.bas_info
    ce_fp32 = basis_layout.ce_fp32
    coords_fp32 = basis_layout.coords_fp32
    ce_fp64 = basis_layout.ce_fp64
    coords_fp64 = basis_layout.coords_fp64
    ao_loc = cp.asarray(basis_layout.ao_loc)
    nbas = basis_layout.nbasis
    group_key, group_offset = basis_layout.group_info

    # Pre-compute constants
    nao = int(ao_loc[-1])
    log_ao_cutoff = math.log(ao_cutoff)

    log_cutoff_fp32 = np.log(cutoff_fp32).astype(np.float32)
    log_cutoff_fp64 = np.log(cutoff_fp64).astype(np.float32)
    log_cutoff_max = np.log(1e100).astype(np.float32)
    logger = lib.logger.new_logger(mol, mol.verbose)

    def compute_ao_sparsity(grid_coords, log_aodm_cutoff):
        """Helper function to compute AO sparsity pattern - avoids code duplication"""
        n_groups = len(group_offset) - 1
        ao_sparsity = {}
        for i in range(n_groups):
            ish0, ish1 = group_offset[i], group_offset[i + 1]
            x = coords_fp32[ish0:ish1]
            ce = ce_fp32[ish0:ish1]
            a = angs[ish0].item()
            n = nprims[ish0].item()
            s = estimate_log_aovalue(
                grid_coords, x, ce, a, n, log_cutoff=log_aodm_cutoff
            )
            log_maxval, indices, nnz = s
            indices += ish0
            ao_sparsity[i] = (log_maxval, indices, nnz)
        return ao_sparsity

    _cache = {"dm_prev": 0, "rho_prev": 0, "wv_prev": 0, "vxcmat_prev": 0}

    def rks_fun(ni, mol, grids, xc_code, dm, **kwargs):
        """rks kernel for PySCF, with incremental DFT implementation

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
        cputime_start = time.perf_counter()
        dm_prev = _cache["dm_prev"]
        rho_prev = _cache["rho_prev"]
        wv_prev = _cache["wv_prev"]
        vxcmat_prev = _cache["vxcmat_prev"]

        xctype = libxc.xc_type(xc_code)

        # Evaluate rho on grids for given density matrix (incremental)
        weights = grids.weights
        dm_diff = dm - dm_prev
        rho = rho_prev + rho_fun(mol, grids, xctype, dm_diff)

        # Evaluate vxc on grids via libxc
        exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[:2]

        # Integrate vxc on grids using fused elementwise kernel for better performance
        vxc = cp.asarray(vxc, dtype=vxc.dtype, order="C")
        exc = cp.asarray(exc, dtype=exc.dtype, order="C")
        nelec_contrib, excsum_contrib = _nelec_excsum_elementwise(
            rho[0], weights, exc[:, 0]
        )
        nelec = float(cp.sum(nelec_contrib))
        excsum = float(cp.sum(excsum_contrib))
        wv = vxc * weights
        wv_diff = wv - wv_prev
        # vxcmat_diff = vxc_fun(mol, grids, xctype, wv_diff)
        vxcmat = vxcmat_prev + vxc_fun(mol, grids, xctype, wv_diff)

        _cache["dm_prev"] = dm.copy()
        _cache["rho_prev"] = rho
        _cache["wv_prev"] = wv
        _cache["vxcmat_prev"] = vxcmat.copy()
        cputime_end = time.perf_counter()
        cputime = cputime_end - cputime_start
        logger.info(f"vxc takes {cputime:.3f} sec")

        return nelec, excsum, vxcmat

    def rho_fun(mol, grids, xctype, dm):
        ngrids = grids.coords.shape[0]
        grid_coords = cp.asarray(grids.coords.T, dtype=np.float64, order="C")

        # Transform density matrix from molecular to cartesian basis
        dm = basis_layout.dm_from_mol(dm)
        dm_fp32 = dm.astype(np.float32)

        # Optimize array operations - combine conversions
        dm_cond = max_block_pooling(dm_fp32, ao_loc)
        log_dm_shell = cp.log(dm_cond + 1e-200, dtype=cp.float32)
        log_dm = cp.max(log_dm_shell).item()
        log_aodm_cutoff = log_ao_cutoff - log_dm

        n_groups = len(group_offset) - 1
        xctype = xctype.upper()
        ndim = DIM_BY_XC[xctype]

        # Use helper function to compute AO sparsity
        ao_sparsity = compute_ao_sparsity(grid_coords, log_aodm_cutoff)

        rho = cp.zeros((ndim, ngrids), dtype=np.float64, order="C")
        # Optimize array conversions - compute both dtypes together
        dm = cp.asarray(dm, dtype=np.float64, order="C")

        for i in range(n_groups):
            for j in range(i, n_groups):
                li, ip = group_key[i]
                lj, jp = group_key[j]
                ang = (li, lj)
                nprim = (ip, jp)

                log_maxval_i, indices_i, nnz_i = ao_sparsity[i]
                log_maxval_j, indices_j, nnz_j = ao_sparsity[j]
                nbas_i = indices_i.shape[1]
                nbas_j = indices_j.shape[1]
                _, _, fun = gen_rho_kernel(ang, nprim, np.float64, ndim)
                fun(
                    grid_coords,
                    coords_fp64,
                    ce_fp64,
                    nbas,
                    dm,
                    log_dm_shell,
                    ao_loc,
                    nao,
                    rho,
                    log_maxval_i,
                    indices_i,
                    nnz_i,
                    nbas_i,
                    log_maxval_j,
                    indices_j,
                    nnz_j,
                    nbas_j,
                    log_cutoff_fp64,
                    log_cutoff_max,
                    ngrids,
                )
                if cutoff_fp64 > cutoff_fp32:
                    _, _, fun = gen_rho_kernel(ang, nprim, np.float32, ndim)
                    fun(
                        grid_coords,
                        coords_fp32,
                        ce_fp32,
                        nbas,
                        dm_fp32,
                        log_dm_shell,
                        ao_loc,
                        nao,
                        rho,
                        log_maxval_i,
                        indices_i,
                        nnz_i,
                        nbas_i,
                        log_maxval_j,
                        indices_j,
                        nnz_j,
                        nbas_j,
                        log_cutoff_fp32,
                        log_cutoff_fp64,
                        ngrids,
                    )
        return rho

    def vxc_fun(mol, grids, xctype, wv):
        ngrids = grids.coords.shape[0]
        grid_coords = cp.asarray(grids.coords.T, dtype=np.float64, order="C")
        vxc = cp.zeros([1, nao, nao], dtype=np.float64, order="C")

        # Optimize wv_max computation
        ngrids_per_atom = ngrids / mol.natm
        wv_max = (cp.max(cp.abs(wv)).item() + 1e-300) * ngrids_per_atom
        log_wv_max = math.log(wv_max)
        log_aodm_cutoff = log_ao_cutoff - log_wv_max

        n_groups = len(group_offset) - 1
        xctype = xctype.upper()
        ndim = DIM_BY_XC[xctype]

        # Use helper function to compute AO sparsity
        ao_sparsity = compute_ao_sparsity(grid_coords, log_aodm_cutoff)

        for i in range(n_groups):
            for j in range(i, n_groups):
                li, ip = group_key[i]
                lj, jp = group_key[j]
                ang = (li, lj)
                nprim = (ip, jp)

                log_maxval_i, indices_i, nnz_i = ao_sparsity[i]
                log_maxval_j, indices_j, nnz_j = ao_sparsity[j]
                nbas_i = indices_i.shape[1]
                nbas_j = indices_j.shape[1]
                _, _, fun = gen_vxc_kernel(ang, nprim, np.float64, ndim)
                fun(
                    grid_coords,
                    coords_fp64,
                    ce_fp64,
                    nbas,
                    vxc,
                    ao_loc,
                    nao,
                    wv,
                    log_maxval_i,
                    indices_i,
                    nnz_i,
                    nbas_i,
                    log_maxval_j,
                    indices_j,
                    nnz_j,
                    nbas_j,
                    log_cutoff_fp64,
                    log_cutoff_max,
                    ngrids,
                )
                if cutoff_fp64 > cutoff_fp32:
                    _, _, fun = gen_vxc_kernel(ang, nprim, np.float32, ndim)
                    fun(
                        grid_coords,
                        coords_fp32,
                        ce_fp32,
                        nbas,
                        vxc,
                        ao_loc,
                        nao,
                        wv,
                        log_maxval_i,
                        indices_i,
                        nnz_i,
                        nbas_i,
                        log_maxval_j,
                        indices_j,
                        nnz_j,
                        nbas_j,
                        log_cutoff_fp32,
                        log_cutoff_fp64,
                        ngrids,
                    )

        # Transform exchange-correlation matrix back to molecular basis
        vxc = basis_layout.dm_to_mol(vxc)
        vxc = inplace_add_transpose(vxc)
        return vxc[0]

    return rks_fun, rho_fun, vxc_fun


def generate_nr_nlc_vxc(basis_layout, cutoff_fp64=1e-13, cutoff_fp32=1e-13):
    _, rho_fun, vxc_fun = generate_rks_kernel(
        basis_layout, cutoff_fp64=cutoff_fp64, cutoff_fp32=cutoff_fp32
    )
    from gpu4pyscf.dft import xc_deriv

    # Cache for incremental computation
    _cache = {"dm_prev": 0, "rho_prev": 0, "wv_prev": 0, "vmat_prev": 0}

    def nr_nlc_vxc(ni, mol, grids, xc_code, dms, **kwargs):
        # Get cached values
        dm_prev = _cache["dm_prev"]
        rho_prev = _cache["rho_prev"]
        wv_prev = _cache["wv_prev"]
        vmat_prev = _cache["vmat_prev"]

        # Compute rho incrementally
        dm_diff = dms - dm_prev
        rho = rho_prev + rho_fun(mol, grids, "GGA", dm_diff)

        # Compute VV10 exchange-correlation
        exc = 0
        vxc = 0
        nlc_coefs = ni.nlc_coeff(xc_code)
        for nlc_pars, fac in nlc_coefs:
            e, v = vv10nlc(
                rho, grids.coords, rho, grids.weights, grids.coords, nlc_pars
            )
            exc += e * fac
            vxc += v * fac

        # Use fused elementwise kernel for density integration
        nelec_contrib, excsum_contrib = _nelec_excsum_elementwise(
            rho[0], grids.weights, exc
        )
        nelec = float(cp.sum(nelec_contrib))
        excsum = float(cp.sum(excsum_contrib))

        # Compute wv incrementally
        vv_vxc = xc_deriv.transform_vxc(rho, vxc, "GGA", spin=0)
        wv = vv_vxc * grids.weights
        wv_diff = wv - wv_prev
        vmat_diff = vxc_fun(mol, grids, "GGA", wv_diff)
        vmat = vmat_prev + vmat_diff

        # Update cache - keep necessary copies for correctness
        _cache["dm_prev"] = dms.copy()
        _cache["rho_prev"] = rho
        _cache["wv_prev"] = wv
        _cache["vmat_prev"] = vmat.copy()

        return nelec, excsum, vmat

    return nr_nlc_vxc


def create_tasks(l_ctr_bas_loc, ovlp, cutoff=1e-13):
    n_groups = len(l_ctr_bas_loc) - 1
    tasks = {}
    for i in range(n_groups):
        for j in range(i + 1):
            ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i + 1]
            jsh0, jsh1 = l_ctr_bas_loc[j], l_ctr_bas_loc[j + 1]
            mask = ovlp[ish0:ish1, jsh0:jsh1] > cp.log(cutoff)
            if i == j:
                mask = cp.tril(mask)
            pairs = cp.argwhere(mask)
            if pairs.shape[0] == 0:
                continue
            pairs[:, 0] += ish0
            pairs[:, 1] += jsh0
            tasks[i, j] = cp.asarray(pairs, dtype=np.int32)
    return tasks
