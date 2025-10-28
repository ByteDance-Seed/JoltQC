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
Generate JK kernels for PySCF
"""

import math
import time
from collections import Counter

import cupy as cp
import numpy as np
from pyscf import lib

from jqc.backend.jk import gen_jk_kernel
from jqc.backend.jk_tasks import MAX_PAIR_SIZE, QUEUE_DEPTH, gen_screen_jk_tasks_kernel
from jqc.backend.linalg_helper import inplace_add_transpose, max_block_pooling
from jqc.constants import TILE
from jqc.pyscf.basis import BasisLayout

__all__ = [
    "get_j",
    "get_jk",
]

ushort4_dtype = np.dtype(
    [("x", np.uint16), ("y", np.uint16), ("z", np.uint16), ("w", np.uint16)]
)

GROUP_SIZE = 256
PAIR_CUTOFF = 1e-13


def generate_get_j(basis_layout, cutoff_fp64=1e-13, cutoff_fp32=1e-13):
    get_jk_kernel = generate_jk_kernel(basis_layout, cutoff_fp64=cutoff_fp64, cutoff_fp32=cutoff_fp32)

    def get_jk(*args, **kwargs):
        return get_jk_kernel(*args, with_j=True, with_k=False, **kwargs)[0]

    return get_jk


def generate_get_k(basis_layout, cutoff_fp64=1e-13, cutoff_fp32=1e-13):
    get_jk_kernel = generate_jk_kernel(basis_layout, cutoff_fp64=cutoff_fp64, cutoff_fp32=cutoff_fp32)

    def get_jk(*args, **kwargs):
        return get_jk_kernel(*args, with_j=False, with_k=True, **kwargs)[1]

    return get_jk


def generate_get_jk(basis_layout, cutoff_fp64=1e-13, cutoff_fp32=1e-13):
    get_jk_kernel = generate_jk_kernel(basis_layout, cutoff_fp64=cutoff_fp64, cutoff_fp32=cutoff_fp32)

    def get_jk(*args, **kwargs):
        return get_jk_kernel(*args, **kwargs)

    return get_jk


def generate_get_veff():
    def get_veff(mf, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
        if dm is None:
            dm = mf.make_rdm1()
        if dm_last is not None and mf.direct_scf:
            dm = cp.asarray(dm) - cp.asarray(dm_last)
        vj, vk = mf.get_jk(mol, dm, hermi)
        vhf = vj - 0.5 * vk
        if vhf_last is not None:
            vhf += cp.asarray(vhf_last)
        return vhf

    return get_veff


def generate_jk_kernel(basis_layout, cutoff_fp64=1e-13, cutoff_fp32=1e-13):
    # Pre-compute log cutoffs
    log_cutoff_fp64 = np.float32(math.log(cutoff_fp64))
    log_cutoff_fp32 = np.float32(math.log(cutoff_fp32))

    # Cache basis_layout data to avoid re-extracting on every call
    group_info = basis_layout.group_info
    nbas = basis_layout.nbasis
    mol = basis_layout._mol
    ce_fp32 = basis_layout.ce_fp32
    coords_fp32 = basis_layout.coords_fp32
    ce_fp64 = basis_layout.ce_fp64
    coords_fp64 = basis_layout.coords_fp64
    ao_loc = cp.asarray(basis_layout.ao_loc)
    # nbas_padded includes padding shells for bounds checking in screening kernel
    nbas_padded = len(ao_loc) - 1

    def get_jk(
        mol_ref,
        dm,
        hermi=0,
        vhfopt=None,
        with_j=True,
        with_k=True,
        omega=None,
        verbose=None,
    ):
        """
        Compute J, K matrices, compatible with jk.get_jk in PySCF

        Args:
            mol_ref: pyscf Mole object (for compatibility, uses pre-generated basis_layout)
            dm: density matrix
            hermi: hermiticity of the density matrix
            vhfopt: vhfopt object
            with_j: whether to compute J matrix
            with_k: whether to compute K matrix
            omega: short ranged J/K parameter
            verbose: verbose level, for compatibility only
        """
        assert with_j or with_k
        if omega is not None:
            assert omega >= 0.0, "short ranged J/K not supported"

        nao = int(ao_loc[-1])

        group_key, group_offset = group_info
        uniq_l = group_key[:, 0]
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
        n_groups = basis_layout.ngroups

        info = cp.empty(4, dtype=np.uint32)

        tasks = [
            (i, j, k, l)
            for i in range(n_groups)
            for j in range(i + 1)
            for k in range(i + 1)
            for l in range(k + 1)
        ]

        group_key = [key.tolist() for key in group_key]

        cputime_start = time.perf_counter()

        logger = lib.logger.new_logger(mol, mol.verbose)
        logger.debug1(
            f"Compute J/K matrices with do_j={with_j} and do_k={with_k}, omega={omega}"
        )

        nao_orig = mol.nao
        dm = cp.asarray(dm, order="C")
        dms = dm.reshape(-1, nao_orig, nao_orig)

        # Convert the density matrix to cartesian basis
        dms = basis_layout.dm_from_mol(dms)
        dms = dms.reshape(-1, nao, nao)

        # Optimize array conversions - compute both fp64 and fp32 versions together
        dms = cp.asarray(dms, dtype=np.float64, order="C")
        dms_fp32 = dms.astype(np.float32)
        dm_cond = max_block_pooling(dms_fp32, ao_loc)

        # Pre-compute omega values
        omega_fp32 = np.float32(omega) if omega is not None else None
        omega_fp64 = np.float64(omega) if omega is not None else None

        if hermi == 0:
            # Wrap the triu contribution to tril
            dm_cond = dm_cond + dm_cond.T
        log_dm_cond = cp.log(dm_cond + 1e-300, dtype=np.float32)
        log_max_dm = log_dm_cond.max().item()
        cutoff = np.log(PAIR_CUTOFF) - log_max_dm
        q_matrix = basis_layout.q_matrix(omega=omega)
        tile_pairs = make_tile_pairs(group_offset, q_matrix, cutoff)

        if hermi == 0:
            # Contract the tril and triu parts separately
            dms = cp.vstack([dms, dms.transpose(0, 2, 1)])
        n_dm = dms.shape[0]

        vj = vk = 0
        if with_k:
            vk = cp.zeros(dms.shape)
        if with_j:
            vj = cp.zeros(dms.shape)

        logger.debug1("Calculate dm_cond and AO pairs")
        _, _, gen_tasks_fun = gen_screen_jk_tasks_kernel(
            do_j=with_j, do_k=with_k, tile=TILE
        )
        logger.debug1("Generate tasks kernel")
        timing_counter = Counter()
        kern_counts = 0
        quartet_list = cp.empty((QUEUE_DEPTH), dtype=ushort4_dtype)

        for task in tasks[::-1]:
            i, j, k, l = task
            li, ip = group_key[i]
            lj, jp = group_key[j]
            lk, kp = group_key[k]
            ll, lp = group_key[l]
            ang = (li, lj, lk, ll)
            nprim = (ip, jp, kp, lp)

            if (i, j) not in tile_pairs or (k, l) not in tile_pairs:
                continue
            tile_ij = tile_pairs[i, j]
            tile_kl = tile_pairs[k, l]
            ntile_ij = tile_ij.size
            ntile_kl = tile_kl.size

            ntasks_fp64 = 0
            ntasks_fp32 = 0

            # Setup events for timing the critical kernels
            start = end = None
            if logger.verbose > lib.logger.INFO:
                start = cp.cuda.Event()
                end = cp.cuda.Event()
                start.record()

            # Pre-generate kernels for this (l_i,l_j,l_k,l_l) once
            jk_fp32_kernel = gen_jk_kernel(
                ang,
                nprim,
                do_j=with_j,
                do_k=with_k,
                dtype=np.float32,
                n_dm=n_dm,
                omega=omega_fp32,
            )
            jk_fp64_kernel = gen_jk_kernel(
                ang,
                nprim,
                do_j=with_j,
                do_k=with_k,
                dtype=np.float64,
                n_dm=n_dm,
                omega=omega_fp64,
            )

            PAIR_TILE_SIZE = MAX_PAIR_SIZE // (TILE * TILE)
            for t_ij0, t_ij1 in lib.prange(0, ntile_ij, PAIR_TILE_SIZE):
                for t_kl0, t_kl1 in lib.prange(0, ntile_kl, PAIR_TILE_SIZE):
                    # Generate tasks for fp32 and fp64
                    # Use nbas_padded for bounds checking in screening kernel
                    gen_tasks_fun(
                        quartet_list,
                        info,
                        nbas_padded,
                        tile_ij[t_ij0:t_ij1],
                        tile_kl[t_kl0:t_kl1],
                        q_matrix,
                        log_dm_cond,
                        log_cutoff_fp32,
                        log_cutoff_fp64,
                    )
                    kern_counts += 1
                    info_cpu = info.get()

                    # Get task counts
                    n_quartets_fp32 = int(info_cpu[1].item())
                    offset = int(info_cpu[2].item())
                    n_quartets_fp64 = QUEUE_DEPTH - offset

                    # Launch FP32 and FP64 kernels asynchronously
                    if n_quartets_fp32 > 0:
                        jk_fp32_kernel(
                            nbas,
                            nao,
                            ao_loc,
                            coords_fp32,
                            ce_fp32,
                            dms_fp32,
                            vj,
                            vk,
                            omega_fp32,
                            quartet_list,
                            n_quartets_fp32,
                        )
                        kern_counts += 1
                        ntasks_fp32 += n_quartets_fp32

                    if n_quartets_fp64 > 0:
                        jk_fp64_kernel(
                            nbas,
                            nao,
                            ao_loc,
                            coords_fp64,
                            ce_fp64,
                            dms,
                            vj,
                            vk,
                            omega_fp64,
                            quartet_list[offset:],
                            n_quartets_fp64,
                        )
                        kern_counts += 1
                        ntasks_fp64 += n_quartets_fp64
                    if np.isnan(cp.linalg.norm(vj)) or np.isnan(cp.linalg.norm(vk)):
                        raise RuntimeError("vj contains NaN values")
                    #print(n_quartets_fp32, n_quartets_fp64, cp.linalg.norm(vj).item(), cp.linalg.norm(vk).item())
            if logger.verbose > lib.logger.INFO:
                end.record()
                end.synchronize()
                elasped_time = cp.cuda.get_elapsed_time(start, end)
                ntasks = max(ntasks_fp64 + ntasks_fp32, 1)
                llll = f"({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})"
                msg_kernel = f"kernel type {llll}/({ip}{jp}{kp}{lp}), "
                msg_fp64 = f"FP64 tasks = {ntasks_fp64:10d}, "
                msg_fp32 = f"FP32 tasks = {ntasks_fp32:10d}, total time = {elasped_time:5.2f} ms, "
                msg_ratio = f"FP64 ratio = {ntasks_fp64/ntasks:.2f}"
                msg = msg_kernel + msg_fp64 + msg_fp32 + msg_ratio
                logger.debug1(msg)
                timing_counter[llll] += elasped_time

        # Transform results back to molecular basis using BasisLayout methods
        # The remove_padding=True parameter filters out padded basis functions
        print(cp.linalg.norm(vj).item(), cp.linalg.norm(vk).item())
        print(nbas, nbas_padded)
        if with_j:
            if hermi == 1:
                vj *= 2.0
            else:
                vj, vjT = vj[: n_dm // 2], vj[n_dm // 2 :]
                vj += vjT.transpose(0, 2, 1)
            vj = inplace_add_transpose(vj)
            vj = basis_layout.dm_to_mol(vj)
            vj = vj.reshape(dm.shape)

        if with_k:
            if hermi == 1:
                vk = inplace_add_transpose(vk)
            else:
                vk, vkT = vk[: n_dm // 2], vk[n_dm // 2 :]
                vk += vkT.transpose(0, 2, 1)
            vk = basis_layout.dm_to_mol(vk)
            vk = vk.reshape(dm.shape)

        if logger.verbose >= lib.logger.DEBUG:
            logger.debug1("kernel launches %d", kern_counts)
            for llll, t in timing_counter.items():
                logger.debug1(f"{llll} wall time {t:.2f} ms")

        cputime_end = time.perf_counter()
        cputime = cputime_end - cputime_start
        logger.info(f"vj = {with_j} and vk = {with_k} take {cputime:.3f} sec")
        return vj, vk

    return get_jk


def make_tile_pairs(l_ctr_bas_loc, q_matrix, cutoff, tile=TILE):
    """Make tile pairs with GTO pairing screening and symmetry
    Filter out the tiles that are not within cutoff
    """
    assert q_matrix.shape[0] % tile == 0
    tile_pairs = {}
    n_groups = len(l_ctr_bas_loc) - 1
    ntiles = q_matrix.shape[0] // tile
    tile_loc = l_ctr_bas_loc // tile
    tiled_q_matrix = q_matrix.reshape([ntiles, tile, ntiles, tile]).max(axis=(1, 3))
    q_idx = tiled_q_matrix

    # Pre-compute tile indices to avoid repeated allocations
    tile_i_indices = cp.arange(ntiles, dtype=np.int32)
    tile_j_indices = cp.arange(ntiles, dtype=np.int32)

    for i in range(n_groups):
        i0, i1 = tile_loc[i], tile_loc[i + 1]
        # Pre-slice i indices
        i_range = tile_i_indices[i0:i1]

        for j in range(i + 1):
            j0, j1 = tile_loc[j], tile_loc[j + 1]
            # Pre-slice j indices
            j_range = tile_j_indices[j0:j1]

            sub_q_idx = q_idx[i0:i1, j0:j1]
            mask = sub_q_idx > cutoff
            if i == j:
                mask = cp.tril(mask)

            # Skip if no valid pairs
            if not mask.any():
                continue

            # Use broadcasting to create index matrix more efficiently
            t_ij = i_range[:, None] * ntiles + j_range[None, :]

            # Apply mask and get valid indices
            valid_pairs = t_ij[mask]

            # Sort by q-values for better cache locality
            if valid_pairs.size > 0:
                sort_idx = cp.argsort(sub_q_idx[mask])
                tile_pairs[i, j] = valid_pairs[sort_idx]

    return tile_pairs
