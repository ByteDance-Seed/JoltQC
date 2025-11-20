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

"""
Generate pair-based JK kernels for PySCF

This module provides a PySCF interface for the pair-based 2D JK algorithm.
The pair-based algorithm uses separate VJ and VK kernels with different
pair formats optimized for each operation:
- VJ uses symmetric pairs (i >= j) without padding
- VK uses tiled pairs with configurable tile width (default 64)
"""

import math
import time
from collections import Counter

import cupy as cp
import numpy as np
from pyscf import lib

from jqc.backend.jk_pair import gen_vj_kernel, gen_vk_kernel, make_pairs, make_pairs_symmetric
from jqc.backend.linalg_helper import inplace_add_transpose
from jqc.pyscf.basis import BasisLayout

__all__ = [
    "generate_get_j",
    "generate_get_k",
    "generate_get_jk",
]

PAIR_CUTOFF = 1e-13
PAIR_WIDE_VJ = 256
PAIR_WIDE_VK = 128


def generate_get_j(basis_layout, cutoff_fp64=1e-13, cutoff_fp32=1e-13, pair_wide_vk=PAIR_WIDE_VK):
    """Generate J-only kernel using pair-based algorithm"""
    get_jk_kernel = generate_jk_kernel(
        basis_layout, cutoff_fp64=cutoff_fp64, cutoff_fp32=cutoff_fp32, pair_wide_vk=pair_wide_vk
    )

    def get_j(*args, **kwargs):
        return get_jk_kernel(*args, with_j=True, with_k=False, **kwargs)[0]

    return get_j


def generate_get_k(basis_layout, cutoff_fp64=1e-13, cutoff_fp32=1e-13, pair_wide_vk=PAIR_WIDE_VK):
    """Generate K-only kernel using pair-based algorithm"""
    get_jk_kernel = generate_jk_kernel(
        basis_layout, cutoff_fp64=cutoff_fp64, cutoff_fp32=cutoff_fp32, pair_wide_vk=pair_wide_vk
    )

    def get_k(*args, **kwargs):
        return get_jk_kernel(*args, with_j=False, with_k=True, **kwargs)[1]

    return get_k


def generate_get_jk(basis_layout, cutoff_fp64=1e-13, cutoff_fp32=1e-13, pair_wide_vk=PAIR_WIDE_VK):
    """Generate JK kernel using pair-based algorithm"""
    get_jk_kernel = generate_jk_kernel(
        basis_layout, cutoff_fp64=cutoff_fp64, cutoff_fp32=cutoff_fp32, pair_wide_vk=pair_wide_vk
    )

    def get_jk(*args, **kwargs):
        return get_jk_kernel(*args, **kwargs)

    return get_jk


def generate_jk_kernel(basis_layout, cutoff_fp64=1e-13, cutoff_fp32=1e-13, pair_wide_vk=PAIR_WIDE_VK):
    """
    Generate pair-based JK kernel for the given basis layout.

    The pair-based algorithm computes J/K matrices using shell pairs where:
    - VJ uses symmetric pairs (i >= j) for efficiency
    - VK uses tiled pairs with configurable width for optimal memory access

    Args:
        basis_layout: BasisLayout object containing basis set information
        cutoff_fp64: Cutoff for FP64 precision calculations
        cutoff_fp32: Cutoff for FP32 precision calculations
        pair_wide_vk: Width of tiled pairs for VK kernel (default: 64)

    Returns:
        Function that computes J and/or K matrices
    """
    # Pre-compute log cutoffs
    log_cutoff_fp64 = np.float32(math.log(cutoff_fp64))
    log_cutoff_fp32 = np.float32(math.log(cutoff_fp32))

    # Cache basis_layout data
    group_info = basis_layout.group_info
    mol = basis_layout._mol

    # Get packed basis data arrays
    basis_data_fp32 = basis_layout.basis_data_fp32['packed']
    basis_data_fp64 = basis_layout.basis_data_fp64['packed']

    ao_loc = cp.asarray(basis_layout.ao_loc)

    # Pre-compute cutoff for pair screening
    cutoff = np.log(PAIR_CUTOFF)

    # Cache for pairs based on omega value
    # Key: omega value (or None), Value: (pairs_vj, q_cond_vj, pairs_vk, q_cond_vk)
    pairs_cache = {}

    def get_or_create_pairs(omega=None):
        """Get cached pairs or create new ones if not cached."""
        # Use omega as cache key (convert to float for hashing if not None)
        cache_key = float(omega) if omega is not None else None

        if cache_key not in pairs_cache:
            # Generate pairs for screening
            q_matrix = basis_layout.q_matrix(omega=omega)
            group_offset = group_info[1]

            # Generate pairs for VJ (symmetric) and VK (tiled)
            pairs_vj, q_cond_vj = make_pairs_symmetric(group_offset, q_matrix, cutoff)
            pairs_vk, q_cond_vk = make_pairs(group_offset, q_matrix, cutoff, column_size=pair_wide_vk)

            pairs_cache[cache_key] = (pairs_vj, q_cond_vj, pairs_vk, q_cond_vk)

        return pairs_cache[cache_key]

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
        Compute J, K matrices using pair-based algorithm.

        Args:
            mol_ref: pyscf Mole object (for compatibility, uses pre-generated basis_layout)
            dm: density matrix
            hermi: hermiticity of the density matrix
            vhfopt: vhfopt object (unused, for compatibility)
            with_j: whether to compute J matrix
            with_k: whether to compute K matrix
            omega: short ranged J/K parameter
            verbose: verbose level (for compatibility only)
        """
        assert with_j or with_k
        if omega is not None:
            assert omega >= 0.0, "short ranged J/K not supported"

        nao = int(ao_loc[-1])

        group_key = group_info[0]
        uniq_l = group_key[:, 0]
        l_symb = [lib.param.ANGULAR[i] for i in uniq_l]

        group_key = [key.tolist() for key in group_key]

        cputime_start = time.perf_counter()

        logger = lib.logger.new_logger(mol, mol.verbose)
        logger.debug1(
            f"Compute J/K matrices (pair-based) with do_j={with_j} and do_k={with_k}, omega={omega}"
        )

        nao_orig = mol.nao
        dm = cp.asarray(dm, order="C")
        dms = dm.reshape(-1, nao_orig, nao_orig)

        # Convert density matrix to cartesian basis
        dms = basis_layout.dm_from_mol(dms)
        dms = dms.reshape(-1, nao, nao)
        dms = cp.ascontiguousarray(dms)

        # Pre-compute omega values
        omega_fp32 = np.float32(omega) if omega is not None else None
        omega_fp64 = np.float64(omega) if omega is not None else None

        if hermi == 0:
            # Contract the tril and triu parts separately
            dms = cp.vstack([dms, dms.transpose(0, 2, 1)])

        # Prepare different precision versions AFTER hermi processing
        dms_fp32 = dms.astype(np.float32)
        dms_fp64 = dms

        n_dm = dms.shape[0]

        # Get or create cached pairs for this omega value
        pairs_vj, q_cond_vj, pairs_vk, q_cond_vk = get_or_create_pairs(omega=omega)

        pair_keys_vj = pairs_vj.keys()
        pair_keys_vk = pairs_vk.keys()

        tasks_vj = []
        for (i,j) in pair_keys_vj:
            for (k, l) in pair_keys_vj:
                tasks_vj.append((i, j, k, l))

        tasks_vk = []
        for i, j in pair_keys_vk:
            for k, l in pair_keys_vk:
                if (i >= k):
                    tasks_vk.append((i, j, k, l))

        use_fp32 = cutoff_fp32 <= PAIR_CUTOFF and cutoff_fp64 > PAIR_CUTOFF
        use_fp64 = cutoff_fp64 <= PAIR_CUTOFF

        vj = vk = 0

        logger.debug1("Generate shell pairs for pair-based algorithm")
        timing_counter = Counter()
        kern_counts = 0
 
        if with_j:
            vj = cp.zeros(dms.shape, dtype=np.float64)
            for task in tasks_vj[::-1]:
                i, j, k, l = task
                li, ip = group_key[i]
                lj, jp = group_key[j]
                lk, kp = group_key[k]
                ll, lp = group_key[l]
                ang = (li, lj, lk, ll)
                nprim = (ip, jp, kp, lp)

                ij_pairs = pairs_vj[i, j]
                kl_pairs = pairs_vj[k, l]
                n_ij_pairs = ij_pairs.shape[0]
                n_kl_pairs = kl_pairs.shape[0]
                if n_ij_pairs == 0 or n_kl_pairs == 0:
                    continue
                
                q_cond_ij = q_cond_vj[i, j]
                q_cond_kl = q_cond_vj[k, l]

                time_fp32 = 0.0
                time_fp64 = 0.0
                if logger.verbose > lib.logger.INFO:
                    start_fp32 = cp.cuda.Event()
                    end_fp32 = cp.cuda.Event()
                    start_fp64 = cp.cuda.Event()
                    end_fp64 = cp.cuda.Event()

                if use_fp32:
                    if logger.verbose > lib.logger.INFO:
                        start_fp32.record()

                    _, _, fun_vj_fp32 = gen_vj_kernel(
                        ang,
                        nprim,
                        dtype=np.float32,
                        n_dm=n_dm,
                        omega=omega_fp32,
                    )

                    fun_vj_fp32(
                        nao,
                        basis_data_fp32,
                        dms_fp32,
                        vj,
                        omega_fp32,
                        ij_pairs,
                        n_ij_pairs,
                        kl_pairs,
                        n_kl_pairs,
                        q_cond_ij,
                        q_cond_kl,
                        np.float32(cutoff),
                    )
                    kern_counts += 1

                    if logger.verbose > lib.logger.INFO:
                        end_fp32.record()
                        end_fp32.synchronize()
                        time_fp32 += cp.cuda.get_elapsed_time(start_fp32, end_fp32)

                if use_fp64:
                    if logger.verbose > lib.logger.INFO:
                        start_fp64.record()

                    _, _, fun_vj_fp64 = gen_vj_kernel(
                        ang,
                        nprim,
                        dtype=np.float64,
                        n_dm=n_dm,
                        omega=omega_fp64,
                    )

                    fun_vj_fp64(
                        nao,
                        basis_data_fp64,
                        dms_fp64,
                        vj,
                        omega_fp64,
                        ij_pairs,
                        n_ij_pairs,
                        kl_pairs,
                        n_kl_pairs,
                        q_cond_ij,
                        q_cond_kl,
                        np.float32(cutoff),
                    )
                    kern_counts += 1

                    if logger.verbose > lib.logger.INFO:
                        end_fp64.record()
                        end_fp64.synchronize()
                        time_fp64 += cp.cuda.get_elapsed_time(start_fp64, end_fp64)

                if logger.verbose > lib.logger.INFO:
                    elapsed_time = time_fp32 + time_fp64
                    llll = f"({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})"
                    msg_kernel = f"kernel type {llll}/({ip}{jp}{kp}{lp}), "
                    msg_time = (
                        f"FP32 = {time_fp32:5.2f} ms, FP64 = {time_fp64:5.2f} ms, "
                        f"total = {elapsed_time:5.2f} ms"
                    )
                    msg = msg_kernel + msg_time
                    logger.debug1(msg)
                    timing_counter[llll] += elapsed_time
            if hermi == 1:
                vj *= 2.0
            else:
                vj, vjT = vj[: n_dm // 2], vj[n_dm // 2 :]
                vj += vjT.transpose(0, 2, 1)
            vj = inplace_add_transpose(vj)
            vj = basis_layout.dm_to_mol(vj)
            vj = vj.reshape(dm.shape)

        if with_k:
            vk = cp.zeros(dms.shape, dtype=np.float64)
            for task in tasks_vk[::-1]:
                i, j, k, l = task
                li, ip = group_key[i]
                lj, jp = group_key[j]
                lk, kp = group_key[k]
                ll, lp = group_key[l]
                ang = (li, lj, lk, ll)
                nprim = (ip, jp, kp, lp)

                ij_pairs = pairs_vk[i, j]
                kl_pairs = pairs_vk[k, l]

                n_ij_pairs = ij_pairs.shape[0] // pair_wide_vk
                n_kl_pairs = kl_pairs.shape[0] // pair_wide_vk
                if n_ij_pairs == 0 or n_kl_pairs == 0:
                    continue

                q_cond_ij = q_cond_vk[i, j]
                q_cond_kl = q_cond_vk[k, l]

                time_fp32 = 0.0
                time_fp64 = 0.0
                if logger.verbose > lib.logger.INFO:
                    start_fp32 = cp.cuda.Event()
                    end_fp32 = cp.cuda.Event()
                    start_fp64 = cp.cuda.Event()
                    end_fp64 = cp.cuda.Event()

                if use_fp32:
                    if logger.verbose > lib.logger.INFO:
                        start_fp32.record()

                    _, _, fun_vk_fp32 = gen_vk_kernel(
                        ang,
                        nprim,
                        dtype=np.float32,
                        n_dm=n_dm,
                        omega=omega_fp32,
                        pair_wide=pair_wide_vk,
                    )

                    fun_vk_fp32(
                        nao,
                        basis_data_fp32,
                        dms_fp32,
                        vk,
                        omega_fp32,
                        ij_pairs,
                        n_ij_pairs,
                        kl_pairs,
                        n_kl_pairs,
                        q_cond_ij,
                        q_cond_kl,
                        np.float32(cutoff),
                    )
                    kern_counts += 1

                    if logger.verbose > lib.logger.INFO:
                        end_fp32.record()
                        end_fp32.synchronize()
                        time_fp32 += cp.cuda.get_elapsed_time(start_fp32, end_fp32)

                if use_fp64:
                    if logger.verbose > lib.logger.INFO:
                        start_fp64.record()

                    _, _, fun_vk_fp64 = gen_vk_kernel(
                        ang,
                        nprim,
                        dtype=np.float64,
                        n_dm=n_dm,
                        omega=omega_fp64,
                        pair_wide=pair_wide_vk,
                    )

                    fun_vk_fp64(
                        nao,
                        basis_data_fp64,
                        dms_fp64,
                        vk,
                        omega_fp64,
                        ij_pairs,
                        n_ij_pairs,
                        kl_pairs,
                        n_kl_pairs,
                        q_cond_ij,
                        q_cond_kl,
                        np.float32(cutoff),
                    )
                    kern_counts += 1

                    if logger.verbose > lib.logger.INFO:
                        end_fp64.record()
                        end_fp64.synchronize()
                        time_fp64 += cp.cuda.get_elapsed_time(start_fp64, end_fp64)

                if logger.verbose > lib.logger.INFO:
                    elapsed_time = time_fp32 + time_fp64
                    llll = f"({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})"
                    msg_kernel = f"kernel type {llll}/({ip}{jp}{kp}{lp}), "
                    msg_time = (
                        f"FP32 = {time_fp32:5.2f} ms, FP64 = {time_fp64:5.2f} ms, "
                        f"total = {elapsed_time:5.2f} ms"
                    )
                    msg = msg_kernel + msg_time
                    logger.debug1(msg)
                    timing_counter[llll] += elapsed_time
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
        logger.info(f"vj = {with_j} and vk = {with_k} take {cputime:.3f} sec (pair-based)")
        return vj, vk

    return get_jk
