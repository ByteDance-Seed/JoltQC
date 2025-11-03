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

import math
import cupy as cp
import numpy as np
from jqc.backend.jk import gen_jk_kernel
from jqc.pyscf.jk import make_pairs, make_tile_pairs
from jqc.pyscf.basis import BasisLayout
from jqc.backend.linalg_helper import inplace_add_transpose

from pyscf import gto, scf
import sys


def benchmark(ang, dtype):
    omega = 0.0
    cutoff = math.log(1e-10)
    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 1.1"
    mol.basis = gto.basis.parse(
    """
    O    S
      0.2700058226E+00      1
    """)
    mol.build()
    nao_mol = mol.nao
    basis_layout = BasisLayout.from_mol(mol)
    nbas = basis_layout.nbasis
    nao = int(basis_layout.ao_loc[-1])
    group_offset = basis_layout.group_info[1]
    q_matrix = basis_layout.q_matrix(omega=omega)
    tile_pairs = make_tile_pairs(group_offset, q_matrix, cutoff)
    ao_loc = cp.asarray(basis_layout.ao_loc)
    coords = cp.asarray(basis_layout.coords_fp64)
    ce_data = cp.asarray(basis_layout.ce_fp64)
    mf = scf.RHF(mol).run()
    dms = cp.asarray(mf.make_rdm1())
    if dms.ndim == 2:
        dms = dms[None, :, :]
    dms_jqc = basis_layout.dm_from_mol(dms)
    vj = cp.zeros_like(dms_jqc)
    vk = cp.zeros_like(dms_jqc)
    omega = 0.0

    group_key = basis_layout.group_key
    nprim_map = {}
    for l, n_prim in group_key:
        if l not in nprim_map:
            nprim_map[l] = n_prim
    
    nprim = (nprim_map.get(ang[0], 1), nprim_map.get(ang[1], 1), nprim_map.get(ang[2], 1), nprim_map.get(ang[3], 1))

    # --- Benchmark 2d ---
    vj.fill(0)
    vk.fill(0)
    fun_2d = gen_jk_kernel(ang, nprim, dtype=dtype, frags=(-2,), n_dm=dms_jqc.shape[0], omega=omega)
    pairs = make_pairs(group_offset, q_matrix, cutoff)

    ij_pairs_2d = pairs.get((ang[0], ang[1]), cp.array([], dtype=cp.int32)).ravel()
    kl_pairs_2d = pairs.get((ang[2], ang[3]), cp.array([], dtype=cp.int32)).ravel()
    n_ij_pairs = len(ij_pairs_2d)
    n_kl_pairs = len(kl_pairs_2d)

    vj_2d = cp.zeros_like(dms_jqc)
    vk_2d = cp.zeros_like(dms_jqc)
    time_2d = 0.0

    if n_ij_pairs == 0 or n_kl_pairs == 0:
        print("No pairs found for 2d algorithm with the given angular momenta.")
    else:
        args_2d = (
            nbas,
            nao,
            ao_loc,
            coords,
            ce_data,
            dms_jqc,
            vj,
            vk,
            omega,
            ij_pairs_2d,
            n_ij_pairs,
            kl_pairs_2d,
            n_kl_pairs,
        )
        fun_2d(*args_2d)
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        fun_2d(*args_2d)
        end.record()
        end.synchronize()
        time_2d = cp.cuda.get_elapsed_time(start, end)
        vj_2d = vj.copy()
        vk_2d = vk.copy()
    print(ij_pairs_2d)
    print(kl_pairs_2d)
    print(f"2d algorithm took: {time_2d:.3f} ms")

    # Post-process the results
    vj_2d = basis_layout.dm_to_mol(vj_2d)
    vk_2d = basis_layout.dm_to_mol(vk_2d)

    # --- Verification ---
    vj_ref, vk_ref = scf.hf.get_jk(mol, dms.get(), hermi=1)
    vj_ref = cp.asarray(vj_ref)
    vk_ref = cp.asarray(vk_ref)

    tolerance = 1e-10 if dtype == np.float64 else 1e-4
    print(f"vj_2d shape: {vj_2d.shape}")
    print(f"vj_ref shape: {vj_ref.shape}")
    print(f"max vj_2d: {cp.abs(vj_2d).max()}")
    print(f"max vj_ref: {cp.abs(vj_ref).max()}")
    vj_diff = cp.abs(vj_2d - vj_ref).max()
    vk_diff = cp.abs(vk_2d - vk_ref).max()
    print(f"max |vj_2d - vj_ref| = {vj_diff:.2e}")
    print(f"max |vk_2d - vk_ref| = {vk_diff:.2e}")
    assert vj_diff < tolerance
    assert vk_diff < tolerance



if __name__ == "__main__":
    ang = tuple(map(int, sys.argv[1:5]))
    dtype = np.float32 if sys.argv[5] == "fp32" else np.float64
    benchmark(ang, dtype)