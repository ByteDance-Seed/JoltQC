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
Benchmark script for testing JK algorithm implementations.

Usage:
    python benchmark_algorithms.py <li> <lj> <lk> <ll> <precision>

Example:
    python benchmark_algorithms.py 0 0 0 0 fp64
    python benchmark_algorithms.py 1 1 1 1 fp32
"""

import math
import sys

import cupy as cp
import numpy as np
from pyscf import gto, scf

from jqc.backend.jk import gen_jk_kernel
from jqc.pyscf.basis import BasisLayout
from jqc.pyscf.jk import make_pairs


def benchmark(ang, dtype):
    """
    Benchmark JK matrix computation for given angular momentum and precision.

    Args:
        ang: Tuple of 4 integers (li, lj, lk, ll) representing angular momenta
        dtype: Data type for computation (np.float32 or np.float64)
    """
    omega = 0.0
    cutoff = math.log(1e-10)

    # Setup test molecule
    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 1.1"
    mol.basis = gto.basis.parse(
        """
        O    S
          1.2700058226E+00      1
          0.2700058226E+00      1
          0.02700058226E+00      1
        """
    )
    mol.build()

    # Setup basis layout and arrays
    basis_layout = BasisLayout.from_mol(mol)
    nbas = basis_layout.nbasis
    nao = int(basis_layout.ao_loc[-1])
    group_offset = basis_layout.group_info[1]
    q_matrix = basis_layout.q_matrix(omega=omega)
    ao_loc = cp.asarray(basis_layout.ao_loc)
    coords = cp.asarray(basis_layout.coords_fp64)
    ce_data = cp.asarray(basis_layout.ce_fp64)

    # Get density matrix from reference calculation
    mf = scf.RHF(mol).run()
    dms = cp.asarray(mf.make_rdm1())
    if dms.ndim == 2:
        dms = dms[None, :, :]
    dms_jqc = basis_layout.dm_from_mol(dms)

    # Determine primitives per angular momentum
    group_key = basis_layout.group_key
    nprim_map = {}
    for l, n_prim in group_key:
        if l not in nprim_map:
            nprim_map[l] = n_prim

    nprim = (
        nprim_map.get(ang[0], 1),
        nprim_map.get(ang[1], 1),
        nprim_map.get(ang[2], 1),
        nprim_map.get(ang[3], 1),
    )

    # Generate JK kernel and compute pairs
    vj = cp.zeros_like(dms_jqc)
    vk = cp.zeros_like(dms_jqc)
    fun = gen_jk_kernel(
        ang, nprim, dtype=dtype, frags=(-2,), n_dm=dms_jqc.shape[0], omega=omega
    )
    pairs = make_pairs(group_offset, q_matrix, cutoff)

    ij_pairs = pairs.get((ang[0], ang[1]), cp.array([], dtype=cp.int32))
    kl_pairs = pairs.get((ang[2], ang[3]), cp.array([], dtype=cp.int32))
    n_ij_pairs = len(ij_pairs)
    n_kl_pairs = len(kl_pairs)

    if n_ij_pairs == 0 or n_kl_pairs == 0:
        print("No pairs found for the given angular momenta.")
        print(f"Angular momenta: {ang}")
        print(f"Group offset: {group_offset}")
        return

    # Execute JK computation
    args = (
        nbas,
        nao,
        ao_loc,
        coords,
        ce_data,
        dms_jqc,
        vj,
        vk,
        omega,
        ij_pairs,
        n_ij_pairs,
        kl_pairs,
        n_kl_pairs,
    )
    fun(*args)

    print(f"Computed JK with {n_ij_pairs} ij-pairs and {n_kl_pairs} kl-pairs")

    # Verify results against PySCF reference
    vj_ref, vk_ref = scf.hf.get_jk(mol, dms.get(), hermi=1)
    vj_ref = cp.asarray(vj_ref)
    vk_ref = cp.asarray(vk_ref)

    tolerance = 1e-10 if dtype == np.float64 else 1e-4
    vj_diff = cp.abs(vj - vj_ref).max()
    vk_diff = cp.abs(vk - vk_ref).max()

    print("\nVerification Results:")
    print(f"  vj shape: {vj.shape}")
    print(f"  vj_ref shape: {vj_ref.shape}")
    print(f"  max |vj - vj_ref| = {vj_diff:.2e}")
    print(f"  max |vk - vk_ref| = {vk_diff:.2e}")
    print(f"  tolerance = {tolerance:.2e}")

    try:
        assert vj_diff < tolerance, f"vj error {vj_diff:.2e} exceeds tolerance {tolerance:.2e}"
        assert vk_diff < tolerance, f"vk error {vk_diff:.2e} exceeds tolerance {tolerance:.2e}"
        print("\n✓ All checks passed!")
    except AssertionError as e:
        print(f"\n✗ Verification failed: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(__doc__)
        print("\nError: Expected 5 arguments (li, lj, lk, ll, precision)")
        print("Example: python benchmark_algorithms.py 0 0 0 0 fp64")
        sys.exit(1)

    try:
        ang = tuple(map(int, sys.argv[1:5]))
        dtype = np.float32 if sys.argv[5] == "fp32" else np.float64
        print(f"Running benchmark with angular momenta {ang} and {sys.argv[5]}")
        benchmark(ang, dtype)
    except ValueError as e:
        print(f"Error: Invalid angular momentum values. Expected integers.")
        print(__doc__)
        sys.exit(1)