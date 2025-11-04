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

This script benchmarks the 2D JK algorithm for specified angular momentum
combinations. It verifies correctness against PySCF reference implementation.

Angular Momentum Support:
- s shell (l=0): simplest case
- p shell (l=1): 3 basis functions
- d shell (l=2): 6 basis functions
- f shell (l=3): 10 basis functions
- g shell (l=4): 15 basis functions

Usage:
    python benchmark_algorithms.py <li> <lj> <lk> <ll> <precision>

Examples:
    python benchmark_algorithms.py 0 0 0 0 fp64   # s-s-s-s shells
    python benchmark_algorithms.py 1 1 1 1 fp32   # p-p-p-p shells
    python benchmark_algorithms.py 0 1 2 3 fp64   # mixed shells
"""

import math
import sys

import cupy as cp
import numpy as np
from pyscf import gto, scf

from jqc.backend.jk import gen_jk_kernel
from jqc.backend.linalg_helper import inplace_add_transpose
from jqc.pyscf.basis import BasisLayout
from jqc.pyscf.jk import make_pairs


def generate_basis_for_angular_momentum(l):
    """
    Generate a custom basis set with the specified angular momentum.

    Args:
        l: Angular momentum (0=s, 1=p, 2=d, 3=f, 4=g)

    Returns:
        Basis string in PySCF format
    """
    shell_types = ['S', 'P', 'D', 'F', 'G']
    shell_type = shell_types[l]

    # Generate appropriate exponents for the given angular momentum
    # Use a sequence of exponents with reasonable spacing
    if l == 0:  # s shell
        exponents = [1.27, 0.27, 0.027]
    elif l == 1:  # p shell
        exponents = [2.5, 0.6, 0.15]
    elif l == 2:  # d shell
        exponents = [3.5, 0.8, 0.2]
    elif l == 3:  # f shell
        exponents = [4.5, 1.0, 0.25]
    else:  # g shell (l=4)
        exponents = [5.5, 1.2, 0.3]

    # Build basis string
    basis_lines = [f"H    {shell_type}"]
    for exp in exponents:
        basis_lines.append(f"      {exp:.10f}      1")

    return "\n".join(basis_lines)


def benchmark(ang, dtype):
    """
    Benchmark JK matrix computation for given angular momentum and precision.

    Args:
        ang: Tuple of 4 integers (li, lj, lk, ll) representing angular momenta
        dtype: Data type for computation (np.float32 or np.float64)
    """
    omega = 0.0
    cutoff = math.log(1e-10)

    li, lj, lk, ll = ang

    # For homogeneous angular momentum, use single L value
    # Since we don't consider mixed shells, all four should be the same
    if not (li == lj == lk == ll):
        print(f"Warning: Mixed angular momentum {ang} detected.")
        print(f"Using L={li} for all shells to ensure basis consistency.")
        L = li
    else:
        L = li

    # Setup test molecule with custom basis for the given angular momentum
    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 1.1"
    mol.unit = "B"

    # Generate custom basis with the exact angular momentum needed
    basis_str = generate_basis_for_angular_momentum(L)
    mol.basis = gto.basis.parse(basis_str)
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

    # Debug: Print basis layout info
    print(f"\nBasis Layout Info:")
    print(f"  group_key: {group_key}")
    print(f"  group_offset: {group_offset}")
    print(f"  nbas: {nbas}, nao: {nao}")

    # Try with 1q1t algorithm first to verify setup
    print(f"\nUsing 1q1t algorithm (frags=[-1]) for testing")
    fun = gen_jk_kernel(
        ang, nprim, dtype=dtype, frags=(-1,), n_dm=dms_jqc.shape[0], omega=omega
    )
    pairs = make_pairs(group_offset, q_matrix, cutoff)

    # Debug: Print available pairs
    print(f"\nAvailable pair keys in dict: {list(pairs.keys())}")
    print(f"Looking for angular momentum: {ang}")

    # BUGFIX: Use group indices (0, 0) instead of angular momentum values
    # Since we have a homogeneous basis with all same L, there's only one group
    # and the group index is 0
    ij_pairs = pairs.get((0, 0), cp.array([], dtype=cp.int32))
    kl_pairs = pairs.get((0, 0), cp.array([], dtype=cp.int32))
    n_ij_pairs = len(ij_pairs)
    n_kl_pairs = len(kl_pairs)

    if n_ij_pairs == 0 or n_kl_pairs == 0:
        print("\nNo pairs found for the given angular momenta.")
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

    # Apply symmetry transformations for hermitian matrices
    # This is required to get the full J and K matrices
    vj = inplace_add_transpose(vj)
    vk = inplace_add_transpose(vk)

    # Convert results back to molecular basis layout
    vj_mol = basis_layout.dm_to_mol(vj)
    vk_mol = basis_layout.dm_to_mol(vk)

    # Verify results against PySCF reference
    vj_ref, vk_ref = scf.hf.get_jk(mol, dms.get(), hermi=1)
    vj_ref = cp.asarray(vj_ref)
    vk_ref = cp.asarray(vk_ref)

    tolerance = 1e-9 if dtype == np.float64 else 1e-4
    vj_diff = cp.abs(vj_mol - vj_ref).max()
    vk_diff = cp.abs(vk_mol - vk_ref).max()

    print("\nVerification Results:")
    print(f"  vj_mol shape: {vj_mol.shape}")
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

        # Validate angular momentum range
        max_l = 4  # Maximum supported angular momentum (g shell)
        if any(l < 0 or l > max_l for l in ang):
            print(f"Error: Angular momentum {ang} out of range")
            print(f"Supported range: 0 (s shell) to {max_l} (g shell)")
            sys.exit(1)

        dtype = np.float32 if sys.argv[5] == "fp32" else np.float64

        # Print configuration
        shell_names = ['s', 'p', 'd', 'f', 'g']
        shell_str = '-'.join(shell_names[l] for l in ang)
        print(f"Benchmarking 2D JK Algorithm")
        print(f"  Angular momentum: {ang} ({shell_str})")
        print(f"  Precision: {sys.argv[5]}")
        print()

        benchmark(ang, dtype)
    except ValueError as e:
        print(f"Error: Invalid angular momentum values. Expected integers.")
        print(__doc__)
        sys.exit(1)