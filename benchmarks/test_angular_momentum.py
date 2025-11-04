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
Test script to verify 2D algorithm support for various angular momentum combinations.

This script tests the 2D JK algorithm with different shell types:
- s shells (l=0)
- p shells (l=1)
- d shells (l=2)
- f shells (l=3)
- g shells (l=4)

Usage:
    python test_angular_momentum.py [--dtype fp32|fp64] [--verbose]
"""

import argparse
import math
import sys

import cupy as cp
import numpy as np
from pyscf import gto, scf

from jqc.backend.jk import gen_jk_kernel
from jqc.pyscf.basis import BasisLayout
from jqc.pyscf.jk import make_pairs


# Test cases: (li, lj, lk, ll, description)
TEST_CASES = [
    # Homogeneous cases
    (0, 0, 0, 0, "s-s-s-s (all s shells)"),
    (1, 1, 1, 1, "p-p-p-p (all p shells)"),
    (2, 2, 2, 2, "d-d-d-d (all d shells)"),
    (3, 3, 3, 3, "f-f-f-f (all f shells)"),

    # Mixed low angular momentum
    (0, 0, 1, 1, "s-s-p-p"),
    (0, 1, 0, 1, "s-p-s-p"),
    (1, 1, 2, 2, "p-p-d-d"),
    (0, 0, 2, 2, "s-s-d-d"),

    # Mixed medium angular momentum
    (1, 2, 1, 2, "p-d-p-d"),
    (2, 2, 3, 3, "d-d-f-f"),
    (0, 2, 0, 2, "s-d-s-d"),

    # Heterogeneous combinations
    (0, 1, 2, 3, "s-p-d-f (all different)"),
    (1, 0, 2, 0, "p-s-d-s"),
    (2, 1, 1, 0, "d-p-p-s"),
]


def create_test_molecule(ang):
    """Create a simple test molecule with appropriate basis for given angular momentum."""
    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 1.1"
    mol.unit = "B"

    # Create a simple basis with the required angular momentum
    li, lj, lk, ll = ang
    max_l = max(ang)

    # Use appropriate basis set based on max angular momentum
    if max_l == 0:
        mol.basis = "sto-3g"
    elif max_l == 1:
        mol.basis = "6-31g"
    elif max_l == 2:
        mol.basis = "def2-svp"
    elif max_l == 3:
        mol.basis = "def2-tzvp"
    else:  # max_l >= 4
        mol.basis = "def2-qzvp"

    mol.build()
    return mol


def test_angular_momentum(ang, dtype, verbose=False):
    """
    Test the 2D JK algorithm for a given angular momentum combination.

    Returns:
        Tuple of (success, vj_error, vk_error, message)
    """
    omega = 0.0
    cutoff = math.log(1e-10)

    try:
        # Setup molecule and basis
        mol = create_test_molecule(ang)
        basis_layout = BasisLayout.from_mol(mol)
        nbas = basis_layout.nbasis
        nao = int(basis_layout.ao_loc[-1])
        group_offset = basis_layout.group_info[1]
        q_matrix = basis_layout.q_matrix(omega=omega)
        ao_loc = cp.asarray(basis_layout.ao_loc)
        coords = cp.asarray(basis_layout.coords_fp64)
        ce_data = cp.asarray(basis_layout.ce_fp64)

        # Get density matrix from reference calculation
        mf = scf.RHF(mol).run(verbose=0)
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

        nprim = tuple(nprim_map.get(l, 1) for l in ang)

        # Generate JK kernel with 2D algorithm
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
            return (False, 0.0, 0.0, f"No pairs found for angular momentum {ang}")

        # Execute JK computation
        args = (
            nbas, nao, ao_loc, coords, ce_data, dms_jqc, vj, vk, omega,
            ij_pairs, n_ij_pairs, kl_pairs, n_kl_pairs,
        )
        fun(*args)

        # Verify against PySCF reference
        vj_ref, vk_ref = scf.hf.get_jk(mol, dms.get(), hermi=1)
        vj_ref = cp.asarray(vj_ref)
        vk_ref = cp.asarray(vk_ref)

        tolerance = 1e-9 if dtype == np.float64 else 1e-4
        vj_diff = float(cp.abs(vj - vj_ref).max())
        vk_diff = float(cp.abs(vk - vk_ref).max())

        success = vj_diff < tolerance and vk_diff < tolerance
        message = f"vj_err={vj_diff:.2e}, vk_err={vk_diff:.2e}"

        if verbose:
            print(f"  Pairs: ij={n_ij_pairs}, kl={n_kl_pairs}")
            print(f"  {message}")

        return (success, vj_diff, vk_diff, message)

    except Exception as e:
        return (False, 0.0, 0.0, f"Exception: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype", choices=["fp32", "fp64"], default="fp64",
        help="Data type for computation (default: fp64)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed output for each test"
    )
    parser.add_argument(
        "--subset", choices=["homogeneous", "mixed", "all"], default="all",
        help="Which subset of tests to run"
    )
    args = parser.parse_args()

    dtype = np.float32 if args.dtype == "fp32" else np.float64

    print(f"Testing 2D JK Algorithm with {args.dtype}")
    print("=" * 70)

    # Filter test cases based on subset
    if args.subset == "homogeneous":
        test_cases = [tc for tc in TEST_CASES if tc[0] == tc[1] == tc[2] == tc[3]]
    elif args.subset == "mixed":
        test_cases = [tc for tc in TEST_CASES if not (tc[0] == tc[1] == tc[2] == tc[3])]
    else:
        test_cases = TEST_CASES

    passed = 0
    failed = 0
    errors = []

    for li, lj, lk, ll, desc in test_cases:
        ang = (li, lj, lk, ll)
        print(f"\nTesting {ang} - {desc}")

        success, vj_err, vk_err, message = test_angular_momentum(ang, dtype, args.verbose)

        if success:
            print(f"  ✓ PASSED: {message}")
            passed += 1
        else:
            print(f"  ✗ FAILED: {message}")
            failed += 1
            errors.append((ang, desc, message))

    # Summary
    print("\n" + "=" * 70)
    print(f"Test Summary: {passed} passed, {failed} failed out of {len(test_cases)} tests")

    if errors:
        print("\nFailed tests:")
        for ang, desc, message in errors:
            print(f"  {ang} ({desc}): {message}")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
