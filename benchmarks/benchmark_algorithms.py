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
combinations. It verifies correctness against JoltQC reference implementation.

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
from pathlib import Path

import cupy as cp
import numpy as np
from pyscf import gto

from jqc.backend.jk import gen_jk_kernel
from jqc.pyscf.basis import BasisLayout
from jqc.pyscf.jk import make_pairs, make_pairs_symmetric, generate_get_jk
from jqc.backend.linalg_helper import inplace_add_transpose

BENCH_DIR = Path(__file__).resolve().parent


def load_xyz(xyz_path):
    """
    Load atomic symbols and coordinates from an XYZ file.

    Args:
        xyz_path: Path to the XYZ file

    Returns:
        List of tuples (symbol, (x, y, z))
    """
    xyz_path = (BENCH_DIR / xyz_path).resolve()
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

    with xyz_path.open("r", encoding="ascii") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        raise ValueError(f"XYZ file {xyz_path} is too short.")

    try:
        natoms = int(lines[0])
    except ValueError as exc:
        raise ValueError(f"First line of {xyz_path} must be an integer.") from exc

    atom_lines = lines[2 : 2 + natoms]
    if len(atom_lines) != natoms:
        raise ValueError(
            f"XYZ file {xyz_path} lists {len(atom_lines)} atoms, expected {natoms}."
        )

    atoms = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ atom line: {line!r}")
        symbol = parts[0]
        try:
            x, y, z = map(float, parts[1:4])
        except ValueError as exc:
            raise ValueError(f"Invalid coordinates in line: {line!r}") from exc
        atoms.append((symbol, (x, y, z)))
    return atoms


def generate_basis_for_angular_momentum(l, symbol="H"):
    """
    Generate a custom basis set with the specified angular momentum.

    Args:
        l: Angular momentum (0=s, 1=p, 2=d, 3=f, 4=g)
        symbol: Atomic symbol to associate with the generated shell

    Returns:
        Basis string in PySCF format
    """
    shell_types = ['S', 'P', 'D', 'F', 'G']
    shell_type = shell_types[l]

    # Generate appropriate exponents for the given angular momentum
    if l == 0:  # s shell
        exponents = [0.27]
    elif l == 1:  # p shell
        exponents = [0.27]
    elif l == 2:  # d shell
        exponents = [3.5, 0.8, 0.2]
    elif l == 3:  # f shell
        exponents = [4.5, 1.0, 0.25]
    else:  # g shell (l=4)
        exponents = [5.5, 1.2, 0.3]

    # Build basis string
    basis_lines = [f"{symbol}    {shell_type}"]
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
    cutoff = math.log(1e-13)

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
    xyz_atoms = load_xyz("molecules/0401-globular-nitrogenous.xyz")
    mol.atom = [(sym, coords) for sym, coords in xyz_atoms]
    mol.unit = "Angstrom"
    unique_symbols = sorted({sym for sym, _ in xyz_atoms})
    basis_blocks = [
        generate_basis_for_angular_momentum(L, symbol=sym) for sym in unique_symbols
    ]
    basis_str = "\n\n".join(basis_blocks)
    mol.basis = gto.basis.parse(basis_str)
    mol.build()

    # Setup basis layout and arrays
    basis_layout = BasisLayout.from_mol(mol)
    nbas = basis_layout.nbasis
    nao = int(basis_layout.ao_loc[-1])  # Cartesian basis size
    nao_mol = mol.nao  # Molecular (spherical) basis size
    group_offset = basis_layout.group_info[1]
    q_matrix = basis_layout.q_matrix(omega=omega)

    # Select appropriate precision for packed basis_data based on dtype
    if dtype == np.float32:
        basis_data_dict = basis_layout.basis_data_fp32
        basis_data = basis_data_dict['packed'].ravel()
    else:
        basis_data_dict = basis_layout.basis_data_fp64
        basis_data = basis_data_dict['packed'].ravel()

    # Generate JoltQC get_jk function for reference comparison
    if dtype == np.float32:
        jqc_get_jk = generate_get_jk(basis_layout, cutoff_fp32=1e-13, cutoff_fp64=1e100)
    else:
        jqc_get_jk = generate_get_jk(basis_layout, cutoff_fp64=1e-13)

    # Cast omega to the correct precision for CUDA kernel
    omega_kernel = dtype(omega)

    # Build a deterministic symmetric density matrix in MOLECULAR basis
    rng = np.random.default_rng(42)
    n_dm = 1
    dm_cpu = rng.standard_normal((n_dm, nao_mol, nao_mol))
    dm_cpu = (dm_cpu + dm_cpu.transpose(0,2,1)) * 0.5
    dm_gpu = cp.asarray(dm_cpu, dtype=np.float64)
    dms = dm_gpu

    # Convert from molecular to Cartesian basis for the 2D kernel
    dms_jqc = basis_layout.dm_from_mol(dms)
    dms_jqc = cp.asarray(dms_jqc, dtype=dtype)

    hermi = 1

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

    if hermi == 0:
        # Contract the tril and triu parts separately
        dms = cp.vstack([dms, dms.transpose(0, 2, 1)])

    # Note: 2D kernel always expects vj/vk in fp64, regardless of dm precision
    vj = cp.zeros(dms_jqc.shape, dtype=np.float64)
    vk = cp.zeros(dms_jqc.shape, dtype=np.float64)

    print(f"\nBasis: {nbas} shells, {nao} AOs")
    print(f"Precision: {dtype.__name__}, omega={omega}")

    # Use 2D algorithm - but call VJ and VK separately since they use different pair formats
    print("\nUsing 2D algorithm with separate VJ and VK invocations")

    # Define pair_wide before kernel generation
    pair_wide = 64

    # Generate separate VJ and VK kernels
    fun_vj = gen_jk_kernel(
        ang,
        nprim,
        dtype=dtype,
        frags=(-2,),
        n_dm=dms_jqc.shape[0],
        omega=omega,
        do_j=True,
        do_k=False,
    )
    fun_vk = gen_jk_kernel(
        ang, nprim, dtype=dtype, frags=(-2,), n_dm=dms_jqc.shape[0], omega=omega, do_j=False, do_k=True, pair_wide=pair_wide
    )

    # Map angular momentum to group indices for pair selection
    group_key_arr = np.asarray(group_key)
    def find_group_idx(lam):
        for idx, (gl, _) in enumerate(group_key_arr.tolist()):
            if gl == lam:
                return idx
        return 0

    gi, gj, gk, gl = map(find_group_idx, ang)

    def pairs_to_int2(pairs_flat):
        if pairs_flat.size == 0:
            return cp.empty((0, 2), dtype=cp.int32)
        # Create (N, 2) array and ensure C-contiguous for CUDA
        result = cp.empty((pairs_flat.size, 2), dtype=cp.int32, order='C')
        result[:, 0] = pairs_flat // nbas
        result[:, 1] = pairs_flat % nbas
        # View as int2-compatible format: flatten to treat as array of 2-int structs
        return cp.ascontiguousarray(result)

    def q_cond_from_pairs(pairs_flat):
        if pairs_flat.size == 0:
            return cp.empty(0, dtype=cp.float32)
        i_idx = pairs_flat // nbas
        j_idx = pairs_flat % nbas
        return q_matrix[i_idx, j_idx].astype(cp.float32, copy=False)

    # ===== VJ Pairs: Use tiled pairs like VK for efficiency =====
    pairs_vj = make_pairs_symmetric(group_offset, q_matrix, cutoff)
    ij_pairs_vj = pairs_vj.get((gi, gj), cp.array([], dtype=cp.int32))
    kl_pairs_vj = pairs_vj.get((gk, gl), cp.array([], dtype=cp.int32))
    n_ij_pairs_vj = len(ij_pairs_vj)
    n_kl_pairs_vj = len(kl_pairs_vj)
    # ===== VK Pairs: Tiled pairs from make_pairs =====
    pairs_vk = make_pairs(group_offset, q_matrix, cutoff, column_size=pair_wide)

    ij_pairs_vk = pairs_vk.get((gi, gj), cp.array([], dtype=cp.int32))
    kl_pairs_vk = pairs_vk.get((gk, gl), cp.array([], dtype=cp.int32))
    n_ij_pairs_vk = len(ij_pairs_vk)  # Number of tiles
    n_kl_pairs_vk = len(kl_pairs_vk)  # Number of tiles
    
    if (
        n_ij_pairs_vj == 0
        or n_kl_pairs_vj == 0
        or n_ij_pairs_vk == 0
        or n_kl_pairs_vk == 0
    ):
        print("\nNo pairs found for the given angular momenta.")
        print(f"Angular momenta: {ang}")
        print(f"Group offset: {group_offset}")
        return

    print(f"VJ pairs: {n_ij_pairs_vj} ij, {n_kl_pairs_vj} kl (tiled pairs)")
    print(f"VK pairs: {n_ij_pairs_vk} ij, {n_kl_pairs_vk} kl (tiled pairs, {pair_wide}-wide)")

    # ===== VJ: Flatten tiled pairs and extract q_cond =====
    ij_pairs_vj_flat = cp.ascontiguousarray(ij_pairs_vj.ravel())
    kl_pairs_vj_flat = cp.ascontiguousarray(kl_pairs_vj.ravel())
    n_ij_pairs_vj_flat = int(ij_pairs_vj_flat.size)  # Total flattened pairs for kernel
    n_kl_pairs_vj_flat = int(kl_pairs_vj_flat.size)  # Total flattened pairs for kernel
    q_cond_ij_vj = q_cond_from_pairs(ij_pairs_vj_flat)
    q_cond_kl_vj = q_cond_from_pairs(kl_pairs_vj_flat)

    # ===== VK: Flatten tiled pairs and extract q_cond =====
    ij_pairs_vk_flat = cp.ascontiguousarray(ij_pairs_vk.ravel())
    kl_pairs_vk_flat = cp.ascontiguousarray(kl_pairs_vk.ravel())
    q_cond_ij_vk = q_cond_from_pairs(ij_pairs_vk_flat)
    q_cond_kl_vk = q_cond_from_pairs(kl_pairs_vk_flat)
    log_cutoff = np.float32(cutoff)

    # ===== VJ: Convert flattened tiled pairs to int2 format =====
    # Packed format: pair = i * nbas + j, so i = pair // nbas, j = pair % nbas
    ij_vj = pairs_to_int2(ij_pairs_vj_flat)
    kl_vj = pairs_to_int2(kl_pairs_vj_flat)

    # ===== VK: Convert flattened tiled pairs to int2 format =====
    # Packed format: pair = i * nbas + j, so i = pair // nbas, j = pair % nbas
    ij_vk = pairs_to_int2(ij_pairs_vk_flat)
    kl_vk = pairs_to_int2(kl_pairs_vk_flat)

    # ===== Execute VJ kernel =====
    vk_dummy = cp.zeros_like(vj)  # Dummy vk for combined signature
    # VJ uses its own tiled pairs
    vj_args = (
        nao,
        basis_data,
        dms_jqc,
        vj,
        vk_dummy,
        omega_kernel,
        ij_vj,  # Use VJ tiled pairs
        n_ij_pairs_vj_flat,  # Total flattened pairs
        kl_vj,  # Use VJ tiled pairs
        n_kl_pairs_vj_flat,  # Total flattened pairs
        q_cond_ij_vj,  # Use VJ q_cond
        q_cond_kl_vj,  # Use VJ q_cond
        log_cutoff,
    )
    
    start_vj = cp.cuda.Event()
    stop_vj = cp.cuda.Event()
    start_vj.record()
    fun_vj(*vj_args)
    stop_vj.record()
    stop_vj.synchronize()
    elapsed_vj = cp.cuda.get_elapsed_time(start_vj, stop_vj) * 1e-3

    # ===== Execute VK kernel =====
    # Note: n_ij_pairs_vk and n_kl_pairs_vk are tile counts, not individual pair counts
    vj_dummy = cp.zeros_like(vk)  # Dummy vj for combined signature
    vk_args = (
        nao,
        basis_data,
        dms_jqc,
        vj_dummy,
        vk,
        omega_kernel,
        ij_vk,
        n_ij_pairs_vk,
        kl_vk,
        n_kl_pairs_vk,
        q_cond_ij_vk,
        q_cond_kl_vk,
        log_cutoff,
    )

    start_vk = cp.cuda.Event()
    stop_vk = cp.cuda.Event()
    start_vk.record()
    fun_vk(*vk_args)
    stop_vk.record()
    stop_vk.synchronize()
    elapsed_vk = cp.cuda.get_elapsed_time(start_vk, stop_vk) * 1e-3

    elapsed_2d = elapsed_vj + elapsed_vk

    # ===== Post-process results =====
    inplace_add_transpose(vj)
    inplace_add_transpose(vk)
    vj *= 2.0

    # Convert results back to molecular basis layout
    vj_mol = basis_layout.dm_to_mol(vj)
    vk_mol = basis_layout.dm_to_mol(vk)

    # ===== Verify against reference implementation =====
    ref_start = cp.cuda.Event()
    ref_stop = cp.cuda.Event()
    ref_start.record()
    vj_ref, vk_ref = jqc_get_jk(mol, dms, hermi=1, omega=omega)
    ref_stop.record()
    ref_stop.synchronize()
    elapsed_ref = cp.cuda.get_elapsed_time(ref_start, ref_stop) * 1e-3

    tolerance = 1e-8 if dtype == np.float64 else 1e-4
    vj_diff = cp.abs(vj_mol - vj_ref).max()
    vk_diff = cp.abs(vk_mol - vk_ref).max()

    print(f"\nVerification (tolerance={tolerance:.0e}):")
    print(f"  vj error: {vj_diff:.2e}")
    print(f"  vk error: {vk_diff:.2e}")
    print(f"\nTiming:")
    print(f"  VJ kernel   : {elapsed_vj:.6f} s")
    print(f"  VK kernel   : {elapsed_vk:.6f} s")
    print(f"  2D total    : {elapsed_2d:.6f} s")
    print(f"  Reference   : {elapsed_ref:.6f} s")
    print(f"  Speedup     : {elapsed_ref/elapsed_2d:.2f}x")

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
    print(f"\n{'='*60}")
    print(f"2D JK Algorithm Benchmark")
    print(f"  Shells: {shell_str} | Precision: {sys.argv[5]}")
    print(f"{'='*60}")

    benchmark(ang, dtype)
