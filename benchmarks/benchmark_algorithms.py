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
from jqc.pyscf.jk import make_pairs, generate_get_jk
from jqc.backend.linalg_helper import inplace_add_transpose

def load_xyz(xyz_path):
    """
    Load atomic symbols and coordinates from an XYZ file.

    Args:
        xyz_path: Path to the XYZ file

    Returns:
        List of tuples (symbol, (x, y, z))
    """
    xyz_path = Path(xyz_path)
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
    # Use a sequence of exponents with reasonable spacing
    if l == 0:  # s shell
        exponents = [0.027, 0.027]#[1.27, 0.27, 0.027]
    elif l == 1:  # p shell
        exponents = [2.5, 0.6, 0.15]
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
    # Use simple H2 molecule for screening tests (set to False to use complex molecule)
    use_simple_molecule = False
    h2_distance = 100.0  # Angstroms - far apart for screening test

    mol = gto.Mole()
    if use_simple_molecule:
        # Simple two-hydrogen molecule far apart to test screening
        mol.atom = [
            ['H', (0.0, 0.0, 0.0)],
            ['H', (h2_distance, 0.0, 0.0)]
        ]
        mol.unit = "Angstrom"
        unique_symbols = ['H']
        print(f"\nUsing simple H2 molecule with separation: {h2_distance} Angstrom")
    else:
        xyz_atoms = load_xyz('molecules/0084-elongated-halogenated.xyz')
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
    ao_loc = cp.asarray(basis_layout.ao_loc)

    # Select appropriate precision for coords and ce_data based on dtype
    if dtype == np.float32:
        coords = cp.asarray(basis_layout.coords_fp32)
        ce_data = cp.asarray(basis_layout.ce_fp32)
    else:
        coords = cp.asarray(basis_layout.coords_fp64)
        ce_data = cp.asarray(basis_layout.ce_fp64)

    # Generate JoltQC get_jk function for reference comparison
    if dtype == np.float32:
        jqc_get_jk = generate_get_jk(basis_layout, cutoff_fp32=1e-13, cutoff_fp64=1e100)
    else:
        jqc_get_jk = generate_get_jk(basis_layout, cutoff_fp64=1e-13)

    # Cast omega to the correct precision for CUDA kernel
    omega_kernel = dtype(omega)
    omega_fp32 = np.float32(omega) if omega is not None else None
    omega_fp64 = np.float64(omega) if omega is not None else None

    # Build a deterministic symmetric density matrix in MOLECULAR basis
    # Then convert it to Cartesian basis for the kernel
    rng = np.random.default_rng(42)
    dm_cpu = rng.standard_normal((nao_mol, nao_mol))
    dm_cpu = (dm_cpu + dm_cpu.T) * 0.5
    dm_gpu = cp.asarray(dm_cpu, dtype=np.float64)
    dms = dm_gpu[None, :, :]
    # Convert from molecular to Cartesian basis for the 2D kernel
    dms_jqc = basis_layout.dm_from_mol(dms)

    # Ensure dms_jqc maintains the correct dtype
    dms_jqc = cp.asarray(dms_jqc, dtype=dtype)

    hermi = 1 # Added hermi definition

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
    n_dm = dms.shape[0]

    # Note: 2D kernel always expects vj/vk in fp64, regardless of dm precision
    vj = cp.zeros(dms_jqc.shape, dtype=np.float64)
    vk = cp.zeros(dms_jqc.shape, dtype=np.float64)

    print(f"\nBasis: {nbas} shells, {nao} AOs")
    print(f"Precision: {dtype.__name__}, omega={omega}")

    # Use 2D algorithm (frags=[-2])
    print("\nUsing 2D algorithm (frags=[-2])")
    fun = gen_jk_kernel(
        ang, nprim, dtype=dtype, frags=(-2,), n_dm=dms_jqc.shape[0], omega=omega, do_j=True, do_k=True
    )

    pairs = make_pairs(group_offset, q_matrix, cutoff)

    # Map angular momentum to group indices for pair selection
    group_key_arr = np.asarray(group_key)
    def find_group_idx(lam):
        for idx, (gl, _) in enumerate(group_key_arr.tolist()):
            if gl == lam:
                return idx
        return 0

    gi, gj, gk, gl = map(find_group_idx, ang)
    ij_pairs = pairs.get((gi, gj), cp.array([], dtype=cp.int32))
    kl_pairs = pairs.get((gk, gl), cp.array([], dtype=cp.int32))

    n_ij_pairs = len(ij_pairs)
    n_kl_pairs = len(kl_pairs)
    if n_ij_pairs == 0 or n_kl_pairs == 0:
        print("\nNo pairs found for the given angular momenta.")
        print(f"Angular momenta: {ang}")
        print(f"Group offset: {group_offset}")
        return

    # Flatten - kernel expects contiguous 1D arrays
    ij_pairs = cp.ascontiguousarray(ij_pairs.ravel())
    kl_pairs = cp.ascontiguousarray(kl_pairs.ravel())

    # Generate per-pair q_cond arrays for Schwarz screening
    def extract_q_cond(pairs_flat, nbas_val, q_mat):
        """Extract q screening values for each pair"""
        n_pairs = len(pairs_flat)
        q_cond = cp.zeros(n_pairs, dtype=cp.float32)
        for idx in range(n_pairs):
            pair = int(pairs_flat[idx].get())
            if pair >= nbas_val * nbas_val:
                # Padding pair - set to very negative value
                q_cond[idx] = -1000.0
            else:
                ish = pair // nbas_val
                jsh = pair % nbas_val
                q_cond[idx] = float(q_mat[ish, jsh].get())
        return q_cond

    q_cond_ij = extract_q_cond(ij_pairs, nbas, q_matrix)
    q_cond_kl = extract_q_cond(kl_pairs, nbas, q_matrix)
    log_cutoff = np.float32(cutoff)

    # Print screening statistics
    if use_simple_molecule:
        total_quartets = len(q_cond_ij) * len(q_cond_kl)
        screened_count = 0
        for q_ij_val in q_cond_ij.get():
            for q_kl_val in q_cond_kl.get():
                if q_ij_val + q_kl_val >= log_cutoff:
                    screened_count += 1
        print(f"Screening: {screened_count}/{total_quartets} quartets screened out ({100*screened_count/total_quartets:.1f}%)")

    # Execute JK computation with per-pair Schwarz screening
    args = (nbas, nao, ao_loc, coords, ce_data, dms_jqc, vj, vk,
            omega_kernel, ij_pairs, n_ij_pairs, kl_pairs, n_kl_pairs,
            q_cond_ij, q_cond_kl, log_cutoff)

    start_evt = cp.cuda.Event()
    stop_evt = cp.cuda.Event()
    start_evt.record()
    fun(*args)
    stop_evt.record()
    stop_evt.synchronize()
    elapsed_2d = cp.cuda.get_elapsed_time(start_evt, stop_evt) * 1e-3

    inplace_add_transpose(vj)
    inplace_add_transpose(vk)

    # Convert results back to molecular basis layout
    vj_mol = basis_layout.dm_to_mol(vj)
    vk_mol = basis_layout.dm_to_mol(vk)

    # Verify results against JoltQC reference implementation
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
    print(f"  2D kernel   : {elapsed_2d:.6f} s")
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
        print(f"\n{'='*60}")
        print(f"2D JK Algorithm Benchmark")
        print(f"  Shells: {shell_str} | Precision: {sys.argv[5]}")
        print(f"{'='*60}")

        benchmark(ang, dtype)
    except ValueError as e:
        print(f"Error: Invalid angular momentum values. Expected integers.")
        print(__doc__)
        sys.exit(1)
