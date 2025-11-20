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

import sys
from pathlib import Path

import cupy as cp
import numpy as np
from pyscf import gto

from jqc.pyscf.basis import BasisLayout
from jqc.pyscf.jk import generate_get_jk
from jqc.pyscf.jk_pair import generate_get_jk as generate_get_jk_pair

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
    exponents = [0.27]

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

    # Setup basis layout
    basis_layout = BasisLayout.from_mol(mol)
    nbas = basis_layout.nbasis
    nao = int(basis_layout.ao_loc[-1])  # Cartesian basis size
    nao_mol = mol.nao  # Molecular (spherical) basis size

    # Generate JoltQC get_jk function for reference comparison (regular JK)
    if dtype == np.float32:
        jqc_get_jk = generate_get_jk(basis_layout, cutoff_fp32=1e-13, cutoff_fp64=1e100)
    else:
        jqc_get_jk = generate_get_jk(basis_layout, cutoff_fp64=1e-13)

    # Generate pair-based JK function
    pair_wide_vk = 64
    if dtype == np.float32:
        jqc_get_jk_pair = generate_get_jk_pair(
            basis_layout,
            cutoff_fp32=1e-13,
            cutoff_fp64=1e100,
            pair_wide_vk=pair_wide_vk
        )
    else:
        jqc_get_jk_pair = generate_get_jk_pair(
            basis_layout,
            cutoff_fp64=1e-13,
            pair_wide_vk=pair_wide_vk
        )

    # Build a deterministic symmetric density matrix in MOLECULAR basis
    rng = np.random.default_rng(42)
    dm_cpu = rng.standard_normal((nao_mol, nao_mol))
    dm_cpu = (dm_cpu + dm_cpu.T) * 0.5
    dm_gpu = cp.asarray(dm_cpu, dtype=np.float64)

    print(f"\nBasis: {nbas} shells, {nao} AOs (cartesian), {nao_mol} AOs (molecular)")
    print(f"Precision: {dtype.__name__}, omega={omega}")
    print(f"\nUsing pair-based JK algorithm (pair_wide_vk={pair_wide_vk})")

    # ===== Execute pair-based JK =====
    start_pair = cp.cuda.Event()
    stop_pair = cp.cuda.Event()
    start_pair.record()
    vj_pair, vk_pair = jqc_get_jk_pair(mol, dm_gpu, hermi=1, omega=omega)
    stop_pair.record()
    stop_pair.synchronize()
    elapsed_pair = cp.cuda.get_elapsed_time(start_pair, stop_pair) * 1e-3

    # ===== Verify against reference implementation =====
    ref_start = cp.cuda.Event()
    ref_stop = cp.cuda.Event()
    ref_start.record()
    vj_ref, vk_ref = jqc_get_jk(mol, dm_gpu, hermi=1, omega=omega)
    ref_stop.record()
    ref_stop.synchronize()
    elapsed_ref = cp.cuda.get_elapsed_time(ref_start, ref_stop) * 1e-3

    tolerance = 1e-8 if dtype == np.float64 else 1e-4
    vj_diff = cp.abs(vj_pair - vj_ref).max()
    vk_diff = cp.abs(vk_pair - vk_ref).max()

    print(f"\nVerification (tolerance={tolerance:.0e}):")
    print(f"  vj error: {vj_diff:.2e}")
    print(f"  vk error: {vk_diff:.2e}")
    print(f"\nTiming:")
    print(f"  Pair-based  : {elapsed_pair:.6f} s")
    print(f"  Reference   : {elapsed_ref:.6f} s")
    print(f"  Speedup     : {elapsed_ref/elapsed_pair:.2f}x")

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
