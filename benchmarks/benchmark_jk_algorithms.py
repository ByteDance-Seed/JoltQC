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
Comprehensive benchmark comparing quartet-based and pair-based JK algorithms.

This script benchmarks both JK algorithms across multiple molecular systems,
basis sets, and precision modes. It provides detailed performance comparisons
and accuracy verification.

IMPORTANT NOTE: The pair-based algorithm may not always be faster than the
quartet-based algorithm. Performance depends on:
- System size (number of atoms)
- Basis set type and size
- Angular momentum composition
- GPU architecture

The pair-based algorithm is typically optimized for larger systems with
higher angular momentum shells, while the quartet-based algorithm may
perform better for smaller systems or simpler basis sets.

Features:
- Multiple molecular systems (small to large)
- Multiple basis sets (STO-3G, 6-31G, def2-SVP, def2-TZVP)
- Multiple precision modes (FP32, FP64, mixed)
- Detailed timing breakdown (J-only, K-only, J+K)
- Accuracy verification against quartet reference
- Optional CSV output and plotting

Usage:
    python benchmark_jk_algorithms.py [--molecules MOLECULES] [--basis BASIS]
                                      [--precision PRECISION] [--output OUTPUT]
                                      [--plot]

Examples:
    # Run all benchmarks
    python benchmark_jk_algorithms.py

    # Test specific molecules
    python benchmark_jk_algorithms.py --molecules small medium

    # Test specific basis sets
    python benchmark_jk_algorithms.py --basis sto-3g 6-31g

    # Test specific precision modes
    python benchmark_jk_algorithms.py --precision fp64 mixed

    # Save results to CSV and generate plots
    python benchmark_jk_algorithms.py --output results.csv --plot
"""

import argparse
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import pyscf

from jqc.constants import TILE
from jqc.pyscf.basis import BasisLayout
from jqc.pyscf.jk import generate_get_jk as generate_quartet_jk
from jqc.pyscf.jk_pair import generate_get_jk as generate_pair_jk

BENCH_DIR = Path(__file__).resolve().parent

# Define test configurations
MOLECULE_CONFIGS = {
    "tiny": "molecules/0031-irregular-nitrogenous.xyz",  # ~30 atoms
    "small": "molecules/0084-elongated-halogenated.xyz",  # ~80 atoms
    "medium": "molecules/0166-irregular-nitrogenous.xyz",  # ~160 atoms
    "large": "molecules/0401-globular-nitrogenous.xyz",  # ~400 atoms
}

BASIS_SETS = {
    "sto-3g": "sto-3g",
    "6-31g": "6-31g",
    "def2-svp": "def2-svp",
    "def2-tzvp": "def2-tzvp",
}

PRECISION_CONFIGS = {
    "fp64": {"cutoff_fp64": 1e-13, "cutoff_fp32": 1e-13},
    "fp32": {"cutoff_fp64": 1e6, "cutoff_fp32": 1e-13},
    "mixed": {"cutoff_fp64": 1e-7, "cutoff_fp32": 1e-13},
}


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(
        self,
        molecule_name,
        basis_name,
        precision,
        natoms,
        nbasis,
        nao,
        algorithm,
        mode,
        time_ms,
        vj_error=None,
        vk_error=None,
    ):
        self.molecule_name = molecule_name
        self.basis_name = basis_name
        self.precision = precision
        self.natoms = natoms
        self.nbasis = nbasis
        self.nao = nao
        self.algorithm = algorithm
        self.mode = mode
        self.time_ms = time_ms
        self.vj_error = vj_error
        self.vk_error = vk_error

    def __repr__(self):
        return (
            f"BenchmarkResult(mol={self.molecule_name}, basis={self.basis_name}, "
            f"prec={self.precision}, algo={self.algorithm}, mode={self.mode}, "
            f"time={self.time_ms:.2f}ms)"
        )


def load_molecule(molecule_path):
    """Load molecule from XYZ file."""
    xyz_path = (BENCH_DIR / molecule_path).resolve()
    if not xyz_path.exists():
        raise FileNotFoundError(f"Molecule file not found: {xyz_path}")

    with xyz_path.open("r", encoding="ascii") as f:
        lines = [line.strip() for line in f if line.strip()]

    natoms = int(lines[0])
    atom_lines = lines[2 : 2 + natoms]

    atoms = []
    for line in atom_lines:
        parts = line.split()
        symbol = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append((symbol, (x, y, z)))

    return atoms


def create_test_molecule(molecule_name, basis_name, verbose=0):
    """Create PySCF molecule object."""
    atoms = load_molecule(MOLECULE_CONFIGS[molecule_name])

    mol = pyscf.M(
        atom=[(sym, coords) for sym, coords in atoms],
        basis=BASIS_SETS[basis_name],
        unit="Angstrom",
        output="/dev/null",
        verbose=verbose,
    )

    return mol


def time_kernel(kernel_fn, args, kwargs, n_warmup=3, n_runs=5):
    """Time kernel execution with warmup."""
    # Warmup
    for _ in range(n_warmup):
        result = kernel_fn(*args, **kwargs)

    # Timing
    times = []
    for _ in range(n_runs):
        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()
        result = kernel_fn(*args, **kwargs)
        end.record()
        end.synchronize()

        elapsed_ms = cp.cuda.get_elapsed_time(start, end)
        times.append(elapsed_ms)

    # Return median time and last result
    return np.median(times), result


def benchmark_configuration(
    molecule_name, basis_name, precision, n_warmup=3, n_runs=5, verbose=False
):
    """Benchmark a single configuration (molecule + basis + precision)."""
    if verbose:
        print(
            f"\n{'='*80}\n"
            f"Benchmarking: {molecule_name} / {basis_name} / {precision}\n"
            f"{'='*80}"
        )

    # Create molecule
    mol = create_test_molecule(molecule_name, basis_name)
    natoms = mol.natm
    nao = mol.nao

    # Create density matrix
    dm = cp.ones((nao, nao), dtype=np.float64)

    # Create basis layout
    layout = BasisLayout.from_mol(mol, alignment=TILE)
    nbasis = layout.nbasis

    if verbose:
        print(f"Atoms: {natoms}, Basis functions: {nbasis}, AOs: {nao}")

    # Get precision cutoffs
    cutoffs = PRECISION_CONFIGS[precision]

    results = []

    try:
        # ===== Quartet-based algorithm =====
        if verbose:
            print("\n--- Quartet-based algorithm ---")

        # Generate kernels
        quartet_jk = generate_quartet_jk(layout, **cutoffs)
        from jqc.pyscf.jk import generate_get_j as generate_quartet_j
        from jqc.pyscf.jk import generate_get_k as generate_quartet_k

        quartet_j = generate_quartet_j(layout, **cutoffs)
        quartet_k = generate_quartet_k(layout, **cutoffs)

        # Benchmark J+K
        time_jk, (vj_quartet, vk_quartet) = time_kernel(
            quartet_jk, (mol, dm), {"hermi": 1}, n_warmup=n_warmup, n_runs=n_runs
        )
        results.append(
            BenchmarkResult(
                molecule_name,
                basis_name,
                precision,
                natoms,
                nbasis,
                nao,
                "quartet",
                "J+K",
                time_jk,
            )
        )

        # Benchmark J-only
        time_j, vj_quartet_only = time_kernel(
            quartet_j, (mol, dm), {"hermi": 1}, n_warmup=n_warmup, n_runs=n_runs
        )
        results.append(
            BenchmarkResult(
                molecule_name,
                basis_name,
                precision,
                natoms,
                nbasis,
                nao,
                "quartet",
                "J-only",
                time_j,
            )
        )

        # Benchmark K-only
        time_k, vk_quartet_only = time_kernel(
            quartet_k, (mol, dm), {"hermi": 1}, n_warmup=n_warmup, n_runs=n_runs
        )
        results.append(
            BenchmarkResult(
                molecule_name,
                basis_name,
                precision,
                natoms,
                nbasis,
                nao,
                "quartet",
                "K-only",
                time_k,
            )
        )

        if verbose:
            print(f"  J+K: {time_jk:.2f} ms")
            print(f"  J-only: {time_j:.2f} ms")
            print(f"  K-only: {time_k:.2f} ms")

        # ===== Pair-based algorithm =====
        if verbose:
            print("\n--- Pair-based algorithm ---")

        # Generate kernels
        pair_jk = generate_pair_jk(layout, **cutoffs, pair_wide_vk=64)
        from jqc.pyscf.jk_pair import generate_get_j as generate_pair_j
        from jqc.pyscf.jk_pair import generate_get_k as generate_pair_k

        pair_j = generate_pair_j(layout, **cutoffs)
        pair_k = generate_pair_k(layout, **cutoffs, pair_wide_vk=64)

        # Benchmark J+K
        time_jk, (vj_pair, vk_pair) = time_kernel(
            pair_jk, (mol, dm), {"hermi": 1}, n_warmup=n_warmup, n_runs=n_runs
        )

        # Compute errors relative to quartet
        vj_error = float(cp.abs(vj_pair - vj_quartet).max())
        vk_error = float(cp.abs(vk_pair - vk_quartet).max())

        results.append(
            BenchmarkResult(
                molecule_name,
                basis_name,
                precision,
                natoms,
                nbasis,
                nao,
                "pair",
                "J+K",
                time_jk,
                vj_error,
                vk_error,
            )
        )

        # Benchmark J-only
        time_j, vj_pair_only = time_kernel(
            pair_j, (mol, dm), {"hermi": 1}, n_warmup=n_warmup, n_runs=n_runs
        )
        vj_error_only = float(cp.abs(vj_pair_only - vj_quartet_only).max())
        results.append(
            BenchmarkResult(
                molecule_name,
                basis_name,
                precision,
                natoms,
                nbasis,
                nao,
                "pair",
                "J-only",
                time_j,
                vj_error_only,
                None,
            )
        )

        # Benchmark K-only
        time_k, vk_pair_only = time_kernel(
            pair_k, (mol, dm), {"hermi": 1}, n_warmup=n_warmup, n_runs=n_runs
        )
        vk_error_only = float(cp.abs(vk_pair_only - vk_quartet_only).max())
        results.append(
            BenchmarkResult(
                molecule_name,
                basis_name,
                precision,
                natoms,
                nbasis,
                nao,
                "pair",
                "K-only",
                time_k,
                None,
                vk_error_only,
            )
        )

        if verbose:
            print(f"  J+K: {time_jk:.2f} ms (vj_err: {vj_error:.2e}, vk_err: {vk_error:.2e})")
            print(f"  J-only: {time_j:.2f} ms (vj_err: {vj_error_only:.2e})")
            print(f"  K-only: {time_k:.2f} ms (vk_err: {vk_error_only:.2e})")

        # Verify accuracy
        tolerance = 1e-8 if precision == "fp64" else 1e-4
        if vj_error > tolerance or vk_error > tolerance:
            print(
                f"WARNING: Accuracy check failed for {molecule_name}/{basis_name}/{precision}"
            )
            print(f"  vj_error: {vj_error:.2e} (tolerance: {tolerance:.2e})")
            print(f"  vk_error: {vk_error:.2e} (tolerance: {tolerance:.2e})")

    except Exception as e:
        print(f"ERROR in {molecule_name}/{basis_name}/{precision}: {e}")
        import traceback

        traceback.print_exc()
        return []

    return results


def print_summary_table(all_results):
    """Print summary table of all results."""
    print("\n" + "=" * 120)
    print("BENCHMARK SUMMARY")
    print("=" * 120)

    # Group by configuration
    configs = {}
    for r in all_results:
        key = (r.molecule_name, r.basis_name, r.precision, r.mode)
        if key not in configs:
            configs[key] = {"quartet": None, "pair": None}
        configs[key][r.algorithm] = r

    # Print header
    print(
        f"{'Molecule':<10} {'Basis':<12} {'Prec':<6} {'Mode':<8} "
        f"{'Quartet(ms)':<12} {'Pair(ms)':<12} {'Speedup':<10} "
        f"{'VJ Error':<12} {'VK Error':<12}"
    )
    print("-" * 120)

    # Print rows
    for key in sorted(configs.keys()):
        mol, basis, prec, mode = key
        quartet = configs[key]["quartet"]
        pair = configs[key]["pair"]

        if quartet and pair:
            speedup = quartet.time_ms / pair.time_ms
            vj_err = f"{pair.vj_error:.2e}" if pair.vj_error is not None else "N/A"
            vk_err = f"{pair.vk_error:.2e}" if pair.vk_error is not None else "N/A"

            print(
                f"{mol:<10} {basis:<12} {prec:<6} {mode:<8} "
                f"{quartet.time_ms:>11.2f} {pair.time_ms:>11.2f} "
                f"{speedup:>9.2f}x {vj_err:>12} {vk_err:>12}"
            )

    print("=" * 120)


def save_results_csv(all_results, output_path):
    """Save results to CSV file."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Molecule",
                "Basis",
                "Precision",
                "NAtoms",
                "NBasis",
                "NAO",
                "Algorithm",
                "Mode",
                "Time(ms)",
                "VJ_Error",
                "VK_Error",
            ]
        )

        for r in all_results:
            writer.writerow(
                [
                    r.molecule_name,
                    r.basis_name,
                    r.precision,
                    r.natoms,
                    r.nbasis,
                    r.nao,
                    r.algorithm,
                    r.mode,
                    f"{r.time_ms:.4f}",
                    f"{r.vj_error:.2e}" if r.vj_error is not None else "",
                    f"{r.vk_error:.2e}" if r.vk_error is not None else "",
                ]
            )

    print(f"\nResults saved to: {output_path}")


def plot_results(all_results, output_prefix="benchmark_jk"):
    """Generate plots comparing algorithms."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    # Extract J+K results
    jk_results = [r for r in all_results if r.mode == "J+K"]

    # Group by precision and mode
    for precision in ["fp64", "fp32", "mixed"]:
        prec_results = [r for r in jk_results if r.precision == precision]
        if not prec_results:
            continue

        # Organize data
        molecules = sorted(set(r.molecule_name for r in prec_results))
        basis_sets = sorted(set(r.basis_name for r in prec_results))

        _, axes = plt.subplots(1, len(basis_sets), figsize=(6 * len(basis_sets), 5))
        if len(basis_sets) == 1:
            axes = [axes]

        for ax, basis in zip(axes, basis_sets):
            quartet_times = []
            pair_times = []
            labels = []

            for mol in molecules:
                quartet = next(
                    (
                        r
                        for r in prec_results
                        if r.molecule_name == mol
                        and r.basis_name == basis
                        and r.algorithm == "quartet"
                    ),
                    None,
                )
                pair = next(
                    (
                        r
                        for r in prec_results
                        if r.molecule_name == mol
                        and r.basis_name == basis
                        and r.algorithm == "pair"
                    ),
                    None,
                )

                if quartet and pair:
                    quartet_times.append(quartet.time_ms)
                    pair_times.append(pair.time_ms)
                    labels.append(f"{mol}\n({quartet.natoms} atoms)")

            if quartet_times:
                x = np.arange(len(labels))
                width = 0.35

                ax.bar(x - width / 2, quartet_times, width, label="Quartet", alpha=0.8)
                ax.bar(x + width / 2, pair_times, width, label="Pair", alpha=0.8)

                ax.set_xlabel("Molecule")
                ax.set_ylabel("Time (ms)")
                ax.set_title(f"{basis.upper()} / {precision.upper()}")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = f"{output_prefix}_{precision}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark quartet vs pair-based JK algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--molecules",
        nargs="+",
        choices=list(MOLECULE_CONFIGS.keys()) + ["all"],
        default=["tiny", "small", "medium"],
        help="Molecules to test (default: tiny small medium)",
    )

    parser.add_argument(
        "--basis",
        nargs="+",
        choices=list(BASIS_SETS.keys()) + ["all"],
        default=["sto-3g", "6-31g"],
        help="Basis sets to test (default: sto-3g 6-31g)",
    )

    parser.add_argument(
        "--precision",
        nargs="+",
        choices=list(PRECISION_CONFIGS.keys()) + ["all"],
        default=["fp64"],
        help="Precision modes to test (default: fp64)",
    )

    parser.add_argument(
        "--warmup", type=int, default=3, help="Number of warmup runs (default: 3)"
    )

    parser.add_argument(
        "--runs", type=int, default=5, help="Number of timing runs (default: 5)"
    )

    parser.add_argument(
        "--output", type=str, help="Output CSV file path (optional)"
    )

    parser.add_argument(
        "--plot", action="store_true", help="Generate comparison plots"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    args = parser.parse_args()

    # Expand "all" selections
    molecules = (
        list(MOLECULE_CONFIGS.keys()) if "all" in args.molecules else args.molecules
    )
    basis_sets = list(BASIS_SETS.keys()) if "all" in args.basis else args.basis
    precisions = (
        list(PRECISION_CONFIGS.keys()) if "all" in args.precision else args.precision
    )

    print("=" * 120)
    print("JK ALGORITHM BENCHMARK: Quartet vs Pair-based")
    print("=" * 120)
    print(f"Molecules: {', '.join(molecules)}")
    print(f"Basis sets: {', '.join(basis_sets)}")
    print(f"Precision modes: {', '.join(precisions)}")
    print(f"Warmup runs: {args.warmup}, Timing runs: {args.runs}")
    print("=" * 120)

    # Run all benchmarks
    all_results = []
    total_configs = len(molecules) * len(basis_sets) * len(precisions)
    current = 0

    for mol in molecules:
        for basis in basis_sets:
            for prec in precisions:
                current += 1
                print(f"\n[{current}/{total_configs}] Testing {mol}/{basis}/{prec}...")

                results = benchmark_configuration(
                    mol, basis, prec, n_warmup=args.warmup, n_runs=args.runs, verbose=args.verbose
                )
                all_results.extend(results)

    # Print summary
    if all_results:
        print_summary_table(all_results)

        # Save to CSV if requested
        if args.output:
            save_results_csv(all_results, args.output)

        # Generate plots if requested
        if args.plot:
            plot_results(all_results)

    else:
        print("\nNo results collected!")
        return 1

    print("\nBenchmark completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
