#!/usr/bin/env python3
"""
Benchmark script for DFT calculations using jqc applied to GPU4PySCF
with the wb97m-v functional. Mirrors benchmark_wb97mv_molecules.py but
applies jqc to the mean-field object before running.

Saves timing results to a JSON file for comparison with GPU4PySCF.
"""

import json
import os
import time
from datetime import datetime

import cupy as cp
import pyscf
from pyscf import lib
from gpu4pyscf import dft
import gpu4pyscf
import jqc.pyscf as jqc_pyscf

# Configuration
BASIS = "def2-tzvpd"
XC_FUNCTIONAL = "wb97m-v"
GRIDS = (99, 590)
USE_MIXED_PRECISION = True
# Mixed-precision thresholds for JK and grid kernels
# Tasks with contribution > cutoff_fp64 run in FP64; between (cutoff_fp32, cutoff_fp64] run in FP32
# Keep cutoff_fp32 tight for accuracy; tune cutoff_fp64 for speed/accuracy tradeoff
CUTOFF_FP32 = 1e-13
CUTOFF_FP64 = 1e-6
VERBOSE = 4

# Molecules to benchmark (same as GPU4PySCF benchmark)
MOLECULES = [
    "0029-elongated-halogenated.xyz",
    "0051-elongated-halogenated.xyz",
    "0084-elongated-halogenated.xyz",
    "0112-elongated-nitrogenous.xyz",
    "0152-elongated-nitrogenous.xyz",
    # "0184-globular-halogenated.xyz",
    # "0224-globular-nitrogenous.xyz",
    # "0357-globular-sulfurous.xyz",
    # "0425-globular-nitrogenous.xyz",
]


def get_gpu4pyscf_version():
    try:
        return gpu4pyscf.__version__
    except AttributeError:
        return "unknown"


def benchmark_molecule(mol_file, warmup=True):
    """Benchmark a single molecule with jqc-applied GPU4PySCF.

    Args:
        mol_file: Filename under benchmarks/molecules/.
        warmup: Whether to run an initial warmup calculation.

    Returns:
        dict with timing and energy information, or None if skipped.
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking (jqc): {os.path.basename(mol_file)}")
    print(f"{'=' * 60}")

    mol_path = os.path.join("molecules", mol_file)
    if not os.path.exists(mol_path):
        print(f"Warning: Molecule file {mol_path} not found!")
        return None

    mol = pyscf.M(atom=mol_path, basis=BASIS, verbose=VERBOSE)

    # Skip odd electron systems for RKS
    if mol.nelectron % 2 == 1:
        print(f"Skipping {mol_file}: odd electron system (RKS only)")
        return None

    print(f"Molecule atoms: {[mol.atom_symbol(i) for i in range(mol.natm)]}")
    print(f"Number of electrons: {mol.nelectron}")
    print(f"Number of basis functions: {mol.nao}")

    # Set up DFT and apply jqc
    mf = dft.RKS(mol, xc=XC_FUNCTIONAL)
    mf.grids.atom_grid = GRIDS
    mf.nlcgrids.atom_grid = (50, 194)  # NLC grids for wb97m-v
    mf.verbose = VERBOSE
    if USE_MIXED_PRECISION:
        print(
            f"Applying JQC with mixed precision: cutoff_fp32={CUTOFF_FP32}, cutoff_fp64={CUTOFF_FP64}"
        )
        mf = jqc_pyscf.apply(mf, cutoff_fp32=CUTOFF_FP32, cutoff_fp64=CUTOFF_FP64)
    else:
        mf = jqc_pyscf.apply(mf)

    # Warmup run
    if warmup:
        print("Running warmup calculation (jqc applied)...")
        e_warmup = mf.kernel()
        print(f"Warmup energy: {e_warmup}")

        # Clear GPU memory
        cp.get_default_memory_pool().free_all_blocks()

        # Recreate for clean timing
        mf = dft.RKS(mol, xc=XC_FUNCTIONAL)
        mf.grids.atom_grid = GRIDS
        mf.nlcgrids.atom_grid = (50, 194)
        mf.verbose = VERBOSE
        if USE_MIXED_PRECISION:
            mf = jqc_pyscf.apply(mf, cutoff_fp32=CUTOFF_FP32, cutoff_fp64=CUTOFF_FP64)
        else:
            mf = jqc_pyscf.apply(mf)

    # Timed calculation
    print("Running timed calculation (jqc applied)...")
    start_time = time.time()

    # CUDA events for GPU timing
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    start_event.record()
    energy = mf.kernel()
    end_event.record()
    end_event.synchronize()

    wall_time = time.time() - start_time
    gpu_time_ms = cp.cuda.get_elapsed_time(start_event, end_event)

    print(f"Final energy: {energy}")
    print(f"Wall time: {wall_time:.3f} s")
    print(f"GPU time: {gpu_time_ms:.1f} ms")

    # Clean up
    mf = None
    cp.get_default_memory_pool().free_all_blocks()

    return {
        "molecule": os.path.basename(mol_file),
        "n_electrons": mol.nelectron,
        "n_basis_functions": mol.nao,
        "energy": float(energy),
        "wall_time_s": wall_time,
        "gpu_time_ms": float(gpu_time_ms),
        "success": True,
    }


def main():
    print("jqc wb97m-v Benchmark Suite (applied to GPU4PySCF)")
    print("=" * 50)

    gpu4pyscf_version = get_gpu4pyscf_version()
    print(f"GPU4PySCF version: {gpu4pyscf_version}")
    print(f"Basis set: {BASIS}")
    print(f"XC functional: {XC_FUNCTIONAL}")
    print(f"Grid points: {GRIDS}")
    print(f"Number of molecules: {len(MOLECULES)}")

    # Set number of OpenMP threads used by PySCF
    lib.num_threads(8)

    results = {
        "timestamp": datetime.now().isoformat(),
        "engine": "jqc",
        "gpu4pyscf_version": gpu4pyscf_version,
        "xc_functional": XC_FUNCTIONAL,
        "basis_set": BASIS,
        "grid_points": GRIDS,
        "molecules": [],
    }

    for i, mol_file in enumerate(MOLECULES):
        print(f"\nProgress: {i + 1}/{len(MOLECULES)}")
        # try:
        result = benchmark_molecule(mol_file)
        if result:
            results["molecules"].append(result)
        else:
            results["molecules"].append(
                {
                    "molecule": mol_file,
                    "success": False,
                    "error": "File not found or skipped",
                }
            )
        # except Exception as e:
        #    print(f"Error benchmarking {mol_file}: {str(e)}")
        #    results["molecules"].append(
        #        {"molecule": mol_file, "success": False, "error": str(e)}
        #    )

    # Save results to JSON
    output_file = (
        f"benchmark_wb97mv_{BASIS}_jqc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Benchmark Complete (jqc)!")
    print(f"Results saved to: {output_file}")

    successful = [r for r in results["molecules"] if r.get("success", False)]
    failed = [r for r in results["molecules"] if not r.get("success", False)]

    print(f"Successful benchmarks: {len(successful)}")
    print(f"Failed benchmarks: {len(failed)}")

    if successful:
        total_time = sum(r["wall_time_s"] for r in successful)
        avg_time = total_time / len(successful)
        print(f"Average wall time: {avg_time:.3f} s")
        print(f"Total benchmark time: {total_time:.3f} s")

    if failed:
        print("\nFailed molecules:")
        for f in failed:
            print(f"  - {f['molecule']}: {f.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
