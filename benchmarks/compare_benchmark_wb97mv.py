#!/usr/bin/env python3
"""
Compare benchmark results between GPU4PySCF baseline and jqc-applied runs.

Usage:
  python benchmarks/compare_benchmark_wb97mv.py \
      --baseline path/to/benchmark_wb97mv_<basis>_<timestamp>.json \
      --jqc      path/to/benchmark_wb97mv_<basis>_jqc_<timestamp>.json

Prints per-molecule energy differences and timing speedups as well as
aggregate statistics.
"""

import argparse
import json
import math
import os
from typing import Dict, Any, List, Tuple


def load_results(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def index_by_molecule(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in items:
        name = it.get("molecule") or it.get("name")
        if name is None:
            continue
        out[name] = it
    return out


def fmt_f(v, digits=6):
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "nan"


def compare(baseline: Dict[str, Any], jqc: Dict[str, Any]) -> None:
    b_meta = {
        "basis": baseline.get("basis_set"),
        "xc": baseline.get("xc_functional"),
        "grid": baseline.get("grid_points"),
    }
    j_meta = {
        "basis": jqc.get("basis_set"),
        "xc": jqc.get("xc_functional"),
        "grid": jqc.get("grid_points"),
    }

    print("Metadata")
    print(f"  Baseline: basis={b_meta['basis']} xc={b_meta['xc']} grid={b_meta['grid']}")
    print(f"  jqc     : basis={j_meta['basis']} xc={j_meta['xc']} grid={j_meta['grid']}")
    if b_meta != j_meta:
        print("  WARNING: Metadata differs between runs; comparisons may be invalid.")

    b_idx = index_by_molecule(baseline.get("molecules", []))
    j_idx = index_by_molecule(jqc.get("molecules", []))

    common = sorted(set(b_idx.keys()) & set(j_idx.keys()))
    if not common:
        print("No common molecules found between the two result files.")
        return

    print("\nPer-molecule comparison")
    print(
        "molecule,energy_baseline,energy_jqc,delta_hartree,wall_baseline_s,wall_jqc_s,"
        "speedup_wall,gpu_baseline_ms,gpu_jqc_ms,speedup_gpu"
    )

    wall_speedups: List[float] = []
    gpu_speedups: List[float] = []
    e_abs_diffs: List[float] = []

    for name in common:
        b = b_idx[name]
        j = j_idx[name]
        if not (b.get("success") and j.get("success")):
            continue

        eb = b.get("energy")
        ej = j.get("energy")
        wb = b.get("wall_time_s") or float("nan")
        wj = j.get("wall_time_s") or float("nan")
        gb = b.get("gpu_time_ms")
        gj = j.get("gpu_time_ms")

        de = None
        if eb is not None and ej is not None:
            de = ej - eb
            e_abs_diffs.append(abs(de))

        speed_wall = None
        if isinstance(wb, (int, float)) and isinstance(wj, (int, float)) and wj > 0:
            speed_wall = wb / wj
            if math.isfinite(speed_wall):
                wall_speedups.append(speed_wall)

        speed_gpu = None
        if (
            isinstance(gb, (int, float))
            and isinstance(gj, (int, float))
            and gj
            and gj > 0
        ):
            speed_gpu = gb / gj
            if math.isfinite(speed_gpu):
                gpu_speedups.append(speed_gpu)

        print(
            f"{name},{fmt_f(eb)},{fmt_f(ej)},{fmt_f(de)},{fmt_f(wb,3)},{fmt_f(wj,3)},"\
            f"{fmt_f(speed_wall,3)},{fmt_f(gb,1)},{fmt_f(gj,1)},{fmt_f(speed_gpu,3)}"
        )

    print("\nSummary")
    matched = len(common)
    b_ok = sum(1 for n in common if b_idx[n].get("success"))
    j_ok = sum(1 for n in common if j_idx[n].get("success"))
    print(f"  Common molecules: {matched}")
    print(f"  Baseline successes: {b_ok}")
    print(f"  jqc successes     : {j_ok}")

    if e_abs_diffs:
        mean_abs = sum(e_abs_diffs) / len(e_abs_diffs)
        max_abs = max(e_abs_diffs)
        print(f"  Mean |ΔE| (Ha): {mean_abs:.6e}")
        print(f"  Max  |ΔE| (Ha): {max_abs:.6e}")

    def gmean(xs: List[float]) -> float:
        if not xs:
            return float("nan")
        return math.exp(sum(math.log(x) for x in xs if x > 0) / len(xs))

    if wall_speedups:
        gm = gmean(wall_speedups)
        am = sum(wall_speedups) / len(wall_speedups)
        print(f"  Speedup (wall)  AM: {am:.3f}x  GM: {gm:.3f}x  n={len(wall_speedups)}")
    if gpu_speedups:
        gm = gmean(gpu_speedups)
        am = sum(gpu_speedups) / len(gpu_speedups)
        print(f"  Speedup (GPU)   AM: {am:.3f}x  GM: {gm:.3f}x  n={len(gpu_speedups)}")


def main():
    ap = argparse.ArgumentParser(description="Compare wb97m-v benchmark JSON files")
    ap.add_argument("--baseline", required=True, help="Path to GPU4PySCF JSON")
    ap.add_argument("--jqc", required=True, help="Path to jqc JSON")
    args = ap.parse_args()

    if not os.path.exists(args.baseline):
        raise FileNotFoundError(args.baseline)
    if not os.path.exists(args.jqc):
        raise FileNotFoundError(args.jqc)

    b = load_results(args.baseline)
    j = load_results(args.jqc)
    compare(b, j)


if __name__ == "__main__":
    main()

