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
Benchmark script for nr_nlc_vxc (VV10 non-local correlation) module
"""

import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf import dft
from jqc.pyscf import rks
from jqc.pyscf.basis import BasisLayout

# Configuration
test_molecules = [
    'molecules/h2o.xyz',
    'molecules/0031-irregular-nitrogenous.xyz',
    'molecules/0051-elongated-halogenated.xyz',
]

test_basis = ['def2-svp', 'def2-tzvpp']
xc_functionals = ['wb97m-v']  # VV10-containing functional
n_warmup = 1
n_benchmark = 3

def benchmark_molecule(atom, basis, xc, grids_level=1, nlcgrids_level=(50, 194)):
    """Benchmark nr_nlc_vxc for a single molecule configuration"""

    print(f"\nBenchmarking: {atom} with {basis} basis and {xc} functional")
    print("=" * 60)

    # Create molecule
    mol = pyscf.M(
        atom=atom,
        basis=basis,
        output=f"benchmark_{atom.split('/')[-1].split('.')[0]}_{basis}_{xc}.log",
        verbose=0
    )

    # Setup DFT calculation
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level = grids_level
    mf.nlcgrids.atom_grid = nlcgrids_level

    # Build grids
    from jqc.pyscf.rks import build_grids
    from types import MethodType
    mf.grids.build = MethodType(build_grids, mf.grids)
    mf.nlcgrids.build = MethodType(build_grids, mf.nlcgrids)

    grids = mf.grids
    nlcgrids = mf.nlcgrids
    grids.build(with_non0tab=False, sort_grids=True)
    nlcgrids.build(with_non0tab=False, sort_grids=True)

    nao = mol.nao
    ngrids = grids.coords.shape[0]
    nnlcgrids = nlcgrids.coords.shape[0]

    print(f"Molecule info: {mol.natm} atoms, {nao} AOs")
    print(f"Grid points: {ngrids} (XC), {nnlcgrids} (NLC)")

    # Create test density matrix
    dm = mf.get_init_guess()
    dm = cp.asarray(dm)

    ni = mf._numint

    # =====================================================
    # Benchmark GPU4PySCF reference implementation
    # =====================================================
    print("\n--- GPU4PySCF Reference ---")

    # Warmup
    for _ in range(n_warmup):
        try:
            n_ref, e_ref, v_ref = ni.nr_nlc_vxc(mol, nlcgrids, xc, dm)
        except Exception as e:
            print(f"GPU4PySCF warmup failed: {e}")
            return None

    # Benchmark
    times_ref = []
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    for _ in range(n_benchmark):
        start.record()
        n_ref, e_ref, v_ref = ni.nr_nlc_vxc(mol, nlcgrids, xc, dm)
        end.record()
        end.synchronize()
        times_ref.append(cp.cuda.get_elapsed_time(start, end))

    time_ref_avg = np.mean(times_ref)
    time_ref_std = np.std(times_ref)

    print(f"GPU4PySCF time: {time_ref_avg:.2f} ± {time_ref_std:.2f} ms")
    print(f"Results: nelec={float(n_ref):.6f}, exc={float(e_ref):.6f}")

    # =====================================================
    # Benchmark JQC implementations with different precisions
    # =====================================================

    basis_layout = BasisLayout.from_mol(mol, alignment=1)

    precision_configs = [
        ("FP64", {"cutoff_fp64": 1e-13, "cutoff_fp32": 1e-13}),
        ("FP32", {"cutoff_fp64": 1e10, "cutoff_fp32": 1e-13}),
        ("Mixed", {"cutoff_fp64": 1e-7, "cutoff_fp32": 1e-13}),
    ]

    results = {}

    for precision_name, precision_config in precision_configs:
        print(f"\n--- JQC {precision_name} ---")

        try:
            # Generate kernel
            nr_nlc_vxc_kernel = rks.generate_nr_nlc_vxc(basis_layout, **precision_config)

            # Warmup
            for _ in range(n_warmup):
                n_jqc, e_jqc, v_jqc = nr_nlc_vxc_kernel(ni, mol, nlcgrids, xc, dm)

            # Benchmark
            times_jqc = []
            for _ in range(n_benchmark):
                start.record()
                n_jqc, e_jqc, v_jqc = nr_nlc_vxc_kernel(ni, mol, nlcgrids, xc, dm)
                end.record()
                end.synchronize()
                times_jqc.append(cp.cuda.get_elapsed_time(start, end))

            time_jqc_avg = np.mean(times_jqc)
            time_jqc_std = np.std(times_jqc)
            speedup = time_ref_avg / time_jqc_avg

            # Accuracy check
            n_diff = float(cp.abs(n_ref - n_jqc))
            e_diff = float(cp.abs(e_ref - e_jqc))
            v_diff = float(cp.linalg.norm(v_ref - v_jqc))

            print(f"JQC time: {time_jqc_avg:.2f} ± {time_jqc_std:.2f} ms")
            print(f"Speedup: {speedup:.2f}x")
            print(f"Results: nelec={float(n_jqc):.6f}, exc={float(e_jqc):.6f}")
            print(f"Accuracy: Δnelec={n_diff:.2e}, Δexc={e_diff:.2e}, Δvmat={v_diff:.2e}")

            results[precision_name] = {
                'time_avg': time_jqc_avg,
                'time_std': time_jqc_std,
                'speedup': speedup,
                'accuracy': {'n_diff': n_diff, 'e_diff': e_diff, 'v_diff': v_diff}
            }

        except Exception as e:
            print(f"JQC {precision_name} failed: {e}")
            results[precision_name] = None

    # Clean up GPU memory
    cp.get_default_memory_pool().free_all_blocks()

    return {
        'molecule': atom,
        'basis': basis,
        'functional': xc,
        'nao': nao,
        'ngrids': ngrids,
        'nnlcgrids': nnlcgrids,
        'reference_time': time_ref_avg,
        'results': results
    }

def main():
    """Run comprehensive benchmark suite"""
    print("JoltQC nr_nlc_vxc Benchmark Suite")
    print("==================================")

    all_results = []

    # Test different molecule/basis combinations
    import os
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))

    test_configs = [
        (os.path.join(benchmark_dir, 'molecules/h2o.xyz'), 'def2-svp'),
    ]

    # Add larger molecules if they exist
    try:
        for molecule in test_molecules[1:]:
            full_path = os.path.join(benchmark_dir, molecule)
            if os.path.exists(full_path):
                test_configs.append((full_path, 'def2-svp'))
                if molecule == 'molecules/0031-irregular-nitrogenous.xyz':
                    test_configs.append((full_path, 'def2-tzvpp'))
    except:
        pass

    for atom, basis in test_configs:
        for xc in xc_functionals:
            try:
                result = benchmark_molecule(atom, basis, xc)
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"Failed to benchmark {atom} with {basis}/{xc}: {e}")
                continue

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    for result in all_results:
        print(f"\n{result['molecule']} ({result['basis']}, {result['functional']})")
        print(f"  Size: {result['nao']} AOs, {result['nnlcgrids']} NLC grid points")
        print(f"  Reference: {result['reference_time']:.2f} ms")

        for precision, data in result['results'].items():
            if data:
                print(f"  {precision}: {data['time_avg']:.2f} ms ({data['speedup']:.2f}x speedup)")
            else:
                print(f"  {precision}: FAILED")

if __name__ == "__main__":
    main()