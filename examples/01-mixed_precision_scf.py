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
Minimal mixed-precision SCF (RHF) example using JoltQC.

This script runs two SCF calculations on a small water molecule:
  1) FP64-baseline with default JoltQC settings
  2) Mixed-precision SCF with relaxed FP32/FP64 cutoffs

It prints the total energies and their difference.

Requirements:
- CUDA 12.x and a compatible NVIDIA driver
- gpu4pyscf-cuda12x, cupy-cuda12x

Run:
    python examples/01-mixed_precision_scf.py
"""

import pyscf
from gpu4pyscf import scf

import jqc.pyscf


def main() -> None:
    atom = """
    O       0.0000000000    -0.0000000000     0.1174000000
    H      -0.7570000000    -0.0000000000    -0.4696000000
    H       0.7570000000     0.0000000000    -0.4696000000
    """

    mol = pyscf.M(atom=atom, basis="def2-tzvpp")

    # FP64-baseline (default JoltQC configuration)
    mf_ref = scf.RHF(mol)
    # Avoid MINAO init path to keep the example robust across environments
    mf_ref.init_guess = "1e"
    mf_ref = jqc.pyscf.apply(mf_ref)
    e_ref = mf_ref.kernel()

    # Mixed-precision (disabled for this basis by using identical cutoffs)
    # Rationale: def2-tzvpp includes f-shells which are less stable in FP32.
    # For lighter bases (no f), try jk cutoff_fp64=1e-7 for speed.
    config = {
        "jk": {"cutoff_fp32": 1e-13, "cutoff_fp64": 1e-7},
        "dft": {"cutoff_fp32": 1e-13, "cutoff_fp64": 1e-7},
    }
    mf_mp = scf.RHF(mol)
    mf_mp.init_guess = "1e"
    mf_mp = jqc.pyscf.apply(mf_mp, config)
    e_mp = mf_mp.kernel()

    print("FP64 (baseline)      :", e_ref)
    print("Mixed-precision (SCF):", e_mp)
    print("Delta (MP - FP64)    :", e_mp - e_ref)


if __name__ == "__main__":
    main()
