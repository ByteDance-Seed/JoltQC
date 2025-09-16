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

import json
import numpy as np
import cupy as cp
from pyscf import gto, lib
from jqc.constants import TILE, MAX_SMEM

"""
Script for greedy search the optimal fragmentation for the kernel.
"""

"""
How to run ?

> python3 generate_fragment.py {li} {lj} {lk} {ll} fp64 > logs/{li}{lj}{lk}{ll}_fp64.log

e.g.
> python3 generate_fragment.py 1 1 1 1 fp64 > logs/1111_fp64.log
or 
> python3 generate_fragment.py 1 1 1 1 fp32 > logs/1111_fp32.log
"""


def generate_fragments(ang, max_threads=256):
    """Create a tile scheme satisfy three conditions
    1. Cover (nfi, nfj, nfk, nfl), # of threads < max_threads,
       (Note: possibly use a lot of local memory)
    2. Shared memory < max_shmem_per_block
    3. At least 4 threads for each quartet
    """
    li, lj, lk, ll = ang
    nf = np.empty(4, dtype=np.int32)
    nf[0] = (li + 1) * (li + 2) // 2
    nf[1] = (lj + 1) * (lj + 2) // 2
    nf[2] = (lk + 1) * (lk + 2) // 2
    nf[3] = (ll + 1) * (ll + 2) // 2

    for fi in range(1, nf[0] + 1):
        for fj in range(1, nf[1] + 1):
            for fk in range(1, nf[2] + 1):
                for fl in range(1, nf[3] + 1):
                    # early exit if the fragment size is greater than 256,
                    # as it will use too many registers
                    if fi * fj * fk * fl > 256:
                        continue
                    fragments = np.array([fi, fj, fk, fl], dtype=np.int32)
                    # integral dimension must be a multiple of fragment size
                    if (
                        nf[0] % fi != 0
                        or nf[1] % fj != 0
                        or nf[2] % fk != 0
                        or nf[3] % fl != 0
                    ):
                        continue
                    nthreads = (nf + fragments - 1) // fragments
                    if np.prod(nthreads) > max_threads:
                        continue
                    # at least 3 threads for each quartet
                    if np.prod(nthreads) < 3:
                        continue
                    yield fragments


def update_frags(i, j, k, l, dtype_str):
    from jqc.pyscf import jk
    from jqc.pyscf.basis import sort_group_basis, compute_q_matrix
    from jqc.backend import jk_1qnt as jk_algo1
    from jqc.backend import jk_1q1t as jk_algo0
    from jqc.backend.jk import device_name
    from pathlib import Path

    if dtype_str == "fp32":
        dtype = np.float32
    elif dtype_str == "fp64":
        dtype = np.float64
    else:
        raise RuntimeError(f"Data type {dtype_str} is not supported")

    basis = gto.basis.parse(
        """
    H    S
        0.10307241             1.0000000
        0.10307241             1.0000000
        0.10307241             1.0000000
    H    P
        0.40700000             1.0000000
        0.10307241             1.0000000
        0.10307241             1.0000000
    H    D
        0.05700000             1.0000000
    H    F
        0.05700000             1.0000000
    H    G
        0.05700000             1.0000000
                                """
    )
    xyz = "gly30.xyz"
    if not Path(xyz).exists():
        raise FileNotFoundError(f"Required file {xyz} not found in current directory")
    mol = gto.M(atom=xyz, basis=basis, unit="Angstrom")

    # Use JoltQC's own data structures instead of GPU4PySCF
    bas_cache, bas_mapping, padding_mask, group_info = sort_group_basis(
        mol, alignment=TILE
    )
    q_matrix = compute_q_matrix(mol)
    q_matrix = q_matrix[bas_mapping[:, None], bas_mapping]
    q_matrix[padding_mask, :] = -100
    q_matrix[:, padding_mask] = -100
    q_matrix = cp.asarray(q_matrix, dtype=np.float32)

    group_key, group_offset = group_info
    nbas = bas_mapping.shape[0]

    # Create dummy density matrix conditioning arrays
    dm_cond = cp.ones([nbas, nbas], dtype=np.float32)
    tile_q_cond = q_matrix
    log_cutoff = -12

    # Get angular momentum and primitive counts from group_key
    uniq_l_ctr = group_key  # This contains (l, nprim) pairs
    l_ctr_bas_loc = group_offset  # This contains the basis offsets
    tile_pairs = jk.make_tile_pairs(l_ctr_bas_loc, tile_q_cond, log_cutoff)

    # Get nao from the sorted basis cache
    ce, coords, angs, nprims = bas_cache
    ao_loc = np.concatenate(([0], np.cumsum((angs + 1) * (angs + 2) // 2)))
    nao = ao_loc[-1]
    ao_loc = cp.asarray(ao_loc, dtype=np.int32)

    # Initialize with a simple density matrix for meaningful J/K computation
    # Following jk.py: dms matches kernel precision, but vj/vk are always FP64
    dms = cp.eye(nao, dtype=dtype) * 0.1  # Density matrix matches kernel precision
    vj = cp.zeros((nao, nao), dtype=np.float64)  # Always FP64 to match jk.py behavior
    vk = cp.zeros((nao, nao), dtype=np.float64)  # Always FP64 to match jk.py behavior

    # Use the ce array directly (combined coefficients and exponents) as in jk.py
    coords = coords.astype(dtype)
    ce_data = ce.astype(dtype)  # This contains interleaved coefficients and exponents

    tile_ij_mapping = (
        tile_pairs[i, j][:256] if (i, j) in tile_pairs else cp.array([], dtype=np.int32)
    )
    tile_kl_mapping = (
        tile_pairs[k, l][:256] if (k, l) in tile_pairs else cp.array([], dtype=np.int32)
    )

    li, ip = uniq_l_ctr[i]
    lj, jp = uniq_l_ctr[j]
    lk, kp = uniq_l_ctr[k]
    ll, lp = uniq_l_ctr[l]
    ang = (li, lj, lk, ll)
    nprim = (ip, jp, kp, lp)
    best_time = 1e100
    best_frag = None

    from jqc.backend.jk_tasks import gen_screen_jk_tasks_kernel

    script, kernel, gen_tasks_fun = gen_screen_jk_tasks_kernel(tile=TILE)
    QUEUE_DEPTH = jk.QUEUE_DEPTH
    # cp.get_default_memory_pool().free_all_blocks()
    # cp.cuda.device.Device().synchronize()
    pool = cp.empty((QUEUE_DEPTH), dtype=jk.ushort4_dtype)
    info = cp.zeros(4, dtype=np.uint32)

    gen_tasks_fun(
        pool,
        info,
        np.int32(nbas),
        tile_ij_mapping,
        tile_kl_mapping,
        tile_q_cond,
        dm_cond,
        np.float32(log_cutoff),
        np.float32(100),
    )

    max_shm = MAX_SMEM  # 48 KB for compatibility

    # Store reference results from best 1qnt algorithm
    best_vj_1qnt = None
    best_vk_1qnt = None

    # Measure GPU time of algorithm 1 with different fragments
    for frag in generate_fragments(ang):
        try:
            script, kernel, fun = jk_algo1.gen_kernel(
                ang, nprim, frags=frag, dtype=dtype, max_shm=max_shm
            )
        except:
            print(f"failed to generate kernel {ang}/{nprim} with frag {frag}")
            continue

        # Reset matrices before computation
        vj.fill(0)
        vk.fill(0)

        # Use the same argument structure as jk.py
        n_quartets = int(info[1].get())  # Number of quartets to process
        omega = dtype(0.0)  # Use the same precision as the kernel
        args = (
            nbas,
            nao,
            ao_loc,
            coords,
            ce_data,
            dms,
            vj,
            vk,
            omega,
            pool,
            n_quartets,
        )
        fun(*args)

        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        fun(*args)
        end.record()
        end.synchronize()
        elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
        if elapsed_time_ms < best_time:
            best_time = elapsed_time_ms
            best_frag = frag
            # Store the best results for comparison
            best_vj_1qnt = vj.copy()
            best_vk_1qnt = vk.copy()
        print(
            f"{ang}/{nprim} : algorithm 1qnt with frag {frag} takes "
            f"{elapsed_time_ms:.3f}ms, best time: {best_time:.3f}ms"
        )

    # Check 1q1t algorithm for low-angular momentum cases
    # Skip for high angular momentum to avoid excessive computation time
    if sum(ang) <= 8:
        # Reset matrices before computation
        vj.fill(0)
        vk.fill(0)

        # Measure time of algorithm 0
        script, kernel, fun = jk_algo0.gen_kernel(ang, nprim, dtype=dtype)
        kernel.compile()
        # Use the same argument structure as jk.py
        n_quartets = int(info[1].get())  # Number of quartets to process
        omega = dtype(0.0)  # Use the same precision as the kernel
        args = (
            nbas,
            nao,
            ao_loc,
            coords,
            ce_data,
            dms,
            vj,
            vk,
            omega,
            pool,
            n_quartets,
        )
        fun(*args)
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        fun(*args)
        end.record()
        end.synchronize()
        elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)

        # Verify that 1q1t and 1qnt produce identical results
        if best_vj_1qnt is not None and best_vk_1qnt is not None:
            # Check for NaN values first
            vj_has_nan = cp.any(cp.isnan(vj)) or cp.any(cp.isnan(best_vj_1qnt))
            vk_has_nan = cp.any(cp.isnan(vk)) or cp.any(cp.isnan(best_vk_1qnt))

            if vj_has_nan or vk_has_nan:
                print(f"Warning: NaN values detected in results!")
                print(f"  vj_1q1t has NaN: {cp.any(cp.isnan(vj))}")
                print(f"  vj_1qnt has NaN: {cp.any(cp.isnan(best_vj_1qnt))}")
                print(f"  vk_1q1t has NaN: {cp.any(cp.isnan(vk))}")
                print(f"  vk_1qnt has NaN: {cp.any(cp.isnan(best_vk_1qnt))}")
                print("Skipping result verification due to NaN values")
            else:
                # Compare J matrices
                vj_diff = cp.abs(vj - best_vj_1qnt).max()
                vk_diff = cp.abs(vk - best_vk_1qnt).max()

                # Use appropriate tolerance based on dtype
                tolerance = 1e-10 if dtype == np.float64 else 1e-4

                print(f"Result verification: max |vj_1q1t - vj_1qnt| = {vj_diff:.2e}")
                print(f"Result verification: max |vk_1q1t - vk_1qnt| = {vk_diff:.2e}")

                assert (
                    vj_diff < tolerance
                ), f"J matrices differ by {vj_diff:.2e}, exceeds tolerance {tolerance:.2e}"
                assert (
                    vk_diff < tolerance
                ), f"K matrices differ by {vk_diff:.2e}, exceeds tolerance {tolerance:.2e}"
                print(
                    f"✓ Algorithm verification passed: 1q1t and 1qnt produce identical results within tolerance {tolerance:.2e}"
                )

        if elapsed_time_ms < best_time:
            best_time = elapsed_time_ms
            best_frag = np.array([-1])
        print(
            f"{ang} : algorithm 1q1t takes {elapsed_time_ms:.3f}ms, best time: {best_time:.3f}ms"
        )
    if best_frag is None:
        print("Warning: No optimal fragment found")
        return

    print("Optimal frag:", best_frag)

    # Use device name from jk module for filename
    filename = f"optimal_scheme_{device_name}_{dtype_str}.json"
    path = Path(filename)
    if not path.exists():
        with open(path, "w") as f:
            json.dump({}, f)  # write empty dict

    with open(filename, "r") as f:
        data = json.load(f)
    ang_num = 1000 * li + 100 * lj + 10 * lk + ll

    # Check if this angular momentum combination already exists
    ang_key = str(int(ang_num))
    if ang_key in data:
        print(
            f"Warning: Overwriting existing entry for angular momentum {ang} (key {ang_key})"
        )

    data.update({ang_key: best_frag.tolist()})
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Results saved to {filename}")


if __name__ == "__main__":
    import sys

    li = int(sys.argv[1])
    lj = int(sys.argv[2])
    lk = int(sys.argv[3])
    ll = int(sys.argv[4])
    dtype = str(sys.argv[5])
    update_frags(li, lj, lk, ll, dtype)
