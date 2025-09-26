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
1qnt algorithm for JK calculations
"""

import warnings

import cupy as cp
import numpy as np

from jqc.backend.cuda_scripts import (
    jk_1qnt_cuda_code,
    rys_roots_data,
    rys_roots_parallel_code,
)
from jqc.backend.util import generate_lookup_table
from jqc.constants import COORD_STRIDE, MAX_SMEM, NPRIM_MAX, PRIM_STRIDE

__all__ = ["gen_kernel"]

# grab shared memory size of the device
dev_id = cp.cuda.runtime.getDevice()
props = cp.cuda.runtime.getDeviceProperties(dev_id)
shm_size = props["sharedMemPerBlock"]
compile_options = ("-std=c++17", "--use_fast_math", "--minimal")


def padded_stride(n):
    """Pad the leading dimension to avoid bank conflict in shared memory.

    Modern GPUs have 32 memory banks in shared memory. Bank conflicts occur
    when multiple threads in a warp access the same bank simultaneously.
    This function adds minimal padding to avoid common conflict patterns.

    Args:
        n (int): Original stride size

    Returns:
        int: Padded stride that avoids bank conflicts
    """
    if n <= 0:
        return 1  # Handle edge case

    # Check for problematic strides that cause bank conflicts
    # Most critical: multiples of 32, 16, 8, 4, 2
    if n % 32 == 0:
        return n + 1
    elif n % 16 == 0:
        return n + 1
    elif n % 8 == 0 and n >= 8:
        return n + 1
    else:
        return n


def create_scheme(
    ang,
    nprim=None,
    frags=None,
    do_j=True,
    do_k=True,
    max_shared_memory=MAX_SMEM,
    max_gout=128,
    max_threads=256,
    dtype=np.double,
):
    """
    Create a scheme for 1QnT kernel.

    Args:
        ang (tuple): Angular momentum of the kernel.
        nprim (tuple): Number of primitives of the kernel.
        frags (np.ndarray): Fragments for the kernel. If not given,
            fragments will be automatically generated.
        do_j (bool): Whether to compute J matrix.
        do_k (bool): Whether to compute K matrix.
        max_shared_memory (int): Maximum shared memory size of the device.
        max_gout (int): Maximum number of output elements.
        max_threads (int): Maximum number of threads per block.
        dtype (np.dtype): Data type of the kernel.
    """
    li, lj, lk, ll = ang
    nroots = (li + lj + lk + ll) // 2 + 1
    nf = np.empty(4, dtype=np.int32)
    nf[0] = (li + 1) * (li + 2) // 2
    nf[1] = (lj + 1) * (lj + 2) // 2
    nf[2] = (lk + 1) * (lk + 2) // 2
    nf[3] = (ll + 1) * (ll + 2) // 2

    if frags is None:
        frags = np.ones(4, dtype=np.int32)
        frags1 = frags.copy()
        while np.prod(frags1) < max_gout:
            frags = frags1.copy()
            nthreads = (nf + frags - 1) // frags
            # If fully covered
            if np.all(nthreads == 1):
                break

            idx = np.argmax(nthreads)
            frags1[idx] += 1

        while np.prod(nthreads) > max_threads:
            idx = np.argmax(nthreads)
            frags[idx] += 1
            nthreads = (nf + frags - 1) // frags

    g_size = (li + 1) * (lj + 1) * (lk + 1) * (ll + 1)
    dtype_size = np.dtype(dtype).itemsize
    smem_per_quartet = g_size * 3 + nroots * 2

    nthreads = (nf + frags - 1) // frags
    nti, ntj, ntk, ntl = nthreads
    nfi, nfj, nfk, nfl = nf

    # for J matrix
    if do_j and nti * ntj > 1:
        smem_per_quartet = max(smem_per_quartet, nti * ntj * nfk * nfl)
    if do_j and ntk * ntl > 1:
        smem_per_quartet = max(smem_per_quartet, ntk * ntl * nfi * nfj)

    # for K matrix
    if do_k and nti * ntk > 1:
        smem_per_quartet = max(smem_per_quartet, nti * ntk * nfj * nfl)
    if do_k and ntj * ntl > 1:
        smem_per_quartet = max(smem_per_quartet, ntj * ntl * nfi * nfk)
    if do_k and nti * ntl > 1:
        smem_per_quartet = max(smem_per_quartet, nti * ntl * nfj * nfk)
    if do_k and ntj * ntk > 1:
        smem_per_quartet = max(smem_per_quartet, ntj * ntk * nfi * nfl)

    # Calculate initial parameters
    dtype_size = np.dtype(dtype).itemsize
    nt = int(np.prod(nthreads))
    nthreads_per_sq = 1 << (nt - 1).bit_length()
    nsq_per_block = max_threads // nthreads_per_sq

    # Account for static shared memory: indices
    static_sm = nfk * nfl * 3 * 4  # for indices
    max_dynamic_sm = max_shared_memory - static_sm

    # If shared memory is not enough, decrease # of quartets in each block
    smem_stride = padded_stride(nsq_per_block)  # reduce bank conflict
    while smem_per_quartet * smem_stride * dtype_size > max_dynamic_sm:
        nsq_per_block >>= 1
        smem_stride = padded_stride(nsq_per_block)
        if nsq_per_block == 0:
            raise RuntimeError("Shared memory is not enough")

    # Recalculate nthreads_per_sq based on final nsq_per_block
    nthreads_per_sq = max_threads // nsq_per_block

    total_shared_memory = smem_per_quartet * smem_stride * dtype_size
    assert nsq_per_block > 0
    return nsq_per_block, nthreads_per_sq, frags, total_shared_memory


def gen_kernel(
    ang,
    nprim,
    frags=None,
    dtype=np.double,
    n_dm=1,
    do_j=True,
    do_k=True,
    omega=None,
    print_log=False,
    use_cache=True,
    max_shm=shm_size,
    force_cache_mode=None,
):
    """
    Generate a 1QNT kernel.

    Args:
        ang (tuple): Angular momentum of the kernel.
        nprim (tuple): Number of primitives of the kernel.
        frags (np.ndarray): Fragments of the kernel.
        dtype (np.dtype): Data type of the kernel.
        n_dm (int): Number of density matrices.
        do_j (bool): Whether to compute J matrix.
        do_k (bool): Whether to compute K matrix.
        omega (float): Angular momentum of the kernel.
        print_log (bool): Whether to print log.
        use_cache (bool): Whether to use cache.
        max_shm (int): Maximum shared memory size of the device.
        force_cache_mode (bool): Force caching on/off for benchmarking. None uses default logic.
    """
    if dtype == np.float64:
        dtype_cuda = "double"
    elif dtype == np.float32:
        dtype_cuda = "float"
    else:
        raise RuntimeError("Data type is not supported")

    if omega is not None and omega < 0:
        raise RuntimeError("Omega value is not supported yet")
    rys_type = 0 if omega is None or omega == 0 else 1

    li, lj, lk, ll = ang
    npi, npj, npk, npl = nprim
    nroots = (li + lj + lk + ll) // 2 + 1
    scheme = create_scheme(ang, nprim, frags, dtype=dtype, max_shared_memory=max_shm)
    nsq_per_block, nthreads_per_sq, frags, dynamic_shared_memory = scheme

    # x,y,z in parallel
    assert nthreads_per_sq >= 3

    fragi, fragj, fragk, fragl = frags
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nfk = (lk + 1) * (lk + 2) // 2
    nfl = (ll + 1) * (ll + 2) // 2
    if nfi % fragi != 0:
        raise RuntimeError(f"Invalid tile size {frags} for {ang}")
    if nfj % fragj != 0:
        raise RuntimeError(f"Invalid tile size {frags} for {ang}")
    if nfk % fragk != 0:
        raise RuntimeError(f"Invalid tile size {frags} for {ang}")
    if nfl % fragl != 0:
        raise RuntimeError(f"Invalid tile size {frags} for {ang}")
    const = f"""
typedef unsigned int uint32_t;
using DataType = {dtype_cuda};
constexpr int li = {li};
constexpr int lj = {lj};
constexpr int lk = {lk};
constexpr int ll = {ll};
constexpr int npi = {npi};
constexpr int npj = {npj};
constexpr int npk = {npk};
constexpr int npl = {npl};
constexpr int fragi = {fragi};
constexpr int fragj = {fragj};
constexpr int fragk = {fragk};
constexpr int fragl = {fragl};
constexpr int n_dm = {n_dm};
constexpr int rys_type = {rys_type};   // 0: omega = 0.0; -1: omega < 0.0; 1 omega > 0.0;
constexpr int nsq_per_block = {nsq_per_block};
constexpr int nthreads_per_sq = {nthreads_per_sq};
constexpr int threads  =  nsq_per_block * nthreads_per_sq;
constexpr int do_j = {int(do_j)};
constexpr int do_k = {int(do_k)};
constexpr int smem_stride = {padded_stride(nsq_per_block)};
// Inject constants to match host-side layout
#define NPRIM_MAX {NPRIM_MAX}
// PRIM_STRIDE here matches host scalar stride; device uses prim_stride = PRIM_STRIDE/2
#define PRIM_STRIDE {PRIM_STRIDE}
#define COORD_STRIDE {COORD_STRIDE}
// for rys_roots
constexpr int nroots = ((li+lj+lk+ll)/2+1);
"""
    # Add caching override if specified
    if force_cache_mode is not None:
        const += f"#define JQC_CACHE_OVERRIDE {int(force_cache_mode)}\n"
    idx_script = generate_lookup_table(li, lj, lk, ll)
    script = (
        const
        + rys_roots_data[nroots]
        + rys_roots_parallel_code
        + idx_script
        + jk_1qnt_cuda_code
    )

    if not use_cache:
        # Generate a random number to change the cuda code,
        # such that the compiler will recompile the code
        import random

        x = random.random()
        script += f" \n#define RANDOM_NUMBER {x}"

    mod = cp.RawModule(code=script, options=compile_options)
    kernel = mod.get_function("rys_jk")
    if kernel.local_size_bytes > 8192:
        msg = f"Local memory usage is high in 1qnt: {kernel.local_size_bytes} Bytes,"
        msg += f"    ang = {ang}, nprim = {nprim}, frags = {frags}, dtype = {dtype}, n_dm = {n_dm}"
        warnings.warn(msg)
    kernel.max_dynamic_shared_size_bytes = dynamic_shared_memory
    if print_log:
        print(
            f"Type: ({li}{lj}|{lk}{ll}), \
primitives: ({npi}{npj}|{npk}{npl}), \
algorithm: 1qnt, \
threads: ({nsq_per_block:3d}, {nthreads_per_sq:3d}), \
frags: ({fragi:2d}, {fragj:2d}, {fragk:2d}, {fragl:2d}), \
shared memory: {dynamic_shared_memory/1024:5.2f} KB, \
registers: {kernel.num_regs:3d}, \
local memory: {kernel.local_size_bytes:4d} Bytes"
        )

    def fun(*args):
        ntasks = args[-1]  # the last argument is ntasks
        blocks = (ntasks + nsq_per_block - 1) // nsq_per_block

        kernel(
            (blocks,),
            (nsq_per_block, nthreads_per_sq),
            args,
            shared_mem=dynamic_shared_memory,
        )

    return script, kernel, fun
