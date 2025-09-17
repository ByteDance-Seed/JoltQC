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
Cartesian to spherical, and spherical to cartesian basis transformations.
"""

import cupy as cp
import numpy as np

from jqc.backend.cuda_scripts import code_path

__all__ = ["cart2sph", "sph2cart"]

compile_options = ("-std=c++17", "--use_fast_math", "--minimal")

with open(f"{code_path}/cuda/cart2sph.cu") as f:
    cart2sph_scripts = f.read()

with open(f"{code_path}/cuda/sph2cart.cu") as f:
    sph2cart_scripts = f.read()

_cart2sph_kernel_cache = {}
_sph2cart_kernel_cache = {}


def cart2sph(dm_cart, angs, cart_offset, sph_offset, nao_sph, out=None):
    """
    Fused kernel for cartesian to spherical transformation and sorting

    Args:
        dm_cart: The cartesian density matrix.
        angs: The angular momentum of each basis.
        cart_offset: The offset of each basis in the cartesian density matrix.
        sph_offset: The offset of each basis in the spherical density matrix.

    Returns:
        The spherical density matrix.
    """
    assert dm_cart.flags["C_CONTIGUOUS"]
    is_2d = dm_cart.ndim == 2
    if is_2d:
        dm_cart = dm_cart[None]
    ndms = dm_cart.shape[0]
    cart_offset = cp.asarray(cart_offset, dtype=cp.int32)
    sph_offset = cp.asarray(sph_offset, dtype=cp.int32)
    buf = cp.zeros_like(dm_cart, order="C")
    nao_cart = dm_cart.shape[-1]
    diff = angs[1:] != angs[:-1]
    offsets = np.concatenate(([0], np.nonzero(diff)[0] + 1, [angs.size]))
    # Initialize output array before any kernel calls
    if out is None:
        dm_sph = cp.zeros((ndms, nao_sph, nao_sph), order="C")
    else:
        dm_sph = out
        dm_sph.fill(0)

    threads = (16, 16)

    # cart2sph for rows
    cart_ao_stride = nao_cart
    sph_ao_stride = nao_sph
    shell_stride = 1
    for p0, p1 in zip(offsets[:-1], offsets[1:]):
        nbatch = p1 - p0
        ang = angs[p0]
        if ang not in _cart2sph_kernel_cache:
            const = f"constexpr int ang = {ang};"
            scripts = const + cart2sph_scripts
            mod = cp.RawModule(code=scripts, options=compile_options)
            _cart2sph_kernel_cache[ang] = mod.get_function("cart2sph")
        kernel = _cart2sph_kernel_cache[ang]

        args = (
            dm_cart,
            buf,
            nao_cart,
            nbatch,
            cart_ao_stride,
            sph_ao_stride,
            shell_stride,
            cart_offset[p0:p1],
            sph_offset[p0:p1],
        )
        blocks = (
            (nao_cart + threads[0] - 1) // threads[0],
            (nbatch + threads[1] - 1) // threads[1],
        )
        kernel(blocks, threads, args)

    # cart2sph for cols
    cart_ao_stride = 1
    sph_ao_stride = 1
    shell_stride = nao_sph
    for p0, p1 in zip(offsets[:-1], offsets[1:]):
        nbatch = p1 - p0
        ang = angs[p0]
        if ang not in _cart2sph_kernel_cache:
            const = f"constexpr int ang = {ang};"
            scripts = const + cart2sph_scripts
            mod = cp.RawModule(code=scripts, options=compile_options)
            _cart2sph_kernel_cache[ang] = mod.get_function("cart2sph")
        kernel = _cart2sph_kernel_cache[ang]

        args = (
            buf,
            dm_sph,
            nao_sph,
            nbatch,
            cart_ao_stride,
            sph_ao_stride,
            shell_stride,
            cart_offset[p0:p1],
            sph_offset[p0:p1],
        )
        blocks = (
            (nao_cart + threads[0] - 1) // threads[0],
            (nbatch + threads[1] - 1) // threads[1],
        )
        kernel(blocks, threads, args)

    if is_2d:
        return dm_sph[0]
    return dm_sph


def sph2cart(dm_sph, angs, sph_offset, cart_offset, nao_cart, out=None):
    """
    Fused kernel for spherical to cartesian transformation and sorting

    Args:
        dm_sph: The spherical density matrix.
        angs: The angular momentum of each basis.
        cart_offset: The offset of each basis in the cartesian density matrix.
        sph_offset: The offset of each basis in the spherical density matrix.

    Returns:
        The cartesian density matrix.
    """
    assert dm_sph.flags["C_CONTIGUOUS"]
    is_2d = dm_sph.ndim == 2
    if is_2d:
        dm_sph = dm_sph[None]
    ndms = dm_sph.shape[0]
    nao_sph = dm_sph.shape[-1]

    cart_offset = cp.asarray(cart_offset, dtype=cp.int32)
    sph_offset = cp.asarray(sph_offset, dtype=cp.int32)
    buf = cp.zeros((ndms, nao_cart, nao_cart), order="C")

    # Initialize output array before any kernel calls
    if out is None:
        dm_cart = cp.zeros((ndms, nao_cart, nao_cart), order="C")
    else:
        dm_cart = out
        dm_cart.fill(0)

    diff = angs[1:] != angs[:-1]
    offsets = np.concatenate(([0], np.nonzero(diff)[0] + 1, [angs.size]))
    threads = (16, 16)

    # sph2cart for rows
    cart_ao_stride = nao_cart
    sph_ao_stride = nao_sph
    shell_stride = 1
    for p0, p1 in zip(offsets[:-1], offsets[1:]):
        nbatch = p1 - p0
        ang = angs[p0]
        if ang not in _sph2cart_kernel_cache:
            const = f"constexpr int ang = {ang};"
            scripts = const + sph2cart_scripts
            mod = cp.RawModule(code=scripts, options=compile_options)
            _sph2cart_kernel_cache[ang] = mod.get_function("sph2cart")
        kernel = _sph2cart_kernel_cache[ang]

        args = (
            buf,
            dm_sph,
            nao_sph,
            nbatch,
            cart_ao_stride,
            sph_ao_stride,
            shell_stride,
            cart_offset[p0:p1],
            sph_offset[p0:p1],
        )
        blocks = (
            (nao_sph + threads[0] - 1) // threads[0],
            (nbatch + threads[1] - 1) // threads[1],
        )
        kernel(blocks, threads, args)

    # sph2cart for cols
    cart_ao_stride = 1
    sph_ao_stride = 1
    shell_stride = nao_cart
    for p0, p1 in zip(offsets[:-1], offsets[1:]):
        nbatch = p1 - p0
        ang = angs[p0]
        if ang not in _sph2cart_kernel_cache:
            const = f"constexpr int ang = {ang};"
            scripts = const + sph2cart_scripts
            mod = cp.RawModule(code=scripts, options=compile_options)
            _sph2cart_kernel_cache[ang] = mod.get_function("sph2cart")
        kernel = _sph2cart_kernel_cache[ang]

        args = (
            dm_cart,
            buf,
            nao_cart,
            nbatch,
            cart_ao_stride,
            sph_ao_stride,
            shell_stride,
            cart_offset[p0:p1],
            sph_offset[p0:p1],
        )
        blocks = (
            (nao_cart + threads[0] - 1) // threads[0],
            (nbatch + threads[1] - 1) // threads[1],
        )
        kernel(blocks, threads, args)

    if is_2d:
        return dm_cart[0]
    return dm_cart


def cart2cart(dm_src, angs, src_offset, dst_offset, nao, out=None):
    """
    Reorder a cartesian density matrix between two basis orderings.
    Handles non one-to-one maps by accumulating contributions when
    destination indices repeat.

    Args:
        dm_src (cp.ndarray): Source cartesian density matrix (2D or 3D).
        angs (np.ndarray): Angular momentum per basis shell (1D).
        src_offset (array-like): AO start indices per shell in source order.
        dst_offset (array-like): AO start indices per shell in destination order.
        nao (int): Total number of AOs in destination order.
        out (cp.ndarray, optional): Destination buffer. If provided, it will be zeroed then accumulated into.

    Returns:
        cp.ndarray: Destination density matrix (3D of shape [ndms, nao, nao]).
    """
    # Ensure CuPy arrays where appropriate
    dm_src_cp = cp.asarray(dm_src)
    if dm_src_cp.ndim == 2:
        dm_src_cp = dm_src_cp[None]
    ndms = dm_src_cp.shape[0]

    # Prepare destination
    if out is None:
        dm_dst = cp.zeros((ndms, nao, nao), dtype=dm_src_cp.dtype)
    else:
        dm_dst = out
        dm_dst.fill(0)

    # Ensure offsets are NumPy arrays for CPU-side iteration
    if isinstance(src_offset, cp.ndarray):
        src_offset_np = src_offset.get()
    else:
        src_offset_np = np.asarray(src_offset)
    if isinstance(dst_offset, cp.ndarray):
        dst_offset_np = dst_offset.get()
    else:
        dst_offset_np = np.asarray(dst_offset)

    # Build comprehensive mapping for all AO pairs
    nbas = len(src_offset_np) - 1  # offset arrays have nbas+1 elements

    # First, build a mapping from each internal AO to its source AO
    internal_to_source = []
    for s in range(nbas):
        ang_s = int(angs[s])
        nf = (ang_s + 1) * (ang_s + 2) // 2
        src_row0 = int(src_offset_np[s])
        dst_row0 = int(dst_offset_np[s])

        for f in range(nf):
            internal_to_source.append((dst_row0 + f, src_row0 + f))

    # Now copy all matrix elements using the mapping
    for b in range(ndms):
        for dst_i, src_i in internal_to_source:
            for dst_j, src_j in internal_to_source:
                if (
                    dst_i < nao
                    and dst_j < nao
                    and src_i < dm_src_cp.shape[-1]
                    and src_j < dm_src_cp.shape[-1]
                ):
                    dm_dst[b][dst_i, dst_j] += dm_src_cp[b][src_i, src_j]

    return dm_dst
