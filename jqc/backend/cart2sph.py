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

''' 
Cartesian to spherical, and spherical to cartesian basis transformations.
'''

import numpy as np
import cupy as cp
from jqc.backend.cuda_scripts import code_path

__all__ = ['cart2sph', 'sph2cart']

compile_options = ('-std=c++17','--use_fast_math', '--minimal')

with open(f'{code_path}/cuda/cart2sph.cu') as f:
    cart2sph_scripts = f.read()

with open(f'{code_path}/cuda/sph2cart.cu') as f:
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
    assert dm_cart.flags['C_CONTIGUOUS']
    if dm_cart.ndim == 2:
        dm_cart = dm_cart[None]
    ndms = dm_cart.shape[0]
    cart_offset = cp.asarray(cart_offset, dtype=cp.int32)
    sph_offset = cp.asarray(sph_offset, dtype=cp.int32)
    buf = cp.empty_like(dm_cart, order='C')
    nao_cart = dm_cart.shape[-1]
    diff = angs[1:] != angs[:-1]
    offsets = np.concatenate(([0], np.nonzero(diff)[0] + 1, [angs.size]))
    threads = (16, 16)

    # cart2sph for rows
    cart_ao_stride = nao_cart
    sph_ao_stride = nao_sph
    shell_stride = 1
    for p0, p1 in zip(offsets[:-1], offsets[1:]):
        nbatch = p1 - p0
        ang = angs[p0]
        if ang not in _cart2sph_kernel_cache:
            const = f'constexpr int ang = {ang};'
            scripts = const + cart2sph_scripts
            mod = cp.RawModule(code=scripts, options=compile_options)
            _cart2sph_kernel_cache[ang] = mod.get_function('cart2sph')
        kernel = _cart2sph_kernel_cache[ang]

        args = (dm_cart, buf, nao_cart, nbatch, cart_ao_stride, sph_ao_stride, shell_stride,
               cart_offset[p0:p1], sph_offset[p0:p1])
        blocks = ((nao_cart + threads[0] - 1) // threads[0], (nbatch + threads[1] - 1) // threads[1])
        kernel(blocks, threads, args)

    if out is None:
        dm_sph = cp.empty((ndms, nao_sph, nao_sph), order='C')
    else:
        dm_sph = out

    # cart2sph for cols
    cart_ao_stride = 1
    sph_ao_stride = 1
    shell_stride = nao_sph
    for p0, p1 in zip(offsets[:-1], offsets[1:]):
        nbatch = p1 - p0
        ang = angs[p0]
        if ang not in _cart2sph_kernel_cache:
            const = f'constexpr int ang = {ang};'
            scripts = const + cart2sph_scripts
            mod = cp.RawModule(code=scripts, options=compile_options)
            _cart2sph_kernel_cache[ang] = mod.get_function('cart2sph')
        kernel = _cart2sph_kernel_cache[ang]

        args = (buf, dm_sph, nao_sph, nbatch, cart_ao_stride, sph_ao_stride, shell_stride,
               cart_offset[p0:p1], sph_offset[p0:p1])
        blocks = ((nao_cart + threads[0] - 1) // threads[0], (nbatch + threads[1] - 1) // threads[1])
        kernel(blocks, threads, args)
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
    assert dm_sph.flags['C_CONTIGUOUS']
    if dm_sph.ndim == 2:
        dm_sph = dm_sph[None]
    ndms = dm_sph.shape[0]
    nao_sph = dm_sph.shape[-1]
    nao_cart = cart_offset[-1].item()

    cart_offset = cp.asarray(cart_offset, dtype=cp.int32)
    sph_offset = cp.asarray(sph_offset, dtype=cp.int32)
    buf = cp.empty((ndms, nao_cart, nao_cart), order='C')
    
    diff = angs[1:] != angs[:-1]
    offsets = np.concatenate(([0], np.nonzero(diff)[0] + 1, [angs.size]))
    threads = (16, 16)

    # cart2sph for rows
    cart_ao_stride = nao_cart
    sph_ao_stride = nao_sph
    shell_stride = 1
    for p0, p1 in zip(offsets[:-1], offsets[1:]):
        nbatch = p1 - p0
        ang = angs[p0]
        if ang not in _sph2cart_kernel_cache:
            const = f'constexpr int ang = {ang};'
            scripts = const + sph2cart_scripts
            mod = cp.RawModule(code=scripts, options=compile_options)
            _sph2cart_kernel_cache[ang] = mod.get_function('sph2cart')
        kernel = _sph2cart_kernel_cache[ang]

        args = (buf, dm_sph, nao_sph, nbatch, cart_ao_stride, sph_ao_stride, shell_stride,
               cart_offset[p0:p1], sph_offset[p0:p1])
        blocks = ((nao_sph + threads[0] - 1) // threads[0], (nbatch + threads[1] - 1) // threads[1])
        kernel(blocks, threads, args)
    
    if out is None:
        dm_cart = cp.empty((ndms, nao_cart, nao_cart), order='C')
    else:
        dm_cart = out

    # cart2sph for cols
    cart_ao_stride = 1
    sph_ao_stride = 1
    shell_stride = nao_cart
    for p0, p1 in zip(offsets[:-1], offsets[1:]):
        nbatch = p1 - p0
        ang = angs[p0]
        if ang not in _sph2cart_kernel_cache:
            const = f'constexpr int ang = {ang};'
            scripts = const + sph2cart_scripts
            mod = cp.RawModule(code=scripts, options=compile_options)
            _sph2cart_kernel_cache[ang] = mod.get_function('sph2cart')
        kernel = _sph2cart_kernel_cache[ang]

        args = (dm_cart, buf, nao_cart, nbatch, cart_ao_stride, sph_ao_stride, shell_stride,
               cart_offset[p0:p1], sph_offset[p0:p1])
        blocks = ((nao_cart + threads[0] - 1) // threads[0], (nbatch + threads[1] - 1) // threads[1])
        kernel(blocks, threads, args)

    return dm_cart

def cart2cart(dm_src, angs, src_offset, dst_offset, nao, out=None):
    """
    Sort density matrix with the order of basis

    Args:
        dm_src: The cartesian density matrix.
        angs: The angular momentum of each basis.
        src_offset: The offset of each basis in the source cartesian density matrix.
        dst_offset: The offset of each basis in the destination cartesian density matrix.

    Returns:
        The cartesian density matrix.
    """
    if isinstance(src_offset, cp.ndarray):
        src_offset = src_offset.get()
    if isinstance(dst_offset, cp.ndarray):
        dst_offset = dst_offset.get()
    
    if dm_src.ndim == 2:
        dm_src = dm_src[None]
    ndms = dm_src.shape[0]
    if out is None:
        dm_dst = cp.empty([ndms, nao, nao])
    else:
        dm_dst = out

    src_idx = []
    dst_idx = []
    diff = angs[1:] != angs[:-1]
    offsets = np.concatenate(([0], np.nonzero(diff)[0] + 1, [angs.size]))
    for p0, p1 in zip(offsets[:-1], offsets[1:]):
        ang_p = angs[p0]
        nf = (ang_p + 1) * (ang_p + 2) // 2
        idx0 = np.arange(nf) + src_offset[p0:p1, None]
        idx1 = np.arange(nf) + dst_offset[p0:p1, None]
        src_idx.append(idx0.ravel())
        dst_idx.append(idx1.ravel())
    src_idx = np.concatenate(src_idx)
    dst_idx = np.concatenate(dst_idx)
    idx = np.arange(ndms)
    dm_dst[np.ix_(idx, dst_idx, dst_idx)] = dm_src[np.ix_(idx, src_idx, src_idx)]
    return dm_dst

def _contration_indices(bas_map):
    """Compute the contraction index (0..nctr-1) for each entry in bas_map.

    bas_map lists original shell ids for each contracted basis entry. For shells
    with multiple contractions (nctr>1), the same shell id appears multiple times.
    This helper returns, for each position, the occurrence count of that shell id
    so far, which serves as the contraction index within the shell.
    """
    import numpy as _np
    ctr = _np.empty(len(bas_map), dtype=_np.int32)
    seen = {}
    for i, s in enumerate(bas_map):
        c = seen.get(int(s), 0)
        ctr[i] = c
        seen[int(s)] = c + 1
    return ctr


def mol2cart(mat, angs, ao_loc, bas_map, mol):
    """
    Transform the matrix from the original basis (mol order) to the cartesian
    basis order used internally. Correctly handles shells with nctr>1 by adding
    per-contraction AO offsets within each shell.
    """
    nao = ao_loc[-1].item()
    # Base offsets per original shell
    mol_ao_loc_base = mol.ao_loc[bas_map]
    # Intra-shell contraction offset in AO units
    import numpy as _np
    bas_map_np = _np.asarray(bas_map)
    ctr_idx = _contration_indices(bas_map_np)
    # Per-shell degeneracy depends on mol.cart (source basis type)
    if mol.cart:
        deg = (angs + 1) * (angs + 2) // 2
    else:
        deg = 2 * angs + 1
    deg = _np.asarray(deg, dtype=_np.int32)
    ctr_shift = ctr_idx * deg
    mol_ao_loc = mol_ao_loc_base + ctr_shift
    if mol.cart:
        mat_cart = cart2cart(mat, angs, mol_ao_loc, ao_loc, nao)
    else:
        mat_cart = sph2cart(mat, angs, mol_ao_loc, ao_loc, nao)
    return mat_cart

def cart2mol(mat, angs, ao_loc, bas_map, mol):
    """
    Transform the matrix from the cartesian basis (internal order) back to the
    original mol basis order. Correctly handles shells with nctr>1.
    """
    nao = mol.nao
    import numpy as _np
    bas_map_np = _np.asarray(bas_map)
    mol_ao_loc_base = mol.ao_loc[bas_map_np]
    ctr_idx = _contration_indices(bas_map_np)
    # Destination basis type depends on mol.cart
    if mol.cart:
        deg = (angs + 1) * (angs + 2) // 2
    else:
        deg = 2 * angs + 1
    deg = _np.asarray(deg, dtype=_np.int32)
    ctr_shift = ctr_idx * deg
    mol_ao_loc = mol_ao_loc_base + ctr_shift
    if mol.cart:
        mat_mol = cart2cart(mat, angs, ao_loc, mol_ao_loc, nao)
    else:
        mat_mol = cart2sph(mat, angs, ao_loc, mol_ao_loc, nao)
    return mat_mol
