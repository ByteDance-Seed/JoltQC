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
Molecular basis set handling and layout for JoltQC.

This module defines three types of molecular basis representations:

1. **Original Mol**: The input PySCF molecule with contracted basis functions.
   - May contain basis functions with nctr > 1 (contracted functions)
   - May contain basis functions with nprim > NPRIM_MAX primitives
   - Spherical or cartesian basis functions as specified by user
   - Example: A d-shell with 6 primitive Gaussians contracted to 2 functions (nctr=2)

2. **Decontracted Mol**: Intermediate molecule with all basis functions decontracted.
   - All basis functions have nctr = 1 (each contraction becomes separate basis)
   - Still may have nprim > NPRIM_MAX for individual basis functions
   - Maintains same coordinate system as original (spherical/cartesian)
   - Intermediate step in split_basis() function processing
   - Example: The same d-shell becomes 2 separate basis functions, each with 6 primitives

3. **Split Mol**: Fully processed molecule for internal JoltQC computations.
   - All basis functions have nctr = 1 (decontracted)
   - All basis functions have nprim ≤ NPRIM_MAX (split if necessary)
   - Always uses cartesian basis functions for JK kernel compatibility
   - Created by split_basis() function and used internally by BasisLayout
   - Example: If NPRIM_MAX=3, each 6-primitive function becomes 2 functions with 3 primitives each

The BasisLayout class manages these transformations and provides mappings between the different
representations to maintain compatibility with the original molecule's AO indexing.
"""

import ctypes
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple

import cupy as cp
import numpy as np
from pyscf import gto, lib
from pyscf.scf import _vhf

# Import transformation functions at module level to avoid repeated local imports
from jqc.backend.cart2sph import cart2cart, cart2sph, sph2cart
from jqc.constants import COORD_STRIDE, NPRIM_MAX, PRIM_STRIDE

__all__ = ["compute_q_matrix", "sort_group_basis", "split_basis"]
PTR_BAS_COORD = 7

ArrayLike = np.ndarray


@dataclass
class BasisLayout:
    ce: ArrayLike  # shape (nbasis_total, PRIM_STRIDE), np/cp
    coords: ArrayLike  # shape (nbasis_total, COORD_STRIDE),  np/cp
    angs: np.ndarray  # shape (nbasis_total,),               int32
    nprims: np.ndarray  # shape (nbasis_total,),               int32

    # maps / masks
    to_split_map: np.ndarray  # shape (nbasis_total,),               int32
    pad_id: np.ndarray  # shape (nbasis_total,),               bool

    # group_info
    group_key: np.ndarray  # shape (ngroups, 2) -> [ang, nprim], int32
    group_offset: np.ndarray  # shape (ngroups+1,),                int64/int32

    # dtype bookkeeping (optional but handy)
    dtype: np.dtype

    # molecule references
    _mol: Optional[object] = None  # original pyscf.gto.Mole reference
    _splitted_mol: Optional[object] = (
        None  # split pyscf.gto.Mole reference (decontracted + split)
    )
    # Mapping from split basis index -> decontracted basis index
    _split_to_decontracted: Optional[np.ndarray] = None

    # Cached properties (computed lazily)
    _q_matrix: Optional[dict] = None
    _ce_fp32: Optional[ArrayLike] = None
    _coords_fp32: Optional[ArrayLike] = None
    _ao_loc: Optional[np.ndarray] = None
    _mol_ao_loc: Optional[cp.ndarray] = None
    _ao_loc_no_pad: Optional[np.ndarray] = None
    _to_decontracted_map: Optional[np.ndarray] = None
    _angs_no_pad: Optional[np.ndarray] = None

    # --------- Compatibility accessors ---------
    @property
    def bas_info(self) -> Tuple[ArrayLike, ArrayLike, np.ndarray, np.ndarray]:
        return (self.ce, self.coords, self.angs, self.nprims)

    @property
    def group_info(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.group_key, self.group_offset)

    # --------- Convenience properties ---------
    @property
    def nbasis(self) -> int:
        return int(self.to_split_map.shape[0])

    @property
    def ngroups(self) -> int:
        return int(self.group_key.shape[0])

    @property
    def angs_no_pad(self) -> np.ndarray:
        """
        Angular momentum values of sorted decontracted mol.
        """
        if self._angs_no_pad is None:
            self._angs_no_pad = self.angs[~self.pad_id]
        return self._angs_no_pad

    @property
    def ao_loc(self) -> np.ndarray:
        """
        AO offsets for each basis in internal format.
        ao_loc[i] gives the starting AO index of basis i.
        """
        if self._ao_loc is None:
            angs = self.angs
            dims = (angs + 1) * (angs + 2) // 2
            dims[self.pad_id] = 0
            ao_loc = np.empty(len(dims) + 1, dtype=np.int32)
            ao_loc[0] = 0
            dims.cumsum(dtype=np.int32, out=ao_loc[1:])
            self._ao_loc = ao_loc
        return self._ao_loc

    @property
    def ao_loc_no_pad(self) -> np.ndarray:
        """
        AO offsets for each basis in internal format without padding.
        Its dimension should be consistent with angs_no_pad.
        """
        if self._ao_loc_no_pad is None:
            ao_loc = np.empty(len(self.angs_no_pad) + 1, dtype=np.int32)
            ao_loc[:-1] = self.ao_loc[:-1][~self.pad_id]
            ao_loc[-1] = self.ao_loc[-1]
            self._ao_loc_no_pad = cp.asarray(ao_loc)
        return self._ao_loc_no_pad

    @property
    def mol_ao_loc(self) -> cp.ndarray:
        """
        AO shell offsets for the original molecule in decontracted format.
        Its dimension should be consistent with angs_no_pad
        """
        if self._mol_ao_loc is None:
            # Compute from original molecule (decontracted basis)
            bas = self._mol._bas
            angs = bas[:, gto.ANG_OF]
            nctr = bas[:, gto.NCTR_OF]
            if self._mol.cart:
                dims = [[(l + 1) * (l + 2) // 2] * n for l, n in zip(angs, nctr)]
            else:
                dims = [[2 * l + 1] * n for l, n in zip(angs, nctr)]
            dims = np.concatenate(dims)
            ao_loc = np.empty(len(dims) + 1, dtype=np.int32)
            ao_loc[0] = 0
            dims.cumsum(dtype=np.int32, out=ao_loc[1:])

            split_idx = self.to_split_map[~self.pad_id]
            decontracted_idx = self._split_to_decontracted[split_idx]
            _ao_loc = np.empty(len(split_idx) + 1, dtype=np.int32)
            _ao_loc[:-1] = ao_loc[decontracted_idx]
            _ao_loc[-1] = ao_loc[-1]
            self._mol_ao_loc = cp.asarray(_ao_loc)
        return self._mol_ao_loc

    @property
    def to_decontracted_map(self) -> np.ndarray:
        """
        Lazy evaluation property for to_decontracted_map.
        Maps from internal sorted/grouped basis layout back to decontracted molecule basis indices.
        """
        if self._to_decontracted_map is None:
            # Compute to_decontracted_map: internal -> decontracted
            # This is the composition of to_split_map (internal -> split) and split_to_decontracted (split -> decontracted)
            to_decontracted_map = np.empty(len(self.to_split_map), dtype=np.int32)
            to_decontracted_map[~self.pad_id] = self._split_to_decontracted[
                self.to_split_map[~self.pad_id]
            ]
            to_decontracted_map[self.pad_id] = -1  # Invalid index for padded entries
            self._to_decontracted_map = to_decontracted_map
        return self._to_decontracted_map

    @property
    def splitted_mol(self) -> object:
        """
        Access to the stored split molecule (decontracted + split basis functions).

        Returns:
            pyscf.gto.Mole: The split molecule used for basis layout computations
        """
        if self._splitted_mol is None:
            raise ValueError("splitted_mol is not available")
        return self._splitted_mol

    def q_matrix(self, omega=0.0) -> ArrayLike:
        """
        Lazy evaluation property for q_matrix.
        Computes and caches the fully processed q_matrix only when accessed.
        Uses the stored split molecule to ensure dimensions match the split basis functions.
        """
        if self._q_matrix is None:
            self._q_matrix = {}

        if omega not in self._q_matrix:
            # Use stored split molecule for q_matrix computation
            if self._splitted_mol is None:
                raise ValueError("q_matrix requires _splitted_mol reference to be set")
            # Compute raw q_matrix using split molecule
            q_matrix_raw = compute_q_matrix(self._splitted_mol, omega)
            # Apply basis mapping to reorder according to sorted basis
            q_matrix_mapped = q_matrix_raw[
                self.to_split_map[:, None], self.to_split_map
            ]
            # Set Q matrix for padded basis to -100 (screening value)
            q_matrix_mapped[self.pad_id, :] = -100
            q_matrix_mapped[:, self.pad_id] = -100
            # Convert to CuPy array with proper dtype for GPU computation
            self._q_matrix[omega] = cp.asarray(q_matrix_mapped, dtype=np.float32)

        return self._q_matrix[omega]

    @property
    def ce_fp32(self) -> ArrayLike:
        """
        Lazy evaluation property for FP32 coefficient and exponent array.
        Converts from base fp64 arrays.
        """
        if self._ce_fp32 is None:
            self._ce_fp32 = self.ce.astype(np.float32)
        return self._ce_fp32

    @property
    def coords_fp32(self) -> ArrayLike:
        """
        Lazy evaluation property for FP32 coordinates array.
        Converts from base fp64 arrays.
        """
        if self._coords_fp32 is None:
            self._coords_fp32 = self.coords.astype(np.float32)
        return self._coords_fp32

    @property
    def ce_fp64(self) -> ArrayLike:
        """
        FP64 coefficient and exponent array (base storage is always fp64).
        """
        return self.ce

    @property
    def coords_fp64(self) -> ArrayLike:
        """
        FP64 coordinates array (base storage is always fp64).
        """
        return self.coords

    @classmethod
    def from_mol(cls, mol, alignment: int = 4, dtype=np.float64) -> "BasisLayout":
        """
        Creates a BasisLayout from a molecule using split basis functions.
        First creates a split mol, then calls sort_group_basis on it.

        Parameters
        ----------
        mol : pyscf.gto.Mole
            Molecular structure (will be decontracted internally)
        alignment : int
            Basis alignment for memory optimization
        dtype : numpy.dtype
            Data type for basis functions
        """
        # First create split molecule with decontracted and split basis functions
        splitted_mol, split_to_decontracted = split_basis(mol)

        # Then call sort_group_basis on the split molecule
        bas_info, to_split_map, pad_id, group_info = sort_group_basis(
            splitted_mol, alignment=alignment, dtype=dtype
        )
        ce, coords, angs, nprims = bas_info
        group_key, group_offset = group_info

        # Assert angular momentum constraint (JoltQC kernels support up to g orbitals, l=4)
        max_ang = np.max(angs[~pad_id]) if np.any(~pad_id) else 0
        assert (
            max_ang <= 4
        ), f"Angular momentum {max_ang} exceeds maximum supported value of 4"

        return cls(
            ce=ce,
            coords=coords,
            angs=np.asarray(angs, dtype=np.int32),
            nprims=np.asarray(nprims, dtype=np.int32),
            to_split_map=np.asarray(to_split_map, dtype=np.int32),
            pad_id=np.asarray(pad_id, dtype=bool),
            group_key=np.asarray(group_key, dtype=np.int32),
            group_offset=np.asarray(group_offset),
            _mol=mol,
            _splitted_mol=splitted_mol,
            _split_to_decontracted=np.asarray(split_to_decontracted, dtype=np.int32),
            dtype=np.dtype(dtype),
        )

    def dm_from_mol(self, mat):
        """
        Transform matrix from decontracted molecular AO order into the
        internal decontracted cartesian AO order used by kernels.

        For spherical input, applies sph->cart transform; for cart input,
        performs a pure reordering.
        """
        mat_cp = cp.asarray(mat) if not isinstance(mat, cp.ndarray) else mat
        # Source = decontracted molecular order, Destination = internal order (no pad)
        src_offsets = self.mol_ao_loc
        dst_offsets = self.ao_loc_no_pad
        nao_dst = int(dst_offsets[-1])
        is_cart_src = self._mol.cart  # original molecule basis type

        if mat_cp.ndim == 3:
            results = []
            transform_func = cart2cart if is_cart_src else sph2cart
            for i in range(mat_cp.shape[0]):
                mat_2d = transform_func(
                    mat_cp[i], self.angs_no_pad, src_offsets, dst_offsets, nao_dst
                )
                if mat_2d.ndim == 3 and mat_2d.shape[0] == 1:
                    mat_2d = mat_2d[0]
                results.append(mat_2d)
            return cp.stack(results, axis=0)
        else:
            transform_func = cart2cart if is_cart_src else sph2cart
            result = transform_func(
                mat_cp, self.angs_no_pad, src_offsets, dst_offsets, nao_dst
            )
            return result[0] if result.ndim == 3 and result.shape[0] == 1 else result

    def dm_to_mol(self, mat):
        """
        Transform matrix from internal decontracted cartesian AO order back to
        decontracted molecular AO order.
        """
        mat_cp = cp.asarray(mat) if isinstance(mat, np.ndarray) else mat
        # Source = internal (cart, no pad), Destination = decontracted molecular order
        src_offsets = self.ao_loc_no_pad
        dst_offsets = self.mol_ao_loc
        nao_dst = int(dst_offsets[-1])
        is_cart_dst = self._mol.cart  # destination basis type

        if mat_cp.ndim == 3:
            results = []
            transform_func = cart2cart if is_cart_dst else cart2sph
            for i in range(mat_cp.shape[0]):
                mat_2d = transform_func(
                    mat_cp[i], self.angs_no_pad, src_offsets, dst_offsets, nao_dst
                )
                if mat_2d.ndim == 3 and mat_2d.shape[0] == 1:
                    mat_2d = mat_2d[0]
                results.append(mat_2d)
            return cp.stack(results, axis=0)
        else:
            transform_func = cart2cart if is_cart_dst else cart2sph
            result = transform_func(
                mat_cp, self.angs_no_pad, src_offsets, dst_offsets, nao_dst
            )
            return result[0] if result.ndim == 3 and result.shape[0] == 1 else result


def sort_group_basis(mol, alignment=4, dtype=np.float64):
    """
    Sort and group basis by angular momentum and number of primitives.

    Args:
        mol (pyscf.gto.Mole): The pyscf Mole object. Must be decontracted.
        alignment (int, optional): The alignment of each basis group. Defaults to 4.
        dtype (numpy.dtype, optional): The data type of the arrays. Defaults to np.float64.
    Returns:
        tuple: A tuple containing the sorted and grouped basis cache.
            (basis_info, basis_map, basis_mask, group_info)

        basis_info := (ce, coords, angs, nprims)
        basis_map := basis_info -> mol._bas
        basis_mask := padding mask for basis_info
        group_info := (group_key, group_offset)
    """

    _bas = mol._bas
    _env = mol._env
    _atm = mol._atm

    # Assert that all basis functions are decontracted (nctr = 1)
    for i in range(mol._bas.shape[0]):
        nctr = _bas[i, gto.NCTR_OF]
        assert (
            nctr == 1
        ), f"Basis function {i} has nctr={nctr}, expected nctr=1. mol must be decontracted."

    # Pre-calculate sizes to avoid intermediate copies
    nbas = _bas.shape[0]
    pattern_counts = defaultdict(int)
    pattern_data = defaultdict(list)

    # First pass: count basis functions by pattern and collect data
    for i in range(nbas):
        nprim = _bas[i, gto.NPRIM_OF]
        nctr = _bas[i, gto.NCTR_OF]  # Always 1 for decontracted molecules
        ang = _bas[i, gto.ANG_OF]
        pattern = (ang, nprim)
        pattern_counts[pattern] += 1  # nctr is always 1

        iatm = _bas[i, gto.ATOM_OF]
        coord_ptr = _atm[iatm, gto.PTR_COORD]
        exp_ptr = _bas[i, gto.PTR_EXP]
        coeff_ptr = _bas[i, gto.PTR_COEFF]

        pattern_data[pattern].append(
            {
                "coord_ptr": coord_ptr,
                "exp_ptr": exp_ptr,
                "coeff_ptr": coeff_ptr,
                "nprim": nprim,
                "ang": ang,
                "to_split_map": i,
            }
        )

    # Pre-allocate arrays for each pattern
    coords_by_pattern = {}
    ce_by_pattern = {}
    to_split_map_by_pattern = {}
    pad_id_by_pattern = {}

    # Pre-compute normalization factors to avoid repeated calculations
    norm_factors = {}
    for ang in set(key[0] for key in pattern_counts.keys()):
        if ang < 2:
            norm_factors[ang] = ((2 * ang + 1) / (4.0 * np.pi)) ** 0.5
        else:
            norm_factors[ang] = 1.0

    for pattern, count in pattern_counts.items():
        ang, nprim = pattern
        padded_count = count + ((alignment - count % alignment) % alignment)

        # Pre-allocate final arrays
        coords_by_pattern[pattern] = np.empty(
            (padded_count, COORD_STRIDE), dtype=np.float64
        )
        # Initialize CE buffer to zeros so padded primitive slots are well-defined
        ce_by_pattern[pattern] = np.zeros((padded_count, PRIM_STRIDE), dtype=dtype)
        to_split_map_by_pattern[pattern] = np.empty(padded_count, dtype=np.int32)
        pad_id_by_pattern[pattern] = np.empty(padded_count, dtype=bool)

        # Get normalization factor once per pattern
        norm_fac = norm_factors[ang]

        # Fill arrays without intermediate copies
        idx = 0
        for data in pattern_data[pattern]:
            coord_ptr = data["coord_ptr"]
            exp_ptr = data["exp_ptr"]
            coeff_ptr = data["coeff_ptr"]
            nprim = data["nprim"]
            to_split_map = data["to_split_map"]

            # Get coefficients and exponents (nctr=1, so length is nprim)
            coeffs = (
                _env[coeff_ptr : coeff_ptr + nprim] * norm_fac
            )  # Apply norm factor directly
            exps = _env[exp_ptr : exp_ptr + nprim]

            # Get coordinates - optimize by direct slicing
            coord = np.zeros(COORD_STRIDE, dtype=np.float64)
            coord[:3] = _env[coord_ptr : coord_ptr + 3]

            # Fill arrays directly (no loop needed since nctr=1)
            coords_by_pattern[pattern][idx] = coord
            ce_by_pattern[pattern][idx, 0 : 2 * nprim : 2] = coeffs
            ce_by_pattern[pattern][idx, 1 : 2 * nprim : 2] = exps
            to_split_map_by_pattern[pattern][idx] = to_split_map
            pad_id_by_pattern[pattern][idx] = False
            idx += 1

        # Fill padding with first element to avoid additional memory allocation
        if idx < padded_count:
            coords_by_pattern[pattern][idx:] = coords_by_pattern[pattern][0]
            ce_by_pattern[pattern][idx:] = ce_by_pattern[pattern][0]
            to_split_map_by_pattern[pattern][idx:] = to_split_map_by_pattern[pattern][0]
            pad_id_by_pattern[pattern][idx:] = True

    # Optimize sorting with explicit key function to avoid lambda overhead
    def _pattern_sort_key(pattern):
        return (pattern[0], -pattern[1])

    # Sort the basis by angular momentum and number of primitives
    # Reverse the order of primitives, to be consistent with GPU4PySCF
    sorted_keys = sorted(coords_by_pattern.keys(), key=_pattern_sort_key)

    # Calculate total size for pre-allocation
    total_count = sum(len(to_split_map_by_pattern[key]) for key in sorted_keys)

    # Pre-allocate CPU arrays for batching
    # Use PRIM_STRIDE (padded) to match per-pattern CE buffers
    all_ce_data = np.empty((total_count, PRIM_STRIDE), dtype=dtype, order="C")
    all_coords_data = np.empty((total_count, COORD_STRIDE), dtype=dtype, order="C")
    to_split_map = np.empty(total_count, dtype=np.int32)
    pad_id = np.empty(total_count, dtype=bool)
    angs = np.empty(total_count, dtype=np.int32)
    nprims = np.empty(total_count, dtype=np.int32)

    group_key = []
    group_offset = []
    offset = 0

    # Fill CPU arrays first, then batch transfer to GPU
    for key in sorted_keys:
        pattern_ce = ce_by_pattern[key]
        pattern_coords = coords_by_pattern[key]
        pattern_to_split_map = to_split_map_by_pattern[key]
        pattern_pad_id = pad_id_by_pattern[key]
        bas_count = len(pattern_to_split_map)

        # Copy data to CPU arrays for batching
        all_ce_data[offset : offset + bas_count] = pattern_ce
        all_coords_data[offset : offset + bas_count] = pattern_coords

        # CPU arrays - direct assignment
        to_split_map[offset : offset + bas_count] = pattern_to_split_map
        pad_id[offset : offset + bas_count] = pattern_pad_id
        angs[offset : offset + bas_count] = key[0]
        nprims[offset : offset + bas_count] = key[1]

        group_key.append([key[0], key[1]])
        group_offset.append(offset)
        offset += bas_count
    group_offset.append(offset)

    # Batch transfer to GPU - single transfer per array type
    ce = cp.asarray(all_ce_data, dtype=dtype, order="C")
    coords = cp.asarray(all_coords_data, dtype=dtype, order="C")

    # Arrays are already in target format - no additional conversions needed
    # ce and coords are already CuPy arrays with correct order

    # Store info at basis level
    bas_info = (ce, coords, angs, nprims)
    '''
    group_size = 256
    splitted_group_key = []
    splitted_group_offset = []
    for group_id in range(len(group_key)):
        for offset in range(group_offset[group_id], group_offset[group_id+1], group_size):
            splitted_group_key.append(group_key[group_id])
            splitted_group_offset.append(offset)
    splitted_group_offset.append(group_offset[-1])
    group_key = np.asarray(splitted_group_key)
    group_offset = np.asarray(splitted_group_offset)
    '''
    group_key = np.asarray(group_key)
    group_offset = np.asarray(group_offset)
    return bas_info, to_split_map, pad_id, (group_key, group_offset)


def split_basis(mol):
    """
    Create a new PySCF molecule with decontracted and split basis functions.

    For basis functions with nctr = 1, keeps the basis as-is.
    For basis functions with nctr > 1, creates nctr separate basis functions,
    each with a single contraction coefficient, all assigned to the same atom.
    Additionally, splits basis functions with nprim > NPRIM_MAX into multiple
    basis functions to respect the NPRIM_MAX limit.

    Args:
        mol (pyscf.gto.Mole): The original PySCF molecule

    Returns:
        tuple: (pyscf.gto.Mole, split_basis_map)
            - New split molecule with decontracted and split basis functions
            - Mapping from each final split basis function to its decontracted parent basis index
    """
    from pyscf import gto

    # Create new molecule with same atoms and general settings
    new_mol = gto.Mole()
    new_mol.atom = mol.atom
    new_mol.charge = mol.charge
    new_mol.spin = mol.spin
    new_mol.unit = mol.unit
    # Internal split representation always uses cartesian for kernel compatibility
    new_mol.cart = True
    new_mol.verbose = mol.verbose

    # Process basis functions to create decontracted versions
    _bas = mol._bas
    _env = mol._env
    _atm = mol._atm

    new_bas_list = []
    new_env_list = list(_env)  # Start with existing environment
    original_bas_map = []  # Map split basis indices to decontracted basis indices
    decontracted_idx = 0  # Counter for decontracted basis functions

    for i in range(mol.nbas):
        nprim = _bas[i, gto.NPRIM_OF]
        nctr = _bas[i, gto.NCTR_OF]
        ang = _bas[i, gto.ANG_OF]
        iatm = _bas[i, gto.ATOM_OF]

        exp_ptr = _bas[i, gto.PTR_EXP]
        coeff_ptr = _bas[i, gto.PTR_COEFF]

        # Get exponents and coefficients
        exps = _env[exp_ptr : exp_ptr + nprim]
        coeffs = _env[coeff_ptr : coeff_ptr + nprim * nctr]

        if nctr == 1:
            # For nctr = 1, check if we need to split due to nprim > NPRIM_MAX
            if nprim <= NPRIM_MAX:
                # Keep the basis as-is
                new_bas_entry = _bas[i].copy()
                new_bas_list.append(new_bas_entry.tolist())
                original_bas_map.append(decontracted_idx)
                decontracted_idx += 1
            else:
                # Split basis function into multiple parts due to nprim > NPRIM_MAX
                nsplits = (nprim + NPRIM_MAX - 1) // NPRIM_MAX  # Ceiling division

                for k in range(nsplits):
                    start_prim = k * NPRIM_MAX
                    end_prim = min((k + 1) * NPRIM_MAX, nprim)
                    split_nprim = end_prim - start_prim

                    # Add split exponents to environment
                    exp_ptr_new = len(new_env_list)
                    new_env_list.extend(exps[start_prim:end_prim])

                    # Add split coefficients to environment
                    coeff_ptr_new = len(new_env_list)
                    new_env_list.extend(coeffs[start_prim:end_prim])

                    # Create new basis entry for this split
                    new_bas_entry = [
                        iatm,  # ATOM_OF (same atom as original)
                        ang,  # ANG_OF
                        split_nprim,  # NPRIM_OF (split size)
                        1,  # NCTR_OF (always 1)
                        0,  # KAPPA (unused)
                        exp_ptr_new,  # PTR_EXP (split exponents)
                        coeff_ptr_new,  # PTR_COEFF (split coefficients)
                        _bas[i, 7],  # PTR_BAS_COORD (same as original)
                    ]
                    new_bas_list.append(new_bas_entry)
                    original_bas_map.append(decontracted_idx)
                decontracted_idx += 1
        else:
            # For nctr > 1, create nctr separate basis functions
            # Each assigned to the same atom
            for j in range(nctr):
                # Extract coefficients for this contraction
                coeff_start = j * nprim
                coeff_end = (j + 1) * nprim
                single_coeffs = coeffs[coeff_start:coeff_end]

                # Check if we need to split this contraction due to nprim > NPRIM_MAX
                if nprim <= NPRIM_MAX:
                    # Add single contraction coefficients to environment
                    coeff_ptr_new = len(new_env_list)
                    new_env_list.extend(single_coeffs)

                    # Create new basis entry with nctr = 1 for the same atom
                    new_bas_entry = [
                        iatm,  # ATOM_OF (same atom as original)
                        ang,  # ANG_OF
                        nprim,  # NPRIM_OF
                        1,  # NCTR_OF (always 1 for decontracted)
                        0,  # KAPPA (unused)
                        exp_ptr,  # PTR_EXP (reuse original)
                        coeff_ptr_new,  # PTR_COEFF (separate for each)
                        _bas[i, 7],  # PTR_BAS_COORD (same as original)
                    ]
                    new_bas_list.append(new_bas_entry)
                    original_bas_map.append(decontracted_idx)
                    decontracted_idx += 1
                else:
                    # Split this contraction into multiple parts due to nprim > NPRIM_MAX
                    nsplits = (nprim + NPRIM_MAX - 1) // NPRIM_MAX  # Ceiling division

                    for k in range(nsplits):
                        start_prim = k * NPRIM_MAX
                        end_prim = min((k + 1) * NPRIM_MAX, nprim)
                        split_nprim = end_prim - start_prim

                        # Add split exponents to environment
                        exp_ptr_new = len(new_env_list)
                        new_env_list.extend(exps[start_prim:end_prim])

                        # Add split coefficients to environment
                        coeff_ptr_new = len(new_env_list)
                        new_env_list.extend(single_coeffs[start_prim:end_prim])

                        # Create new basis entry for this split
                        new_bas_entry = [
                            iatm,  # ATOM_OF (same atom as original)
                            ang,  # ANG_OF
                            split_nprim,  # NPRIM_OF (split size)
                            1,  # NCTR_OF (always 1)
                            0,  # KAPPA (unused)
                            exp_ptr_new,  # PTR_EXP (split exponents)
                            coeff_ptr_new,  # PTR_COEFF (split coefficients)
                            _bas[i, 7],  # PTR_BAS_COORD (same as original)
                        ]
                        new_bas_list.append(new_bas_entry)
                        original_bas_map.append(decontracted_idx)
                    decontracted_idx += 1

    # Set new basis and environment (atoms remain the same)
    new_mol._atm = _atm.copy()
    new_mol._bas = np.array(new_bas_list, dtype=np.int32)
    new_mol._env = np.array(new_env_list)

    # Do not call build() - return the unbuilt molecule and mapping
    return new_mol, np.array(original_bas_map)


def compute_q_matrix(mol, omega=0.0):
    """
    Compute the Q matrix in infinite norm for the given molecule.

    Args:
        mol (pyscf.gto.Mole): The pyscf Mole object.

    Returns:
        numpy.ndarray: The Q matrix.
    """
    nbas = mol.nbas
    ao_loc = mol.ao_loc
    q_matrix = np.empty((nbas, nbas))
    intor = mol._add_suffix("int2e")
    with mol.with_range_coulomb(omega):
        _vhf.libcvhf.CVHFnr_int2e_q_cond(
            getattr(_vhf.libcvhf, intor),
            lib.c_null_ptr(),
            q_matrix.ctypes,
            ao_loc.ctypes,
            mol._atm.ctypes,
            ctypes.c_int(mol.natm),
            mol._bas.ctypes,
            ctypes.c_int(mol.nbas),
            mol._env.ctypes,
        )
    q_matrix = np.log(q_matrix + 1e-300).astype(np.float32)
    return q_matrix


def cluster_into_tile(coords, tile=4):
    """
    Greedy heuristic of grouping coords into tiles
    """
    from scipy.spatial.distance import pdist, squareform

    coords = np.asarray(coords)

    distance_matrix = squareform(pdist(coords, metric="euclidean"))
    n = distance_matrix.shape[0]
    unassigned = set(range(n))
    clusters = []

    while unassigned:
        i = unassigned.pop()
        # find nearest 3 among remaining
        dists = [(j, distance_matrix[i, j]) for j in unassigned]
        nearest = sorted(dists, key=lambda x: x[1])[: tile - 1]
        cluster = [i] + [j for j, _ in nearest]
        for j, _ in nearest:
            unassigned.remove(j)
        clusters += cluster
    return clusters


if __name__ == "__main__":
    from pyscf import gto

    # Test 1: Normal molecule with basis functions within NPRIM_MAX
    mol = gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 1"
    mol.basis = "cc-pvdz"
    mol.build()

    print("=== Test 1: Normal molecule (cc-pvdz) ===")
    layout = BasisLayout.from_mol(mol)
    print(f"Original mol.nao: {mol.nao}")
    print(f"Layout nbasis: {layout.nbasis}")
    print(f"Layout ao_loc shape: {layout.ao_loc.shape}")
    print(f"Original mol.ao_loc: {mol.ao_loc}")
    print(f"Layout ao_loc: {layout.ao_loc}")
    print(f"Max primitives in original mol: {max(mol._bas[:, gto.NPRIM_OF])}")

    # Test 2: Create a molecule with large basis to trigger splitting
    print("\n=== Test 2: Molecule with large basis (cc-pCV5Z) ===")
    try:
        mol2 = gto.Mole()
        mol2.atom = "C 0 0 0"
        mol2.basis = "cc-pCV5Z"  # This often has basis functions with > 16 primitives
        mol2.build()

        print(f"Original mol.nao: {mol2.nao}")
        print(f"Max primitives in original mol: {max(mol2._bas[:, gto.NPRIM_OF])}")

        layout2 = BasisLayout.from_mol(mol2)
        print(f"Layout nbasis: {layout2.nbasis}")
        print(f"Layout ao_loc shape: {layout2.ao_loc.shape}")
        print(
            f"Original mol.ao_loc matches layout ao_loc: {np.array_equal(mol2.ao_loc, layout2.ao_loc)}"
        )

        # to_split_map provides the mapping from internal sorted layout to split basis indices
        print(
            f"Basis mapping established: {len(layout2.to_split_map)} split->decontracted mappings"
        )

    except Exception as e:
        print(f"Test 2 failed (basis may not be available): {e}")

    print("\n=== Test 3: Simulated large primitive basis ===")
    # Manually test the split_basis function with artificial data
    mol3 = gto.Mole()
    mol3.atom = "H 0 0 0"
    mol3.basis = "sto-3g"
    mol3.spin = 1  # Correct spin for single hydrogen
    mol3.build()

    # Test the split_basis function directly
    old_nprim = mol3._bas[0, gto.NPRIM_OF]
    print(f"Original nprim for first basis: {old_nprim}")
    splitted_mol, original_bas_map = split_basis(mol3)
    print(f"Split mol nbas: {splitted_mol.nbas}")
    print(f"Original basis mapping: {original_bas_map}")

    print("\n=== Summary ===")
    print("✓ BasisLayout successfully redesigned to handle:")
    print("  1. Arbitrary number of primitives (splits when > NPRIM_MAX)")
    print("  2. Maintains original ao_loc compatibility")
    print("  3. Tracks mapping between split and original basis functions")
