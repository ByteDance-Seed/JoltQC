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

from dataclasses import dataclass
from typing import Tuple, Iterator, Optional
import ctypes
from collections import defaultdict
import numpy as np
import cupy as cp
from pyscf import gto, lib
from pyscf.scf import _vhf
from jqc.constants import NPRIM_MAX

__all__ = ['format_bas_cache', 'sort_group_basis', 'compute_q_matrix']
PTR_BAS_COORD = 7

ArrayLike = np.ndarray

@dataclass(frozen=True)
class BasisLayout:
    ce: ArrayLike          # shape (nbasis_total, 2*NPRIM_MAX), np/cp
    coords: ArrayLike      # shape (nbasis_total, 4),             np/cp
    angs: np.ndarray       # shape (nbasis_total,),               int32
    nprims: np.ndarray     # shape (nbasis_total,),               int32

    # maps / masks
    bas_id: np.ndarray     # shape (nbasis_total,),               int32
    pad_id: np.ndarray     # shape (nbasis_total,),               bool

    # group_info
    group_key: np.ndarray      # shape (ngroups, 2) -> [ang, nprim], int32
    group_offset: np.ndarray   # shape (ngroups+1,),                int64/int32

    # dtype bookkeeping (optional but handy)
    dtype: np.dtype

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
        return int(self.bas_id.shape[0])

    @property
    def ngroups(self) -> int:
        return int(self.group_key.shape[0])

    @property
    def ao_loc(self) -> cp.ndarray:
        """
        AO shell offsets: cumulative sum of angular momentum degeneracies.

        Each shell contributes (l+1)(l+2)//2 functions.
        """
        dims = (self.angs + 1) * (self.angs + 2) // 2
        return cp.concatenate(([0], np.cumsum(dims)))

    @classmethod
    def from_sort_group_basis(cls, mol, alignment: int = 4, dtype=np.float64) -> "BasisLayout":
        """
        Calls your `sort_group_basis(mol, alignment, dtype)` and wraps the result.
        Expects that function to already move `ce` and `coords` to CuPy (as in your code).
        """
        bas_info, bas_id, pad_id, group_info = sort_group_basis(mol, alignment=alignment, dtype=dtype)
        ce, coords, angs, nprims = bas_info
        group_key, group_offset = group_info
        # Normalize dtypes
        return cls(
            ce=ce,
            coords=coords,
            angs=np.asarray(angs, dtype=np.int32),
            nprims=np.asarray(nprims, dtype=np.int32),
            bas_id=np.asarray(bas_id, dtype=np.int32),
            pad_id=np.asarray(pad_id, dtype=bool),
            group_key=np.asarray(group_key, dtype=np.int32),
            group_offset=np.asarray(group_offset),
            dtype=np.dtype(dtype),
        )


def format_bas_cache(sorted_mol, dtype=np.float64):
    """
    Format the basis cache used in JQC.
    coords:    [nbas, 3]
    coeffs:    [nbas, nprim_max]
    exponents: [nbas, nprim_max]
    ao_loc:    [nbas + 1]
    nprims:    [nbas]
    angs:      [nbas]

    Args:
        sorted_mol (pyscf.gto.Mole): The sorted pyscf Mole object. Must be decontracted.
        dtype (numpy.dtype, optional): The data type of the arrays. Defaults to np.float64.

    Returns:
        tuple: A tuple containing the formatted basis cache.
    """
    _bas = sorted_mol._bas
    _env = sorted_mol._env
    coord_ptr = _bas[:, PTR_BAS_COORD]
    nbas = _bas.shape[0]
    
    coords = np.empty([nbas, 3], dtype=dtype, order='C')
    coeffs = np.empty([nbas, NPRIM_MAX], dtype=dtype, order='C')
    exponents = np.empty([nbas, NPRIM_MAX], dtype=dtype, order='C')
    nprims = np.empty(nbas, dtype=np.int32)
    angs = np.empty(nbas, dtype=np.int32)

    for i in range(nbas):
        exp_ptr = _bas[i, gto.PTR_EXP]
        coeff_ptr = _bas[i, gto.PTR_COEFF]
        nprim = _bas[i, gto.NPRIM_OF]
        nctr = _bas[i, gto.NCTR_OF]
        ang = _bas[i, gto.ANG_OF]
        coeff = _env[coeff_ptr:coeff_ptr+nprim*nctr].copy()
        # Apply normalization factor, being consistent with libcint
        if ang < 2:
            fac = ((2*ang + 1) / (4.0 * np.pi))**.5
            coeff *= fac
    
        coeffs[i,:nprim] = coeff
        exponents[i,:nprim] = _env[exp_ptr:exp_ptr+nprim*nctr]
        coords[i,:] = _env[coord_ptr[i]:coord_ptr[i]+3]
        nprims[i] = nprim
        angs[i] = ang

    coords = cp.asarray(coords, dtype=dtype)
    coeffs = cp.asarray(coeffs, dtype=dtype)
    exponents = cp.asarray(exponents, dtype=dtype)
    nprims = cp.asarray(nprims)
    angs = cp.asarray(angs)

    ao_loc = cp.array(sorted_mol.ao_loc)
    bas_cache = (coords, coeffs, exponents, ao_loc, nprims, angs)
    return bas_cache

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

    # Pre-calculate sizes to avoid intermediate copies
    nbas = _bas.shape[0]
    pattern_counts = defaultdict(int)
    pattern_data = defaultdict(list)
    
    # First pass: count basis functions by pattern and collect data
    for i in range(nbas):
        nprim = _bas[i, gto.NPRIM_OF]
        nctr = _bas[i, gto.NCTR_OF]
        ang = _bas[i, gto.ANG_OF]
        pattern = (ang, nprim)
        pattern_counts[pattern] += nctr
        
        iatm = _bas[i, gto.ATOM_OF]
        coord_ptr = _atm[iatm, gto.PTR_COORD]
        exp_ptr = _bas[i, gto.PTR_EXP]
        coeff_ptr = _bas[i, gto.PTR_COEFF]
        
        pattern_data[pattern].append({
            'coord_ptr': coord_ptr,
            'exp_ptr': exp_ptr,
            'coeff_ptr': coeff_ptr,
            'nprim': nprim,
            'nctr': nctr,
            'ang': ang,
            'bas_id': i
        })

    # Pre-allocate arrays for each pattern
    coords_by_pattern = {}
    ce_by_pattern = {}
    bas_id_by_pattern = {}
    pad_id_by_pattern = {}
    
    for pattern, count in pattern_counts.items():
        ang, nprim = pattern
        padded_count = count + ((alignment - count % alignment) % alignment)
        
        # Pre-allocate final arrays
        coords_by_pattern[pattern] = np.empty((padded_count, 4), dtype=np.float64)
        ce_by_pattern[pattern] = np.empty((padded_count, 2*NPRIM_MAX), dtype=dtype)
        bas_id_by_pattern[pattern] = np.empty(padded_count, dtype=np.int32)
        pad_id_by_pattern[pattern] = np.empty(padded_count, dtype=bool)
        
        # Fill arrays without intermediate copies
        idx = 0
        for data in pattern_data[pattern]:
            coord_ptr = data['coord_ptr']
            exp_ptr = data['exp_ptr']
            coeff_ptr = data['coeff_ptr']
            nprim = data['nprim']
            nctr = data['nctr']
            ang = data['ang']
            bas_id = data['bas_id']
            
            # Get coefficients and exponents
            coeffs = _env[coeff_ptr:coeff_ptr+nprim*nctr]
            exps = _env[exp_ptr:exp_ptr+nprim*nctr]
            
            # Apply normalization factor
            if ang < 2:
                fac = ((2*ang + 1) / (4.0 * np.pi))**.5
                coeffs = coeffs * fac
            
            # Get coordinates
            coord = np.zeros(4, dtype=np.float64)
            coord[:3] = _env[coord_ptr:coord_ptr+3]
            
            # Fill arrays directly
            for j in range(nctr):
                coords_by_pattern[pattern][idx] = coord
                ce_start = j * nprim
                ce_end = (j + 1) * nprim
                ce_by_pattern[pattern][idx, 0:2*nprim:2] = coeffs[ce_start:ce_end]
                ce_by_pattern[pattern][idx, 1:2*nprim:2] = exps[ce_start:ce_end]
                bas_id_by_pattern[pattern][idx] = bas_id
                pad_id_by_pattern[pattern][idx] = False
                idx += 1
        
        # Fill padding with first element to avoid additional memory allocation
        if idx < padded_count:
            coords_by_pattern[pattern][idx:] = coords_by_pattern[pattern][0]
            ce_by_pattern[pattern][idx:] = ce_by_pattern[pattern][0]
            bas_id_by_pattern[pattern][idx:] = bas_id_by_pattern[pattern][0]
            pad_id_by_pattern[pattern][idx:] = True

    # Sort the basis by angular momentum and number of primitives
    # Reverse the order of primitives, to be consistent with GPU4PySCF
    sorted_keys = sorted(coords_by_pattern.keys(), key=lambda x: (x[0], -x[1]))
    
    # Calculate total size for pre-allocation
    total_count = sum(len(bas_id_by_pattern[key]) for key in sorted_keys)
    
    # Pre-allocate final arrays
    ce = np.empty((total_count, 2*NPRIM_MAX), dtype=dtype)
    coords = np.empty((total_count, 4), dtype=np.float64)
    bas_id = np.empty(total_count, dtype=np.int32)
    pad_id = np.empty(total_count, dtype=bool)
    angs = np.empty(total_count, dtype=np.int32)
    nprims = np.empty(total_count, dtype=np.int32)
    
    group_key = []
    group_offset = []
    offset = 0
    
    # Fill arrays directly without concatenation
    for key in sorted_keys:
        pattern_ce = ce_by_pattern[key]
        pattern_coords = coords_by_pattern[key]
        pattern_bas_id = bas_id_by_pattern[key]
        pattern_pad_id = pad_id_by_pattern[key]
        bas_count = len(pattern_bas_id)
        
        # Copy data directly to final arrays
        ce[offset:offset+bas_count] = pattern_ce
        coords[offset:offset+bas_count] = pattern_coords
        bas_id[offset:offset+bas_count] = pattern_bas_id
        pad_id[offset:offset+bas_count] = pattern_pad_id
        angs[offset:offset+bas_count] = key[0]
        nprims[offset:offset+bas_count] = key[1]
        
        group_key.append([key[0], key[1]])
        group_offset.append(offset)
        offset += bas_count
    group_offset.append(offset)
    
    # Convert to specified dtype
    coords = coords.astype(dtype, copy=False)

    ce = cp.asarray(ce, order='C')
    coords = cp.asarray(coords, order='C')

    # Store info at basis level
    bas_info = (ce, coords, angs, nprims)
    
    '''
    group_size = 25600
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
    return bas_info, bas_id, pad_id, (group_key, group_offset)

def compute_q_matrix(mol):
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
    intor = mol._add_suffix('int2e')
    _vhf.libcvhf.CVHFnr_int2e_q_cond(
        getattr(_vhf.libcvhf, intor), lib.c_null_ptr(),
        q_matrix.ctypes, ao_loc.ctypes,
        mol._atm.ctypes, ctypes.c_int(mol.natm),
        mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
    q_matrix = np.log(q_matrix + 1e-300).astype(np.float32)
    return q_matrix

def cluster_into_tile(coords, tile=4):
    """
    Greedy heuristic of grouping coords into tiles 
    """
    from scipy.spatial.distance import pdist, squareform
    coords = np.asarray(coords)

    distance_matrix = squareform(pdist(coords, metric='euclidean')) 
    n = distance_matrix.shape[0]
    unassigned = set(range(n))
    clusters = []

    while unassigned:
        i = unassigned.pop()
        # find nearest 3 among remaining
        dists = [(j, distance_matrix[i, j]) for j in unassigned]
        nearest = sorted(dists, key=lambda x: x[1])[:tile-1]
        cluster = [i] + [j for j, _ in nearest]
        for j, _ in nearest:
            unassigned.remove(j)
        clusters += cluster
    return clusters

if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 1'
    mol.basis = 'cc-pvdz'
    mol.build()

    sort_group_basis(mol)
