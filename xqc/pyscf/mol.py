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

import ctypes
from collections import defaultdict
import numpy as np
import cupy as cp
from pyscf import gto, lib
from pyscf.scf import _vhf

__all__ = ['format_bas_cache', 'create_sorted_basis']

NPRIM_MAX = 16
PTR_BAS_COORD = 7


def format_bas_cache(sorted_mol, dtype=np.float64):
    """
    Format the basis cache used in xQC.
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

def create_sorted_basis(mol, alignment=4, dtype=np.float64):
    _bas = mol._bas
    _env = mol._env
    _atm = mol._atm

    coords_by_pattern = defaultdict(list)
    exponents_by_pattern = defaultdict(list)
    coeffs_by_pattern = defaultdict(list)
    bas_id_by_pattern = defaultdict(list)
    pad_id_by_pattern = defaultdict(list)

    # Extract basis information from PySCF format
    nbas = _bas.shape[0]
    for i in range(nbas):
        nprim = _bas[i, gto.NPRIM_OF]
        nctr = _bas[i, gto.NCTR_OF]
        ang = _bas[i, gto.ANG_OF]

        iatm = _bas[i, gto.ATOM_OF]
        coord_ptr = _atm[iatm, gto.PTR_COORD]

        exp_ptr = _bas[i, gto.PTR_EXP]
        coeff_ptr = _bas[i, gto.PTR_COEFF]
        coeff = _env[coeff_ptr:coeff_ptr+nprim*nctr].copy()
        exp = _env[exp_ptr:exp_ptr+nprim*nctr]
        if ang < 2:
            fac = ((2*ang + 1) / (4.0 * np.pi))**.5
            coeff *= fac
        coord = _env[coord_ptr:coord_ptr+3]
        coeff = coeff.reshape(nctr, nprim)
        exp = exp.reshape(nctr, nprim)
        coeff = np.concatenate((coeff, np.zeros((nctr, NPRIM_MAX-nprim))), axis=1)
        exp = np.concatenate((exp, np.zeros((nctr, NPRIM_MAX-nprim))), axis=1)

        coords_by_pattern[ang, nprim].append(np.tile(coord, (nctr, 1)))
        exponents_by_pattern[ang, nprim].append(exp)
        coeffs_by_pattern[ang, nprim].append(coeff)
        bas_id_by_pattern[ang, nprim].append(np.ones(nctr, dtype=np.int32) * i)

    # Pad the arrays for basis information
    for key in coords_by_pattern:
        nbas = sum([len(x) for x in coords_by_pattern[key]])
        pad = (alignment - nbas % alignment) % alignment
        exp = exponents_by_pattern[key]
        coeff = coeffs_by_pattern[key]
        coord = coords_by_pattern[key]
        bas_id = bas_id_by_pattern[key]
        
        # Pad the arrays with first basis in the group
        exp.append(np.tile(exp[0], (pad, 1)))
        coeff.append(np.tile(coeff[0], (pad, 1)))
        coord.append(np.tile(coord[0], (pad, 1)))
        bas_id.append(np.ones(pad, dtype=np.int32) * bas_id[0])
        
        exponents_by_pattern[key] = np.concatenate(exp, axis=0)
        coeffs_by_pattern[key] = np.concatenate(coeff, axis=0)
        bas_id_by_pattern[key] = np.concatenate(bas_id, axis=0)
        coords_by_pattern[key] = np.concatenate(coord, axis=0)

        active_id = np.zeros(nbas, dtype=np.bool)
        pad_id = np.ones(pad, dtype=np.bool)
        pad_id_by_pattern[key] = np.concatenate([active_id, pad_id])

    # Sort the basis by angular momentum and number of primitives
    # Reverse the order of primitives, to be consistent with GPU4PySCF
    sorted_keys = sorted(coords_by_pattern.keys(), key=lambda x: (x[0], -x[1]))
    exponents = []
    coeffs = []
    coords = []
    bas_id = []
    pad_id = []
    angs = []
    nprims = []
    group_key = []
    group_offset = []
    offset = 0
    for key in sorted_keys:
        exponents.append(exponents_by_pattern[key])
        coeffs.append(coeffs_by_pattern[key])
        coords.append(coords_by_pattern[key])
        bas_id.append(bas_id_by_pattern[key])
        pad_id.append(pad_id_by_pattern[key])
        bas_count = len(bas_id_by_pattern[key])
        angs.append(np.full(bas_count, key[0], dtype=np.int32))
        nprims.append(np.full(bas_count, key[1], dtype=np.int32))
        group_key.append([key[0], key[1]])
        group_offset.append(offset)
        offset += bas_count
    group_offset.append(offset)

    exponents = np.concatenate(exponents, axis=0, dtype=dtype)
    coeffs = np.concatenate(coeffs, axis=0, dtype=dtype)
    coords = np.concatenate(coords, axis=0, dtype=dtype)
    bas_id = np.concatenate(bas_id, axis=0, dtype=np.int32)
    pad_id = np.concatenate(pad_id, axis=0, dtype=np.bool)
    angs = np.concatenate(angs, axis=0)
    nprims = np.concatenate(nprims, axis=0)

    exponents = cp.asarray(exponents)
    coeffs = cp.asarray(coeffs)
    coords = cp.asarray(coords)

    # Store info at basis level
    bas_cache = (coeffs, exponents, coords, angs, nprims)
    
    group_key = np.asarray(group_key)
    group_offset = np.asarray(group_offset)
    
    return bas_cache, bas_id, pad_id, (group_key, group_offset)

def compute_q_matrix(mol):
    """ 
    Compute the Q matrix for the given molecule.

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


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 1'
    mol.basis = 'cc-pvdz'
    mol.build()

    create_sorted_basis(mol)
