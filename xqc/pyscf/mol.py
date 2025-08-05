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

import numpy as np
import cupy as cp
from pyscf import gto

__all__ = ['format_bas_cache']

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
    coords = cp.empty([nbas, 3], dtype=dtype, order='C')
    coords[:,0] = cp.asarray(_env[coord_ptr], dtype=dtype)
    coords[:,1] = cp.asarray(_env[coord_ptr+1], dtype=dtype)
    coords[:,2] = cp.asarray(_env[coord_ptr+2], dtype=dtype)
    ao_loc = cp.array(sorted_mol.ao_loc)
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
        nprims[i] = nprim
        angs[i] = ang
    coeffs = cp.asarray(coeffs)
    exponents = cp.asarray(exponents)
    nprims = cp.asarray(nprims)
    angs = cp.asarray(angs)
    bas_cache = (coords, coeffs, exponents, ao_loc, nprims, angs)
    return bas_cache
