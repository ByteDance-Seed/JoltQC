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

from types import MethodType
from jqc.pyscf import rks, jk
from jqc.pyscf.rks import build_grids
from jqc.pyscf.mol import BasisLayout
from jqc.constants import TILE

def _generate_basis_layouts(mol):
    """
    Generate basis layouts for both RKS (alignment=1) and JK (alignment=TILE) operations.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecular structure
        
    Returns
    -------
    layout_rks : BasisLayout
        Basis layout for RKS operations (alignment=1)
    layout_jk : BasisLayout  
        Basis layout for JK operations (alignment=TILE)
    """
    layout_rks = BasisLayout.from_sort_group_basis(mol, alignment=1)
    layout_jk = BasisLayout.from_sort_group_basis(mol, alignment=TILE)
    return layout_rks, layout_jk

def apply(obj, cutoff_fp32=None, cutoff_fp64=None):
    '''
    Apply JIT kernels to the corresponding PySCF Object.
    If no JIT kernel is found, return the unchanged object

    Note: this is a in-place operation.

    Parameters
    ----------
    obj : PySCF Object
        PySCF Object to apply JIT kernels to.
    cutoff_fp32 : float, optional
            Cutoff for single precision. The default is obj.direct_scf_tol.
    cutoff_fp64 : float, optional
            Cutoff for double precision. The default is obj.direct_scf_tol.

    Returns
    -------
    obj : PySCF Object
        PySCF Object with JIT kernels applied.

    '''
    # TODO: check supported object
    
    if 'gpu4pyscf' not in obj.__class__.__module__:
        obj = obj.to_gpu()

    if cutoff_fp32 is None:
        cutoff_fp32 = obj.direct_scf_tol
    
    if cutoff_fp64 is None:
        cutoff_fp64 = obj.direct_scf_tol

    if not obj.istype('RHF'):
        return obj

    assert hasattr(obj, 'mol')
    mol = obj.mol
    
    # Generate basis layouts once and reuse them
    layout_rks, layout_jk = _generate_basis_layouts(mol)

    if obj.istype('RKS'):
        get_rho = rks.generate_get_rho(mol, layout=layout_rks, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64)
        obj._numint.get_rho = get_rho

        obj.grids.build = MethodType(build_grids, obj.grids)
        nr_rks = rks.generate_nr_rks(mol, layout=layout_rks, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64)
        obj._numint.nr_rks = MethodType(nr_rks, obj._numint)

        nr_nlc_vxc = rks.generate_nr_nlc_vxc(mol, layout=layout_rks, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64)
        obj._numint.nr_nlc_vxc = MethodType(nr_nlc_vxc, obj._numint)

    if not obj.istype('DFRHF'):
        # TODO: cache intermediate variables
        if hasattr(obj, 'get_jk'):
            get_jk = jk.generate_jk_kernel(mol, layout=layout_jk, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64)
            obj.get_jk = get_jk
        
        if hasattr(obj, 'get_j'):
            get_j = jk.generate_get_j(mol, layout=layout_jk, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64)
            obj.get_j = get_j

        if hasattr(obj, 'get_k'):
            get_k = jk.generate_get_k(mol, layout=layout_jk, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64)
            obj.get_k = get_k
        
        if obj.istype('RHF'):
            get_veff = jk.generate_get_veff(mol, layout=layout_jk)
            obj.get_veff = MethodType(get_veff, obj)
        
        if obj.istype('RKS'):
            get_veff = rks.generate_get_veff(mol, layout=layout_rks)
            obj.get_veff = MethodType(get_veff, obj)
    return obj
