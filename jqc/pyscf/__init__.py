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

def compile(obj, cutoff_fp32=1e-13, cutoff_fp64=1e-13):
    '''
    Compile kernels and assign them to the corresponding PySCF Object.
    If no JIT kernel is found, return the unchanged object

    Note: this is a in-place operation.

    Parameters
    ----------
    obj : PySCF Object
        PySCF Object to be compiled.
    cutoff_fp32 : float, optional
            Cutoff for single precision. The default is 1e-13.
    cutoff_fp64 : float, optional
            Cutoff for double precision. The default is 1e-13.

    Returns
    -------
    obj : PySCF Object
        PySCF Object with compiled kernels.

    '''
    
    if 'gpu4pyscf' not in obj.__class__.__module__:
        obj = obj.to_gpu()

    if not obj.istype('RHF'):
        return obj

    assert hasattr(obj, 'mol')
    mol = obj.mol

    if obj.istype('RKS'):
        get_rho = rks.generate_get_rho(mol, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64)
        obj._numint.get_rho = get_rho

        obj.grids.build = MethodType(build_grids, obj.grids)
        nr_rks = rks.generate_nr_rks(mol, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64)
        obj._numint.nr_rks = MethodType(nr_rks, obj._numint)

    if not obj.istype('DFRHF'):
        # TODO: cache intermediate variables
        if hasattr(obj, 'get_jk'):
            get_jk = jk.generate_jk_kernel(mol, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64)
            obj.get_jk = get_jk
        
        if hasattr(obj, 'get_j'):
            get_j = jk.generate_get_j(mol, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64)
            obj.get_j = get_j

        if hasattr(obj, 'get_k'):
            get_k = jk.generate_get_k(mol, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64)
            obj.get_k = get_k
        
        if obj.istype('RHF'):
            get_veff = jk.generate_get_veff(mol)
            obj.get_veff = MethodType(get_veff, obj)
        
        if obj.istype('RKS'):
            get_veff = rks.generate_get_veff(mol)
            obj.get_veff = MethodType(get_veff, obj)
    return obj
