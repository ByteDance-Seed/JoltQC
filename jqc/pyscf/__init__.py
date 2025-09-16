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
from jqc.constants import TILE


def apply(obj, cutoff_fp32=None, cutoff_fp64=None):
    """
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

    """
    # TODO: check supported object

    if "gpu4pyscf" not in obj.__class__.__module__:
        obj = obj.to_gpu()

    if cutoff_fp32 is None:
        cutoff_fp32 = obj.direct_scf_tol

    if cutoff_fp64 is None:
        cutoff_fp64 = obj.direct_scf_tol

    if not obj.istype("RHF"):
        return obj

    assert hasattr(obj, "mol")
    mol = obj.mol

    # Lazy imports to avoid initializing CUDA at import time
    from jqc.pyscf.basis import BasisLayout
    from jqc.pyscf import rks as _rks
    from jqc.pyscf import jk as _jk
    from jqc.pyscf.rks import build_grids

    # Generate basis layouts once and reuse them
    layout_rks = BasisLayout.from_mol(mol, alignment=1)
    layout_jk = BasisLayout.from_mol(mol, alignment=TILE)

    if obj.istype("RKS"):
        get_rho = _rks.generate_get_rho(
            layout_rks, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64
        )
        obj._numint.get_rho = get_rho

        obj.grids.build = MethodType(build_grids, obj.grids)
        nr_rks = _rks.generate_nr_rks(
            layout_rks, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64
        )
        obj._numint.nr_rks = MethodType(nr_rks, obj._numint)

        nr_nlc_vxc = _rks.generate_nr_nlc_vxc(
            layout_rks, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64
        )
        obj._numint.nr_nlc_vxc = MethodType(nr_nlc_vxc, obj._numint)

    if not obj.istype("DFRHF"):
        # TODO: cache intermediate variables
        if hasattr(obj, "get_jk"):
            get_jk = _jk.generate_jk_kernel(
                layout_jk, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64
            )
            obj.get_jk = get_jk

        if hasattr(obj, "get_j"):
            get_j = _jk.generate_get_j(
                layout_jk, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64
            )
            obj.get_j = get_j

        if hasattr(obj, "get_k"):
            get_k = _jk.generate_get_k(
                layout_jk, cutoff_fp32=cutoff_fp32, cutoff_fp64=cutoff_fp64
            )
            obj.get_k = get_k

        if obj.istype("RHF"):
            get_veff = _jk.generate_get_veff()
            obj.get_veff = MethodType(get_veff, obj)

        if obj.istype("RKS"):
            get_veff = _rks.generate_get_veff()
            obj.get_veff = MethodType(get_veff, obj)
    return obj
