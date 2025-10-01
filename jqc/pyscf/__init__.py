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
from typing import Dict, Optional, Any
from functools import wraps

__all__ = ["apply", "reset", "get_default_config"]


def create_reset_function(original_reset, config):
    """
    Create a reset function that captures the original reset method.

    Parameters
    ----------
    original_reset : method
        The original reset method to capture.
    config : dict
        Configuration dictionary to pass to apply().

    Returns
    -------
    function
        A new reset function that uses the captured original method.
    """
    @wraps(original_reset)
    def reset(self, mol=None):
        """
        Reset the object with a new molecule and reapply JIT kernels.

        Parameters
        ----------
        obj : PySCF Object
            PySCF Object to reset.
        mol : PySCF Molecule
            New molecule to use for the object.

        Returns
        -------
        obj : PySCF Object
            PySCF Object with new molecule and JIT kernels reapplied.
        """
        # Call the captured original reset method
        mf = original_reset(self, mol)
        return apply(mf, config)

    return reset


def create_scanner_wrapper(original_as_scanner, config):
    """
    Create an as_scanner wrapper that ensures scanners have proper reset functionality.

    Parameters
    ----------
    original_as_scanner : method
        The original as_scanner method to capture.
    config : dict
        Configuration dictionary to pass to apply().

    Returns
    -------
    function
        A new as_scanner function that creates scanners with JoltQC reset functionality.
    """
    @wraps(original_as_scanner)
    def as_scanner(self, **kwargs):
        """
        Create a scanner with JoltQC reset functionality.
        """
        # Create the scanner using the original method
        scanner = original_as_scanner(self, **kwargs)

        # Mark scanner as having JoltQC applied
        scanner._joltqc_applied = True

        # Override the scanner's reset method if it exists
        if hasattr(scanner, 'reset'):
            original_scanner_reset = scanner.reset.__func__
            custom_scanner_reset = create_reset_function(original_scanner_reset, config)
            scanner.reset = MethodType(custom_scanner_reset, scanner)
        return scanner

    return as_scanner


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration for JIT kernel application.

    Returns
    -------
    dict
        Default configuration with separate cutoffs for JK and DFT operations.
    """
    return {
        "jk": {
            "cutoff_fp32": None,  # Will use obj.direct_scf_tol if None
            "cutoff_fp64": None,  # Will use obj.direct_scf_tol if None
        },
        "dft": {
            "cutoff_fp32": 1e-13,  # Default for DFT operations
            "cutoff_fp64": 1e-6,  # Default for DFT operations
        },
    }


def apply(obj, config: Optional[Dict[str, Any]] = None):
    """
    Apply JIT kernels to the corresponding PySCF Object.
    If no JIT kernel is found, return the unchanged object

    Note: this is a in-place operation.

    Parameters
    ----------
    obj : PySCF Object
        PySCF Object to apply JIT kernels to.
    config : dict, optional
        Configuration dictionary with separate cutoffs for JK and DFT operations.
        If None, uses default configuration. Structure:
        {
            "jk": {"cutoff_fp32": float, "cutoff_fp64": float},
            "dft": {"cutoff_fp32": float, "cutoff_fp64": float}
        }

    Returns
    -------
    obj : PySCF Object
        PySCF Object with JIT kernels applied.

    """
    # TODO: check supported object

    if "gpu4pyscf" not in obj.__class__.__module__:
        obj = obj.to_gpu()

    if config is None:
        config = get_default_config()
    # Extract cutoffs for JK operations
    jk_cutoff_fp32 = config.get("jk", {}).get("cutoff_fp32")
    jk_cutoff_fp64 = config.get("jk", {}).get("cutoff_fp64")

    # Extract cutoffs for DFT operations
    dft_cutoff_fp32 = config.get("dft", {}).get("cutoff_fp32")
    dft_cutoff_fp64 = config.get("dft", {}).get("cutoff_fp64")

    # Use obj.direct_scf_tol as fallback for JK operations only
    if jk_cutoff_fp32 is None:
        jk_cutoff_fp32 = getattr(obj, 'direct_scf_tol', 1e-12)
    if jk_cutoff_fp64 is None:
        jk_cutoff_fp64 = getattr(obj, 'direct_scf_tol', 1e-12)

    # DFT cutoffs should never be None due to defaults, but add safety check
    if dft_cutoff_fp32 is None:
        dft_cutoff_fp32 = 1e-13
    if dft_cutoff_fp64 is None:
        dft_cutoff_fp64 = 1e-6

    # Check if object supports istype method and is RHF-like
    if hasattr(obj, 'istype') and not obj.istype("RHF"):
        return obj
    elif not hasattr(obj, 'istype'):
        # For objects like gradient scanners that don't have istype, we can still apply reset
        pass
    
    # Lazy imports to avoid initializing CUDA at import time
    from jqc.pyscf import jk as _jk
    from jqc.pyscf import rks as _rks
    from jqc.pyscf.rks import build_grids
    from jqc.pyscf.basis import BasisLayout
    from jqc.constants import TILE

    # Generate separate basis layouts for RKS (alignment=1) and JK (alignment=TILE=4)
    basis_layout_rks = BasisLayout.from_mol(obj.mol, alignment=1)
    basis_layout_jk = BasisLayout.from_mol(obj.mol, alignment=TILE)

    if hasattr(obj, 'istype') and obj.istype("RKS"):
        get_rho = _rks.generate_get_rho(
            basis_layout_rks, cutoff_fp32=dft_cutoff_fp32, cutoff_fp64=dft_cutoff_fp64
        )
        obj._numint.get_rho = get_rho

        obj.grids.build = MethodType(build_grids, obj.grids)
        nr_rks = _rks.generate_nr_rks(
            basis_layout_rks, cutoff_fp32=dft_cutoff_fp32, cutoff_fp64=dft_cutoff_fp64
        )
        obj._numint.nr_rks = MethodType(nr_rks, obj._numint)

        nr_nlc_vxc = _rks.generate_nr_nlc_vxc(
            basis_layout_rks, cutoff_fp32=dft_cutoff_fp32, cutoff_fp64=dft_cutoff_fp64
        )
        obj._numint.nr_nlc_vxc = MethodType(nr_nlc_vxc, obj._numint)

    # Apply JK kernels if not density fitting
    if hasattr(obj, 'istype') and not obj.istype("DFRHF") and not obj.istype("DFRKS"):
        if hasattr(obj, "get_jk"):
            get_jk = _jk.generate_jk_kernel(
                basis_layout_jk, cutoff_fp32=jk_cutoff_fp32, cutoff_fp64=jk_cutoff_fp64
            )
            obj.get_jk = get_jk

        if hasattr(obj, "get_j"):
            get_j = _jk.generate_get_j(
                basis_layout_jk, cutoff_fp32=jk_cutoff_fp32, cutoff_fp64=jk_cutoff_fp64
            )
            obj.get_j = get_j

        if hasattr(obj, "get_k"):
            get_k = _jk.generate_get_k(
                basis_layout_jk, cutoff_fp32=jk_cutoff_fp32, cutoff_fp64=jk_cutoff_fp64
            )
            obj.get_k = get_k

        if hasattr(obj, 'istype') and obj.istype("RHF"):
            get_veff = _jk.generate_get_veff()
            obj.get_veff = MethodType(get_veff, obj)

        if hasattr(obj, 'istype') and obj.istype("RKS"):
            get_veff = _rks.generate_get_veff()
            obj.get_veff = MethodType(get_veff, obj)
    
    # Mark that JoltQC has been applied to this object
    obj._joltqc_applied = True

    # Store the original reset method and create a closure-based reset function
    # Only wrap reset if we haven't already stored the original
    if not hasattr(obj, '_jqc_original_reset'):
        original_reset = obj.reset.__func__
        obj._jqc_original_reset = original_reset
        custom_reset = create_reset_function(original_reset, config)
        # Overwrite obj.reset() with our custom reset function
        obj.reset = MethodType(custom_reset, obj)
    
    # Override as_scanner method to ensure scanners have proper reset functionality
    if hasattr(obj, 'as_scanner'):
        original_as_scanner = obj.as_scanner.__func__
        custom_as_scanner = create_scanner_wrapper(original_as_scanner, config)
        obj.as_scanner = MethodType(custom_as_scanner, obj)
    
    return obj
