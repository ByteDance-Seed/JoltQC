#!/usr/bin/env python3
"""
Example: Patch gpu4pyscf ECP integrals to use JoltQC's CUDA implementation.

This script demonstrates how to monkey‑patch the ECP functions exposed by
gpu4pyscf.gto.ecp with the JoltQC equivalents from jqc.backend.ecp.

It compares norms before and after patching for a small Na2 system.

Requirements:
- PySCF + gpu4pyscf installed and importable
- A CUDA GPU available (for CuPy)

Run:
  python examples/01-patch_gpu4pyscf_ecp.py
"""

import cupy as cp
import numpy as np
from pyscf import gto

# JoltQC ECP implementations
from jqc.backend.ecp import get_ecp as jqc_get_ecp
from jqc.backend.ecp import get_ecp_ip as jqc_get_ecp_ip
from jqc.backend.ecp import get_ecp_ipip as jqc_get_ecp_ipip


def build_na2_mol(cart=True):
    """Construct a tiny Na2 molecule with an ECP and a minimal H-like basis."""
    cu1_basis = gto.basis.parse(
        """
        H    S
               1.8000000              1.0000000
        H    S
               2.8000000              0.0210870             -0.0045400              0.0000000
               1.3190000              0.3461290             -0.1703520              0.0000000
               0.9059000              0.0393780              0.1403820              1.0000000
        """
    )

    ecp_basis = gto.basis.parse_ecp(
        """
        Na nelec 10
        Na ul
        2       1.0                   0.5
        """
    )

    return gto.M(
        atom="Na 0 0 0; Na 0 0 1.7",
        basis=cu1_basis,
        ecp=ecp_basis,
        cart=1 if cart else 0,
        output="/dev/null",
        verbose=0,
    )


def main():
    # Ensure GPU context
    cp.cuda.Device().use()

    # Import gpu4pyscf ECP module
    try:
        from gpu4pyscf.gto import ecp as gpu_ecp
    except Exception as e:
        raise RuntimeError("gpu4pyscf is not available on PYTHONPATH") from e

    mol = build_na2_mol(cart=True)

    # Grab originals to allow comparison and restore
    orig_get_ecp = gpu_ecp.get_ecp
    orig_get_ecp_ip = gpu_ecp.get_ecp_ip
    orig_get_ecp_ipip = gpu_ecp.get_ecp_ipip

    # Compute baseline (gpu4pyscf original)
    h_ecp_gpu4 = orig_get_ecp(mol).get()
    ip_ecp_gpu4 = orig_get_ecp_ip(mol).get()
    ipipv_ecp_gpu4 = orig_get_ecp_ipip(mol, "ipipv").get()
    ipvip_ecp_gpu4 = orig_get_ecp_ipip(mol, "ipvip").get()

    print("Before patching gpu4pyscf:")
    print(f"  get_ecp   norm: {np.linalg.norm(h_ecp_gpu4):.6e}")
    print(f"  get_ecp_ip   norm: {np.linalg.norm(ip_ecp_gpu4):.6e}")
    print(f"  get_ecp_ipip[ipipv] norm: {np.linalg.norm(ipipv_ecp_gpu4):.6e}")
    print(f"  get_ecp_ipip[ipvip] norm: {np.linalg.norm(ipvip_ecp_gpu4):.6e}")

    # Patch module functions to use JoltQC implementations
    def patched_get_ecp(xmol):
        return jqc_get_ecp(xmol, precision="fp64")

    def patched_get_ecp_ip(xmol, ip_type="ip", ecp_atoms=None):
        return jqc_get_ecp_ip(xmol, ip_type=ip_type, ecp_atoms=ecp_atoms, precision="fp64")

    def patched_get_ecp_ipip(xmol, ip_type="ipipv", ecp_atoms=None):
        return jqc_get_ecp_ipip(xmol, ip_type=ip_type, ecp_atoms=ecp_atoms, precision="fp64")

    gpu_ecp.get_ecp = patched_get_ecp
    gpu_ecp.get_ecp_ip = patched_get_ecp_ip
    gpu_ecp.get_ecp_ipip = patched_get_ecp_ipip

    # Evaluate again with patched functions
    h_ecp_patched = gpu_ecp.get_ecp(mol).get()
    ip_ecp_patched = gpu_ecp.get_ecp_ip(mol).get()
    ipipv_ecp_patched = gpu_ecp.get_ecp_ipip(mol, "ipipv").get()
    ipvip_ecp_patched = gpu_ecp.get_ecp_ipip(mol, "ipvip").get()

    print("\nAfter patching to JoltQC:")
    print(f"  get_ecp   norm: {np.linalg.norm(h_ecp_patched):.6e}")
    print(f"  get_ecp_ip   norm: {np.linalg.norm(ip_ecp_patched):.6e}")
    print(f"  get_ecp_ipip[ipipv] norm: {np.linalg.norm(ipipv_ecp_patched):.6e}")
    print(f"  get_ecp_ipip[ipvip] norm: {np.linalg.norm(ipvip_ecp_patched):.6e}")

    # Differences vs original
    print("\nDifferences (patched - original):")
    print(f"  get_ecp   diff: {np.linalg.norm(h_ecp_patched - h_ecp_gpu4):.6e}")
    print(f"  get_ecp_ip   diff: {np.linalg.norm(ip_ecp_patched - ip_ecp_gpu4):.6e}")
    print(
        f"  get_ecp_ipip[ipipv] diff: {np.linalg.norm(ipipv_ecp_patched - ipipv_ecp_gpu4):.6e}"
    )
    print(
        f"  get_ecp_ipip[ipvip] diff: {np.linalg.norm(ipvip_ecp_patched - ipvip_ecp_gpu4):.6e}"
    )

    # Restore originals if desired
    gpu_ecp.get_ecp = orig_get_ecp
    gpu_ecp.get_ecp_ip = orig_get_ecp_ip
    gpu_ecp.get_ecp_ipip = orig_get_ecp_ipip


if __name__ == "__main__":
    main()

