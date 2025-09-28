#!/usr/bin/env python3

import sys
import cupy as cp
import numpy as np
from pyscf import gto

sys.path.insert(0, '/home/xiaojie/Documents/JoltQC')

def test_compilation_only():
    """Test that IP kernel compilation works in isolation."""
    print("=== Testing IP Kernel Compilation ===")

    from jqc.backend.ecp import _compile_ecp_type2_ip_kernel
    import jqc.backend.ecp

    # Clear any cached kernels
    jqc.backend.ecp._ecp_kernel_cache.clear()

    test_cases = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]

    for li, lj, lc in test_cases:
        try:
            print(f"Compiling (li={li}, lj={lj}, lc={lc})...")
            kernel = _compile_ecp_type2_ip_kernel(li, lj, lc, 'fp64')
            print(f"  ✓ Success")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return False

    return True

def test_simple_execution():
    """Test minimal kernel execution."""
    print("\n=== Testing Simple Kernel Execution ===")

    from jqc.backend.ecp import _compile_ecp_type2_ip_kernel
    from jqc.pyscf.basis import BasisLayout

    # Create simple molecule
    mol = gto.M(
        atom="Cu 0 0 0",
        basis="sto-3g",
        ecp="crenbl",
        spin=1,
        charge=0,
        cart=1,
        output="/dev/null",
        verbose=0,
    )

    try:
        # Create basis layout
        basis_layout = BasisLayout.from_mol(mol)
        print("✓ Basis layout created")

        # Prepare parameters
        nao = mol.nao_nr()
        n_ecp_atoms = 1

        # Create output buffer
        gctr = cp.zeros((n_ecp_atoms, 3, nao, nao), dtype=cp.float64)

        # Compile kernel for simplest case
        kernel_func = _compile_ecp_type2_ip_kernel(0, 0, 0, 'fp64')
        print("✓ Kernel compilation successful")

        # Minimal execution parameters
        ao_loc = cp.array(mol.ao_loc_nr(), dtype=cp.int32)
        tasks = cp.array([[0, 0, 4]], dtype=cp.int32)  # minimal task
        ntasks = 1
        ecpbas = cp.array(mol._ecpbas, dtype=cp.int32)
        ecploc = cp.arange(len(mol._ecpbas), dtype=cp.int32)
        atm = cp.array(mol._atm, dtype=cp.int32)
        env = cp.array(mol._env, dtype=cp.float64)
        npi = npj = 1

        print("Attempting kernel execution...")

        # Execute kernel
        kernel_func(
            gctr.ravel(),
            ao_loc, nao, tasks.ravel(), ntasks,
            ecpbas.ravel(), ecploc,
            basis_layout.coords, basis_layout.ce,
            atm.ravel(), env, npi, npj
        )

        # Synchronize
        cp.cuda.Device().synchronize()
        print("✓ Kernel execution successful")

        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_get_ecp_ip():
    """Test the full get_ecp_ip function."""
    print("\n=== Testing Full get_ecp_ip Function ===")

    from jqc.backend.ecp import get_ecp_ip

    mol = gto.M(
        atom="Cu 0 0 0",
        basis="sto-3g",
        ecp="crenbl",
        spin=1,
        charge=0,
        cart=1,
        output="/dev/null",
        verbose=0,
    )

    try:
        result = get_ecp_ip(mol)
        print("✓ get_ecp_ip successful")
        print(f"  Result shape: {result.shape}")
        print(f"  Result norm: {float(result.norm()):.6f}")
        return True

    except Exception as e:
        print(f"✗ get_ecp_ip failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔧 ECP IP Kernel Isolated Debug")
    print("=" * 40)

    # Test compilation
    if not test_compilation_only():
        print("❌ Compilation test failed, stopping.")
        return 1

    # Test simple execution
    if not test_simple_execution():
        print("❌ Simple execution test failed, stopping.")
        return 2

    # Test full function
    if not test_full_get_ecp_ip():
        print("❌ Full function test failed.")
        return 3

    print("\n🎉 All tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())