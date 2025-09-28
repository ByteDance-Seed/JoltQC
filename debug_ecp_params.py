#!/usr/bin/env python3
"""
Debug script to trace exact parameters passed to ECP IP kernel compilation
to identify differences between working isolated test and failing real get_ecp_ip.
"""

import sys
import cupy as cp
import numpy as np
from pyscf import gto

sys.path.insert(0, '/home/xiaojie/Documents/JoltQC')

def patch_kernel_compilation():
    """Patch the kernel compilation to add debug output."""
    from jqc.backend import ecp

    # Store original functions
    original_compile_type2 = ecp._compile_ecp_type2_ip_kernel
    original_compile_type1 = ecp._compile_ecp_type1_ip_kernel

    def debug_compile_type2(li, lj, lc, precision='fp64'):
        print(f"🔍 TYPE2 IP KERNEL COMPILATION CALLED:")
        print(f"  li={li}, lj={lj}, lc={lc}, precision='{precision}'")

        # Estimate shared memory
        shared_mem = ecp._estimate_type2_ip_shared_memory(li, lj, lc, precision)
        print(f"  Estimated shared memory: {shared_mem} bytes ({shared_mem/1024:.1f} KB)")

        try:
            result = original_compile_type2(li, lj, lc, precision)
            print(f"  ✓ Compilation successful")
            return result
        except Exception as e:
            print(f"  ✗ Compilation failed: {e}")
            raise

    def debug_compile_type1(li, lj, precision='fp64'):
        print(f"🔍 TYPE1 IP KERNEL COMPILATION CALLED:")
        print(f"  li={li}, lj={lj}, precision='{precision}'")

        # Estimate shared memory
        shared_mem = ecp._estimate_type1_ip_shared_memory(li, lj, precision)
        print(f"  Estimated shared memory: {shared_mem} bytes ({shared_mem/1024:.1f} KB)")

        try:
            result = original_compile_type1(li, lj, precision)
            print(f"  ✓ Compilation successful")
            return result
        except Exception as e:
            print(f"  ✗ Compilation failed: {e}")
            raise

    # Patch the functions
    ecp._compile_ecp_type2_ip_kernel = debug_compile_type2
    ecp._compile_ecp_type1_ip_kernel = debug_compile_type1
    return original_compile_type2, original_compile_type1

def test_real_get_ecp_ip():
    """Test the real get_ecp_ip with debug tracing."""
    print("=== Testing Real get_ecp_ip with Debug Tracing ===")

    # Patch compilation for debugging
    original_compile_type2, original_compile_type1 = patch_kernel_compilation()

    try:
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

        print(f"Molecule info:")
        print(f"  nao: {mol.nao_nr()}")
        print(f"  natm: {mol.natm}")
        print(f"  ECP shells: {len(mol._ecpbas)}")
        print(f"  ECP basis shape: {mol._ecpbas.shape}")

        # Check what lk values we have
        uniq_lecp = set()
        for ksh in range(len(mol._ecpbas)):
            lk = mol._ecpbas[ksh, 1]  # Angular momentum of ECP shell
            uniq_lecp.add(lk)

        print(f"  Unique ECP lk values: {sorted(uniq_lecp)}")

        print("\nCalling get_ecp_ip...")
        result = get_ecp_ip(mol)
        print("✓ get_ecp_ip successful!")
        print(f"  Result shape: {result.shape}")
        print(f"  Result norm: {float(result.norm()):.6f}")

    except Exception as e:
        print(f"✗ get_ecp_ip failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Restore original functions
        import jqc.backend.ecp
        jqc.backend.ecp._compile_ecp_type2_ip_kernel = original_compile_type2
        jqc.backend.ecp._compile_ecp_type1_ip_kernel = original_compile_type1

def test_isolated_compilation():
    """Test isolated kernel compilation for comparison."""
    print("\n=== Testing Isolated Compilation ===")

    from jqc.backend.ecp import _compile_ecp_type2_ip_kernel
    import jqc.backend.ecp

    # Clear cache
    jqc.backend.ecp._ecp_kernel_cache.clear()

    # Test cases that might be used in real molecule
    test_cases = [
        (0, 0, 0), (0, 0, 1), (0, 0, 2),  # s orbital with different ECP l
        (1, 0, 0), (1, 0, 1), (1, 0, 2),  # p orbital with different ECP l
        (0, 1, 0), (0, 1, 1), (0, 1, 2),  # s-p combinations
        (1, 1, 0), (1, 1, 1), (1, 1, 2),  # p-p combinations
        (2, 2, 0), (2, 2, 1), (2, 2, 2),  # d-d combinations
    ]

    for li, lj, lc in test_cases:
        try:
            shared_mem = jqc.backend.ecp._estimate_type2_ip_shared_memory(li, lj, lc, 'fp64')
            print(f"(li={li}, lj={lj}, lc={lc}): {shared_mem} bytes ({shared_mem/1024:.1f} KB)", end=" ")

            kernel = _compile_ecp_type2_ip_kernel(li, lj, lc, 'fp64')
            print("✓")
        except Exception as e:
            print(f"✗ {e}")

def main():
    print("🔧 ECP IP Parameter Debug")
    print("=" * 50)

    # Test isolated compilation first
    test_isolated_compilation()

    # Test real get_ecp_ip with tracing
    test_real_get_ecp_ip()

if __name__ == "__main__":
    main()