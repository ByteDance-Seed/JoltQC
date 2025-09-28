#!/usr/bin/env python3
"""
Minimal test for debugging type2 IP kernel issues.

This script creates a minimal, isolated test to reproduce and debug
the CUDA_ERROR_ILLEGAL_ADDRESS issue in type2 IP kernels.
"""

import sys
import os
import traceback
import cupy as cp
import numpy as np
from pyscf import gto

# Add JoltQC to path
sys.path.insert(0, '/home/xiaojie/Documents/JoltQC')

def test_minimal_molecule():
    """Create the smallest possible molecule with ECP for testing."""
    print("=== Creating minimal test molecule ===")

    mol = gto.M(
        atom="Cu 0 0 0",
        basis="sto-3g",
        ecp="crenbl",
        spin=1,  # Fix: Cu has 19 electrons after ECP, needs odd spin
        charge=0,
        cart=1,
        output="/dev/null",
        verbose=0,
    )

    print(f"✓ Molecule created:")
    print(f"  Atoms: {mol.natm}")
    print(f"  Basis functions: {mol.nbas}")
    print(f"  ECP shells: {mol._ecpbas.shape[0] if mol._ecpbas is not None else 0}")

    return mol

def test_cuda_environment():
    """Test basic CUDA environment and operations."""
    print("\n=== Testing CUDA Environment ===")

    try:
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        print(f"✓ Device: {props['name'].decode()}")

        # Test basic memory operations
        test_data = cp.array([1.0, 2.0, 3.0])
        print(f"✓ Basic memory allocation: {test_data.sum()}")

        # Test basic kernel compilation
        simple_kernel = '''
        extern "C" __global__ void add_one(double* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) data[idx] += 1.0;
        }
        '''
        mod = cp.RawModule(code=simple_kernel)
        kernel = mod.get_function('add_one')
        kernel((1,), (3,), (test_data, 3))
        cp.cuda.Device().synchronize()
        print(f"✓ Basic kernel execution: {test_data}")

        return True

    except Exception as e:
        print(f"✗ CUDA environment test failed: {e}")
        return False

def test_type2_ip_compilation_stages():
    """Test type2 IP kernel compilation in stages to isolate issues."""
    print("\n=== Testing Type2 IP Compilation Stages ===")

    from jqc.backend.ecp import _compile_ecp_type2_ip_kernel
    import jqc.backend.ecp

    # Clear any cached kernels
    jqc.backend.ecp._ecp_kernel_cache.clear()

    test_cases = [
        (0, 0, 0, "simplest case: s-s-s"),
        (1, 0, 0, "p-s-s case"),
        (0, 1, 0, "s-p-s case"),
        (1, 1, 0, "p-p-s case"),
        (1, 1, 1, "p-p-p case"),
    ]

    results = []

    for li, lj, lc, description in test_cases:
        try:
            print(f"Testing (li={li}, lj={lj}, lc={lc}) - {description}...")

            # Test compilation
            kernel_func = _compile_ecp_type2_ip_kernel(li, lj, lc, 'fp64')
            print(f"  ✓ Compilation successful")

            # Test if we can access the cached kernel
            cache_key = f"type2_ip_{li}_{lj}_{lc}_fp64"
            if cache_key in jqc.backend.ecp._ecp_kernel_cache:
                print(f"  ✓ Kernel cached successfully")

            results.append((li, lj, lc, True, "Success"))

        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ Failed: {error_msg}")

            # Categorize the error
            if "CUDA_ERROR_ILLEGAL_ADDRESS" in error_msg:
                error_type = "ILLEGAL_ADDRESS"
            elif "register" in error_msg.lower():
                error_type = "REGISTER_ISSUE"
            elif "memory" in error_msg.lower():
                error_type = "MEMORY_ISSUE"
            else:
                error_type = "OTHER"

            results.append((li, lj, lc, False, error_type))

            # Print detailed traceback for first failure
            if len([r for r in results if not r[3]]) == 1:
                print(f"  Detailed error trace:")
                traceback.print_exc()

    # Summary
    print(f"\n=== Compilation Test Summary ===")
    successes = [r for r in results if r[3]]
    failures = [r for r in results if not r[3]]

    print(f"✓ Successful: {len(successes)}/{len(results)}")
    for li, lj, lc, _, _ in successes:
        print(f"  (li={li}, lj={lj}, lc={lc})")

    if failures:
        print(f"✗ Failed: {len(failures)}/{len(results)}")
        for li, lj, lc, _, error_type in failures:
            print(f"  (li={li}, lj={lj}, lc={lc}) - {error_type}")

    return len(failures) == 0

def test_type2_ip_execution():
    """Test actual type2 IP kernel execution with minimal data."""
    print("\n=== Testing Type2 IP Execution ===")

    try:
        from jqc.backend.ecp import get_ecp_ip

        # Create minimal molecule
        mol = test_minimal_molecule()

        print("Attempting get_ecp_ip execution...")

        # This is where the actual failure typically occurs
        result = get_ecp_ip(mol)

        print(f"✓ Execution successful!")
        print(f"  Result shape: {result.shape}")
        print(f"  Result type: {type(result)}")

        return True

    except Exception as e:
        print(f"✗ Execution failed: {e}")

        # Analyze the error location
        tb = traceback.extract_tb(e.__traceback__)
        for frame in tb:
            if 'ecp' in frame.filename:
                print(f"  Error in: {frame.filename}:{frame.lineno}")
                print(f"  Function: {frame.name}")
                print(f"  Code: {frame.line}")
                break

        return False

def test_shared_memory_estimation():
    """Test our shared memory estimation vs. actual requirements."""
    print("\n=== Testing Shared Memory Estimation ===")

    from jqc.backend.ecp import _estimate_type2_ip_shared_memory

    test_cases = [(0, 0, 0), (1, 1, 0), (2, 2, 1)]

    for li, lj, lc in test_cases:
        estimated = _estimate_type2_ip_shared_memory(li, lj, lc, 'fp64')

        # Check if size is reasonable
        max_shared_per_block = 48 * 1024  # 48KB typical limit

        status = "✓" if estimated <= max_shared_per_block else "⚠"
        print(f"{status} (li={li}, lj={lj}, lc={lc}): {estimated} bytes ({estimated/1024:.1f} KB)")

        if estimated > max_shared_per_block:
            print(f"    WARNING: Exceeds typical 48KB limit!")

def main():
    """Run all debugging tests."""
    print("🔍 Type2 IP Kernel Debug Suite")
    print("=" * 50)

    # Test environment
    if not test_cuda_environment():
        print("❌ CUDA environment issues detected. Stopping.")
        return 1

    # Test shared memory estimation
    test_shared_memory_estimation()

    # Test compilation stages
    compilation_ok = test_type2_ip_compilation_stages()

    if compilation_ok:
        print("\n✅ All compilation tests passed!")

        # Test execution
        execution_ok = test_type2_ip_execution()

        if execution_ok:
            print("\n🎉 All tests passed! Type2 IP kernels are working correctly.")
            return 0
        else:
            print("\n❌ Execution test failed, but compilation works.")
            print("   This suggests the issue is in the kernel execution or data setup.")
            return 2
    else:
        print("\n❌ Compilation test failures detected.")
        print("   This suggests the issue is in the kernel compilation process.")
        return 3

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)