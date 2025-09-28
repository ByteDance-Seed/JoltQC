#!/usr/bin/env python3
"""
Debug kernel execution issues for type2 IP kernels.

This focuses on the actual kernel execution step to identify
memory access or parameter passing issues.
"""

import sys
import cupy as cp
import numpy as np
from pyscf import gto

sys.path.insert(0, '/home/xiaojie/Documents/JoltQC')

def create_minimal_execution_test():
    """Create minimal kernel execution test with controlled data."""
    print("=== Testing Kernel Execution with Minimal Data ===")

    from jqc.backend.ecp import _compile_ecp_type2_ip_kernel
    from jqc.pyscf.basis import BasisLayout

    # Create molecule
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

    # Create basis layout
    try:
        print("Creating basis layout...")
        basis_layout = BasisLayout.from_mol(mol)
        print(f"✓ Basis layout created: {basis_layout.coords.shape}")
    except Exception as e:
        print(f"✗ Basis layout creation failed: {e}")
        return False

    # Test with smallest possible parameters
    li, lj, lc = 0, 0, 0  # Simplest case
    precision = 'fp64'

    try:
        # Compile kernel
        print(f"Compiling kernel (li={li}, lj={lj}, lc={lc})...")
        kernel_func = _compile_ecp_type2_ip_kernel(li, lj, lc, precision)
        print("✓ Kernel compilation successful")

        # Prepare minimal execution parameters
        nao = mol.nao_nr()
        print(f"Number of AOs: {nao}")

        # Create minimal output buffer
        n_ecp_atoms = 1  # Only one ECP atom (Cu)
        gctr = cp.zeros((n_ecp_atoms, 3, nao, nao), dtype=cp.float64)
        print(f"✓ Output buffer created: {gctr.shape}")

        # Get AO location info
        ao_loc = cp.array(mol.ao_loc_nr(), dtype=cp.int32)
        print(f"✓ AO locations: {ao_loc}")

        # Create minimal task list (just one task)
        # Task format: [ish, jsh, ksh] where ksh refers to ECP shell
        tasks = cp.array([[0, 0, 4]], dtype=cp.int32)  # Use first non-local ECP shell (index 4)
        ntasks = 1
        print(f"✓ Tasks created: {tasks.shape}")

        # ECP basis info
        ecpbas = cp.array(mol._ecpbas, dtype=cp.int32)
        ecploc = cp.array(mol._ecploc, dtype=cp.int32) if hasattr(mol, '_ecploc') else cp.arange(len(mol._ecpbas), dtype=cp.int32)
        print(f"✓ ECP basis info: ecpbas.shape={ecpbas.shape}, ecploc.shape={ecploc.shape}")

        # Environment and atom info
        atm = cp.array(mol._atm, dtype=cp.int32)
        env = cp.array(mol._env, dtype=cp.float64)
        print(f"✓ Molecule data: atm.shape={atm.shape}, env.shape={env.shape}")

        # Primitive info (minimal)
        npi = npj = 1  # Single primitive
        print(f"✓ Primitive counts: npi={npi}, npj={npj}")

        print("\\n--- Attempting kernel execution ---")

        # Execute kernel with minimal parameters
        kernel_func(
            gctr.ravel(),      # Flattened output
            ao_loc,            # AO locations
            nao,               # Number of AOs
            tasks.ravel(),     # Task list
            ntasks,            # Number of tasks
            ecpbas.ravel(),    # ECP basis info
            ecploc,            # ECP shell locations
            basis_layout.coords,  # Coordinates
            basis_layout.ce,   # Coefficients and exponents
            atm.ravel(),       # Atom info
            env,               # Environment
            npi,               # Number of i primitives
            npj                # Number of j primitives
        )

        # Synchronize to catch any execution errors
        cp.cuda.Device().synchronize()

        print("✓ Kernel execution successful!")
        print(f"Result norm: {cp.linalg.norm(gctr)}")

        return True

    except Exception as e:
        print(f"✗ Kernel execution failed: {e}")

        # Analyze the specific failure point
        import traceback
        tb = traceback.extract_tb(e.__traceback__)

        print("\\nError analysis:")
        for frame in tb:
            if 'cupy' in frame.filename or 'cuda' in frame.filename:
                print(f"  CUDA error in: {frame.filename}:{frame.lineno}")
                print(f"  Function: {frame.name}")
                break

        # Check if it's a parameter/data size mismatch
        if 'illegal' in str(e).lower():
            print("\\n🔍 Illegal memory access suggests:")
            print("  - Buffer size mismatch")
            print("  - Wrong indexing in kernel")
            print("  - Shared memory overflow")
            print("  - Invalid pointer arithmetic")

        return False

def test_parameter_validation():
    """Test if the parameters being passed are valid."""
    print("\\n=== Parameter Validation Test ===")

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

    from jqc.pyscf.basis import BasisLayout

    basis_layout = BasisLayout.from_mol(mol)

    print(f"Molecular parameters:")
    print(f"  nao: {mol.nao_nr()}")
    print(f"  nbas: {mol.nbas}")
    print(f"  natm: {mol.natm}")
    print(f"  ECP shells: {mol._ecpbas.shape[0]}")

    print(f"\\nBasis layout:")
    print(f"  coords.shape: {basis_layout.coords.shape}")
    print(f"  ce.shape: {basis_layout.ce.shape}")

    # Check for any obvious issues
    if mol.nao_nr() == 0:
        print("⚠ WARNING: No AO functions!")

    if mol._ecpbas.shape[0] == 0:
        print("⚠ WARNING: No ECP basis functions!")

    print("✓ Parameters appear reasonable")

def main():
    """Run kernel execution debugging."""
    print("🔧 Type2 IP Kernel Execution Debug")
    print("=" * 40)

    # Test parameter validation first
    test_parameter_validation()

    # Test minimal execution
    success = create_minimal_execution_test()

    if success:
        print("\\n🎉 Kernel execution test passed!")
        print("   The execution issue may be context-dependent or")
        print("   related to specific parameter combinations.")
    else:
        print("\\n❌ Kernel execution test failed.")
        print("   This confirms there's a real execution issue")
        print("   that needs to be investigated further.")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())