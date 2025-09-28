#!/usr/bin/env python3
"""
Test to isolate CUDA compilation issues by trying to compile the kernel
with verbose error reporting.
"""

import sys
import cupy as cp
import traceback

sys.path.insert(0, '/home/xiaojie/Documents/JoltQC')

def test_type2_ip_compilation_detailed():
    """Test Type2 IP kernel compilation with detailed error reporting."""
    print("=== Testing Type2 IP CUDA Compilation ===")

    from jqc.backend.ecp import _compile_ecp_type2_ip_kernel
    import jqc.backend.ecp

    # Clear cache
    jqc.backend.ecp._ecp_kernel_cache.clear()

    # Try to compile the simplest case
    li, lj, lc = 0, 0, 0
    precision = 'fp64'

    print(f"Attempting compilation: li={li}, lj={lj}, lc={lc}")

    try:
        # Get the source code that would be compiled
        from jqc.backend.cuda_scripts import get_kernel_source

        # This is roughly what _compile_ecp_type2_ip_kernel does internally
        sources = []
        sources.append(('common.cu', get_kernel_source('common.cu')))
        sources.append(('ecp_type2_kernel.cu', get_kernel_source('ecp_type2_kernel.cu')))
        sources.append(('ecp_type2.cu', get_kernel_source('ecp_type2.cu')))
        sources.append(('ecp_type2_ip.cu', get_kernel_source('ecp_type2_ip.cu')))

        # Template parameters
        template_params = {
            'LI': li,
            'LJ': lj,
            'LC': lc,
            'DataType': 'double',
            'DataType2': 'double2',
            'DataType4': 'double4',
        }

        combined_source = ""
        for name, source in sources:
            # Apply template substitutions
            for param, value in template_params.items():
                source = source.replace(f"#{param}#", str(value))
            combined_source += f"\n// === {name} ===\n" + source + "\n"

        print("✓ Source code generated successfully")
        print(f"Source length: {len(combined_source)} characters")

        # Try to compile with CuPy
        try:
            print("Attempting CUDA compilation...")
            mod = cp.RawModule(code=combined_source)
            print("✓ CUDA module compiled successfully")

            try:
                kernel = mod.get_function('type2_cart_ip1')
                print("✓ Kernel function retrieved successfully")
                return True

            except Exception as e:
                print(f"✗ Failed to get kernel function: {e}")
                return False

        except Exception as e:
            print(f"✗ CUDA compilation failed: {e}")
            print("\nTrying to get more detailed error info...")

            # Try with more verbose compilation
            try:
                # Save source to file for inspection
                with open('/tmp/debug_cuda_source.cu', 'w') as f:
                    f.write(combined_source)
                print("✓ Source saved to /tmp/debug_cuda_source.cu for inspection")
            except:
                pass

            return False

    except Exception as e:
        print(f"✗ Source generation failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("🔧 CUDA Compilation Debug")
    print("=" * 40)

    success = test_type2_ip_compilation_detailed()

    if success:
        print("\n🎉 Compilation test passed!")
    else:
        print("\n❌ Compilation test failed.")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())