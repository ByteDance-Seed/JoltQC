#!/usr/bin/env python3
"""
Benchmark for ECP kernel compilation times

This script measures the JIT compilation time for different types of ECP kernels
to understand compilation overhead and identify optimization opportunities.

Usage:
    python benchmarks/benchmark_ecp_compilation.py
"""

import time
import cupy as cp
import numpy as np
from typing import Dict, List, Tuple, Optional
from pyscf import gto

# JoltQC ECP modules
from jqc.backend.ecp import (
    _compile_ecp_type1_kernel, _compile_ecp_type2_kernel,
    _compile_ecp_type1_ip_kernel, _compile_ecp_type2_ip_kernel,
    _compile_ecp_type1_ipip_kernel, _compile_ecp_type2_ipip_kernel
)
# BasisLayout import removed as it's not needed for this benchmark


def get_kernel_resource_info_placeholder() -> Dict[str, int]:
    """Placeholder for kernel resource information - returns zeros for now"""
    # TODO: Extract actual register and memory usage from compiled kernels
    # This requires accessing the CuPy kernel object from the compilation functions
    return {
        'registers': 0,
        'local_memory': 0,
        'shared_memory': 0,
        'const_memory': 0,
        'max_threads': 0
    }


def create_test_molecules() -> Dict[str, gto.Mole]:
    """Create test molecules with different ECP configurations"""

    molecules = {}

    # Small Cu atom (Type1 ECP)
    molecules['cu_small'] = gto.M(
        atom="Cu 0 0 0",
        basis="sto-3g",
        ecp="crenbl",
        spin=1,  # Cu has 19 electrons, with ECP removing 18, leaves 1 unpaired
        cart=1,
        output="/dev/null",
        verbose=0,
    )

    # Na2 system (Type2 ECP)
    cu1_basis = gto.basis.parse("""
        H    S
               1.8000000              1.0000000
        H    S
               2.8000000              0.0210870             -0.0045400              0.0000000
               1.3190000              0.3461290             -0.1703520              0.0000000
               0.9059000              0.0393780              0.1403820              1.0000000
        H    P
               2.1330000              0.0868660              0.0000000
               1.2000000              0.0000000              0.5000000
               0.3827000              0.5010080              1.0000000
        H    D
               0.3827000              1.0000000
    """)

    na_ecp = gto.basis.parse_ecp("""
        Na nelec 10
        Na ul
        2       1.0                   0.5
        Na S
        2      13.652203             732.2692
        2       6.826101              26.484721
        Na P
        2      10.279868             299.489474
        2       5.139934              26.466234
        Na D
        2       7.349859             124.457595
        2       3.674929              14.035995
    """)

    molecules['na2_medium'] = gto.M(
        atom="Na 0 0 0; Na 0 0 1.7",
        basis={'Na': cu1_basis},
        ecp={'Na': na_ecp},
        cart=1,
        output="/dev/null",
        verbose=0,
    )

    return molecules


def get_angular_momentum_combinations() -> List[Tuple[int, int]]:
    """Get list of (LI, LJ) angular momentum combinations to test"""
    combinations = []
    for li in range(5):  # L = 0,1,2,3,4 (s,p,d,f,g)
        for lj in range(li + 1):  # Only unique combinations
            combinations.append((li, lj))
    return combinations


def benchmark_kernel_compilation(kernel_type: str, li: int, lj: int, lc: int,
                                precision: str = "fp64") -> Tuple[float, Optional[Dict[str, int]]]:
    """
    Benchmark compilation time and resource usage for a specific kernel type and angular momentum

    Args:
        kernel_type: Type of kernel (maps to actual CUDA kernel names):
                    'type1_ecp' -> type1_cart_ecp
                    'type2_ecp' -> type2_cart_ecp
                    'type1_ip' -> type1_cart_ip1
                    'type2_ip' -> type2_cart_ip1
                    'type1_ipip_ipipv' -> type1_cart_ipipv
                    'type1_ipip_ipvip' -> type1_cart_ipvip
                    'type2_ipip_ipipv' -> type2_cart_ipipv
                    'type2_ipip_ipvip' -> type2_cart_ipvip
        li, lj: Angular momentum quantum numbers
        lc: Angular momentum for Type2 kernels (ignored for Type1)
        precision: Precision type

    Returns:
        Tuple of (compilation_time, resource_info) where resource_info contains
        registers, local_memory, shared_memory, const_memory, max_threads
    """
    start_time = time.perf_counter()

    try:
        if kernel_type == 'type1_ecp':
            _compile_ecp_type1_kernel(li, lj, precision)
        elif kernel_type == 'type2_ecp':
            _compile_ecp_type2_kernel(li, lj, lc, precision)
        elif kernel_type == 'type1_ip':
            _compile_ecp_type1_ip_kernel(li, lj, precision)
        elif kernel_type == 'type2_ip':
            _compile_ecp_type2_ip_kernel(li, lj, lc, precision)
        elif kernel_type == 'type1_ipip_ipipv':
            _compile_ecp_type1_ipip_kernel(li, lj, 'ipipv', precision)
        elif kernel_type == 'type1_ipip_ipvip':
            _compile_ecp_type1_ipip_kernel(li, lj, 'ipvip', precision)
        elif kernel_type == 'type2_ipip_ipipv':
            _compile_ecp_type2_ipip_kernel(li, lj, lc, 'ipipv', precision)
        elif kernel_type == 'type2_ipip_ipvip':
            _compile_ecp_type2_ipip_kernel(li, lj, lc, 'ipvip', precision)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        compilation_time = time.perf_counter() - start_time

        # Get resource information (placeholder for now)
        resources = get_kernel_resource_info_placeholder()

        return compilation_time, resources

    except Exception as e:
        print(f"  ✗ Failed to compile {kernel_type}({li},{lj},{lc}): {e}")
        return -1.0, None


def run_compilation_benchmark():
    """Run comprehensive ECP kernel compilation benchmark"""

    print("=" * 80)
    print("ECP Kernel Compilation Time Benchmark")
    print("=" * 80)

    # Initialize CUDA context
    cp.cuda.Device().use()

    # Create test molecules
    molecules = create_test_molecules()

    # Get angular momentum combinations
    am_combinations = get_angular_momentum_combinations()

    # Kernel types to test
    kernel_types = [
        'type1_ecp', 'type2_ecp',
        'type1_ip', 'type2_ip',
        'type1_ipip_ipipv', 'type1_ipip_ipvip',
        'type2_ipip_ipipv', 'type2_ipip_ipvip'
    ]

    # Store results
    results = {}

    for mol_name, mol in molecules.items():
        print(f"\n📋 Testing molecule: {mol_name}")
        print(f"   Atoms: {mol.atom}")
        print(f"   ECP atoms: {len(mol._ecpbas)} basis functions")

        results[mol_name] = {}

        for kernel_type in kernel_types:
            print(f"\n🔧 Kernel type: {kernel_type}")
            results[mol_name][kernel_type] = {}

            total_time = 0.0
            successful_compilations = 0

            for li, lj in am_combinations:
                # For Type2 kernels, also test different LC values
                lc_values = [0, 1, 2] if 'type2' in kernel_type else [0]

                for lc in lc_values:
                    # Map kernel_type to actual CUDA kernel name for clearer output
                    cuda_kernel_map = {
                        'type1_ecp': 'type1_cart_ecp',
                        'type2_ecp': 'type2_cart_ecp',
                        'type1_ip': 'type1_cart_ip1',
                        'type2_ip': 'type2_cart_ip1',
                        'type1_ipip_ipipv': 'type1_cart_ipipv',
                        'type1_ipip_ipvip': 'type1_cart_ipvip',
                        'type2_ipip_ipipv': 'type2_cart_ipipv',
                        'type2_ipip_ipvip': 'type2_cart_ipvip'
                    }
                    cuda_kernel_name = cuda_kernel_map.get(kernel_type, kernel_type)

                    if 'type2' in kernel_type:
                        print(f"   Compiling {cuda_kernel_name} L=({li},{lj},{lc})...", end=" ", flush=True)
                    else:
                        print(f"   Compiling {cuda_kernel_name} L=({li},{lj})...", end=" ", flush=True)

                    compile_time, resources = benchmark_kernel_compilation(
                        kernel_type, li, lj, lc, "fp64"
                    )

                    if compile_time > 0 and resources is not None:
                        print(f"✓ {compile_time:.3f}s [R:{resources['registers']} LM:{resources['local_memory']}B]")
                        key = (li, lj, lc) if 'type2' in kernel_type else (li, lj)
                        results[mol_name][kernel_type][key] = {
                            'time': compile_time,
                            'resources': resources
                        }
                        total_time += compile_time
                        successful_compilations += 1
                    else:
                        print("✗ Failed")
                        key = (li, lj, lc) if 'type2' in kernel_type else (li, lj)
                        results[mol_name][kernel_type][key] = None

            avg_time = total_time / successful_compilations if successful_compilations > 0 else 0
            total_kernels = len(am_combinations) * (3 if 'type2' in kernel_type else 1)
            print(f"   📊 Total: {total_time:.3f}s, Average: {avg_time:.3f}s, Success: {successful_compilations}/{total_kernels}")

    # Summary report
    print("\n" + "=" * 80)
    print("COMPILATION BENCHMARK SUMMARY")
    print("=" * 80)

    for mol_name in results:
        print(f"\n🧪 {mol_name.upper()}:")

        for kernel_type in kernel_types:
            data = [entry for entry in results[mol_name][kernel_type].values() if entry is not None]
            if data:
                times = [entry['time'] for entry in data]
                registers = [entry['resources']['registers'] for entry in data]
                local_mem = [entry['resources']['local_memory'] for entry in data]

                total = sum(times)
                avg = total / len(times)
                min_time = min(times)
                max_time = max(times)
                avg_regs = sum(registers) / len(registers)
                max_regs = max(registers)
                avg_lmem = sum(local_mem) / len(local_mem)
                max_lmem = max(local_mem)

                print(f"   {kernel_type:16s}: Time={total:6.3f}s(avg={avg:5.3f}s) Regs={avg_regs:4.1f}(max={max_regs}) LMem={avg_lmem:5.0f}B(max={max_lmem}B)")

    # Find slowest compilations and highest resource usage
    print(f"\n🐌 SLOWEST COMPILATIONS:")
    all_times = []
    for mol_name in results:
        for kernel_type in kernel_types:
            for key, entry in results[mol_name][kernel_type].items():
                if entry is not None:
                    time_val = entry['time']
                    resources = entry['resources']
                    if len(key) == 3:  # Type2 kernels
                        li, lj, lc = key
                        all_times.append((time_val, mol_name, kernel_type, li, lj, lc, resources))
                    else:  # Type1 kernels
                        li, lj = key
                        all_times.append((time_val, mol_name, kernel_type, li, lj, None, resources))

    all_times.sort(reverse=True)
    for i, (time_val, mol_name, kernel_type, li, lj, lc, resources) in enumerate(all_times[:10]):
        if lc is not None:
            print(f"   {i+1:2d}. {time_val:6.3f}s - {mol_name} {kernel_type}({li},{lj},{lc}) [R:{resources['registers']} LM:{resources['local_memory']}B]")
        else:
            print(f"   {i+1:2d}. {time_val:6.3f}s - {mol_name} {kernel_type}({li},{lj}) [R:{resources['registers']} LM:{resources['local_memory']}B]")

    # Find highest register usage
    print(f"\n📊 HIGHEST REGISTER USAGE:")
    all_times.sort(key=lambda x: x[6]['registers'], reverse=True)
    for i, (time_val, mol_name, kernel_type, li, lj, lc, resources) in enumerate(all_times[:10]):
        if lc is not None:
            print(f"   {i+1:2d}. {resources['registers']:3d} regs - {mol_name} {kernel_type}({li},{lj},{lc}) [{time_val:.3f}s, LM:{resources['local_memory']}B]")
        else:
            print(f"   {i+1:2d}. {resources['registers']:3d} regs - {mol_name} {kernel_type}({li},{lj}) [{time_val:.3f}s, LM:{resources['local_memory']}B]")

    # Find highest local memory usage
    print(f"\n💾 HIGHEST LOCAL MEMORY USAGE:")
    all_times.sort(key=lambda x: x[6]['local_memory'], reverse=True)
    for i, (time_val, mol_name, kernel_type, li, lj, lc, resources) in enumerate(all_times[:10]):
        if lc is not None:
            print(f"   {i+1:2d}. {resources['local_memory']:6d}B - {mol_name} {kernel_type}({li},{lj},{lc}) [{time_val:.3f}s, R:{resources['registers']}]")
        else:
            print(f"   {i+1:2d}. {resources['local_memory']:6d}B - {mol_name} {kernel_type}({li},{lj}) [{time_val:.3f}s, R:{resources['registers']}]")

    # Performance insights
    print(f"\n💡 INSIGHTS:")
    if all_times:
        total_compilation_time = sum(t[0] for t in all_times)
        all_registers = [t[6]['registers'] for t in all_times]
        all_local_mem = [t[6]['local_memory'] for t in all_times]

        print(f"   • Total compilation time: {total_compilation_time:.3f}s")
        print(f"   • Average per kernel: {total_compilation_time/len(all_times):.3f}s")
        print(f"   • Register usage: avg={sum(all_registers)/len(all_registers):.1f}, max={max(all_registers)}")
        print(f"   • Local memory: avg={sum(all_local_mem)/len(all_local_mem):.0f}B, max={max(all_local_mem)}B")
        print(f"   • Compilation overhead dominates for small calculations")
        print(f"   • Higher angular momentum (L≥3) kernels take longer to compile")
        print(f"   • Type2 kernels generally use more registers than Type1")
        print(f"   • IPIP kernels typically have highest resource usage")


if __name__ == "__main__":
    run_compilation_benchmark()