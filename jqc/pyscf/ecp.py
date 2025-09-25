"""
JoltQC ECP PySCF Integration

Provides ECP integral replacement for PySCF/GPU4PySCF methods
"""

import numpy as np
import cupy as cp
from typing import Dict, Any, Optional
from jqc.backend.ecp import ecp_generator, get_ecp

def apply_ecp(mol, cutoff_fp32: float = 1e-8, cutoff_fp64: float = 1e-12) -> Dict[str, Any]:
    """
    Apply JoltQC ECP kernels to PySCF molecule

    Args:
        mol: PySCF molecule object
        cutoff_fp32: Precision cutoff for FP32 calculations
        cutoff_fp64: Precision cutoff for FP64 calculations

    Returns:
        Dictionary with patched ECP methods
    """
    # Check if molecule has ECP
    if not hasattr(mol, '_ecpbas') or len(mol._ecpbas) == 0:
        return {}

    # Generate ECP kernels
    ecp_kernel = ecp_generator(mol, precision)

    # Store original methods for potential restoration
    original_methods = {}

    def jit_get_ecp():
        """JIT-compiled ECP integral evaluation"""
        ecp_mat = get_ecp(mol, precision)
        return ecp_mat

    # Patch methods
    patched_methods = {
        'get_ecp': jit_get_ecp,
        'ecp_kernel': ecp_kernel,
        'precision': precision,
        'original_methods': original_methods
    }

    return patched_methods


def patch_ecp_integrals(mol, **kwargs) -> None:
    """
    In-place patch of ECP integral methods

    Args:
        mol: PySCF molecule object
        **kwargs: Additional arguments for precision control
    """
    patches = apply_ecp(mol, **kwargs)

    if not patches:
        return  # No ECP in molecule

    # Apply patches to molecule
    for method_name, method_func in patches.items():
        if method_name not in ['original_methods', 'ecp_kernel', 'precision']:
            setattr(mol, method_name, method_func)

    # Store metadata
    mol._jqc_ecp_info = {
        'precision': patches['precision'],
        'kernel_type': 'ecp_type1',
        'original_methods': patches['original_methods']
    }

    print(f"JoltQC ECP: Patched molecule with {patches['precision']} precision kernels")


def restore_ecp_methods(mol) -> None:
    """
    Restore original ECP methods

    Args:
        mol: PySCF molecule object
    """
    if not hasattr(mol, '_jqc_ecp_info'):
        return

    info = mol._jqc_ecp_info
    for method_name, original_method in info['original_methods'].items():
        setattr(mol, method_name, original_method)

    delattr(mol, '_jqc_ecp_info')
    print("JoltQC ECP: Restored original methods")


def ecp_performance_info(mol) -> Dict[str, Any]:
    """
    Get performance information for ECP calculations

    Args:
        mol: PySCF molecule object

    Returns:
        Dictionary with performance metrics
    """
    if not hasattr(mol, '_jqc_ecp_info'):
        return {'status': 'No JoltQC ECP applied'}

    info = mol._jqc_ecp_info
    nao = mol.nao_nr()
    necpbas = len(mol._ecpbas) if hasattr(mol, '_ecpbas') else 0

    return {
        'status': 'JoltQC ECP active',
        'precision': info['precision'],
        'kernel_type': info['kernel_type'],
        'nao': nao,
        'n_ecp_shells': necpbas,
        'estimated_memory_mb': nao * nao * 8 / 1024 / 1024,  # Rough estimate
    }


# Convenience functions for specific ECP use cases
def enable_ecp_jit(mol, precision: str = 'auto', **kwargs) -> None:
    """
    Enable JIT-compiled ECP integrals with automatic precision selection

    Args:
        mol: PySCF molecule object
        precision: 'auto', 'fp64', 'fp32', or 'mixed'
        **kwargs: Additional precision control arguments
    """
    if precision == 'auto':
        nao = mol.nao_nr()
        if nao < 300:
            precision = 'fp64'
        elif nao < 800:
            precision = 'mixed'
        else:
            precision = 'fp32'

    # Override precision in kwargs
    kwargs['precision'] = precision

    patch_ecp_integrals(mol, **kwargs)


def benchmark_ecp(mol, n_trials: int = 5) -> Dict[str, float]:
    """
    Benchmark ECP integral evaluation

    Args:
        mol: PySCF molecule object
        n_trials: Number of timing trials

    Returns:
        Timing results dictionary
    """
    import time

    if not hasattr(mol, '_ecpbas') or len(mol._ecpbas) == 0:
        return {'error': 'No ECP in molecule'}

    # Test different precisions
    results = {}

    for precision in ['fp64', 'fp32', 'mixed']:
        times = []
        try:
            for _ in range(n_trials):
                cp.cuda.Stream.null.synchronize()  # Ensure GPU is ready
                start_time = time.time()

                ecp_mat = get_ecp(mol, precision)
                cp.cuda.Stream.null.synchronize()  # Wait for completion

                end_time = time.time()
                times.append(end_time - start_time)

            results[precision] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'trials': n_trials
            }
        except Exception as e:
            results[precision] = {'error': str(e)}

    return results