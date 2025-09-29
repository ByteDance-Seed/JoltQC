#!/usr/bin/env python3

import cupy as cp
import numpy as np
from pyscf import gto

from jqc.backend.ecp import get_ecp_ipip

# Use the exact same setup as test_ecp_small.py
cu1_basis_small = gto.basis.parse(
    '''
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
    ''')

na_ecp_type2 = gto.basis.parse_ecp('''
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
    ''')

mol_type2 = gto.M(
    atom='''
        Na 0.5 0.5 0.
        Na  0.  1.  1.
    ''',
    basis={'Na': cu1_basis_small, 'H': cu1_basis_small},
    ecp={'Na': na_ecp_type2},
    output="/dev/null",
    verbose=0,
)

print("Testing ECP IPIP ipvip differences...")

# Get CPU reference
h1_cpu = mol_type2.intor('ECPscalar_ipnucip', comp=9)
print(f"CPU result shape: {h1_cpu.shape}")

# Get GPU result
h1_gpu = get_ecp_ipip(mol_type2, 'ipvip').sum(axis=0).get()
print(f"GPU result shape: {h1_gpu.shape}")

diff = h1_cpu - h1_gpu
print(f"IPVIP Difference norm: {np.linalg.norm(diff):.3e}")

print("\nTesting ECP IPIP ipipv differences...")

# Get CPU reference for ipipv
h1_cpu2 = mol_type2.intor('ECPscalar_ipipnuc', comp=9)
print(f"CPU result shape: {h1_cpu2.shape}")

# Get GPU result for ipipv
h1_gpu2 = get_ecp_ipip(mol_type2, 'ipipv').sum(axis=0).get()
print(f"GPU result shape: {h1_gpu2.shape}")

diff2 = h1_cpu2 - h1_gpu2
print(f"IPIPV Difference norm: {np.linalg.norm(diff2):.3e}")

print(f"\nIPVIP - CPU [0,0,0]: {h1_cpu[0,0,0]:.6e}")
print(f"IPVIP - GPU [0,0,0]: {h1_gpu[0,0,0]:.6e}")
print(f"IPIPV - CPU [0,0,0]: {h1_cpu2[0,0,0]:.6e}")
print(f"IPIPV - GPU [0,0,0]: {h1_gpu2[0,0,0]:.6e}")

# Return early, remove the old analysis code
mol_type2.stdout.close()
exit()

diff = h1_cpu - h1_gpu
print(f"Difference norm: {np.linalg.norm(diff):.3e}")
print(f"CPU max value: {np.max(np.abs(h1_cpu)):.3e}")
print(f"GPU max value: {np.max(np.abs(h1_gpu)):.3e}")
print(f"Difference max value: {np.max(np.abs(diff)):.3e}")

# Check for specific patterns
print(f"CPU result contains NaN: {np.any(np.isnan(h1_cpu))}")
print(f"GPU result contains NaN: {np.any(np.isnan(h1_gpu))}")
print(f"CPU result contains Inf: {np.any(np.isinf(h1_cpu))}")
print(f"GPU result contains Inf: {np.any(np.isinf(h1_gpu))}")

# Show first few values
print(f"CPU [0,0,0]: {h1_cpu[0,0,0]:.6e}")
print(f"GPU [0,0,0]: {h1_gpu[0,0,0]:.6e}")
print(f"CPU [0,0,1]: {h1_cpu[0,0,1]:.6e}")
print(f"GPU [0,0,1]: {h1_gpu[0,0,1]:.6e}")

mol_type2.stdout.close()