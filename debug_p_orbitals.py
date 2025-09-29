#!/usr/bin/env python3

import cupy as cp
import numpy as np
from pyscf import gto

from jqc.backend.ecp import get_ecp_ipip

# P orbitals only test case
cu1_basis_p_only = gto.basis.parse(
    '''
 H    P
       0.3827000              1.0000000
    ''')

# ECP with P channel only
na_ecp_p_only = gto.basis.parse_ecp('''
Na nelec 10
Na ul
2       1.0                   0.5
Na P
2      10.279868             299.489474
    ''')

mol_p_only = gto.M(
    atom='Na 0.0 0.0 0.0',
    basis={'Na': cu1_basis_p_only},
    ecp={'Na': na_ecp_p_only},
    spin=1,  # 1 unpaired electron
    output="/dev/null",
    verbose=0,
)

print("Testing P orbitals only...")
print(f"Mol shape: {mol_p_only.nao}")

print("\n=== IPVIP TEST ===")
# Get CPU reference for ipvip
h1_cpu_ipvip = mol_p_only.intor('ECPscalar_ipnucip', comp=9)
print(f"CPU ipvip shape: {h1_cpu_ipvip.shape}")
print(f"CPU ipvip max value: {np.max(np.abs(h1_cpu_ipvip)):.6e}")
print(f"CPU ipvip diagonal values:")
for i in range(min(3, h1_cpu_ipvip.shape[1])):
    print(f"  [0,{i},{i}]: {h1_cpu_ipvip[0,i,i]:.6e}")

# Get GPU result for ipvip
h1_gpu_ipvip = get_ecp_ipip(mol_p_only, 'ipvip').sum(axis=0).get()
print(f"\nGPU ipvip shape: {h1_gpu_ipvip.shape}")
print(f"GPU ipvip max value: {np.max(np.abs(h1_gpu_ipvip)):.6e}")
print(f"GPU ipvip diagonal values:")
for i in range(min(3, h1_gpu_ipvip.shape[1])):
    print(f"  [0,{i},{i}]: {h1_gpu_ipvip[0,i,i]:.6e}")

diff_ipvip = h1_cpu_ipvip - h1_gpu_ipvip
print(f"\nIPVIP Difference norm: {np.linalg.norm(diff_ipvip):.3e}")
print(f"IPVIP Max difference: {np.max(np.abs(diff_ipvip)):.3e}")
print(f"IPVIP Relative error: {np.linalg.norm(diff_ipvip)/np.linalg.norm(h1_cpu_ipvip):.3e}")

print("\n=== IPIPV TEST ===")
# Get CPU reference for ipipv
h1_cpu_ipipv = mol_p_only.intor('ECPscalar_ipipnuc', comp=9)
print(f"CPU ipipv shape: {h1_cpu_ipipv.shape}")
print(f"CPU ipipv max value: {np.max(np.abs(h1_cpu_ipipv)):.6e}")

# Get GPU result for ipipv
h1_gpu_ipipv = get_ecp_ipip(mol_p_only, 'ipipv').sum(axis=0).get()
print(f"\nGPU ipipv shape: {h1_gpu_ipipv.shape}")
print(f"GPU ipipv max value: {np.max(np.abs(h1_gpu_ipipv)):.6e}")

diff_ipipv = h1_cpu_ipipv - h1_gpu_ipipv
print(f"\nIPIPV Difference norm: {np.linalg.norm(diff_ipipv):.3e}")
print(f"IPIPV Max difference: {np.max(np.abs(diff_ipipv)):.3e}")
print(f"IPIPV Relative error: {np.linalg.norm(diff_ipipv)/np.linalg.norm(h1_cpu_ipipv):.3e}")

mol_p_only.stdout.close()