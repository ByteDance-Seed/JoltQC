#!/usr/bin/env python3

import cupy as cp
import numpy as np
from pyscf import gto

from jqc.backend.ecp import get_ecp_ipip

# S orbitals only test case
cu1_basis_s_only = gto.basis.parse(
    '''
 H    S
       1.8000000              1.0000000
    ''')

# Simple ECP with S channel only
na_ecp_s_only = gto.basis.parse_ecp('''
Na nelec 10
Na ul
2       1.0                   0.5
Na S
2      13.652203             732.2692
    ''')

mol_s_only = gto.M(
    atom='Na 0.0 0.0 0.0',
    basis={'Na': cu1_basis_s_only},
    ecp={'Na': na_ecp_s_only},
    spin=1,  # 1 unpaired electron
    output="/dev/null",
    verbose=0,
)

print("Testing S orbitals only...")
print(f"Mol shape: {mol_s_only.nao}")

print("\n=== IPVIP TEST ===")
# Get CPU reference for ipvip
h1_cpu_ipvip = mol_s_only.intor('ECPscalar_ipnucip', comp=9)
print(f"CPU ipvip shape: {h1_cpu_ipvip.shape}")
print(f"CPU ipvip values:")
for i in range(9):
    print(f"  Component {i}: {h1_cpu_ipvip[i,0,0]:.6e}")

# Get GPU result for ipvip
h1_gpu_ipvip = get_ecp_ipip(mol_s_only, 'ipvip').sum(axis=0).get()
print(f"\nGPU ipvip shape: {h1_gpu_ipvip.shape}")
print(f"GPU ipvip values:")
for i in range(9):
    print(f"  Component {i}: {h1_gpu_ipvip[i,0,0]:.6e}")

diff_ipvip = h1_cpu_ipvip - h1_gpu_ipvip
print(f"\nIPVIP Difference norm: {np.linalg.norm(diff_ipvip):.3e}")
print(f"IPVIP Max difference: {np.max(np.abs(diff_ipvip)):.3e}")

print("\n=== IPIPV TEST ===")
# Get CPU reference for ipipv
h1_cpu_ipipv = mol_s_only.intor('ECPscalar_ipipnuc', comp=9)
print(f"CPU ipipv shape: {h1_cpu_ipipv.shape}")
print(f"CPU ipipv values:")
for i in range(9):
    print(f"  Component {i}: {h1_cpu_ipipv[i,0,0]:.6e}")

# Get GPU result for ipipv
h1_gpu_ipipv = get_ecp_ipip(mol_s_only, 'ipipv').sum(axis=0).get()
print(f"\nGPU ipipv shape: {h1_gpu_ipipv.shape}")
print(f"GPU ipipv values:")
for i in range(9):
    print(f"  Component {i}: {h1_gpu_ipipv[i,0,0]:.6e}")

diff_ipipv = h1_cpu_ipipv - h1_gpu_ipipv
print(f"\nIPIPV Difference norm: {np.linalg.norm(diff_ipipv):.3e}")
print(f"IPIPV Max difference: {np.max(np.abs(diff_ipipv)):.3e}")

mol_s_only.stdout.close()