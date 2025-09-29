#!/usr/bin/env python3

import cupy as cp
import numpy as np
from pyscf import gto

# Add debugging for IPIP issues
from jqc.backend.ecp import get_ecp_ipip

# Skip module if no CUDA device is available
try:
    _ndev = cp.cuda.runtime.getDeviceCount()
except Exception:
    _ndev = 0

if _ndev > 0:
    print(f"Found {_ndev} CUDA devices")
    print(f"Current device: {cp.cuda.Device()}")

    # Test case with exact basis from test_ecp_small.py
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

    # Type2 ECP: semi-local S, P, D channels like in test_ecp_small.py
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

    # Test with two atoms like in test_ecp_small.py
    mol_type2 = gto.M(
        atom='''
            Na 0.5 0.5 0.
            Na  0.  1.  1.
        ''',
        basis={'Na': cu1_basis_small},
        ecp={'Na': na_ecp_type2},
        output="/dev/null",
        verbose=0,
    )

    print("Molecule created successfully")
    print(f"Number of ECP atoms: {len(set(mol_type2._ecpbas[:, gto.ATOM_OF]))}")
    print(f"Number of shells: {mol_type2.nbas}")

    try:
        print("Testing basic ECP...")
        from jqc.backend.ecp import get_ecp
        h1_gpu = get_ecp(mol_type2).get()
        print(f"Basic ECP shape: {h1_gpu.shape}")

        print("Testing ECP IP...")
        from jqc.backend.ecp import get_ecp_ip
        h1_ip = get_ecp_ip(mol_type2)
        print(f"ECP IP completed")

        print("Testing ECP IPIP ipvip...")
        h1_ipip = get_ecp_ipip(mol_type2, 'ipvip')
        print(f"ECP IPIP ipvip shape: {h1_ipip.shape}")

        print("Testing ECP IPIP ipipv...")
        h1_ipip2 = get_ecp_ipip(mol_type2, 'ipipv')
        print(f"ECP IPIP ipipv shape: {h1_ipip2.shape}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    mol_type2.stdout.close()
else:
    print("No CUDA devices found")