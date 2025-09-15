import numpy as np
import pytest

try:
    import pyscf as _pyscf
    from pyscf.scf import hf as _hf
    HAS_PYSCF = True
except Exception:
    HAS_PYSCF = False


def test_minimal_h2_jk():
    # H2 with a minimal 1s orbital per H (STO-3G gives 1 contracted s AO per atom)
    # Local imports to avoid ImportError during collection when optional deps missing
    from jqc.pyscf.jk import generate_jk_kernel
    from jqc.pyscf.mol import BasisLayout
    from jqc.constants import TILE

    mol = _pyscf.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis={"H": "sto-3g"},
        cart=True,
        output="/dev/null",
        verbose=0,
    )

    # Simple hermitian test density (identity in AO basis)
    nao = mol.nao
    dm = np.eye(nao, dtype=np.float64)

    # JoltQC JK kernel
    layout = BasisLayout.from_mol(mol, alignment=TILE)
    print(layout.angs)
    print(layout.angs_no_pad)
    print(layout.to_split_map)
    print(layout.nprims)
    print(layout.pad_id)
    print(layout.ao_loc)
    print(layout.mol_ao_loc)
    print(layout.ao_loc_no_pad)
    #exit()
    get_jk = generate_jk_kernel(layout)
    vj_jqc, vk_jqc = get_jk(mol, dm, hermi=1)
    # Convert to numpy if they are CuPy arrays
    try:
        import cupy as cp  # type: ignore
        vj_jqc = vj_jqc.get() if isinstance(vj_jqc, cp.ndarray) else vj_jqc
        vk_jqc = vk_jqc.get() if isinstance(vk_jqc, cp.ndarray) else vk_jqc
    except Exception:
        pass
    
    # PySCF reference
    mf_ref = _hf.RHF(mol)
    vj_ref, vk_ref = mf_ref.get_jk(mol, dm, hermi=1)

    print(vj_ref)
    print(vj_jqc)
    exit()
    # Accuracy checks (double precision tolerance)
    assert vj_jqc.shape == vj_ref.shape == (nao, nao)
    assert vk_jqc.shape == vk_ref.shape == (nao, nao)
    assert np.max(np.abs(vj_jqc - vj_ref)) < 1e-7
    assert np.max(np.abs(vk_jqc - vk_ref)) < 1e-7

test_minimal_h2_jk()
