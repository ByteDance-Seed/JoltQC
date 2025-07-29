# xQC

**xQC** is a JIT-compiled GPU backend for executing key quantum chemistry algorithms with high performance. It is not intended to be a complete quantum chemistry package, but rather a collection of standalone, optimized CUDA kernels designed specifically for quantum chemistry workloads.

> ⚠️ **Experimental Project**  
> This project is under active development and may change significantly. Use at your own risk.

## Key Features

- High-performance GPU kernels for quantum chemistry
- JIT compilation to reduce dependency on the CUDA toolkit
- Python interface via [GPU4PySCF](https://github.com/pyscf/gpu4pyscf)
- Minimal runtime dependencies when used standalone

> The kernels are designed to be **self-contained**.  
> Python bindings for use with PySCF are provided via GPU4PySCF.  
> While xQC itself does **not** rely on the CUDA toolkit, additional dependencies are required when used with GPU4PySCF.  
> See [GPU4PySCF installation instructions](https://github.com/pyscf/gpu4pyscf) for full setup.

---

## Recommendations

- Use **PySCF/GPU4PySCF** for general-purpose workflows; xQC is a JIT backend
- NVIDIA **Ampere or newer** GPUs are recommended
- **Newer CUDA versions** (e.g., CUDA 12+) improve JIT compilation speed

---

## Python Interface

This example shows how to use xQC as a JIT backend with GPU4PySCF:

```python
import numpy as np
import pyscf
from gpu4pyscf import scf
from xqc.pyscf import jk

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-tzvpp')
mf = scf.RHF(mol)
mf.verbose = 1
mf.conv_tol = 1e-10
mf.max_cycle = 50

# Overwrite PySCF's get_jk with the JIT-compiled kernel from xQC
mf.get_jk = jk.generate_jk_kernel(dtype=np.float64) 
e_tot = mf.kernel()
print('Total energy with double precision:', e_tot)
```
See more examples in the examples/ directory.

## Limitations
- No support for density-fitting (DF); DF does not benefit significantly from JIT at this stage
- First runs may be slow due to JIT compilation (especially with large basis sets)
- Only RHF and RKS are currently supported

## Disclaimer

- The integral evaluation kernels are based on the [GPU4PySCF v1.4](https://github.com/gpu4pyscf/gpu4pyscf) project.

## Citation

If you use this project in your research, please cite:
```
@misc{wu2025designingquantumchemistryalgorithms,
      title={Designing quantum chemistry algorithms with just-in-time compilation}, 
      author={Xiaojie Wu, Qiming Sun and Yuanheng Wang},
      year={2025},
      eprint={2507.09772},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2507.09772}, 
}
```
