<div align="center">
 👋 Hi, everyone! 
    <br>
    We are <b>ByteDance Seed team.</b>
</div>

<p align="center">
  You can get to know us better through the following channels👇
  <br>
  <a href="https://seed.bytedance.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/5793e67c-79bb-4a59-811a-fcc7ed510bd4">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
</p>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)

# JoltQC: a JIT-compiled GPU backend for quantum chemistry calculations
*formerly xQC*
<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="https://github.com/ByteDance-Seed/joltqc/actions/workflows/lint.yml">
    <img src="https://github.com/ByteDance-Seed/joltqc/actions/workflows/lint.yml/badge.svg"></a>
</p>

> ⚠️ **Experimental Project**  
> This project is under active development and may change significantly. Use at your own risk.

## Overview
JoltQC is not intended to be a complete quantum chemistry package, but rather a collection of standalone, optimized CUDA kernels designed specifically for quantum chemistry workloads. Please see [GPU4PySCF installation instructions](https://github.com/pyscf/gpu4pyscf) for full capabilities of quantum chemistry calculations.

## Key Features

- High-performance GPU kernels for quantum chemistry
- Completely JIT compilation, without any pre-compiled kernels
- Support FP64, FP32, and mixed-precision schemes
- Python interface to [GPU4PySCF](https://github.com/pyscf/gpu4pyscf)

## Recommendations

- Use **PySCF/GPU4PySCF** for general-purpose workflows; JoltQC is a JIT backend
- NVIDIA **Ampere or newer** GPUs are recommended
- **Newer CUDA versions** (e.g., CUDA 12.4+) improve JIT compilation speed

## Installation

```bash
pip3 install -e .
```

## Python Interface

This example shows how to use JoltQC as a JIT backend with GPU4PySCF:

```python
import numpy as np
import pyscf
from gpu4pyscf import scf
import jqc.pyscf

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

# In-place overwrite PySCF kernels with JIT-compiled versions
mf = jqc.pyscf.apply(mf)
e_tot = mf.kernel()
print('Total energy with double precision:', e_tot)
```
See more examples in the examples/ directory.

## Supported JIT-Compiled Kernels

| Kernel Type | Description | Precision Support |
|-------------|-------------|-------------------|
| **J/K Potentials** | Coulomb/exchange matrix construction | FP64 (default), FP32, Mixed |
| **DFT Density** | Electron density evaluation on grids | FP64, FP32, Mixed (default) |
| **XC Potentials** | Exchange-correlation evaluation | FP64, FP32, Mixed (default) |
| **ECP & Derivatives** | Effective Core Potentials | FP64 only |

![JoltQC vs GPU4PySCF Speedup](benchmarks/media/jqc_vs_gpu4pyscf_speedup.png)


## Limitations
- No support for density-fitting (DF); DF does not benefit significantly from JIT at this stage.
- First run may be slow due to JIT compilation (especially with large basis sets), the compiled kernels will be cached in disk for following runs.
- Only RHF and RKS methods are currently supported.
- The performance of small systems is bounded by Python overhead and kernel launch overhead.
- Support up to 65535 atomic basis.
- Multi-GPU is not supported yet.

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

## About [ByteDance Seed Team](https://seed.bytedance.com/)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society.
