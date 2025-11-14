# Gemini Guide to JoltQC

This document provides a guide for developers and large language models (LLMs) to understand, navigate, and contribute to the JoltQC project.

## 1. Project Overview

JoltQC is a high-performance, Just-In-Time (JIT) compiled GPU backend for quantum chemistry calculations. It is not a standalone quantum chemistry package but rather a collection of optimized CUDA kernels designed to accelerate existing CPU-based packages like PySCF. It is intended to be used as a backend for [GPU4PySCF](https://github.com/pyscf/gpu4pyscf).

**Key Features**:

*   **JIT Compilation**: Kernels are compiled on-the-fly, allowing for specialization and optimization based on the specific calculation and hardware.
*   **High Performance**: Optimized for modern NVIDIA GPUs (Ampere or newer recommended).
*   **Precision Control**: Supports double (FP64), single (FP32), and mixed-precision calculations.
*   **PySCF Integration**: Provides a Python interface to seamlessly patch and accelerate PySCF calculations.

## 2. Getting Started

### Prerequisites

*   Python (>= 3.8)
*   NVIDIA GPU with CUDA Toolkit (12.x recommended)
*   `cupy-cuda12x`
*   `pyscf`
*   `gpu4pyscf-cuda12x`

### Installation

Clone the repository and install the package in editable mode:

```bash
pip3 install -e .
```

### Environment Setup

The project includes a `setenv.sh` script to configure the environment. This script adds the project root to the `PYTHONPATH`.

To set up your environment, run:

```bash
source setenv.sh
```

## 3. Development

### Code Style and Linting

This project uses [Black](https://github.com/psf/black) for code formatting and [Ruff](https://github.com/astral-sh/ruff) for linting.

To check for formatting issues:

```bash
black --check --diff .
```

To run the linter:

```bash
ruff check .
```

### Running Tests

The project uses [pytest](https://docs.pytest.org/) for testing. The tests are located in the `jqc/backend/tests` and `jqc/pyscf/tests` directories.

To run the test suite, execute the following command from the project root:

```bash
source setenv.sh
python3 -m pytest -v --tb=long
```

## 4. Project Structure

```
├── jqc/                  # Main source code
│   ├── backend/          # Core CUDA kernels and backend logic
│   │   ├── dft/          # CUDA source files for DFT
│   │   ├── ecp/          # CUDA source files for ECP
│   │   ├── rys/          # CUDA source files for Rys quadrature
│   │   └── tests/        # Tests for backend components
│   └── pyscf/            # Integration with PySCF
│       └── tests/        # Tests for PySCF integration
├── examples/             # Example scripts demonstrating usage
├── benchmarks/           # Benchmarking scripts
├── pyproject.toml        # Project metadata and dependencies
└── README.md             # Project overview
```

## 5. Key Concepts

### JIT Compilation

The core of JoltQC is its JIT compilation engine. Instead of shipping pre-compiled kernels, JoltQC compiles the CUDA code at runtime. This has several advantages:

*   **Specialization**: Kernels can be optimized for the specific molecular system (e.g., basis set, angular momentum) and hardware.
*   **Flexibility**: New features and optimizations can be added without requiring users to recompile the entire package.
*   **Reduced Binary Size**: The distributed package is smaller as it only contains the source code for the kernels.

The first time a calculation is run, there will be a compilation delay. The compiled kernels are cached on disk for subsequent runs.

### Integration with PySCF

JoltQC is designed to work with PySCF. The `jqc.pyscf.apply(mf)` function "patches" a PySCF mean-field object (`mf`) by replacing its integral functions with JoltQC's JIT-compiled GPU versions. This allows for a significant speedup in the most computationally intensive parts of the calculation, such as the construction of the Coulomb (J) and Exchange (K) matrices.
