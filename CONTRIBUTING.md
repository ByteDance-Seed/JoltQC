# Contributing to xQC

Thank you for your interest in contributing to xQC! We welcome all contributions. By participating, you agree to abide by our guidelines.

## How to Contribute

### Reporting Bugs or Requesting Features
The best way to contribute is by opening an issue on GitHub.
- For bugs, please provide a clear title, steps to reproduce, and any relevant logs.
- For feature requests, describe the enhancement and its potential benefits.

### Pull Requests
1.  **Fork & Clone:** Fork the repository and clone it locally.
    ```bash
    git clone https://github.com/your-username/xqc.git
    ```
2.  **Create a Branch:** Create a new branch for your changes.
    ```bash
    git checkout -b feature/my-new-feature
    ```
3.  **Make Changes:** Write your code, following the project's coding style.
4.  **Commit:** Commit your changes with a descriptive message.
5.  **Push & Open PR:** Push your branch to your fork and open a pull request.

## Development

### Prerequisites
- Python 3.x
- PySCF
- Cupy
- CUDA Toolkit

### Coding Style
- **Python:** Please follow PEP 8 guidelines.
- **C++/CUDA:** Our coding style is loosely based on the Google style. Please match the style of the existing codebase to maintain consistency.

### Testing
Please ensure that your changes do not break existing functionality. You can run the benchmarks in the `benchmarks/` directory to validate your changes. Add new tests for new features if applicable.

## License
By contributing, you agree that your contributions will be licensed under the Apache License 2.0, as found in the [LICENSE](LICENSE) file.
