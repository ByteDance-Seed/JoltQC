FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_ARCH_LIST="100;120"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    wget \
    curl \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install numpy scipy h5py pyscf cupy-cuda12x cutensor-cu12 basis-set-exchange
RUN pip3 install pyscf-dispersion geometric

# Clone gpu4pyscf v1.4.0
WORKDIR /opt
RUN git clone --branch v1.4.0 --depth 1 https://github.com/pyscf/gpu4pyscf.git

# Build gpu4pyscf with cmake
WORKDIR /opt/gpu4pyscf
RUN cmake -B build -S gpu4pyscf/lib \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ARCHITECTURES="${CUDA_ARCH_LIST}" \
    -DBUILD_LIBXC=ON

RUN cd build && make -j1

# Set environment variables for gpu4pyscf
ENV PYTHONPATH="/opt/gpu4pyscf:${PYTHONPATH}"

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]