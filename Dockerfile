# This Dockerfile is used for automated testing of WarpX on Shippable

FROM ubuntu:14.04

# Install a few packages, as root
RUN apt-get update \
    && apt-get install -y \
    wget \
    make \
    git \
    gcc \
    gfortran \
    g++ \
    python \
    python-numpy

WORKDIR /home/

# Install MPI
RUN apt-get install -y \
    openmpi-bin \
libopenmpi-dev

# Install FFTW
RUN apt-get install -y \
    libfftw3-dev \
    libfftw3-mpi-dev

# Clone amrex
RUN git clone https://github.com/AMReX-Codes/amrex.git \
    && cd amrex/ \
    && git checkout development \
    && cd ..

# Clone their regression test utility
RUN git clone https://github.com/AMReX-Codes/regression_testing.git

# Clone picsar
RUN git clone https://bitbucket.org/berkeleylab/picsar.git

# Copy warpx
RUN mkdir -p /home/warpx/
COPY ./ /home/warpx/

# Prepare regression tests
RUN mkdir -p rt-WarpX/WarpX-benchmarks \
    && cd warpx/Regression \
    && python prepare_file_shippable.py \
    && cp shippable-tests.ini ../../rt-WarpX
