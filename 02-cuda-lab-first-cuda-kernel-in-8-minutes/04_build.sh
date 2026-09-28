#!/usr/bin/env bash
# Your First CUDA Kernel in 8 Minutes - slide 4: compile and run with nvcc
# Look at: the four Hello lines arrive only at the synchronization. Comment out cudaDeviceSynchronize()
# in 03_hello_kernel.cu, build again, and they may never appear.
set -eu

# one file, two compilers: nvcc splits host code from device code
nvcc -std=c++20 -O2 03_hello_kernel.cu -o 03_hello_kernel

# run it: the four lines arrive at the synchronization (typical output)
./03_hello_kernel
#   Hello GPU from thread 0
#   Hello GPU from thread 1
#   Hello GPU from thread 2
#   Hello GPU from thread 3

# or build every sample of the folder with the Makefile
make
