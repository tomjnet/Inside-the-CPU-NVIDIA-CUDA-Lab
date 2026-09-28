#!/usr/bin/env bash
# Install and Verify CUDA in 7 Minutes: Driver, Toolkit and Runtime - slide 4: step 1: the driver, with nvidia-smi
# Look at: the CUDA version in the header is the newest CUDA this driver supports (the ceiling), not the
# installed toolkit. If nvidia-smi is missing or shows no GPU, fix the GPU driver before anything else.
set -eu

# 1. the driver: ships with the GPU driver (on WSL: the Windows one)
nvidia-smi
#   NVIDIA-SMI 615.71.08  KMD Version: 616.92  CUDA UMD Version: 13.4
#   0  NVIDIA GeForce GTX 1650    0MiB / 4096MiB    0%    Default
#   typical lines, trimmed: 13.4 is the newest CUDA this driver runs
# the same facts as one CSV line, easy to keep next to a result
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
#   NVIDIA GeForce GTX 1650, 616.92, 7.5
# libcuda belongs to the driver: on WSL 2 Windows maps it in here
ls /usr/lib/wsl/lib/libcuda.so.1
