#!/usr/bin/env bash
# Install and Verify CUDA in 7 Minutes: Driver, Toolkit and Runtime - slide 5: step 2: the toolkit, with nvcc
# Look at: the release nvcc prints is the toolkit version; it must not be newer than the CUDA version that
# nvidia-smi showed. On native Ubuntu replace wsl-ubuntu with ubuntu2404 in the repository path.
set -eu

# 2. the toolkit: nvcc, headers, libraries, tools (never the driver)
repo=https://developer.download.nvidia.com/compute/cuda/repos
wget $repo/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install -y cuda-toolkit-13-3
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version
#   Cuda compilation tools, release 13.3, V13.3.73   (typical)
# 3. the runtime: libcudart, linked statically unless you ask
nvcc -std=c++20 -O2 06_device_query.cu -o 06_device_query
./06_device_query
