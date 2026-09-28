#!/usr/bin/env bash
# Profile CUDA with NVIDIA Nsight in 10 Minutes - slide 11: measure again, then Nsight Compute
# Look at: the shorter wait in the after report, then Duration, Memory Throughput and Achieved Occupancy of one
# launch of scale. ncu needs GPU performance counter access; WSL and some drivers refuse it (ERR_NVGPUCTRPERM).
set -eu

# 1. measure again: the same command, the other mode
nsys stats -r cuda_api_sum after.nsys-rep
#       84.2       9,630,208          1  cudaDeviceSynchronize
# 2. zoom into one kernel: Nsight Compute replays it with counters
ncu -k scale -c 1 --set basic ./07_pipeline_nvtx after
#   Duration                       us        488
#   Memory Throughput               %      81.4
#   Achieved Occupancy              %      89.7
#   typical lines, trimmed; WSL may refuse the counters
