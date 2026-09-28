#!/usr/bin/env bash
# Profile CUDA with NVIDIA Nsight in 10 Minutes - slide 9: read the summary with nsys stats
# Look at: kernel time against copy time (roughly half each in before), and the share of cudaDeviceSynchronize
# in the API table: that is the CPU waiting for a GPU that does one thing at a time. Run 08_nsys_profile.sh first.
set -eu

# where does the GPU time go? kernels first, then copies
nsys stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum before.nsys-rep
#   Time (%)  Total Time (ns)  Instances  Name
#       48.9       7,812,416         16  scale(float *, int, float)
#       25.8       4,121,088         16  [CUDA memcpy Host-to-Device]
#       25.3       4,040,192         16  [CUDA memcpy Device-to-Host]
# and the CPU side: which API call does the waiting?
nsys stats -r cuda_api_sum before.nsys-rep
#       91.7      16,502,784          1  cudaDeviceSynchronize
#   typical lines, trimmed: your GPU gives other numbers
