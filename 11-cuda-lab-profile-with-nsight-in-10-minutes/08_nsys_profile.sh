#!/usr/bin/env bash
# Profile CUDA with NVIDIA Nsight in 10 Minutes - slide 8: record a timeline with nsys
# Look at: the Generated line names one .nsys-rep report per mode; in the GUI, compare the kernel row
# of before and after: empty during every copy in before, busy next to the copies in after. WSL may miss GPU rows.
set -eu

# build once, then record one timeline per mode
make 07_pipeline_nvtx
nsys profile -t cuda,nvtx -f true -o before ./07_pipeline_nvtx before
nsys profile -t cuda,nvtx -f true -o after ./07_pipeline_nvtx after
#   Generated:
#       /home/developer/lab/before.nsys-rep
#   typical line: one report file per run
# open both reports in the Nsight Systems GUI, one above the other
nsys-ui before.nsys-rep after.nsys-rep
