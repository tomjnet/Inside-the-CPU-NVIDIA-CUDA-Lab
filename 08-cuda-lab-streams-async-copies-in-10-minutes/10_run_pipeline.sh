#!/usr/bin/env bash
# CUDA Streams and Asynchronous Copies in 10 Minutes - slide 10: run the pipeline, read the timeline
# Look at: the two end to end times, the ratio, and the rows of the streams timeline, where the U, K and D
# of neighbouring chunks share the same columns while the serial rows never do.
set -eu

# build the pipeline of slide 8, then run one mode at a time
nvcc -std=c++20 -O2 08_pipeline.cu -o 08_pipeline
./08_pipeline serial
./08_pipeline streams
#   typical, trimmed: U upload, K kernel, D download, time runs right
#   serial   chunk 0 stream 0 |UUKKKDDD................|
#   serial   chunk 1 stream 0 |........UUKKKDDD........|
#   streams  chunk 0 stream 0 |UUKKKDDD................|
#   streams  chunk 1 stream 1 |.UUUKKKKDDD.............|

# both modes in one run: the ratio is end to end, steady_clock
./08_pipeline
