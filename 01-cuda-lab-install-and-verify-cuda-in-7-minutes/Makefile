# Inside the CPU: NVIDIA CUDA Lab - code samples (one program per .cu or .cpp file)
#
#   make                 build every sample (.cu with nvcc, .cpp with g++)
#   make run             build and run every sample, one after the other
#   make 03_vector_add   build one sample
#   make run-03_vector_add
#   make clean           remove the binaries
#
# Linux or WSL 2 with the CUDA Toolkit installed. nvcc is searched in PATH, then in $(CUDA_HOME)/bin.
# Pick the GPU generation with ARCH, for example make ARCH=-arch=native (default: the toolkit default).

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(shell command -v nvcc 2>/dev/null || echo $(CUDA_HOME)/bin/nvcc)
STD       ?= c++20
ARCH      ?=
NVCCFLAGS ?= -std=$(STD) -O2 $(ARCH) -Xcompiler -Wall,-Wextra
CXXFLAGS  ?= -std=$(STD) -Wall -Wextra -pedantic -O2 -pthread

CU_SRCS  := $(sort $(wildcard *.cu))
CPP_SRCS := $(sort $(wildcard *.cpp))
BINS     := $(CU_SRCS:.cu=) $(CPP_SRCS:.cpp=)

all: $(BINS)
	@echo built $(words $(BINS)) samples with $(NVCC) and $(CXX)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

run: $(addprefix run-,$(BINS)) py

run-%: %
	@echo == $< ==
	./$< || exit 1

# Python twins: NN_name.py beside NN_name.cu or NN_name.cpp. The CUDA ones compile the same kernel text with
# NVRTC through NVIDIA's cuda-python and run it on the GPU:  pip install cuda-python numpy
#   make py               run every Python sample       make py-03_vector_add   run one
PYTHON ?= python3
PYS := $(sort $(wildcard *.py))

py: $(addprefix py-,$(PYS:.py=))

py-%: %.py
	@echo == $< ==
	$(PYTHON) $< || exit 1

clean:
	rm -f $(BINS) *.o *.nsys-rep *.sqlite
	rm -rf __pycache__

.PHONY: all run clean py
