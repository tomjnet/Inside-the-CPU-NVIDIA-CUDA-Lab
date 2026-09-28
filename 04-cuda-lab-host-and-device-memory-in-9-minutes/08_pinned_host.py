#!/usr/bin/env python3
"""CUDA Host and Device Memory in 9 Minutes - slide 8: cudamallochost (Python version of 08_pinned_host.cu)

Run: python3 08_pinned_host.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
"""
import ctypes
import sys

# ---- CUDA from Python: NVIDIA's cuda-python (pip install cuda-python numpy) ----------------
# The kernel is the same CUDA C++ text as in the .cu sample; NVRTC compiles it at run time.
try:
    import numpy as np
    from cuda.bindings import driver as cu, nvrtc
except ImportError as e:                       # samples exit 0 without the libraries
    print(f"needs cuda-python and numpy: {e.name} is missing (pip install cuda-python numpy)")
    sys.exit(0)


def check(result, what="CUDA call"):
    """Every cuda-python call returns (status, values...): stop on the first error, like check() in C++."""
    err, *rest = result
    if isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            sys.exit(f"{what}: {nvrtc.nvrtcGetErrorString(err)[1].decode()}")
    elif err != cu.CUresult.CUDA_SUCCESS:
        name = cu.cuGetErrorName(err)[1]
        sys.exit(f"{what}: {name.decode() if name else err}")
    return rest[0] if len(rest) == 1 else (tuple(rest) if rest else None)


def require_device():
    """Samples exit 0 without a GPU; returns (device, context) or None."""
    err, = cu.cuInit(0)
    if err != cu.CUresult.CUDA_SUCCESS:
        print(f"no CUDA device: {cu.cuGetErrorName(err)[1].decode()}")
        return None
    if check(cu.cuDeviceGetCount(), "cuDeviceGetCount") == 0:
        print("no CUDA device: count is 0")
        return None
    dev = check(cu.cuDeviceGet(0), "cuDeviceGet")
    ctx = check(cu.cuDevicePrimaryCtxRetain(dev), "cuDevicePrimaryCtxRetain")
    check(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    return dev, ctx


def compile_kernels(source, names, dev):
    """NVRTC: CUDA C++ text to a module for this GPU; returns {name: CUfunction}."""
    major = check(cu.cuDeviceGetAttribute(cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "attr")
    minor = check(cu.cuDeviceGetAttribute(cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "attr")
    prog = check(nvrtc.nvrtcCreateProgram(source.encode(), b"sample.cu", 0, [], []), "nvrtcCreateProgram")
    opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"-std=c++17"]
    err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        size = check(nvrtc.nvrtcGetProgramLogSize(prog), "log size")
        log = b" " * size
        check(nvrtc.nvrtcGetProgramLog(prog, log), "log")
        sys.exit("kernel does not compile:\n" + log.decode(errors="replace"))
    size = check(nvrtc.nvrtcGetCUBINSize(prog), "nvrtcGetCUBINSize")
    cubin = b" " * size
    check(nvrtc.nvrtcGetCUBIN(prog, cubin), "nvrtcGetCUBIN")
    module = check(cu.cuModuleLoadData(cubin), "cuModuleLoadData")
    return {n: check(cu.cuModuleGetFunction(module, n.encode()), f"cuModuleGetFunction {n}") for n in names}


def launch(fn, grid, block, args, shared_bytes=0, stream=0):
    """kernel<<<grid, block>>>(args...): args is a list of (value, ctypes type) pairs."""
    grid = (grid, 1, 1) if isinstance(grid, int) else tuple(grid) + (1,) * (3 - len(grid))
    block = (block, 1, 1) if isinstance(block, int) else tuple(block) + (1,) * (3 - len(block))
    values = tuple(v for v, _ in args)
    types = tuple(t for _, t in args)
    check(cu.cuLaunchKernel(fn, *grid, *block, shared_bytes, stream, (values, types), 0), "cuLaunchKernel")


def check_kernel(name):
    """A launch returns nothing useful: the fault shows up at the next synchronization."""
    check(cu.cuCtxSynchronize(), name)


# ---- the sample ----------------------------------------------------------------------------
# The slide shows the runtime API; the driver API names differ: cudaMallocHost is cuMemAllocHost, cudaFreeHost is
# cuMemFreeHost, cudaMalloc is cuMemAlloc, cudaMemcpy is cuMemcpyHtoD / cuMemcpyDtoH and the cudaEvent calls are
# cuEventCreate, cuEventRecord, cuEventSynchronize, cuEventElapsedTime and cuEventDestroy.
KERNELS = r"""
namespace std { using size_t = ::size_t; }   // NVRTC has no <cstddef>: give the kernel its std::size_t

extern "C" __global__ void add_one(float* p, std::size_t n) {
    std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < n) p[i] = p[i] + 1.0f;
}
"""


def time_copy_ms(d_dst, h_src: int, nbytes: int) -> float:
    """Device time of one host to device copy, measured with CUDA events: one warm up, then the best of five."""
    start = check(cu.cuEventCreate(0), "cudaEventCreate")
    stop = check(cu.cuEventCreate(0), "cudaEventCreate")
    check(cu.cuMemcpyHtoD(d_dst, h_src, nbytes), "warm up copy")
    best = 1e30
    for _ in range(5):
        check(cu.cuEventRecord(start, 0), "cudaEventRecord")
        check(cu.cuMemcpyHtoD(d_dst, h_src, nbytes), "timed copy")
        check(cu.cuEventRecord(stop, 0), "cudaEventRecord")
        check(cu.cuEventSynchronize(stop), "cudaEventSynchronize")
        best = min(best, float(check(cu.cuEventElapsedTime(start, stop), "cudaEventElapsedTime")))
    check(cu.cuEventDestroy(start), "cudaEventDestroy")
    check(cu.cuEventDestroy(stop), "cudaEventDestroy")
    return best


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["add_one"], dev)
    n = 1 << 22                                  # 4M floats: 16 MiB
    nbytes = n * 4
    d_a = check(cu.cuMemAlloc(nbytes), "cudaMalloc")

    pinned_ptr = int(check(cu.cuMemAllocHost(nbytes), "cudaMallocHost"))   # page locked host memory:
    h_pinned = np.ctypeslib.as_array((ctypes.c_float * n).from_address(pinned_ptr))   # the GPU can DMA from it
    h_pageable = np.full(n, 1.0, dtype=np.float32)   # ordinary numpy memory, pageable like new or malloc
    h_pinned[:] = 1.0

    # same bytes, same call; pageable pays one extra copy to a staging
    # buffer inside the driver, pinned goes straight across PCIe
    ms_pageable = time_copy_ms(d_a, h_pageable.ctypes.data, nbytes)
    ms_pinned = time_copy_ms(d_a, pinned_ptr, nbytes)

    launch(fns["add_one"], (n + 255) // 256, 256, [(d_a, None), (n, ctypes.c_size_t)])   # the device saw the pinned upload
    check_kernel("add_one")
    h_back = np.zeros(n, dtype=np.float32)      # CPU check: what arrived is what was sent, plus one
    check(cu.cuMemcpyDtoH(h_back.ctypes.data, d_a, nbytes), "download")
    wrong = int(np.count_nonzero(h_back != 2.0))

    # CUDA events time the copy on the device, so Python adds nothing to these numbers
    gib = 1024.0 * 1024.0 * 1024.0
    print(f"host to device, {nbytes >> 20} MiB, CUDA events (device time), best of 5, this machine:")
    print(f"  pageable  {ms_pageable:8.3f} ms  {nbytes / gib / (ms_pageable / 1000.0):6.2f} GiB/s")
    print(f"  pinned    {ms_pinned:8.3f} ms  {nbytes / gib / (ms_pinned / 1000.0):6.2f} GiB/s")
    print(f"  ratio pageable / pinned: {ms_pageable / ms_pinned:.2f} (a timing, never a pass condition)")

    del h_pinned                                 # drop the view before the memory goes away
    check(cu.cuMemFreeHost(pinned_ptr), "cudaFreeHost")   # never free() or delete
    check(cu.cuMemFree(d_a), "cudaFree")

    ok = wrong == 0
    print(f"elements that differ after the round trip: {wrong}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
