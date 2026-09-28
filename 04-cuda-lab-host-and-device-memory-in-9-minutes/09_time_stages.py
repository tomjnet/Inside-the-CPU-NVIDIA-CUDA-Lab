#!/usr/bin/env python3
"""CUDA Host and Device Memory in 9 Minutes - slide 9: timing each stage separately (Python version of 09_time_stages.cu)

Run: python3 09_time_stages.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
"""
import ctypes
import sys
import time

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
# The slide shows the runtime API; the driver API names differ: cudaMalloc is cuMemAlloc, cudaMemcpy is
# cuMemcpyHtoD / cuMemcpyDtoH, the cudaEvent calls are cuEventCreate, cuEventRecord, cuEventSynchronize,
# cuEventElapsedTime and cuEventDestroy, and kernel<<<grid, block>>> is cuLaunchKernel, which returns the
# launch error itself (the runtime keeps it for cudaGetLastError).
KERNELS = r"""
extern "C" __global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global index, as in episode 3
    if (i < n) c[i] = a[i] + b[i];
}
"""


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["add"], dev)
    n = 1 << 22                                  # 4M floats: 16 MiB per vector
    nbytes = n * 4
    threads = 256
    blocks = (n + threads - 1) // threads
    h_a = np.full(n, 1.0, dtype=np.float32)
    h_b = np.full(n, 2.0, dtype=np.float32)
    h_c = np.zeros(n, dtype=np.float32)
    d_a = check(cu.cuMemAlloc(nbytes), "cudaMalloc a")
    d_b = check(cu.cuMemAlloc(nbytes), "cudaMalloc b")
    d_c = check(cu.cuMemAlloc(nbytes), "cudaMalloc c")
    check(cu.cuMemcpyHtoD(d_a, h_a.ctypes.data, nbytes), "upload a")
    check(cu.cuMemcpyHtoD(d_b, h_b.ctypes.data, nbytes), "upload b")
    args = [(d_a, None), (d_b, None), (d_c, None), (n, ctypes.c_int)]

    launch(fns["add"], blocks, threads, args)    # warm up: the first launch loads the kernel
    check_kernel("add warm up")

    # the wrong way: a host clock around the launch measures the launch, not the kernel
    # (from Python the launch call itself is slower than from C++, the lesson is the same)
    h0 = time.perf_counter_ns()
    launch(fns["add"], blocks, threads, args)
    h1 = time.perf_counter_ns()
    check_kernel("add, host clock")
    launch_ms = (h1 - h0) / 1e6

    start = check(cu.cuEventCreate(0), "cudaEventCreate")   # markers in the GPU queue
    stop = check(cu.cuEventCreate(0), "cudaEventCreate")

    check(cu.cuEventRecord(start, 0), "cudaEventRecord")    # device time, not host time
    launch(fns["add"], blocks, threads, args)                # launch() already checked that it was accepted
    check(cu.cuEventRecord(stop, 0), "cudaEventRecord")
    check(cu.cuEventSynchronize(stop), "cudaEventSynchronize")   # 1 synchronization point

    kernel_ms = float(check(cu.cuEventElapsedTime(start, stop), "cudaEventElapsedTime"))   # resolution about 0.5 us

    check(cu.cuMemcpyDtoH(h_c.ctypes.data, d_c, nbytes), "download c")
    wrong = int(np.count_nonzero(h_c != 3.0))

    print(f"add over {n} elements, this machine:")
    print(f"  perf_counter_ns around the launch only: {launch_ms:8.4f} ms  "
          "(host time: the call returned, the GPU had not finished)")
    print(f"  CUDA events around the kernel:          {kernel_ms:8.4f} ms  "
          "(device time: what the kernel really took)")
    print(f"elements that differ from 3: {wrong}")

    check(cu.cuEventDestroy(start), "cudaEventDestroy")
    check(cu.cuEventDestroy(stop), "cudaEventDestroy")
    for d in (d_a, d_b, d_c):
        check(cu.cuMemFree(d), "cudaFree")
    ok = wrong == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
