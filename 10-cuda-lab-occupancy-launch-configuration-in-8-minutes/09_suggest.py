#!/usr/bin/env python3
"""CUDA Occupancy and Launch Configuration in 8 Minutes - slide 9: the toolkit's suggestion (Python version of 09_suggest.cu)

Run: python3 09_suggest.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
# The slide shows the runtime API; cuda-python calls the driver API: cudaMalloc is cuMemAlloc,
# cudaMemcpy is cuMemcpyHtoD / cuMemcpyDtoH, cudaFree is cuMemFree, cudaDeviceSynchronize is cuCtxSynchronize.


def time_ms(launch_once) -> float:
    """Device time of one launch with CUDA events: 1 warm up, then the median of 7 runs.

    The events are stamped on the device, so Python's slower host side mostly stays out of the
    number; a very short kernel can still read a little longer than from C++ (this machine).
    """
    start = check(cu.cuEventCreate(0), "event")
    stop = check(cu.cuEventCreate(0), "event")
    launch_once()                                # warm up: first launch pays setup
    runs: list[float] = []
    for _ in range(7):
        check(cu.cuEventRecord(start, 0), "record")
        launch_once()
        check(cu.cuEventRecord(stop, 0), "record")
        check(cu.cuEventSynchronize(stop), "sync")   # 1 synchronization point per run
        runs.append(float(check(cu.cuEventElapsedTime(start, stop), "elapsed")))
    check(cu.cuEventDestroy(start), "event")
    check(cu.cuEventDestroy(stop), "event")
    runs.sort()
    return runs[len(runs) // 2]


KERNELS = r"""
extern "C" __global__ void scale_add(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global index
    if (i < n) y[i] = a * x[i] + y[i];               // 2 global reads, 1 write
}
"""


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["scale_add"], dev)
    scale_add = fns["scale_add"]
    n = 1 << 22                                 # 4 M floats, 16 MiB per array
    h_x = (np.arange(n) % 7).astype(np.float32)
    h_y = np.empty(n, dtype=np.float32)
    d_x = check(cu.cuMemAlloc(h_x.nbytes), "cudaMalloc x")
    d_y = check(cu.cuMemAlloc(h_y.nbytes), "cudaMalloc y")
    check(cu.cuMemcpyHtoD(d_x, h_x.ctypes.data, h_x.nbytes), "upload")
    check(cu.cuMemsetD32(d_y, 0, n), "memset")  # cudaMemset of zero bytes: 32 bit zeros here

    # the toolkit's advice: cudaOccupancyMaxPotentialBlockSize is cuOccupancyMaxPotentialBlockSize;
    # the two zeros say "no dynamic shared memory", the last zero "any block size the kernel allows"
    min_grid, suggested = check(cu.cuOccupancyMaxPotentialBlockSize(scale_add, 0, 0, 0), "suggest")
    # suggested: highest occupancy, then the fewest blocks
    # min_grid: blocks that fill every SM once
    # typical GTX 1650: suggested 1024, min_grid 16
    blocks = (n + suggested - 1) // suggested   # still cover n
    args = [(n, ctypes.c_int), (2.0, ctypes.c_float), (d_x, None), (d_y, None)]
    launch(scale_add, blocks, suggested, args)
    check_kernel("scale_add")                   # then time it: sweep

    check(cu.cuMemcpyDtoH(h_y.ctypes.data, d_y, h_y.nbytes), "download")
    ok = suggested > 0 and min_grid > 0 and bool(np.all(h_y == np.float32(2.0) * h_x))

    print(f"suggested block size {suggested}, minimum grid {min_grid} blocks, our grid {blocks} blocks")
    ts = time_ms(lambda: launch(scale_add, blocks, suggested, args))
    check_kernel("scale_add suggested")
    t256 = time_ms(lambda: launch(scale_add, (n + 255) // 256, 256, args))
    check_kernel("scale_add 256")
    print(f"median device time: suggested {ts:.3f} ms, 256 threads {t256:.3f} ms (this machine)")
    print("the suggestion knows the resource tables, not the memory traffic: measure")

    check(cu.cuMemFree(d_x), "cudaFree x")
    check(cu.cuMemFree(d_y), "cudaFree y")
    print(f"y = 2 x checked on the CPU for {n} elements")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
