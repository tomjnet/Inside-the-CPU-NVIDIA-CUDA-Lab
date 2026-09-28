#!/usr/bin/env python3
"""CUDA Grids, Blocks and Threads in 8 Minutes - slide 5: vector addition (Python version of 05_vector_add.cu)

Run: python3 05_vector_add.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
# The slide shows the runtime API; the driver API names differ: cudaMalloc is cuMemAlloc, cudaMemcpy is
# cuMemcpyHtoD / cuMemcpyDtoH, cudaFree is cuMemFree and kernel<<<grid, block>>> is cuLaunchKernel.
KERNELS = r"""
extern "C" __global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];   // 2 global reads, 1 global write
}
"""


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["add"], dev)

    n = 1_000_000                               # not a multiple of 256
    threads = 256                               # a multiple of the warp
    blocks = (n + threads - 1) // threads       # 3907 blocks

    index = np.arange(n, dtype=np.int64)        # 4 MB per float array, 12 MB in total
    a = index.astype(np.float32)                # 3 * i stays exact in a float
    b = (2 * index).astype(np.float32)
    c = np.full(n, -1.0, dtype=np.float32)
    nbytes = a.nbytes

    # da, db, dc: device arrays, cudaMalloc and cudaMemcpy as in episode 2
    da = check(cu.cuMemAlloc(nbytes), "cudaMalloc a")
    db = check(cu.cuMemAlloc(nbytes), "cudaMalloc b")
    dc = check(cu.cuMemAlloc(nbytes), "cudaMalloc c")
    check(cu.cuMemcpyHtoD(da, a.ctypes.data, nbytes), "upload a")
    check(cu.cuMemcpyHtoD(db, b.ctypes.data, nbytes), "upload b")

    launch(fns["add"], blocks, threads,          # 1 kernel launch
           [(da, None), (db, None), (dc, None), (n, ctypes.c_int)])
    check_kernel("add")                          # 1 synchronization point
    # then 1 device to host copy of dc and a CPU check of every element
    check(cu.cuMemcpyDtoH(c.ctypes.data, dc, nbytes), "download c")
    check(cu.cuMemFree(da), "cudaFree a")
    check(cu.cuMemFree(db), "cudaFree b")
    check(cu.cuMemFree(dc), "cudaFree c")

    wrong = int(np.count_nonzero(c != a + b))    # the CPU sum, in float32 like the C++ check

    print(f"n = {n}, threads per block = {threads}, blocks = {blocks}")
    print(f"threads launched = {blocks * threads}, stopped by the guard = {blocks * threads - n}")
    print(f"c[0] = {float(c[0]):.0f}, c[1] = {float(c[1]):.0f}, c[n - 1] = {float(c[n - 1]):.0f}")
    print(f"elements that differ from the CPU sum: {wrong}")
    print("PASS" if wrong == 0 else "FAIL")
    return 0 if wrong == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
