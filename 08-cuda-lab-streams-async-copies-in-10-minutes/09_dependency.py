#!/usr/bin/env python3
"""CUDA Streams and Asynchronous Copies in 10 Minutes - slide 9: dependencies between streams (Python version of 09_dependency.cu)

Run: python3 09_dependency.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
# Stream a uploads a lookup table, stream b runs the kernel that reads it. Two streams promise no order
# between them; cudaStreamWaitEvent makes b wait for the marker on a, on the device, without a host wait.
# The slide shows the runtime API; the driver API names differ: cudaStreamWaitEvent is cuStreamWaitEvent,
# cudaEventRecord is cuEventRecord, cudaMemset is cuMemsetD32 and cudaMemcpyAsync is cuMemcpyHtoDAsync.
KERNELS = r"""
extern "C" __global__ void lookup(const int* table, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = table[(n - 1) - i] + 1;              // reads the far end of the table first
}
"""


def pinned_ints(n: int) -> tuple[int, "np.ndarray"]:
    """cudaMallocHost: the pinned address and a numpy view of its n ints."""
    ptr = int(check(cu.cuMemAllocHost(n * 4), "cudaMallocHost"))
    return ptr, np.ctypeslib.as_array((ctypes.c_int * n).from_address(ptr))


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["lookup"], dev)
    n = 1 << 22                                              # 16 MiB table
    nbytes = n * 4                                           # sizeof(int)
    block = 256
    grid = (n + block - 1) // block

    a = check(cu.cuStreamCreate(0), "cudaStreamCreate a")
    b = check(cu.cuStreamCreate(0), "cudaStreamCreate b")
    ready = check(cu.cuEventCreate(0), "cudaEventCreate")
    table_ptr, h_table = pinned_ints(n)
    out_ptr, h_out = pinned_ints(n)
    d_table = check(cu.cuMemAlloc(nbytes), "cudaMalloc table")
    d_out = check(cu.cuMemAlloc(nbytes), "cudaMalloc out")
    check(cu.cuMemsetD32(d_table, 0, n), "cudaMemset")       # a stale table: all zeros
    h_table[:] = np.arange(n, dtype=np.int32)
    check(cu.cuCtxSynchronize(), "idle")

    # stream a uploads the table, stream b must not start before it
    check(cu.cuMemcpyHtoDAsync(d_table, table_ptr, nbytes, a), "upload")
    check(cu.cuEventRecord(ready, a), "cudaEventRecord")     # marker after the upload
    check(cu.cuStreamWaitEvent(b, ready, 0), "cudaStreamWaitEvent")   # b waits on the device,
    launch(fns["lookup"], grid, block, [(d_table, None), (d_out, None), (n, ctypes.c_int)],
           stream=b)                                         # the host does not
    check(cu.cuMemcpyDtoHAsync(out_ptr, d_out, nbytes, b), "download")
    check(cu.cuStreamSynchronize(b), "cudaStreamSynchronize")   # 1 synchronization point

    want = (n - 1 - np.arange(n, dtype=np.int64)) + 1
    stale = int(np.count_nonzero(h_out != want))
    print(f"with cudaStreamWaitEvent: {stale} of {n} results read a stale table")
    print("the dependency cost 1 event and 0 host waits: stream b waited on the device")
    print("without the wait, the kernel on b may race the upload on a: two streams promise no order")
    ok = stale == 0

    check(cu.cuStreamSynchronize(a), "cudaStreamSynchronize a")
    check(cu.cuEventDestroy(ready), "cudaEventDestroy")
    check(cu.cuMemFree(d_table), "cudaFree")
    check(cu.cuMemFree(d_out), "cudaFree")
    del h_table, h_out                                       # drop the views before the memory goes away
    check(cu.cuMemFreeHost(table_ptr), "cudaFreeHost")
    check(cu.cuMemFreeHost(out_ptr), "cudaFreeHost")
    check(cu.cuStreamDestroy(a), "cudaStreamDestroy")
    check(cu.cuStreamDestroy(b), "cudaStreamDestroy")
    check_kernel("final check")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
