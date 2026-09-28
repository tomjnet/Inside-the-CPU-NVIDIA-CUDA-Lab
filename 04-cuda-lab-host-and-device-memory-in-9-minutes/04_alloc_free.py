#!/usr/bin/env python3
"""CUDA Host and Device Memory in 9 Minutes - slide 4: cudamalloc and cudafree (Python version of 04_alloc_free.cu)

Run: python3 04_alloc_free.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
# The slide shows the runtime API; the driver API names differ: cudaMalloc is cuMemAlloc, cudaFree is
# cuMemFree, cudaMemGetInfo is cuMemGetInfo, cudaMemcpy is cuMemcpyDtoH and kernel<<<grid, block>>> is cuLaunchKernel.
KERNELS = r"""
namespace std { using size_t = ::size_t; }   // NVRTC has no <cstddef>: give the kernel its std::size_t

extern "C" __global__ void fill(float* p, float v, std::size_t n) {   // a kernel may use the device address
    std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < n) p[i] = v;
}
"""


def error_text(err) -> str:
    """cudaGetErrorString: the readable text of one status."""
    message = cu.cuGetErrorString(err)[1]
    return message.decode() if message else str(err)


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["fill"], dev)
    n = 1 << 20
    mib = 1024.0 * 1024.0

    nbytes = n * 4                               # 1M floats: 4 MiB of VRAM
    d_a = check(cu.cuMemAlloc(nbytes), "cudaMalloc")   # d_ prefix: device address, never dereferenced on host

    free_b, total_b = check(cu.cuMemGetInfo(), "cudaMemGetInfo")   # how much VRAM is left?

    err, d_huge = cu.cuMemAlloc(1 << 40)         # a refused allocation is a status, not an exception
    print(f"1 TiB: {error_text(err)}")
    # the runtime also records the error for cudaGetLastError; the driver API only returns it, nothing to clear

    print(f"allocated {nbytes} bytes ({nbytes / mib:.1f} MiB) at device address 0x{int(d_a):x}")
    print(f"VRAM now: {free_b / mib:.0f} MiB free of {total_b / mib:.0f} MiB (other programs share this number)")
    # a Python int holding d_a is only a number: there is no way to write through it on the host
    launch(fns["fill"], (n + 255) // 256, 256,   # valid on the device: the same address
           [(d_a, None), (1.0, ctypes.c_float), (n, ctypes.c_size_t)])
    check_kernel("fill")
    first = np.zeros(1, dtype=np.float32)        # read it back the only legal way: a copy
    check(cu.cuMemcpyDtoH(first.ctypes.data, d_a, 4), "cudaMemcpy")
    print(f"a kernel wrote 1.0 through that address; copied back: {float(first[0]):g}")

    check(cu.cuMemFree(d_a), "cudaFree")         # one cudaFree per cudaMalloc
    check(cu.cuMemFree(0), "cudaFree null")      # freeing a null pointer is allowed and does nothing

    # cuda-python hands back None as the value of a failed call: the Python spelling of a null pointer
    huge_is_null = d_huge is None or int(d_huge) == 0
    ok = (err != cu.CUresult.CUDA_SUCCESS and huge_is_null
          and total_b >= free_b and float(first[0]) == 1.0)
    print(f"refused allocation left the pointer null and the program alive: {'yes' if ok else 'no'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
