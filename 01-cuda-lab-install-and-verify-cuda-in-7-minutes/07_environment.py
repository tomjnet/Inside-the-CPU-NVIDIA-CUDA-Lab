#!/usr/bin/env python3
"""Install and Verify CUDA in 7 Minutes: Driver, Toolkit and Runtime - slide 7: record the environment (Python version of 07_environment.cu)

Run: python3 07_environment.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
Keep the output next to every benchmark: python3 07_environment.py > environment.txt
"""
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
# The slide shows the runtime API; the driver API names differ: cudaDriverGetVersion is cuDriverGetVersion,
# cudaGetDeviceCount is cuDeviceGetCount and cudaGetDeviceProperties is one cuDeviceGetAttribute per field.
# The runtime half lives in cuda.bindings.runtime: cudaRuntimeGetVersion, and CUDART_VERSION, the toolkit
# version the bindings were built with (in C++ it is the header the program was compiled against).
# Host calls only: like the .cu, this sample launches no kernel, so nothing is compiled.
from cuda.bindings import runtime as cudart

A = cu.CUdevice_attribute


def attribute(dev, which) -> int:
    return int(check(cu.cuDeviceGetAttribute(which, dev), str(which)))


def main() -> int:
    if require_device() is None:
        return 0

    # major * 1000 + minor * 10
    err, drv = cu.cuDriverGetVersion()
    if err != cu.CUresult.CUDA_SUCCESS:
        return 1
    err, rt = cudart.cudaRuntimeGetVersion()
    if err != cudart.cudaError_t.cudaSuccess:
        return 1
    built_with = int(cudart.CUDART_VERSION)
    print(f"driver: CUDA {drv // 1000}.{drv % 1000 // 10}")
    print(f"runtime: CUDA {rt // 1000}.{rt % 1000 // 10}")
    print(f"built with toolkit {built_with // 1000}.{built_with % 1000 // 10}")
    ok = drv >= rt                           # the one rule: driver >= runtime
    print("compatible" if ok else "update the GPU driver")
    # keep these lines, plus the device query, with every benchmark

    # the device half of the record, one line per GPU
    n = int(check(cu.cuDeviceGetCount(), "cudaGetDeviceCount"))
    for d in range(n):
        dev = check(cu.cuDeviceGet(d), "cuDeviceGet")
        name = check(cu.cuDeviceGetName(256, dev), "cudaGetDeviceProperties").split(b"\0")[0].decode()
        major = attribute(dev, A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        minor = attribute(dev, A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        total_global_mem = int(check(cu.cuDeviceTotalMem(dev), "cudaGetDeviceProperties"))
        warp_size = attribute(dev, A.CU_DEVICE_ATTRIBUTE_WARP_SIZE)
        print(f"gpu {d}: {name}, compute capability {major}.{minor}, "
              f"{total_global_mem >> 20} MiB, warpSize {warp_size}")

    # CPU check of the result: the rule holds, and the runtime has the major version of the headers
    passed = ok and rt // 1000 == built_with // 1000 and n > 0
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
