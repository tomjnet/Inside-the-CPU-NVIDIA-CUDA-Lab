#!/usr/bin/env python3
"""Your First CUDA Kernel in 8 Minutes - slide 6: check and check_kernel (Python version of 06_check.cu)

Run: python3 06_check.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)

The helper block below is pasted unchanged into every CUDA Python sample of the playlist. check stops on
the first error like check() in C++; check_kernel only synchronizes, because cuLaunchKernel already
returns the launch error that the runtime keeps for cudaGetLastError.
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
# The slide shows the runtime API; the driver API names differ: cudaError_t is CUresult, cudaGetErrorName
# and cudaGetErrorString are cuGetErrorName and cuGetErrorString, cudaMalloc is cuMemAlloc, cudaMemcpy is
# cuMemcpyDtoH, cudaFree is cuMemFree and kernel<<<1, 1>>> is cuLaunchKernel.
KERNELS = r"""
extern "C" __global__ void store(int* out, int value) { *out = value; }
"""


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["store"], dev)

    # what check would print: the status codes are an enum, cuGetErrorString gives the readable text
    samples = [cu.CUresult.CUDA_SUCCESS, cu.CUresult.CUDA_ERROR_INVALID_VALUE,
               cu.CUresult.CUDA_ERROR_OUT_OF_MEMORY, cu.CUresult.CUDA_ERROR_ILLEGAL_ADDRESS]
    for e in samples:
        name = check(cu.cuGetErrorName(e), "cuGetErrorName").decode()
        text = check(cu.cuGetErrorString(e), "cuGetErrorString").decode()
        print(f"status {int(e.value):3d}  {name:<28} {text}")

    # the pattern of every later sample: check around each driver call, check_kernel after each launch
    result = np.zeros(1, dtype=np.int32)
    d_result = check(cu.cuMemAlloc(result.nbytes), "cudaMalloc")
    launch(fns["store"], 1, 1, [(d_result, None), (2026, ctypes.c_int)])
    check_kernel("store")                            # launch accepted, then kernel finished
    check(cu.cuMemcpyDtoH(result.ctypes.data, d_result, result.nbytes), "download")
    check(cu.cuMemFree(d_result))                    # the label is optional: "CUDA call"

    value = int(result[0])
    print(f"kernel stored {value}, every call checked, 1 launch and 1 synchronization point")
    ok = value == 2026
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
