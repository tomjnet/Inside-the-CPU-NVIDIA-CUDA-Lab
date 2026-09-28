#!/usr/bin/env python3
"""CUDA Host and Device Memory in 9 Minutes - slide 5: cudamemcpy in both directions (Python version of 05_copy_both_ways.cu)

Run: python3 05_copy_both_ways.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
# The slide shows the runtime API; the driver API names differ: cudaMalloc is cuMemAlloc, cudaFree is cuMemFree,
# and cudaMemcpy with a direction argument is one call per direction: cuMemcpyHtoD, cuMemcpyDtoH, cuMemcpyDtoD.
KERNELS = r"""
extern "C" __global__ void scale(float* a, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global index, as in episode 3
    if (i < n) a[i] = a[i] * factor;                 // guard the threads past the end
}
"""


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["scale"], dev)
    n = 1 << 20
    threads = 256
    blocks = (n + threads - 1) // threads
    nbytes = n * 4
    d_a = check(cu.cuMemAlloc(nbytes), "cudaMalloc")

    h_a = np.full(n, 1.5, dtype=np.float32)      # h_ prefix: host RAM
    h_back = np.zeros(n, dtype=np.float32)

    # destination first, like memcpy; the driver API puts the direction in the name
    check(cu.cuMemcpyHtoD(d_a, h_a.ctypes.data, nbytes), "cudaMemcpy host to device")

    launch(fns["scale"], blocks, threads,        # 1 kernel launch
           [(d_a, None), (2.0, ctypes.c_float), (n, ctypes.c_int)])
    check_kernel("scale")

    # 1 PCIe crossing each way; the call blocks until the bytes arrived
    check(cu.cuMemcpyDtoH(h_back.ctypes.data, d_a, nbytes), "cudaMemcpy device to host")
    ok = float(h_back[0]) == 3.0 and float(h_back[n - 1]) == 3.0   # CPU check

    wrong = int(np.count_nonzero(h_back != 3.0))   # the full check: every element, on the CPU
    print(f"uploaded {nbytes} bytes, scaled by 2 on the device, downloaded {nbytes} bytes")
    print(f"h_a[0] = {float(h_a[0]):g} stayed on the host, h_back[0] = {float(h_back[0]):g} came from the device")
    print(f"elements that differ from 3: {wrong} of {n}")

    # the wrong direction is an error status, not silent corruption: host addresses as device addresses
    err, = cu.cuMemcpyDtoD(h_back.ctypes.data, h_a.ctypes.data, nbytes)
    message = cu.cuGetErrorString(err)[1]
    print(f"host pointers with cudaMemcpyDeviceToDevice: {message.decode() if message else err}")
    # the runtime also records it for cudaGetLastError; the driver API only returns it, nothing to clear

    check(cu.cuMemFree(d_a), "cudaFree")
    ok = ok and wrong == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
