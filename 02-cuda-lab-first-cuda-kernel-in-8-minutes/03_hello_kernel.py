#!/usr/bin/env python3
"""Your First CUDA Kernel in 8 Minutes - slide 3: hello gpu: a kernel and a launch (Python version of 03_hello_kernel.cu)

Run: python3 03_hello_kernel.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
# cuMemcpyHtoD / cuMemcpyDtoH, cudaFree is cuMemFree, cudaDeviceSynchronize is cuCtxSynchronize and
# kernel<<<blocks, threads>>> is cuLaunchKernel.
KERNELS = r"""
// device code: the kernel, run once by every thread of the launch
extern "C" __global__ void hello() {
    printf("Hello GPU from thread %d\n", threadIdx.x);
}

// the same launch shape, but every thread leaves a mark the CPU can verify
extern "C" __global__ void sign_in(int* seen) {
    seen[threadIdx.x] = static_cast<int>(threadIdx.x) + 1;
}
"""


def flush_c_stdout() -> None:
    """Kernel printf lines reach the C library stdout, not sys.stdout: flush it so they keep their place."""
    try:
        ctypes.CDLL(None).fflush(None)
    except (OSError, AttributeError, TypeError):
        pass


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["hello", "sign_in"], dev)

    print("host: before the launch")
    sys.stdout.flush()
    # host code: the launch, <<<blocks in the grid, threads per block>>>
    launch(fns["hello"], 1, 4, [])              # 1 kernel launch, 4 threads, async
    print("host: the launch returned, the kernel output is still on the device")
    sys.stdout.flush()
    cu.cuCtxSynchronize()                       # 1 synchronization point, flushes printf
    flush_c_stdout()
    print("host: after the synchronization (comment it out and the four lines may never appear)")

    # CPU verification: did all four threads really run? Each one writes its own slot.
    seen = np.zeros(4, dtype=np.int32)
    d_seen = check(cu.cuMemAlloc(seen.nbytes), "cudaMalloc")
    check(cu.cuMemcpyHtoD(d_seen, seen.ctypes.data, seen.nbytes), "upload")
    launch(fns["sign_in"], 1, 4, [(d_seen, None)])
    check_kernel("sign_in")
    check(cu.cuMemcpyDtoH(seen.ctypes.data, d_seen, seen.nbytes), "download")
    check(cu.cuMemFree(d_seen), "cudaFree")

    ok = True
    for i in range(4):
        print(f"thread {i} signed in with {int(seen[i])}")
        if seen[i] != i + 1:
            ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
