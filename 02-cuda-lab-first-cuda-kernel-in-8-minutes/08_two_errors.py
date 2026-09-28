#!/usr/bin/env python3
"""Your First CUDA Kernel in 8 Minutes - slide 8: two mistakes, two places (Python version of 08_two_errors.cu)

Run: python3 08_two_errors.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)

This sample breaks a kernel on purpose. The second mistake leaves a sticky error in the CUDA context of
this process only; the process ends right after, so nothing else is affected.
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
# The slide shows the runtime API; the driver API names differ: cudaGetDeviceProperties is one
# cuDeviceGetAttribute per field, cudaMalloc is cuMemAlloc, cudaDeviceSynchronize is cuCtxSynchronize and
# kernel<<<grid, block>>> is cuLaunchKernel. cuLaunchKernel returns the launch error itself, where the
# runtime keeps it for cudaGetLastError: the first mistake still surfaces at once, the second at the sync.
KERNELS = r"""
extern "C" __global__ void hello() { printf("Hello GPU from thread %d\n", threadIdx.x); }
extern "C" __global__ void add_one(int* x) { *x = *x + 1; }
"""


def flush_c_stdout() -> None:
    """Kernel printf lines reach the C library stdout, not sys.stdout: flush it so they keep their place."""
    try:
        ctypes.CDLL(None).fflush(None)
    except (OSError, AttributeError, TypeError):
        pass


def raw_launch(fn, threads: int, args) -> "cu.CUresult":
    """kernel<<<1, threads>>>(args...) without check(): return the status instead of stopping."""
    values = tuple(v for v, _ in args)
    types = tuple(t for _, t in args)
    err, = cu.cuLaunchKernel(fn, 1, 1, 1, threads, 1, 1, 0, 0, (values, types), 0)
    return err


def report(label: str, e) -> None:
    name = cu.cuGetErrorName(e)[1]
    text = cu.cuGetErrorString(e)[1]
    print(f"{label:<44} {name.decode() if name else str(e):<24} {text.decode() if text else ''}")


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["hello", "add_one"], dev)

    # a healthy launch first, with the helpers: the device and the kernels are fine
    name = check(cu.cuDeviceGetName(128, dev), "cuDeviceGetName").split(b"\0")[0].decode()
    max_threads = int(check(cu.cuDeviceGetAttribute(
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev), "cudaGetDeviceProperties"))
    print(f"{name}: at most {max_threads} threads per block")
    sys.stdout.flush()
    launch(fns["hello"], 1, 2, [])
    check_kernel("hello")
    flush_c_stdout()

    # from here on without the helpers, to see where each error surfaces
    e1 = raw_launch(fns["hello"], 4096, [])      # too many threads per block: invalid argument, right away

    null = cu.CUdeviceptr(0)
    e2 = raw_launch(fns["add_one"], 1, [(null, None)])   # a legal launch, a bad pointer: CUDA_SUCCESS
    e3, = cu.cuCtxSynchronize()                  # illegal memory access

    e4, d = cu.cuMemAlloc(4)                     # the fault is sticky: same error until the exit
    d_value = int(d) if d is not None else 0

    report("e1 cuLaunchKernel <<<1, 4096>>>", e1)
    report("e2 cuLaunchKernel with the bad pointer", e2)
    report("e3 cuCtxSynchronize", e3)
    report("e4 cuMemAlloc after the fault", e4)
    print(f"d is still {d_value:#x}: nothing was allocated")

    # CPU verification of the lesson: launch error at once, execution error at the sync, then sticky
    success = cu.CUresult.CUDA_SUCCESS
    ok = e1 != success and e2 == success and e3 != success and e4 == e3 and d_value == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
