#!/usr/bin/env python3
"""CUDA Unified Memory in 7 Minutes - slide 4: the first run pays the migration (Python version of 04_first_vs_warm.cu)

Run: python3 04_first_vs_warm.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
# The slide shows the runtime API; the driver API names differ: cudaMallocManaged is cuMemAllocManaged,
# cudaFree is cuMemFree, cudaGetDeviceProperties is one cuDeviceGetAttribute call per field and
# kernel<<<grid, block>>> is cuLaunchKernel.
KERNELS = r"""
extern "C" __global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];   // 2 global reads, 1 global write
}
"""

THREADS = 256


def blocks_for(n: int) -> int:
    return (n + THREADS - 1) // THREADS


def managed(n: int):
    """cudaMallocManaged: one allocation of n floats, one address, valid on the host and on the device."""
    return check(cu.cuMemAllocManaged(n * 4, cu.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL.value), "cudaMallocManaged")


def host_view(ptr, n: int) -> "np.ndarray":
    """The same pointer seen from the CPU: a numpy array over that address, no copy made."""
    return np.ctypeslib.as_array((ctypes.c_float * n).from_address(int(ptr)))


def count_wrong(c: "np.ndarray", expected: float) -> int:
    """CPU check of every element."""
    return int(np.count_nonzero(c != np.float32(expected)))


def ms_since(t0: int) -> float:
    return (time.perf_counter_ns() - t0) / 1e6


def run_ms(add, a, b, c, n: int) -> float:
    """Same kernel launch twice: only the first one pays the migration."""
    t0 = time.perf_counter_ns()                  # end to end, like steady_clock in the .cu
    launch(add, blocks_for(n), THREADS,          # 1 kernel launch
           [(a, None), (b, None), (c, None), (n, ctypes.c_int)])
    check_kernel("add")                          # 1 sync point
    return ms_since(t0)


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["add"], dev)

    n = 1 << 22                                  # 4 M floats, 16 MiB per array, 48 MiB in total
    a, b, c = managed(n), managed(n), managed(n)
    ha, hb, hc = host_view(a, n), host_view(b, n), host_view(c, n)
    ha[:] = 1.0                                  # CPU touch: pages in host RAM
    hb[:] = 2.0
    hc[:] = 0.0

    concurrent = check(cu.cuDeviceGetAttribute(
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, dev), "concurrentManagedAccess")
    if not concurrent:
        print("concurrentManagedAccess is 0 (Windows hosts, WSL): every launch moves the whole allocations,\n"
              "so the warmed run pays too and the gap is small; on native Linux only the first run pays")
    first = run_ms(fns["add"], a, b, c, n)       # page faults + host to device
    warm = run_ms(fns["add"], a, b, c, n)        # pages already in device memory
    print(f"first {first:.2f} ms, warm {warm:.2f} ms")

    # the launches and the migration run in the driver, so these match the C++ run; the CPU check
    # below runs in numpy, not in a compiled loop, so its time is not the C++ time
    print("end to end times from perf_counter_ns on this machine, never asserted")
    if warm > 0.0:
        print(f"first / warm = {first / warm:.1f}x: the difference is the migration, not the kernel")

    t0 = time.perf_counter_ns()
    wrong = count_wrong(hc, 3.0)                 # the CPU touches c: the return trip
    back = ms_since(t0)
    print(f"CPU check of {n} elements: {back:.2f} ms, it includes the migration of c back to the host")
    print(f"{wrong} wrong")

    del ha, hb, hc                               # no host view may outlive the memory
    check(cu.cuMemFree(a), "cudaFree a")
    check(cu.cuMemFree(b), "cudaFree b")
    check(cu.cuMemFree(c), "cudaFree c")
    print("PASS" if wrong == 0 else "FAIL")
    return 0 if wrong == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
