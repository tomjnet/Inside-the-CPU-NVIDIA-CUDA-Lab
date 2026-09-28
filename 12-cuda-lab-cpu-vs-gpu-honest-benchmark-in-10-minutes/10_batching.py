#!/usr/bin/env python3
"""CPU vs GPU: An Honest Benchmark in 10 Minutes - slide 10: batching amortizes the fixed cost (Python version of 10_batching.cu)

Run: python3 10_batching.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
"""
import ctypes
import sys
import time
from typing import Callable

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
# cuMemcpyHtoD / cuMemcpyDtoH, cudaFree is cuMemFree and saxpy<<<grid, block>>> is cuLaunchKernel.
KERNELS = r"""
// one thread per element: 2 flops, 12 bytes of global memory
extern "C" __global__ void saxpy(float a, const float* x, float* y,
                      int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];   // same math as the CPU
}
"""

FLOAT = 4                                      # sizeof(float): d_x + b * 1024 in C++ is a byte offset here


def uniform_inputs(n: int, seed: int = 42) -> tuple["np.ndarray", "np.ndarray"]:
    """std::mt19937(seed) with std::uniform_real_distribution<float>(-1, 1), drawn x[i] then y0[i] as the C++ loop
    does. numpy's legacy RandomState(seed) is the same MT19937 stream; libstdc++ turns one 32 bit draw u into
    u / 2^32 in float (kept below 1), then r * 2 - 1, all in float32 as here."""
    v = np.random.RandomState(seed).randint(0, 1 << 32, size=2 * n, dtype=np.uint32).astype(np.float32)
    v /= np.float32(1 << 32)
    np.minimum(v, np.nextafter(np.float32(1), np.float32(0)), out=v)
    v *= np.float32(2)
    v += np.float32(-1)
    return v[0::2].copy(), v[1::2].copy()


def median_ms(f: Callable[[], None], runs: int = 7) -> float:
    """perf_counter_ns (the steady_clock twin), 1 warm up, median of runs."""
    f()
    s: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        f()
        s.append((time.perf_counter_ns() - t0) / 1e6)
    s.sort()
    return s[len(s) // 2]


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fn = compile_kernels(KERNELS, ["saxpy"], dev)["saxpy"]

    jobs = 1000
    total = jobs * 1024                        # 1 M floats: 4 MiB per buffer
    x, y0 = uniform_inputs(total)
    y = np.empty_like(y0)
    nbytes = x.nbytes
    d_x = check(cu.cuMemAlloc(nbytes), "cudaMalloc x")
    d_y = check(cu.cuMemAlloc(nbytes), "cudaMalloc y")
    check(cu.cuMemcpyHtoD(d_x, x.ctypes.data, nbytes), "upload x")
    base_x, base_y = int(d_x), int(d_y)

    def saxpy_at(grid: int, off: int, n: int) -> None:
        """saxpy<<<grid, 256>>>(2.0f, d_x + off, d_y + off, n): offset device pointers as plain addresses."""
        launch(fn, grid, 256, [(2.0, ctypes.c_float), (base_x + off * FLOAT, ctypes.c_void_p),
                               (base_y + off * FLOAT, ctypes.c_void_p), (n, ctypes.c_int)])

    def many() -> None:
        # 1000 jobs of 1 K floats, one launch each: 1000 launch costs
        for b in range(jobs):
            saxpy_at(4, b * 1024, 1024)
        check_kernel("1000 launches")

    def one() -> None:
        # the same 1 M floats as one batch: 1 launch, cost paid once
        n = jobs * 1024
        saxpy_at((n + 255) // 256, 0, n)
        check_kernel("1 launch")               # same bytes, same flops, 1 launch

    # the check first: each way once on fresh data, every element against the CPU
    ref = 2.0 * x.astype(np.float64) + y0.astype(np.float64)
    wrong = 0
    for way in (many, one):
        check(cu.cuMemcpyHtoD(d_y, y0.ctypes.data, nbytes), "reset y")
        way()
        check(cu.cuMemcpyDtoH(y.ctypes.data, d_y, nbytes), "download y")
        wrong += int(np.count_nonzero(np.abs(y.astype(np.float64) - ref) > 1e-6 * np.abs(ref) + 1e-6))

    # kernels only: the data stays on the device
    k_many = median_ms(many)
    k_one = median_ms(one)

    # with the copies: per job 2 small copies in and 1 out, or 3 large copies for the whole batch
    job_bytes = 1024 * FLOAT

    def many_with_copies() -> None:
        for b in range(jobs):
            off = b * 1024
            check(cu.cuMemcpyHtoD(base_x + off * FLOAT, x.ctypes.data + off * FLOAT, job_bytes), "upload x job")
            check(cu.cuMemcpyHtoD(base_y + off * FLOAT, y0.ctypes.data + off * FLOAT, job_bytes), "upload y job")
            saxpy_at(4, off, 1024)             # launch() stops on a refused launch, like cudaGetLastError
            check(cu.cuMemcpyDtoH(y.ctypes.data + off * FLOAT, base_y + off * FLOAT, job_bytes), "download job")

    def one_with_copies() -> None:
        check(cu.cuMemcpyHtoD(d_x, x.ctypes.data, nbytes), "upload x")
        check(cu.cuMemcpyHtoD(d_y, y0.ctypes.data, nbytes), "upload y")
        saxpy_at((total + 255) // 256, 0, total)
        check(cu.cuMemcpyDtoH(y.ctypes.data, d_y, nbytes), "download y")

    e_many = median_ms(many_with_copies, 3)
    e_one = median_ms(one_with_copies, 3)
    check(cu.cuMemFree(d_x), "cudaFree x")
    check(cu.cuMemFree(d_y), "cudaFree y")

    # every launch and copy from Python adds interpreter time on top of the driver's own fixed cost,
    # so the 1000 jobs column is higher than in the C++ run; the lesson (pay the fixed cost once) is the same
    print(f"saxpy on {total} floats as {jobs} jobs of 1024, perf_counter_ns, median after 1 warm up")
    print(f"{'':<26} {'1000 jobs ms':>14} {'1 batch ms':>14} {'ratio':>8}")
    print(f"{'kernels only':<26} {k_many:14.3f} {k_one:14.3f} {k_many / k_one:7.1f}x")
    print(f"{'with copies, end to end':<26} {e_many:14.3f} {e_one:14.3f} {e_many / e_one:7.1f}x")
    print("same bytes, same flops: the difference is fixed cost per launch and per copy on this machine")
    print(f"elements outside the tolerance: {wrong}")
    print("PASS" if wrong == 0 else "FAIL")
    return 0 if wrong == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
