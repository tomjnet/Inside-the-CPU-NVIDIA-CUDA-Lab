#!/usr/bin/env python3
"""CPU vs GPU: An Honest Benchmark in 10 Minutes - slide 11: the whole benchmark (Python version of 11_sweep.cu)

Run: python3 11_sweep.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
"""
import ctypes
import sys
import time
from dataclasses import dataclass

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
# cuMemcpyHtoD / cuMemcpyDtoH, cudaFree is cuMemFree, saxpy<<<blocks, threads>>> is cuLaunchKernel,
# cudaEvent* are cuEvent* and each cudaDeviceProp field is one cuDeviceGet* call (the name: cuDeviceGetName).
KERNELS = r"""
// one thread per element: 2 flops, 12 bytes of global memory
extern "C" __global__ void saxpy(float a, const float* x, float* y,
                      int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];   // same math as the CPU
}
"""

A = 2.0
K_WARMUP = 3


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


def saxpy_cpu(a: "np.float32", x: "np.ndarray", y: "np.ndarray") -> None:
    # the -O2 baseline of slide 3; a numpy CPU loop is not the C++ -O2 loop (own compiled loop, one temporary,
    # and a few microseconds of call overhead that dominate the smallest sizes)
    y += a * x


# samples in us: median and p99 by nearest rank, as on slide 5
def pct(s: list[float], p: float) -> float:
    s.sort()
    k = int(p / 100.0 * len(s))
    return s[min(k, len(s) - 1)]


@dataclass
class Stats:
    median: float
    p99: float
    runs: int


def stats_of(s: list[float]) -> Stats:
    return Stats(pct(s, 50.0), pct(s, 99.0), len(s))


def runs_for(n: int) -> int:                   # many cheap runs for small sizes, few for large ones
    if n <= (1 << 18):
        return 201                             # 201 runs: p99 is the 3rd slowest
    if n <= (1 << 20):
        return 51
    if n <= (1 << 22):
        return 21
    return 11                                  # 11 runs: p99 is the max, say so


def us_since(t0_ns: int) -> float:
    return (time.perf_counter_ns() - t0_ns) / 1e3


class Sweep:
    """The C++ globals g_x, g_y0, g_y (host inputs, pageable, sized for the largest n) plus the compiled kernel."""

    def __init__(self, fn, max_n: int) -> None:
        self.fn = fn
        self.x, self.y0 = uniform_inputs(max_n)   # fixed seed: same work on every run
        self.y = self.y0.copy()

    def run_cpu(self, n: int) -> Stats:        # numpy loop, perf_counter_ns (the steady_clock twin)
        a = np.float32(A)
        x, y = self.x[:n], self.y[:n]
        s: list[float] = []
        for r in range(K_WARMUP + runs_for(n)):
            t0 = time.perf_counter_ns()
            saxpy_cpu(a, x, y)
            us = us_since(t0)
            if r >= K_WARMUP:
                s.append(us)
        return stats_of(s)

    def run_kernel_only(self, n: int) -> Stats:   # CUDA events, data resident on the device
        nbytes = n * 4
        d_x = check(cu.cuMemAlloc(nbytes), "cudaMalloc x")
        d_y = check(cu.cuMemAlloc(nbytes), "cudaMalloc y")
        check(cu.cuMemcpyHtoD(d_x, self.x.ctypes.data, nbytes), "upload x")
        check(cu.cuMemcpyHtoD(d_y, self.y0.ctypes.data, nbytes), "upload y")
        threads = 256
        blocks = (n + threads - 1) // threads
        args = [(A, ctypes.c_float), (d_x, None), (d_y, None), (n, ctypes.c_int)]
        flags = cu.CUevent_flags.CU_EVENT_DEFAULT
        t0 = check(cu.cuEventCreate(flags), "cudaEventCreate")
        t1 = check(cu.cuEventCreate(flags), "cudaEventCreate")
        s: list[float] = []
        for r in range(K_WARMUP + runs_for(n)):
            check(cu.cuEventRecord(t0, 0), "cudaEventRecord")
            launch(self.fn, blocks, threads, args)   # stops on a refused launch, like cudaGetLastError
            check(cu.cuEventRecord(t1, 0), "cudaEventRecord")
            check(cu.cuEventSynchronize(t1), "cudaEventSynchronize")
            ms = float(check(cu.cuEventElapsedTime(t0, t1), "cudaEventElapsedTime"))
            if r >= K_WARMUP:
                s.append(ms * 1000.0)
        check_kernel("saxpy kernel only")
        check(cu.cuEventDestroy(t0), "cudaEventDestroy")
        check(cu.cuEventDestroy(t1), "cudaEventDestroy")
        check(cu.cuMemFree(d_x), "cudaFree x")
        check(cu.cuMemFree(d_y), "cudaFree y")
        return stats_of(s)

    def gpu_job(self, n: int) -> None:
        """One complete job: allocate, 2 copies in, launch, sync, 1 copy out, free."""
        nbytes = n * 4
        d_x = check(cu.cuMemAlloc(nbytes), "cudaMalloc x")
        d_y = check(cu.cuMemAlloc(nbytes), "cudaMalloc y")
        check(cu.cuMemcpyHtoD(d_x, self.x.ctypes.data, nbytes), "upload x")
        check(cu.cuMemcpyHtoD(d_y, self.y.ctypes.data, nbytes), "upload y")
        launch(self.fn, (n + 255) // 256, 256, [(A, ctypes.c_float), (d_x, None), (d_y, None), (n, ctypes.c_int)])
        check_kernel("saxpy")
        check(cu.cuMemcpyDtoH(self.y.ctypes.data, d_y, nbytes), "download y")
        check(cu.cuMemFree(d_x), "cudaFree x")
        check(cu.cuMemFree(d_y), "cudaFree y")

    def run_end_to_end(self, n: int) -> Stats:   # perf_counter_ns around the whole job
        s: list[float] = []
        for r in range(K_WARMUP + runs_for(n)):
            t0 = time.perf_counter_ns()
            self.gpu_job(n)
            us = us_since(t0)
            if r >= K_WARMUP:
                s.append(us)
        return stats_of(s)

    def verify(self, n: int) -> bool:          # fresh inputs, one GPU job, every element checked
        self.y[:n] = self.y0[:n]
        self.gpu_job(n)
        ref = 2.0 * self.x[:n].astype(np.float64) + self.y0[:n].astype(np.float64)
        got = self.y[:n].astype(np.float64)
        wrong = int(np.count_nonzero(np.abs(got - ref) > 1e-6 * np.abs(ref) + 1e-6))
        if wrong != 0:
            print(f"n = {n}: {wrong} elements outside the tolerance\nFAIL")
            return False
        return True


def size_label(n: int) -> str:
    return f"{n >> 20} M" if n >= (1 << 20) else f"{n >> 10} K"


def print_row(n: int, cpu: Stats, gk: Stats, ge: Stats) -> None:
    resident = "GPU" if gk.median < cpu.median else "CPU"
    e2e = "GPU" if ge.median < cpu.median else "CPU"
    print(f"{size_label(n):<6} {cpu.runs:5d} | {cpu.median:9.1f} {cpu.p99:9.1f} | {gk.median:9.1f} {gk.p99:9.1f} | "
          f"{ge.median:9.1f} {ge.p99:9.1f} | {resident:<8} {e2e:<8}")


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    name = check(cu.cuDeviceGetName(256, dev), "cudaGetDeviceProperties").split(b"\0")[0].decode()
    fn = compile_kernels(KERNELS, ["saxpy"], dev)["saxpy"]
    max_n = 1 << 24                            # 16 M floats: 64 MiB per buffer, 128 MiB on the device
    sweep = Sweep(fn, max_n)

    # Python adds a few microseconds per driver call and per numpy call: the small sizes show more fixed cost
    # than the C++ run, on both sides; the break even size moves, the shape of the table does not
    print(f"saxpy y = 2x + y on {name}, CPU one core numpy; times in microseconds, median and p99")
    print("kernel only: CUDA events, data resident. end to end: perf_counter_ns around malloc, copies, launch, free")
    print(f"{'size':<6} {'runs':>5} | {'cpu med':>9} {'cpu p99':>9} | {'krn med':>9} {'krn p99':>9} | "
          f"{'e2e med':>9} {'e2e p99':>9} | {'resident':<8} {'e2e':<8}")

    # 1 K to 16 M floats, x4 per step: find the break even size
    n = 1 << 10
    while n <= 1 << 24:
        cpu = sweep.run_cpu(n)                 # numpy loop, perf_counter_ns
        gk = sweep.run_kernel_only(n)          # events, data resident
        ge = sweep.run_end_to_end(n)           # malloc, 3 copies, sync
        if not sweep.verify(n):                # CPU check before any number
            return 1
        print_row(n, cpu, gk, ge)              # median and p99 each
        n <<= 2                                # winner per size, 2 ways

    print("with 11 runs the p99 is the slowest run; every number is for this machine only")
    print("resident: GPU kernel only against the CPU. e2e: the whole GPU job against the CPU")
    print("PASS")                              # every size passed the CPU check
    return 0


if __name__ == "__main__":
    sys.exit(main())
