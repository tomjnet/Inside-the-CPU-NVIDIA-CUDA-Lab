#!/usr/bin/env python3
"""Memory Coalescing in CUDA in 8 Minutes - slide 8: array of structures against structure of arrays (Python version of 08_aos_soa.cu)

Run: python3 08_aos_soa.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
"""
import ctypes
import sys
from dataclasses import dataclass
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
# cuMemcpyHtoD / cuMemcpyDtoH, cudaFree is cuMemFree, kernel<<<grid, block>>> is cuLaunchKernel and
# cudaEventCreate / Record / Synchronize / ElapsedTime / Destroy are cuEventCreate / Record / Synchronize /
# ElapsedTime / Destroy.
KERNELS = r"""
struct Particle { float x, y, z, w; };  // AoS: x values 16 B apart
extern "C" __global__ void scale_aos(Particle* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i].x *= 2.0f;          // stride 4: 1 float in 4 used
}
extern "C" __global__ void scale_soa(float* x, int n) {  // SoA: x adjacent
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 2.0f;            // 4 times fewer bytes moved
}
"""

# struct Particle on the host side: a numpy structured type has the same 16 byte layout as the C++ struct,
# so an array of it is the AoS buffer the kernel sees (a Python dataclass would not be contiguous floats).
PARTICLE = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32), ("w", np.float32)])


def device_ms(run: Callable[[], None]) -> float:
    """Device time only: two events around the launch, one synchronization point."""
    flags = cu.CUevent_flags.CU_EVENT_DEFAULT
    t0 = check(cu.cuEventCreate(flags), "cudaEventCreate")
    t1 = check(cu.cuEventCreate(flags), "cudaEventCreate")
    check(cu.cuEventRecord(t0, 0), "cudaEventRecord")       # marker into stream 0
    run()                                                   # asynchronous: returns at once
    check(cu.cuEventRecord(t1, 0), "cudaEventRecord")
    check(cu.cuEventSynchronize(t1), "cudaEventSynchronize")  # 1 synchronization point
    ms = float(check(cu.cuEventElapsedTime(t0, t1), "cudaEventElapsedTime"))
    check(cu.cuEventDestroy(t0), "cudaEventDestroy")
    check(cu.cuEventDestroy(t1), "cudaEventDestroy")
    return ms


@dataclass
class Stat:
    min_ms: float
    median_ms: float


def measure(run: Callable[[], None], warmup: int = 3, repeats: int = 21) -> Stat:
    for _ in range(warmup):                                 # warm up: clocks, caches, lazy setup
        run()
    check(cu.cuCtxSynchronize(), "warm up")
    s = sorted(device_ms(run) for _ in range(repeats))      # keep minimum and median, as the Low Latency C++ Lab does
    return Stat(s[0], s[len(s) // 2])


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["scale_aos", "scale_soa"], dev)

    n = 1 << 22                                             # 4 M particles: 64 MiB as AoS, 16 MiB for x alone
    start_x = (np.arange(n) % 1000).astype(np.float32)
    aos = np.zeros(n, dtype=PARTICLE)
    aos["x"] = start_x
    aos["y"] = 1.0
    aos["z"] = 2.0
    aos["w"] = 3.0
    soa_x = start_x.copy()
    abytes = aos.nbytes
    xbytes = soa_x.nbytes

    d_aos = check(cu.cuMemAlloc(abytes), "cudaMalloc aos")
    d_x = check(cu.cuMemAlloc(xbytes), "cudaMalloc x")

    threads = 256
    blocks = (n + threads - 1) // threads
    runs = 3 + 21                                           # warm ups plus repeats: every run doubles x again

    check(cu.cuMemcpyHtoD(d_aos, aos.ctypes.data, abytes), "upload aos")
    a = measure(lambda: launch(fns["scale_aos"], blocks, threads, [(d_aos, None), (n, ctypes.c_int)]))
    check_kernel("scale_aos")
    check(cu.cuMemcpyDtoH(aos.ctypes.data, d_aos, abytes), "download aos")

    check(cu.cuMemcpyHtoD(d_x, soa_x.ctypes.data, xbytes), "upload x")
    s = measure(lambda: launch(fns["scale_soa"], blocks, threads, [(d_x, None), (n, ctypes.c_int)]))
    check_kernel("scale_soa")
    check(cu.cuMemcpyDtoH(soa_x.ctypes.data, d_x, xbytes), "download x")

    check(cu.cuMemFree(d_aos), "cudaFree")
    check(cu.cuMemFree(d_x), "cudaFree")

    # CPU check: x doubled `runs` times, y z w untouched
    factor = np.float32(1 << runs)                          # a power of two: exact in float
    expect = start_x * factor
    bad = (aos["x"] != expect) | (soa_x != expect)
    bad_yzw = (aos["y"] != 1.0) | (aos["z"] != 2.0) | (aos["w"] != 3.0)
    wrong = int(np.count_nonzero(bad)) + int(np.count_nonzero(bad_yzw))

    size_p = PARTICLE.itemsize
    size_f = np.dtype(np.float32).itemsize
    print(f"sizeof(Particle) = {size_p} bytes: neighbouring threads touch x values {size_p} bytes apart in AoS, "
          f"{size_f} in SoA")
    # events time the kernels on the GPU, so Python's host overhead stays outside the numbers
    print(f"{'layout':<6} {'min ms':>10} {'median ms':>12}")
    print(f"{'AoS':<6} {a.min_ms:10.3f} {a.median_ms:12.3f}")
    print(f"{'SoA':<6} {s.min_ms:10.3f} {s.median_ms:12.3f}   AoS / SoA = {a.median_ms / s.median_ms:.2f}x on this GPU")
    print(f"wrong elements: {wrong}")
    ok = wrong == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
