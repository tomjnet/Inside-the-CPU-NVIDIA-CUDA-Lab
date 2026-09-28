#!/usr/bin/env python3
"""Memory Coalescing in CUDA in 8 Minutes - slide 9: two dimensions: row major or stride (Python version of 09_row_major.cu)

Run: python3 09_row_major.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
# cuMemcpyHtoD / cuMemcpyDtoH, cudaMemset is cuMemsetD8, cudaFree is cuMemFree, kernel<<<grid, block>>> is
# cuLaunchKernel and cudaEventCreate / Record / Synchronize / ElapsedTime / Destroy are cuEventCreate /
# Record / Synchronize / ElapsedTime / Destroy.
KERNELS = r"""
extern "C" __global__ void invert(const float* in, float* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // fast index
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int good = y * w + x;   // x neighbours are 1 apart: coalesced
    // int bad = x * w + y; // x neighbours are w apart: strided
    out[good] = 1.0f - in[good];
}

// The commented line of the slide as a second kernel: same arithmetic, the two indices swapped.
// On a square image it still visits every pixel exactly once, so the result is identical.
extern "C" __global__ void invert_bad(const float* in, float* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int bad = x * w + y;    // x neighbours are w apart: strided
    out[bad] = 1.0f - in[bad];
}
"""


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
    fns = compile_kernels(KERNELS, ["invert", "invert_bad"], dev)

    w = h = 2048                                            # square, 4 M pixels: 16 MiB per buffer
    n = w * h
    host_in = ((np.arange(n) % 256).astype(np.float32) / np.float32(256.0))
    out_good = np.zeros(n, dtype=np.float32)
    out_bad = np.zeros(n, dtype=np.float32)
    nbytes = host_in.nbytes

    d_in = check(cu.cuMemAlloc(nbytes), "cudaMalloc in")
    d_out = check(cu.cuMemAlloc(nbytes), "cudaMalloc out")
    check(cu.cuMemcpyHtoD(d_in, host_in.ctypes.data, nbytes), "upload")

    threads = (16, 16)                                      # threadIdx.x is the index that changes fastest
    blocks = ((w + 15) // 16, (h + 15) // 16)
    args = [(d_in, None), (d_out, None), (w, ctypes.c_int), (h, ctypes.c_int)]

    g = measure(lambda: launch(fns["invert"], blocks, threads, args))
    check_kernel("invert")
    check(cu.cuMemcpyDtoH(out_good.ctypes.data, d_out, nbytes), "download")

    check(cu.cuMemsetD8(d_out, 0, nbytes), "cudaMemset")
    b = measure(lambda: launch(fns["invert_bad"], blocks, threads, args))
    check_kernel("invert_bad")
    check(cu.cuMemcpyDtoH(out_bad.ctypes.data, d_out, nbytes), "download")

    check(cu.cuMemFree(d_in), "cudaFree")
    check(cu.cuMemFree(d_out), "cudaFree")

    expect = np.float32(1.0) - host_in                      # CPU check of both results
    wrong = int(np.count_nonzero((out_good != expect) | (out_bad != expect)))

    # events time the kernels on the GPU, so Python's host overhead stays outside the numbers
    print(f"invert {w} x {h}, blocks of 16 x 16, device time from events")
    print(f"{'index':<22} {'min ms':>10} {'median ms':>12}")
    print(f"{'y * w + x (row major)':<22} {g.min_ms:10.3f} {g.median_ms:12.3f}")
    print(f"{'x * w + y (strided)':<22} {b.min_ms:10.3f} {b.median_ms:12.3f}   "
          f"bad / good = {b.median_ms / g.median_ms:.2f}x on this GPU")
    print(f"wrong pixels: {wrong}")
    ok = wrong == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
