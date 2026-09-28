#!/usr/bin/env python3
"""Profile CUDA with NVIDIA Nsight in 10 Minutes - slide 5: profiling noise and warm up (Python version of 05_warm_up.cu)

Run: python3 05_warm_up.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)

The first launch pays for context and module setup, so a profile or a timing never starts there:
one warm up launch, then 20 launches timed with events, then 7 single launches (keep the median).
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
# The slide shows the runtime API; the driver API names differ: cudaMalloc is cuMemAlloc, cudaMemcpy is
# cuMemcpyHtoD / cuMemcpyDtoH, the cudaEvent calls are cuEventCreate, cuEventRecord and cuEventElapsedTime.
KERNELS = r"""
// one read and one write per element: x = x * a + 1
extern "C" __global__ void scale(float* x, int n, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * a + 1.0f;
}
"""


def elapsed_ms(a, b) -> float:
    return float(check(cu.cuEventElapsedTime(a, b), "cudaEventElapsedTime"))


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    scale = compile_kernels(KERNELS, ["scale"], dev)["scale"]

    n = 1 << 20                                      # 4 MiB of floats
    blocks = (n + 255) // 256
    h = (np.arange(n) % 1000).astype(np.float32)

    d = check(cu.cuMemAlloc(h.nbytes), "cudaMalloc")
    args = [(d, None), (n, ctypes.c_int), (1.0, ctypes.c_float)]    # scale(d, n, 1.0f)
    check(cu.cuMemcpyHtoD(d, h.ctypes.data, h.nbytes), "upload")
    t0 = check(cu.cuEventCreate(0), "cudaEventCreate")
    t1 = check(cu.cuEventCreate(0), "cudaEventCreate")

    # sample only: time the warm up to show its cost (Python adds its own call overhead to this number)
    w0 = time.perf_counter_ns()
    # launch 1 pays context and module setup: never time it
    launch(scale, blocks, 256, args)         # warm up: 1 launch
    check_kernel("warm up")                  # 1 sync, not timed
    w1 = time.perf_counter_ns()                      # sample only
    check(cu.cuEventRecord(t0, 0), "cudaEventRecord")
    for _ in range(20):                      # 20 launches, 0 syncs
        launch(scale, blocks, 256, args)
    check(cu.cuEventRecord(t1, 0), "cudaEventRecord")
    check_kernel("timed")                    # device time = t1 - t0

    loop_ms = elapsed_ms(t0, t1)
    warm_us = (w1 - w0) / 1000.0

    # seven more single launches, each timed with events: keep the median
    one: list[float] = []
    for _ in range(7):
        check(cu.cuEventRecord(t0, 0), "cudaEventRecord")
        launch(scale, blocks, 256, args)
        check(cu.cuEventRecord(t1, 0), "cudaEventRecord")
        check_kernel("single")
        one.append(elapsed_ms(t0, t1))
    one.sort()

    print(f"warm up launch, end to end (perf_counter): {warm_us:9.1f} us")
    print(f"20 timed launches, device time (events):   {1000.0 * loop_ms / 20.0:9.1f} us each")
    print(f"single launches, device time, median of 7: {1000.0 * one[3]:9.1f} us "
          f"(min {1000.0 * one[0]:.1f}, max {1000.0 * one[-1]:.1f})")
    print("this machine: the warm up is typically many times one steady launch")

    out = np.empty_like(h)
    check(cu.cuMemcpyDtoH(out.ctypes.data, d, out.nbytes), "download")
    check(cu.cuEventDestroy(t0), "cudaEventDestroy")
    check(cu.cuEventDestroy(t1), "cudaEventDestroy")
    check(cu.cuMemFree(d), "cudaFree")

    ok = bool(np.array_equal(out, h + np.float32(28.0)))   # 1 + 20 + 7 launches, each adds 1
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
