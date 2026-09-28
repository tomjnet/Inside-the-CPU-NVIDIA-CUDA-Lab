#!/usr/bin/env python3
"""CUDA Streams and Asynchronous Copies in 10 Minutes - slide 7: events: markers in the queue (Python version of 07_events.cu)

Run: python3 07_events.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
# Device time from events against end to end time from perf_counter_ns, for one upload, kernel and download.
# The slide shows the runtime API; the driver API names differ: cudaEventCreate is cuEventCreate, cudaEventRecord
# is cuEventRecord, cudaEventSynchronize is cuEventSynchronize, cudaEventElapsedTime is cuEventElapsedTime,
# cudaEventQuery is cuEventQuery and cudaMemcpyAsync is cuMemcpyHtoDAsync / cuMemcpyDtoHAsync.
KERNELS = r"""
extern "C" __global__ void work(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];
    float v = x;
    for (int k = 0; k < 32; ++k) v = 0.5f * (v + x / v);     // Newton steps toward sqrt(x)
    out[i] = v;
}
"""


def pinned_floats(n: int) -> tuple[int, "np.ndarray"]:
    """cudaMallocHost: the pinned address and a numpy view of its n floats."""
    ptr = int(check(cu.cuMemAllocHost(n * 4), "cudaMallocHost"))
    return ptr, np.ctypeslib.as_array((ctypes.c_float * n).from_address(ptr))


def new_event():
    return check(cu.cuEventCreate(0), "cudaEventCreate")


def elapsed_ms(a, b) -> float:
    return float(check(cu.cuEventElapsedTime(a, b), "cudaEventElapsedTime"))


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["work"], dev)
    n = 1 << 21                                              # 8 MiB each way
    nbytes = n * 4
    block = 256
    grid = (n + block - 1) // block

    st = check(cu.cuStreamCreate(0), "cudaStreamCreate")
    in_ptr, h_in = pinned_floats(n)
    out_ptr, h_out = pinned_floats(n)
    d_in = check(cu.cuMemAlloc(nbytes), "cudaMalloc in")
    d_out = check(cu.cuMemAlloc(nbytes), "cudaMalloc out")
    h_in[:] = 1.0 + (np.arange(n) % 500).astype(np.float32)
    args = [(d_in, None), (d_out, None), (n, ctypes.c_int)]
    check(cu.cuMemcpyHtoDAsync(d_in, in_ptr, nbytes, st), "warm up upload")
    launch(fns["work"], grid, block, args, stream=st)        # warm up
    check_kernel("warm up")

    up0, down1 = new_event(), new_event()                    # two more markers, around the copies
    h0 = time.perf_counter_ns()
    check(cu.cuEventRecord(up0, st), "cudaEventRecord")
    check(cu.cuMemcpyHtoDAsync(d_in, in_ptr, nbytes, st), "upload")

    t0, t1 = new_event(), new_event()
    check(cu.cuEventRecord(t0, st), "cudaEventRecord")       # a marker in the queue of st
    launch(fns["work"], grid, block, args, stream=st)
    check(cu.cuEventRecord(t1, st), "cudaEventRecord")       # stamped when st gets here
    check(cu.cuMemcpyDtoHAsync(out_ptr, d_out, nbytes, st), "download")
    check(cu.cuEventRecord(down1, st), "cudaEventRecord")
    check(cu.cuEventSynchronize(t1), "cudaEventSynchronize")   # wait for this marker only
    ms = elapsed_ms(t0, t1)                                  # device time, not host
    print(f"kernel: {ms:.3f} ms on the device clock")

    # t1 is complete, the download behind it may still be running: an event waits for a point, not a stream
    err, = cu.cuEventQuery(down1)
    name = cu.cuGetErrorName(err)[1]
    print(f"the marker after the download says right now: {name.decode() if name else err}")
    check(cu.cuStreamSynchronize(st), "cudaStreamSynchronize")
    # the host clock includes the Python call overhead, which is larger than from C++
    host_ms = (time.perf_counter_ns() - h0) / 1e6
    up_ms = elapsed_ms(up0, t0)
    down_ms = elapsed_ms(t1, down1)
    all_ms = elapsed_ms(up0, down1)
    print(f"upload {up_ms:.3f} ms + kernel {ms:.3f} ms + download {down_ms:.3f} ms = {all_ms:.3f} ms "
          "on the device clock")
    print(f"end to end {host_ms:.3f} ms with perf_counter_ns: queueing and the final wait included (this machine)")

    ok = bool(np.all(np.abs(h_out - np.sqrt(h_in)) < np.float32(1e-3) * h_out))
    for ev in (up0, t0, t1, down1):
        check(cu.cuEventDestroy(ev), "cudaEventDestroy")
    check(cu.cuMemFree(d_in), "cudaFree")
    check(cu.cuMemFree(d_out), "cudaFree")
    del h_in, h_out                                          # drop the views before the memory goes away
    check(cu.cuMemFreeHost(in_ptr), "cudaFreeHost")
    check(cu.cuMemFreeHost(out_ptr), "cudaFreeHost")
    check(cu.cuStreamDestroy(st), "cudaStreamDestroy")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
