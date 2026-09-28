#!/usr/bin/env python3
"""CUDA Streams and Asynchronous Copies in 10 Minutes - slide 6: asynchronous copies on a stream (Python version of 06_async_copy.cu)

Run: python3 06_async_copy.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
# The same three operations twice. From pinned memory the calls return at once and the host keeps working;
# from pageable memory the driver has to stage the data first, so the calls themselves do the waiting.
# The slide shows the runtime API; the driver API names differ: cudaMemcpyAsync is cuMemcpyHtoDAsync /
# cuMemcpyDtoHAsync, cudaStreamQuery is cuStreamQuery (CUDA_ERROR_NOT_READY), cudaMallocHost is cuMemAllocHost.
KERNELS = r"""
extern "C" __global__ void work(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 2.0f * in[i] + 1.0f;
}
"""

g_prepared = 0


def prepare_next_chunk() -> None:
    """Stands for useful host work."""
    global g_prepared
    g_prepared += 1


def use(h_out: "np.ndarray") -> float:
    return float(h_out[0])


def ms_since(t0: int) -> float:
    return (time.perf_counter_ns() - t0) / 1e6


def pinned_floats(n: int) -> tuple[int, "np.ndarray"]:
    """cudaMallocHost: the pinned address and a numpy view of its n floats."""
    ptr = int(check(cu.cuMemAllocHost(n * 4), "cudaMallocHost"))
    return ptr, np.ctypeslib.as_array((ctypes.c_float * n).from_address(ptr))


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["work"], dev)
    n = 1 << 22                                              # 16 MiB each way
    nbytes = n * 4
    block = 256
    grid = (n + block - 1) // block

    st = check(cu.cuStreamCreate(0), "cudaStreamCreate")
    in_ptr, h_in = pinned_floats(n)
    out_ptr, h_out = pinned_floats(n)
    d_in = check(cu.cuMemAlloc(nbytes), "cudaMalloc in")
    d_out = check(cu.cuMemAlloc(nbytes), "cudaMalloc out")
    h_in[:] = (np.arange(n) % 100).astype(np.float32)
    args = [(d_in, None), (d_out, None), (n, ctypes.c_int)]
    launch(fns["work"], grid, block, args)                   # warm up: load the kernel once
    check_kernel("warm up")

    # Python adds its own call overhead to the queueing time; the device work is the same as from C++
    t0 = time.perf_counter_ns()
    # 3 calls, 0 waits: each one only appends to the queue of st
    check(cu.cuMemcpyHtoDAsync(d_in, in_ptr, nbytes, st), "upload")      # 1 upload
    launch(fns["work"], grid, block, args, stream=st)                    # 1 launch
    check(cu.cuMemcpyDtoHAsync(out_ptr, d_out, nbytes, st), "download")  # 1 download
    queued_ms = ms_since(t0)
    while cu.cuStreamQuery(st)[0] == cu.CUresult.CUDA_ERROR_NOT_READY:   # the slide calls it once; here it fills the wait
        prepare_next_chunk()                                 # the host is free meanwhile
    check(cu.cuStreamSynchronize(st), "cudaStreamSynchronize")   # 1 synchronization point
    done_ms = ms_since(t0)
    first = use(h_out)                                       # only now is h_out valid

    ok = bool(np.all(np.abs(h_out - (np.float32(2.0) * h_in + np.float32(1.0))) < 1e-4))
    print(f"pinned  : 3 calls returned after {queued_ms:.3f} ms, all done after {done_ms:.3f} ms (this machine)")
    print(f"          the host prepared {g_prepared} chunks while the device worked, h_out[0] = {first:.1f}")

    # the same with pageable memory: legal, but the driver stages the data through its own pinned buffer
    p_in = h_in.copy()                                       # ordinary numpy memory: pageable
    p_out = np.zeros(n, dtype=np.float32)
    t0 = time.perf_counter_ns()
    check(cu.cuMemcpyHtoDAsync(d_in, p_in.ctypes.data, nbytes, st), "pageable upload")
    launch(fns["work"], grid, block, args, stream=st)
    check(cu.cuMemcpyDtoHAsync(p_out.ctypes.data, d_out, nbytes, st), "pageable download")
    p_queued_ms = ms_since(t0)
    check(cu.cuStreamSynchronize(st), "cudaStreamSynchronize")
    p_done_ms = ms_since(t0)
    ok = ok and bool(np.array_equal(p_out, h_out))
    print(f"pageable: 3 calls returned after {p_queued_ms:.3f} ms, all done after {p_done_ms:.3f} ms (this machine)")
    print("          when the calls return late, the copy was asynchronous in name only")

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
