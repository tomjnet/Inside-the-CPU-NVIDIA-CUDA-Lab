#!/usr/bin/env python3
"""CUDA Streams and Asynchronous Copies in 10 Minutes - slide 5: create streams and pinned buffers (Python version of 05_create_streams.cu)

Run: python3 05_create_streams.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
# The slide shows the runtime API; the driver API names differ: cudaStreamCreate is cuStreamCreate,
# cudaMallocHost is cuMemAllocHost, cudaMalloc is cuMemAlloc, cudaMemcpyAsync is cuMemcpyDtoHAsync,
# cudaStreamQuery is cuStreamQuery and the status names read CUDA_SUCCESS / CUDA_ERROR_NOT_READY.
KERNELS = r"""
extern "C" __global__ void fill(float* out, int n, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = value;
}
"""


def status_name(err) -> str:
    """cudaGetErrorName: the name of a status, for example CUDA_ERROR_NOT_READY."""
    name = cu.cuGetErrorName(err)[1]
    return name.decode() if name else str(err)


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["fill"], dev)
    n = 1 << 20
    nbytes = n * 4                                           # sizeof(float)

    S = 4                                                    # 4 ordered queues
    stream = [check(cu.cuStreamCreate(0), "cudaStreamCreate") for _ in range(S)]   # cheap: create once, reuse

    h_ptr = int(check(cu.cuMemAllocHost(nbytes), "cudaMallocHost"))   # pinned host memory:
    h_in = np.ctypeslib.as_array((ctypes.c_float * n).from_address(h_ptr))   # page locked, DMA ready

    # four fills of the same buffer, all on stream 0: one queue, so they run in the order they were queued
    d_buf = check(cu.cuMemAlloc(nbytes), "cudaMalloc")
    block = 256
    grid = (n + block - 1) // block
    for k in range(1, 5):
        launch(fns["fill"], grid, block, [(d_buf, None), (n, ctypes.c_int), (float(k), ctypes.c_float)],
               stream=stream[0])
    check(cu.cuMemcpyDtoHAsync(h_ptr, d_buf, nbytes, stream[0]), "download")

    # a stream is a queue you can ask without waiting: cudaStreamQuery
    busy, = cu.cuStreamQuery(stream[0])
    idle, = cu.cuStreamQuery(stream[1])
    print(f"stream 0 right after queueing 5 operations: {status_name(busy)}")
    print(f"stream 1, which got no work:                {status_name(idle)}")
    check(cu.cuStreamSynchronize(stream[0]), "cudaStreamSynchronize")
    after, = cu.cuStreamQuery(stream[0])
    print(f"stream 0 after its synchronization point:   {status_name(after)}")

    ok = (busy in (cu.CUresult.CUDA_SUCCESS, cu.CUresult.CUDA_ERROR_NOT_READY)
          and idle == cu.CUresult.CUDA_SUCCESS)
    ok = ok and bool(np.all(h_in == 4.0))                    # queue order: the last fill wins
    print(f"4 fills on one stream ran in order: every element is {float(h_in[0]):.0f}")

    check(cu.cuMemFree(d_buf), "cudaFree")
    for st in stream:
        check(cu.cuStreamDestroy(st), "cudaStreamDestroy")   # queued work still finishes
    del h_in                                                 # drop the view before the memory goes away
    check(cu.cuMemFreeHost(h_ptr), "cudaFreeHost")
    check_kernel("final check")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
