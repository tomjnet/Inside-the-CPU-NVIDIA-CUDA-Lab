#!/usr/bin/env python3
"""CUDA Synchronization Without Deadlocks in 8 Minutes - slide 8: stream and device synchronization (Python version of 08_stream_device.cu)

Run: python3 08_stream_device.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
# The slide shows the runtime API; the driver API names differ: cudaMalloc is cuMemAlloc, cudaMemcpy is
# cuMemcpyHtoD / cuMemcpyDtoH, cudaStreamCreate / Synchronize / Destroy are cuStreamCreate /
# cuStreamSynchronize / cuStreamDestroy, cudaDeviceSynchronize is cuCtxSynchronize, and
# kernel<<<grid, block, 0, st>>> is cuLaunchKernel with the stream, which returns the launch error
# itself (the runtime keeps it for cudaGetLastError).
KERNELS = r"""
extern "C" __global__ void block_sum(const int* x, int* out, int n) {
    __shared__ int cell[256];              // 1 cell per thread
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    cell[t] = i < n ? x[i] : 0;            // no early return: stay in
    __syncthreads();                       // barrier 1: all cells ready
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (t < s) cell[t] += cell[t + s]; // work inside the branch
        __syncthreads();                   // barrier outside: s done
    }
    if (t == 0) out[blockIdx.x] = cell[0]; // 9 barriers, 1 global write
}
"""


def sum_on_cpu(x: "np.ndarray") -> int:
    """The host works while the GPU runs."""
    return int(x.sum(dtype=np.int64))


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["block_sum"], dev)

    n = 1 << 20
    blocks = (n + 255) // 256
    x = (np.arange(n, dtype=np.int32) % 9 + 1).astype(np.int32)
    out = np.zeros(blocks, dtype=np.int32)
    nbytes = out.nbytes
    d_x = check(cu.cuMemAlloc(x.nbytes), "cudaMalloc x")
    d_out = check(cu.cuMemAlloc(nbytes), "cudaMalloc out")
    check(cu.cuMemcpyHtoD(d_x, x.ctypes.data, x.nbytes), "upload")
    args = [(d_x, None), (d_out, None), (n, ctypes.c_int)]
    launch(fns["block_sum"], blocks, 256, args)   # warm up on the default stream: first launch costs more
    check_kernel("warm up")

    # the slide, in driver API names:
    st = check(cu.cuStreamCreate(0), "create")                     # an ordered queue
    launch(fns["block_sum"], blocks, 256, args, stream=st)         # 1 kernel launch, accepted or exit
    cpu_sum = sum_on_cpu(x)                                        # the host did not wait
    check(cu.cuStreamSynchronize(st), "stream")                    # waits for st only
    check(cu.cuMemcpyDtoH(out.ctypes.data, d_out, nbytes), "copy")
    check(cu.cuCtxSynchronize(), "device")                         # waits for every stream
    check(cu.cuStreamDestroy(st), "destroy")                       # 2 sync points, 1 copy

    check(cu.cuMemFree(d_x), "cudaFree x")
    check(cu.cuMemFree(d_out), "cudaFree out")
    gpu_sum = int(out.sum(dtype=np.int64))
    print(f"CPU sum, computed while the kernel ran on the stream: {cpu_sum}")
    print(f"GPU block sums after cudaStreamSynchronize and one copy: {gpu_sum}")
    ok = gpu_sum == cpu_sum
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
