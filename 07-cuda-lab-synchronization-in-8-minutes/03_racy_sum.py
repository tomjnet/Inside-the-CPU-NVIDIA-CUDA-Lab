#!/usr/bin/env python3
"""CUDA Synchronization Without Deadlocks in 8 Minutes - slide 3: a race on one shared cell (Python version of 03_racy_sum.cu)

Run: python3 03_racy_sum.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
# cuMemcpyHtoD / cuMemcpyDtoH, cudaFree is cuMemFree and kernel<<<grid, block>>> is cuLaunchKernel.
KERNELS = r"""
extern "C" __global__ void racy_sum(const int* x, int* out, int n) {
    __shared__ int total;                  // 1 cell for the whole block
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0) total = 0;
    __syncthreads();                       // barrier: total is 0 now
    if (i < n) total += x[i];              // RACE: read, add, write
    __syncthreads();                       // barrier: every add is over
    if (threadIdx.x == 0) out[blockIdx.x] = total;
}
// 256 threads read the same old total, 255 updates are lost
// a barrier orders phases, it does not make += atomic
"""


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["racy_sum"], dev)

    n = 1 << 20
    threads = 256
    blocks = (n + threads - 1) // threads
    x = (np.arange(n, dtype=np.int32) % 7 + 1).astype(np.int32)   # every element is 1 to 7
    expected = x.reshape(blocks, threads).sum(axis=1, dtype=np.int64)   # n is a multiple of 256
    out = np.zeros(blocks, dtype=np.int32)
    d_x = check(cu.cuMemAlloc(x.nbytes), "cudaMalloc x")
    d_out = check(cu.cuMemAlloc(out.nbytes), "cudaMalloc out")
    check(cu.cuMemcpyHtoD(d_x, x.ctypes.data, x.nbytes), "upload")

    wrong_runs = 0
    lost = 0
    only_lost = True                        # a lost update can only make a sum smaller
    for _ in range(20):
        launch(fns["racy_sum"], blocks, threads, [(d_x, None), (d_out, None), (n, ctypes.c_int)])
        check_kernel("racy_sum")
        check(cu.cuMemcpyDtoH(out.ctypes.data, d_out, out.nbytes), "download")
        gpu = out.astype(np.int64)
        if bool(np.any(gpu != expected)):
            wrong_runs += 1
        if bool(np.any(gpu > expected)):
            only_lost = False
        lost += int((expected - gpu).sum())
    check(cu.cuMemFree(d_x), "cudaFree x")
    check(cu.cuMemFree(d_out), "cudaFree out")

    print(f"block 0: GPU {int(out[0])}, CPU {int(expected[0])}")
    cpu_total = int(expected.sum())
    avg_lost = int(lost / 20)               # truncates toward zero, like C++ integer division
    print(f"{wrong_runs} of 20 runs wrong; per run the GPU total misses {avg_lost} of {cpu_total} "
          "on average (this GPU)")
    print("the barriers are in place and the sum is still wrong: += is read, add, write")
    print("every GPU sum is at most the CPU sum (updates lost, never invented): "
          f"{'yes' if only_lost else 'no'}")
    print("PASS" if only_lost else "FAIL")
    return 0 if only_lost else 1


if __name__ == "__main__":
    sys.exit(main())
