#!/usr/bin/env python3
"""CUDA Grids, Blocks and Threads in 8 Minutes - slide 3: the global index (Python version of 03_global_index.cu)

Run: python3 03_global_index.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
# The slide shows the runtime API; the driver API names differ: cudaMalloc is cuMemAlloc, cudaMemset is
# cuMemsetD8, cudaMemcpy is cuMemcpyDtoH, cudaFree is cuMemFree and kernel<<<grid, block>>> is cuLaunchKernel.
KERNELS = r"""
// every thread runs the same kernel and computes its own global index
extern "C" __global__ void show_index() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // unique per thread
    printf("block %u thread %u of %u -> i = %d\n",
           blockIdx.x, threadIdx.x, blockDim.x, i);
}

// the same index, written down so the CPU can verify it: out[i] = block * 100 + thread
extern "C" __global__ void record_index(int* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = static_cast<int>(blockIdx.x * 100 + threadIdx.x);
}
"""


def flush_c_stdout() -> None:
    """Kernel printf lines reach the C library stdout, not sys.stdout: flush it so they keep their place."""
    try:
        ctypes.CDLL(None).fflush(None)
    except (OSError, AttributeError, TypeError):
        pass


def c_divmod(v: int, d: int) -> tuple[int, int]:
    """C integer division truncates toward zero; Python // floors. Same result for the values we expect."""
    q = abs(v) // d * (1 if v >= 0 else -1)
    return q, v - q * d


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["show_index", "record_index"], dev)

    print("the blocks may print in any order: nothing promises which block runs first")
    sys.stdout.flush()
    launch(fns["show_index"], 3, 4, [])       # 1 kernel launch: 3 blocks of 4 threads
    check_kernel("show_index")                 # 1 sync point: the device printf buffer is written now
    flush_c_stdout()
    # gridDim.x = 3, blockDim.x = 4: 12 threads, i runs from 0 to 11

    blocks, threads = 3, 4
    total = blocks * threads
    host = np.zeros(12, dtype=np.int32)
    d_out = check(cu.cuMemAlloc(host.nbytes), "cudaMalloc")
    check(cu.cuMemsetD8(d_out, 0xFF, host.nbytes), "cudaMemset")        # every int is -1: not written yet
    launch(fns["record_index"], blocks, threads, [(d_out, None)])
    check_kernel("record_index")
    check(cu.cuMemcpyDtoH(host.ctypes.data, d_out, host.nbytes), "download")
    check(cu.cuMemFree(d_out), "cudaFree")

    ok = True
    print("\nCPU check: element i must come from block i / 4, thread i % 4")
    for i in range(total):
        value = int(host[i])
        expected = (i // threads) * 100 + i % threads
        if value != expected:
            ok = False
        block, thread = c_divmod(value, 100)
        mark = "" if value == expected else "  (wrong)"
        print(f"  i = {i:2d}  block {block} thread {thread}{mark}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
