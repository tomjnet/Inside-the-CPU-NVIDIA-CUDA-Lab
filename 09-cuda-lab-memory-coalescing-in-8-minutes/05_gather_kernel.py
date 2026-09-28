#!/usr/bin/env python3
"""Memory Coalescing in CUDA in 8 Minutes - slide 5: one kernel, one index table (Python version of 05_gather_kernel.cu)

Run: python3 05_gather_kernel.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
// One thread per output element; the table decides the read address.
extern "C" __global__ void gather(const float* in, const int* map, float* out,
                       int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;                 // guard the rounded up grid
    out[i] = 2.0f * in[map[i]] + 1.0f;  // map[i], out[i]: coalesced
}                                       // in[map[i]]: up to the table
"""


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["gather"], dev)

    n = 1 << 20                                             # 1 M floats: 4 MiB per buffer
    idx = np.arange(n, dtype=np.int32)
    host_in = (idx % 1000).astype(np.float32)
    host_out = np.zeros(n, dtype=np.float32)
    host_map = idx.copy()                                   # identity table: thread i reads address i
    fbytes = host_in.nbytes
    ibytes = host_map.nbytes

    d_in = check(cu.cuMemAlloc(fbytes), "cudaMalloc in")
    d_out = check(cu.cuMemAlloc(fbytes), "cudaMalloc out")
    d_map = check(cu.cuMemAlloc(ibytes), "cudaMalloc map")
    check(cu.cuMemcpyHtoD(d_in, host_in.ctypes.data, fbytes), "upload in")
    check(cu.cuMemcpyHtoD(d_map, host_map.ctypes.data, ibytes), "upload map")

    threads = 256
    blocks = (n + threads - 1) // threads
    launch(fns["gather"], blocks, threads,                  # 1 launch
           [(d_in, None), (d_map, None), (d_out, None), (n, ctypes.c_int)])
    check_kernel("gather")                                  # the helpers from episode 2

    check(cu.cuMemcpyDtoH(host_out.ctypes.data, d_out, fbytes), "download")
    check(cu.cuMemFree(d_in), "cudaFree")
    check(cu.cuMemFree(d_out), "cudaFree")
    check(cu.cuMemFree(d_map), "cudaFree")

    # CPU check of every element, in float32 like the C++ check
    expect = np.float32(2.0) * host_in[host_map] + np.float32(1.0)
    wrong = int(np.count_nonzero(host_out != expect))
    m = host_map
    o = host_out
    print(f"gather over {n} elements in {blocks} blocks of {threads} threads, identity table")
    print(f"threads 0..3 read addresses {m[0]} {m[1]} {m[2]} {m[3]}: neighbours, so their reads can coalesce")
    print(f"out[0..3] = {o[0]:.0f} {o[1]:.0f} {o[2]:.0f} {o[3]:.0f}, wrong elements: {wrong}")
    ok = wrong == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
