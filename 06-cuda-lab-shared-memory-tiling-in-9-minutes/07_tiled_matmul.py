#!/usr/bin/env python3
"""CUDA Shared Memory and Tiling in 9 Minutes - slide 7: tiled multiply, part two: reuse and the second barrier (Python version of 07_tiled_matmul.cu)

Run: python3 07_tiled_matmul.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
TILE = 16                                       # tile width = block width, same as #define TILE in the kernel

KERNELS = r"""
#define TILE 16                           // tile width = block width

extern "C" __global__ void matmul_tiled(const float* a, const float* b,
                             float* c, int n) {
    __shared__ float ta[TILE][TILE];      // tile of a, one per block
    __shared__ float tb[TILE][TILE];      // tile of b
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float sum = 0.0f;
    for (int t = 0; t < n / TILE; ++t) {  // walk the row and the column
        ta[ty][tx] = a[row * n + t * TILE + tx];   // 2 global reads
        tb[ty][tx] = b[(t * TILE + ty) * n + col]; // per thread, tile
        __syncthreads();                  // barrier 1: tiles loaded
        for (int k = 0; k < TILE; ++k)    // 32 reads, all shared memory
            sum += ta[ty][k] * tb[k][tx];
        __syncthreads();                  // barrier 2: all have read
    }                                     // before the next load
    c[row * n + col] = sum;               // 1 global write per thread
}
"""


def fill(n: int) -> tuple["np.ndarray", "np.ndarray"]:
    """Small exact values: every product and every sum is an integer below 2^24, so the GPU result must
    match the CPU result bit for bit, whatever the order of the additions."""
    i = np.arange(n * n, dtype=np.int64)
    return (i % 7).astype(np.float32), ((i % 5) - 2).astype(np.float32)


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["matmul_tiled"], dev)

    n = 512                                     # 3 matrices of 1 MiB each
    count = n * n
    a, b = fill(n)
    c = np.zeros(count, dtype=np.float32)
    nbytes = a.nbytes

    da = check(cu.cuMemAlloc(nbytes), "cudaMalloc a")
    db = check(cu.cuMemAlloc(nbytes), "cudaMalloc b")
    dc = check(cu.cuMemAlloc(nbytes), "cudaMalloc c")
    check(cu.cuMemcpyHtoD(da, a.ctypes.data, nbytes), "copy a")
    check(cu.cuMemcpyHtoD(db, b.ctypes.data, nbytes), "copy b")

    # launch: one block per tile, n = 512 gives a 32 by 32 grid
    block = (TILE, TILE)
    grid = (n // TILE, n // TILE)
    launch(fns["matmul_tiled"], grid, block, [(da, None), (db, None), (dc, None), (n, ctypes.c_int)])
    check_kernel("matmul_tiled")                # launch, then execution
    check(cu.cuMemcpyDtoH(c.ctypes.data, dc, nbytes), "copy c")

    print(f"grid {grid[0]} by {grid[1]} blocks of {block[0]} by {block[1]} threads: "
          f"{n // TILE} steps of {2 * TILE} shared memory reads each")
    print(f"per thread: {2 * n // TILE} global reads, {2 * n} shared memory reads, 1 global write")

    # CPU check of a sample of the result (the full product costs n^3 = 134 M multiply adds)
    am = a.reshape(n, n).astype(np.float64)
    bm = b.reshape(n, n).astype(np.float64)
    cm = c.reshape(n, n)
    bad = 0
    checked = 0
    for r in range(0, n, 37):
        for q in range(0, n, 41):
            s = float(np.dot(am[r], bm[:, q]))
            checked += 1
            if abs(s - float(cm[r, q])) > 1e-3:
                bad += 1
    print(f"CPU check: {checked} elements compared, {bad} wrong, c[n*n-1] = {float(c[count - 1]):g}")

    check(cu.cuMemFree(da), "cudaFree a")
    check(cu.cuMemFree(db), "cudaFree b")
    check(cu.cuMemFree(dc), "cudaFree c")
    print("PASS" if bad == 0 else "FAIL")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
