#!/usr/bin/env python3
"""CUDA Shared Memory and Tiling in 9 Minutes - slide 5: declaring shared memory (Python version of 05_shared_tile.cu)

Run: python3 05_shared_tile.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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

extern "C" __global__ void mirror_tiles(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE];    // one copy per block, on chip
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * TILE + tx;     // n is a multiple of TILE
    int row = blockIdx.y * TILE + ty;
    tile[ty][tx] = in[row * n + col];     // 1 global read per thread
    __syncthreads();                      // barrier: tile is complete
    out[row * n + col] = tile[ty][TILE - 1 - tx]; // a neighbour
}
"""


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["mirror_tiles"], dev)

    n = 256                                     # 16 by 16 tiles, 256 blocks
    count = n * n
    inp = np.arange(count, dtype=np.float32)
    out = np.zeros(count, dtype=np.float32)
    nbytes = inp.nbytes

    din = check(cu.cuMemAlloc(nbytes), "cudaMalloc in")
    dout = check(cu.cuMemAlloc(nbytes), "cudaMalloc out")
    check(cu.cuMemcpyHtoD(din, inp.ctypes.data, nbytes), "copy in")

    block = (TILE, TILE)                        # one block per tile
    grid = (n // TILE, n // TILE)
    launch(fns["mirror_tiles"], grid, block, [(din, None), (dout, None), (n, ctypes.c_int)])
    check_kernel("mirror_tiles")
    check(cu.cuMemcpyDtoH(out.ctypes.data, dout, nbytes), "copy out")

    # the first row of the first tile: the input and what the neighbours' slots gave back
    print("in  row 0, tile 0:" + "".join(f" {float(v):3.0f}" for v in inp[:TILE]))
    print("out row 0, tile 0:" + "".join(f" {float(v):3.0f}" for v in out[:TILE]))

    # CPU check: every tile is mirrored left to right, nothing crosses a tile border
    q = np.arange(n)
    mirrored = (q // TILE) * TILE + (TILE - 1 - q % TILE)
    expected = inp.reshape(n, n)[:, mirrored]
    bad = int(np.count_nonzero(out.reshape(n, n) != expected))
    print(f"CPU check: {bad} of {count} elements wrong")
    print(f"shared memory per block: {4 * TILE * TILE} bytes, {TILE * TILE} threads loaded one element each")

    check(cu.cuMemFree(din), "cudaFree in")
    check(cu.cuMemFree(dout), "cudaFree out")
    print("PASS" if bad == 0 else "FAIL")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
