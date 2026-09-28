#!/usr/bin/env python3
"""CUDA Occupancy and Launch Configuration in 8 Minutes - slide 7: shared memory per block (Python version of 07_shared_limit.cu)

Run: python3 07_shared_limit.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
"""
import ctypes
import sys
from dataclasses import dataclass

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
# The slide shows the runtime API; cuda-python calls the driver API: cudaMalloc is cuMemAlloc,
# cudaMemcpy is cuMemcpyHtoD / cuMemcpyDtoH, cudaFree is cuMemFree, cudaDeviceSynchronize is cuCtxSynchronize.


@dataclass
class DeviceProp:
    """The fields of cudaDeviceProp this sample reads; each one is one cuDeviceGetAttribute call."""
    name: str
    major: int
    minor: int
    multiProcessorCount: int
    warpSize: int
    maxThreadsPerMultiProcessor: int
    maxBlocksPerMultiProcessor: int
    regsPerMultiprocessor: int
    sharedMemPerMultiprocessor: int
    maxThreadsPerBlock: int
    regsPerBlock: int
    sharedMemPerBlock: int


def get_device_properties(dev) -> DeviceProp:
    """cudaGetDeviceProperties: the runtime fills a struct, the driver answers one attribute per call."""
    def attr(what: str) -> int:
        a = getattr(cu.CUdevice_attribute, f"CU_DEVICE_ATTRIBUTE_{what}")
        return int(check(cu.cuDeviceGetAttribute(a, dev), what))

    name = check(cu.cuDeviceGetName(256, dev), "cuDeviceGetName").split(b"\0")[0].decode()
    return DeviceProp(
        name=name,
        major=attr("COMPUTE_CAPABILITY_MAJOR"),
        minor=attr("COMPUTE_CAPABILITY_MINOR"),
        multiProcessorCount=attr("MULTIPROCESSOR_COUNT"),
        warpSize=attr("WARP_SIZE"),
        maxThreadsPerMultiProcessor=attr("MAX_THREADS_PER_MULTIPROCESSOR"),
        maxBlocksPerMultiProcessor=attr("MAX_BLOCKS_PER_MULTIPROCESSOR"),
        regsPerMultiprocessor=attr("MAX_REGISTERS_PER_MULTIPROCESSOR"),
        sharedMemPerMultiprocessor=attr("MAX_SHARED_MEMORY_PER_MULTIPROCESSOR"),
        maxThreadsPerBlock=attr("MAX_THREADS_PER_BLOCK"),
        regsPerBlock=attr("MAX_REGISTERS_PER_BLOCK"),
        sharedMemPerBlock=attr("MAX_SHARED_MEMORY_PER_BLOCK"),
    )


@dataclass
class FuncAttributes:
    """cudaFuncAttributes: what one kernel asks for; each field is one cuFuncGetAttribute call."""
    numRegs: int                 # registers per thread
    sharedSizeBytes: int         # static shared memory per block
    localSizeBytes: int          # local memory per thread: stack and spills
    maxThreadsPerBlock: int      # largest block this kernel can launch with


def get_func_attributes(fn) -> FuncAttributes:
    def attr(what: str) -> int:
        a = getattr(cu.CUfunction_attribute, f"CU_FUNC_ATTRIBUTE_{what}")
        return int(check(cu.cuFuncGetAttribute(a, fn), what))

    return FuncAttributes(
        numRegs=attr("NUM_REGS"),
        sharedSizeBytes=attr("SHARED_SIZE_BYTES"),
        localSizeBytes=attr("LOCAL_SIZE_BYTES"),
        maxThreadsPerBlock=attr("MAX_THREADS_PER_BLOCK"),
    )


def max_active_blocks(fn, threads: int, dynamic_smem: int = 0) -> int:
    """cudaOccupancyMaxActiveBlocksPerMultiprocessor: resident blocks per SM, no launch needed."""
    return int(check(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn, threads, dynamic_smem), "occupancy"))


KERNELS = r"""
extern "C" __global__ void reverse_tile(const float* x, float* y) {
    extern __shared__ float tile[];         // 4096 floats: 16 KiB
    int base = blockIdx.x * 4096;           // one tile per block
    for (int k = threadIdx.x; k < 4096; k += blockDim.x)
        tile[k] = x[base + k];              // coalesced global reads
    __syncthreads();                        // the tile is complete
    for (int k = threadIdx.x; k < 4096; k += blockDim.x)
        y[base + k] = tile[4095 - k];       // another thread's load
}
"""


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    p = get_device_properties(dev)
    fns = compile_kernels(KERNELS, ["reverse_tile"], dev)
    reverse_tile = fns["reverse_tile"]
    tiles = 256                                 # 256 tiles of 4096 floats: 4 MiB
    n = tiles * 4096
    h_x = np.arange(n, dtype=np.float32)
    h_y = np.empty(n, dtype=np.float32)
    d_x = check(cu.cuMemAlloc(h_x.nbytes), "cudaMalloc x")
    d_y = check(cu.cuMemAlloc(h_y.nbytes), "cudaMalloc y")
    check(cu.cuMemcpyHtoD(d_x, h_x.ctypes.data, h_x.nbytes), "upload")

    smem = 4096 * 4                             # 3rd launch parameter: 4096 floats
    launch(reverse_tile, tiles, 128, [(d_x, None), (d_y, None)], shared_bytes=smem)
    check_kernel("reverse_tile")                # 64 / 16 KiB: 4 blocks

    max_warps = p.maxThreadsPerMultiProcessor // p.warpSize
    print(f"shared memory: {p.sharedMemPerMultiprocessor // 1024} KiB per SM, tile {smem // 1024} KiB per block, "
          f"default cap {p.sharedMemPerBlock // 1024} KiB per block")
    print("threads  blocks/SM  warps  occupancy")
    t = 64
    while t <= 1024:
        b = max_active_blocks(reverse_tile, t, smem)
        w = b * t // p.warpSize
        print(f"{t:7d}  {b:9d}  {w:5d}  {100.0 * w / max_warps:8.0f}%")
        t *= 2
    print("the tile does not shrink with the block: small blocks leave warp slots empty")

    check(cu.cuMemcpyDtoH(h_y.ctypes.data, d_y, h_y.nbytes), "download")
    check(cu.cuMemFree(d_x), "cudaFree x")
    check(cu.cuMemFree(d_y), "cudaFree y")
    # element i of a tile holds element 4095 - i of the same tile: reverse every row of a tiles x 4096 view
    want = h_x.reshape(tiles, 4096)[:, ::-1].reshape(n)
    ok = bool(np.array_equal(h_y, want))
    print(f"every tile reversed, checked on the CPU for {n} elements")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
