#!/usr/bin/env python3
"""CUDA Occupancy and Launch Configuration in 8 Minutes - slide 3: the budget of one sm (Python version of 03_sm_limits.cu)

Run: python3 03_sm_limits.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
"""
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


KERNELS = r"""
extern "C" __global__ void read_warp_size(int* out) { *out = warpSize; }  // the device's own view
"""


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got

    p = get_device_properties(dev)              # the budget of one SM
    warps = p.maxThreadsPerMultiProcessor // p.warpSize
    blocks = p.maxBlocksPerMultiProcessor       # block slots
    regs = p.regsPerMultiprocessor              # 32 bit registers
    shm = p.sharedMemPerMultiprocessor          # bytes
    # typical (Turing, GTX 1650): 32 warps, 16 blocks,
    # 65536 registers, 64 KiB shared memory, 16 SMs
    # occupancy = resident warps / warps, per kernel and launch

    print(f"device {p.name}, compute capability {p.major}.{p.minor}, {p.multiProcessorCount} SMs")
    print(f"per SM: {p.maxThreadsPerMultiProcessor} threads = {warps} warp slots of {p.warpSize}, "
          f"{blocks} block slots")
    print(f"per SM: {regs} registers ({regs // p.maxThreadsPerMultiProcessor} per thread if every slot is full), "
          f"{shm // 1024} KiB shared memory")
    print(f"per block: at most {p.maxThreadsPerBlock} threads, {p.regsPerBlock} registers, "
          f"{p.sharedMemPerBlock // 1024} KiB shared memory by default")

    fns = compile_kernels(KERNELS, ["read_warp_size"], dev)
    d = check(cu.cuMemAlloc(4), "cudaMalloc")
    launch(fns["read_warp_size"], 1, 1, [(d, None)])   # 1 launch, 1 thread
    check_kernel("read_warp_size")
    device_warp = np.zeros(1, dtype=np.int32)
    check(cu.cuMemcpyDtoH(device_warp.ctypes.data, d, 4), "download")
    check(cu.cuMemFree(d), "cudaFree")
    print(f"warpSize inside a kernel: {int(device_warp[0])}, from the properties: {p.warpSize}")

    ok = int(device_warp[0]) == p.warpSize and warps > 0 and blocks > 0 and regs > 0 and shm > 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
