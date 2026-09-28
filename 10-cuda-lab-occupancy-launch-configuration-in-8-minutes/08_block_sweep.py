#!/usr/bin/env python3
"""CUDA Occupancy and Launch Configuration in 8 Minutes - slide 8: the block size sweep (Python version of 08_block_sweep.cu)

Run: python3 08_block_sweep.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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


def time_ms(launch_once) -> float:
    """Device time of one launch with CUDA events: 1 warm up, then the median of 7 runs.

    The events are stamped on the device, so Python's slower host side mostly stays out of the
    number; a very short kernel can still read a little longer than from C++ (this machine).
    """
    start = check(cu.cuEventCreate(0), "event")
    stop = check(cu.cuEventCreate(0), "event")
    launch_once()                                # warm up: first launch pays setup
    runs: list[float] = []
    for _ in range(7):
        check(cu.cuEventRecord(start, 0), "record")
        launch_once()
        check(cu.cuEventRecord(stop, 0), "record")
        check(cu.cuEventSynchronize(stop), "sync")   # 1 synchronization point per run
        runs.append(float(check(cu.cuEventElapsedTime(start, stop), "elapsed")))
    check(cu.cuEventDestroy(start), "event")
    check(cu.cuEventDestroy(stop), "event")
    runs.sort()
    return runs[len(runs) // 2]


KERNELS = r"""
extern "C" __global__ void scale_add(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global index
    if (i < n) y[i] = a * x[i] + y[i];               // 2 global reads, 1 write
}
"""


@dataclass
class Row:
    threads: int
    per_sm: int
    occupancy: float
    limit: str
    ms: float


def print_row(table: list[Row], p: DeviceProp, attr: FuncAttributes, threads: int, per_sm: int,
              ms: float) -> None:
    """Collects one row: occupancy from blocks per SM, and the ceiling that matches it."""
    by_threads = p.maxThreadsPerMultiProcessor // threads
    by_slots = p.maxBlocksPerMultiProcessor
    by_regs = p.regsPerMultiprocessor // (attr.numRegs * threads)
    if per_sm == by_threads:
        limit = "threads"
    elif per_sm == by_slots:
        limit = "block slots"
    elif per_sm == by_regs:
        limit = "registers"
    else:
        limit = "other"
    occ = per_sm * threads / p.maxThreadsPerMultiProcessor
    table.append(Row(threads, per_sm, occ, limit, ms))


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    p = get_device_properties(dev)
    fns = compile_kernels(KERNELS, ["scale_add"], dev)
    scale_add = fns["scale_add"]
    attr = get_func_attributes(scale_add)

    n = 1 << 22                                 # 4 M floats, 16 MiB per array
    h_x = (np.arange(n) % 7).astype(np.float32)
    h_y = np.empty(n, dtype=np.float32)
    d_x = check(cu.cuMemAlloc(h_x.nbytes), "cudaMalloc x")
    d_y = check(cu.cuMemAlloc(h_y.nbytes), "cudaMalloc y")
    check(cu.cuMemcpyHtoD(d_x, h_x.ctypes.data, h_x.nbytes), "upload")
    check(cu.cuMemsetD32(d_y, 0, n), "memset")  # cudaMemset of zero bytes: 32 bit zeros here
    args = [(n, ctypes.c_int), (2.0, ctypes.c_float), (d_x, None), (d_y, None)]
    table: list[Row] = []

    threads = 32
    while threads <= 1024:
        blocks = (n + threads - 1) // threads   # round up
        per_sm = max_active_blocks(scale_add, threads, 0)
        ms = time_ms(lambda: launch(scale_add, blocks, threads, args))   # CUDA events, median
        check_kernel("scale_add")
        print_row(table, p, attr, threads, per_sm, ms)   # occupancy, limit, time
        threads *= 2

    best = min(range(len(table)), key=lambda r: table[r].ms)
    print(f"{p.name}, {p.multiProcessorCount} SMs, scale_add uses {attr.numRegs} registers, n = {n}")
    print("threads  blocks/SM  occupancy  limit        median ms  (device time, this machine)")
    for r, w in enumerate(table):
        if r == best:
            mark = "  fastest"
        elif w.ms <= table[best].ms * 1.05:
            mark = "  within 5 percent"
        else:
            mark = ""
        print(f"{w.threads:7d}  {w.per_sm:9d}  {w.occupancy * 100.0:8.0f}%  {w.limit:<11}  {w.ms:9.3f}{mark}")
    print("on a plateau the fastest row can change from run to run: tune by measurement")

    # one clean run with the fastest configuration, checked on the CPU
    t = table[best].threads
    check(cu.cuMemsetD32(d_y, 0, n), "memset")
    launch(scale_add, (n + t - 1) // t, t, args)
    check_kernel("scale_add best")
    check(cu.cuMemcpyDtoH(h_y.ctypes.data, d_y, h_y.nbytes), "download")
    check(cu.cuMemFree(d_x), "cudaFree x")
    check(cu.cuMemFree(d_y), "cudaFree y")
    ok = len(table) == 6 and bool(np.all(h_y == np.float32(2.0) * h_x))
    print(f"{t} threads per block, y = 2 x checked on the CPU for {n} elements")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
