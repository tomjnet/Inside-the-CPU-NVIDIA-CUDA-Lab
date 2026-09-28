#!/usr/bin/env python3
"""CUDA Occupancy and Launch Configuration in 8 Minutes - slide 6: register pressure and launch bounds (Python version of 06_launch_bounds.cu)

Run: python3 06_launch_bounds.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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


def heavy_body_cpu(n: int, x: "np.ndarray") -> "np.ndarray":
    """heavy_body on the CPU for every i at once, in float32 like the kernel.

    The .cu calls the same __host__ __device__ function once per element; numpy runs the
    same 64 loads and 8 x 64 multiply adds on the whole array per step instead.
    """
    i = np.arange(n, dtype=np.int64)
    v = [x[(i + 97 * k) % n] for k in range(64)]   # 64 live values per thread
    s = np.zeros(n, dtype=np.float32)
    decay = np.float32(0.999)
    for r in range(8):
        for k in range(64):
            s = s * decay + v[k] * v[(k + r) % 64]
    return s


def close_to(got: "np.ndarray", want: "np.ndarray") -> "np.ndarray":
    """The GPU may fuse multiply and add: compare with a relative tolerance."""
    return np.abs(got - want) <= np.float32(1e-3) * np.maximum(np.abs(want), np.float32(1.0))


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
// 64 live values per thread: the compiler keeps them in registers if it may
__host__ __device__ inline float heavy_body(int i, int n, const float* x) {
    float v[64];
    for (int k = 0; k < 64; ++k) v[k] = x[(i + 97 * k) % n];
    float s = 0.0f;
    for (int r = 0; r < 8; ++r)
        for (int k = 0; k < 64; ++k) s = s * 0.999f + v[k] * v[(k + r) % 64];
    return s;
}
extern "C" __global__ void heavy(int n, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = heavy_body(i, n, x);
}

extern "C" __global__ void __launch_bounds__(256, 4)   // 256 threads, 4 blocks
heavy_lb(int n, const float* x, float* y);  // same body as heavy

extern "C" __global__ void __launch_bounds__(256, 4) heavy_lb(int n, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = heavy_body(i, n, x);
}
"""


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    p = get_device_properties(dev)
    fns = compile_kernels(KERNELS, ["heavy", "heavy_lb"], dev)
    heavy, heavy_lb = fns["heavy"], fns["heavy_lb"]
    n = 1 << 18                                 # 1 MiB per array
    h_x = ((np.arange(n) % 13).astype(np.float32) / np.float32(13.0)).astype(np.float32)
    h_y = np.empty(n, dtype=np.float32)
    h_z = np.empty(n, dtype=np.float32)
    d_x = check(cu.cuMemAlloc(h_x.nbytes), "cudaMalloc x")
    d_y = check(cu.cuMemAlloc(h_y.nbytes), "cudaMalloc y")
    check(cu.cuMemcpyHtoD(d_x, h_x.ctypes.data, h_x.nbytes), "upload")
    args = [(n, ctypes.c_int), (d_x, None), (d_y, None)]

    a = get_func_attributes(heavy)
    # typical: numRegs 128, 2 blocks of 256 per SM: 50 percent
    b = get_func_attributes(heavy_lb)
    # typical: numRegs 64, localSizeBytes grows: values spill
    # 4 blocks per SM: 100 percent, and often several times slower

    max_warps = p.maxThreadsPerMultiProcessor // p.warpSize
    ba = max_active_blocks(heavy, 256, 0)
    bb = max_active_blocks(heavy_lb, 256, 0)
    blocks = (n + 255) // 256
    ta = time_ms(lambda: launch(heavy, blocks, 256, args))
    check_kernel("heavy")
    check(cu.cuMemcpyDtoH(h_y.ctypes.data, d_y, h_y.nbytes), "download heavy")
    tb = time_ms(lambda: launch(heavy_lb, blocks, 256, args))
    check_kernel("heavy_lb")
    check(cu.cuMemcpyDtoH(h_z.ctypes.data, d_y, h_z.nbytes), "download heavy_lb")

    print("kernel     registers  local bytes  blocks/SM  occupancy  median ms (device, this machine)")
    print(f"heavy      {a.numRegs:9d}  {a.localSizeBytes:11d}  {ba:9d}  "
          f"{100.0 * ba * 256 / p.warpSize / max_warps:8.0f}%  {ta:9.3f}")
    print(f"heavy_lb   {b.numRegs:9d}  {b.localSizeBytes:11d}  {bb:9d}  "
          f"{100.0 * bb * 256 / p.warpSize / max_warps:8.0f}%  {tb:9.3f}")
    print(f"heavy_lb / heavy time: {tb / ta:.2f} (above 1: more occupancy, slower kernel)")

    # heavy<<<1, 1024>>>: 1024 x 128 registers. The runtime keeps the refusal for cudaGetLastError;
    # the driver returns it straight from cuLaunchKernel, so call it here instead of launch(), which exits.
    values = tuple(v for v, _ in args)
    types = tuple(t for _, t in args)
    err, = cu.cuLaunchKernel(heavy, 1, 1, 1, 1024, 1, 1, 0, 0, (values, types), 0)
    # typical: too many resources requested for launch
    message = cu.cuGetErrorString(err)[1]       # a launch error is not sticky
    print(f"heavy with 1024 threads per block: {message.decode() if message else err}")
    check(cu.cuCtxSynchronize(), "after the refused launch")

    check(cu.cuMemFree(d_x), "cudaFree x")
    check(cu.cuMemFree(d_y), "cudaFree y")
    want = heavy_body_cpu(n, h_x)
    ok = bool(np.all(close_to(h_y, want)) and np.all(close_to(h_z, want)))
    print(f"both kernels checked on the CPU for {n} elements")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
