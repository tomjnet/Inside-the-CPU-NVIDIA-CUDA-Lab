#!/usr/bin/env python3
"""CUDA Host and Device Memory in 9 Minutes - slide 10: complete vector addition (Python version of 10_vector_add.cu)

Run: python3 10_vector_add.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
"""
import ctypes
import sys
import time
from contextlib import ExitStack
from typing import Callable

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
# The slide shows the runtime API; the driver API names differ: cudaMalloc / cudaFree are cuMemAlloc / cuMemFree,
# cudaMallocHost / cudaFreeHost are cuMemAllocHost / cuMemFreeHost, cudaMemcpy is cuMemcpyHtoD / cuMemcpyDtoH,
# the cudaEvent calls are cuEvent calls and kernel<<<grid, block>>> is cuLaunchKernel.
KERNELS = r"""
extern "C" __global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global index, as in episode 3
    if (i < n) c[i] = a[i] + b[i];
}
"""

FLOAT_BYTES = 4


class DeviceBuffer:
    """Slide 6: owns one device allocation; the `with` block is the C++ scope that frees it."""

    def __init__(self, n: int) -> None:
        self._n = n
        self._p = check(cu.cuMemAlloc(n * FLOAT_BYTES), "cudaMalloc")

    def __enter__(self) -> "DeviceBuffer":
        return self

    def __exit__(self, *exc) -> None:
        check(cu.cuMemFree(self._p), "cudaFree")

    def get(self):
        return self._p

    def bytes(self) -> int:
        return self._n * FLOAT_BYTES


class PinnedBuffer:
    """The same idea for cudaMallocHost and cudaFreeHost; array() is a numpy view of the pinned bytes."""

    def __init__(self, n: int) -> None:
        self._p = int(check(cu.cuMemAllocHost(n * FLOAT_BYTES), "cudaMallocHost"))
        self._view = np.ctypeslib.as_array((ctypes.c_float * n).from_address(self._p))

    def __enter__(self) -> "PinnedBuffer":
        return self

    def __exit__(self, *exc) -> None:
        self._view = None                        # drop the view before the memory goes away
        check(cu.cuMemFreeHost(self._p), "cudaFreeHost")

    def get(self) -> int:
        return self._p

    def array(self) -> "np.ndarray":
        return self._view


def device_ms(f: Callable[[], None]) -> float:
    """Slide 9: one event pair around a stage, device time."""
    start = check(cu.cuEventCreate(0), "cudaEventCreate")
    stop = check(cu.cuEventCreate(0), "cudaEventCreate")
    check(cu.cuEventRecord(start, 0), "cudaEventRecord")
    f()
    check(cu.cuEventRecord(stop, 0), "cudaEventRecord")
    check(cu.cuEventSynchronize(stop), "cudaEventSynchronize")
    ms = float(check(cu.cuEventElapsedTime(start, stop), "cudaEventElapsedTime"))
    check(cu.cuEventDestroy(start), "cudaEventDestroy")
    check(cu.cuEventDestroy(stop), "cudaEventDestroy")
    return ms


def verify_on_cpu(a: "np.ndarray", b: "np.ndarray", c: "np.ndarray") -> bool:
    wrong = int(np.count_nonzero(c != a + b))   # float32 sum on the CPU, like the C++ check
    print(f"CPU verification: {wrong} of {c.size} elements differ")
    return wrong == 0


def host_ms(t0: int) -> float:
    return (time.perf_counter_ns() - t0) / 1e6


def main() -> int:
    # the runtime creates its context at the first call (cudaFree(nullptr) in the .cu); the driver API creates
    # it in cuInit plus cuDevicePrimaryCtxRetain, both inside require_device(), so that call is the one timed
    t_context = time.perf_counter_ns()
    got = require_device()
    ms_context = host_ms(t_context)
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["add"], dev)  # NVRTC at run time: the .cu has this done by nvcc, not timed

    n = 1 << 22                                  # 4M floats: 16 MiB per vector
    nbytes = n * FLOAT_BYTES
    threads = 256
    blocks = (n + threads - 1) // threads
    with ExitStack() as scope:                   # every buffer entered here frees itself when the block ends
        h_a = scope.enter_context(PinnedBuffer(n))
        h_b = scope.enter_context(PinnedBuffer(n))
        h_c = scope.enter_context(PinnedBuffer(n))
        h_a.array()[:] = np.arange(n) % 1024
        h_b.array()[:] = 0.5
        h_c.array()[:] = 0.0

        # Python adds call overhead to the steady_clock stages below; the CUDA event stages are device time
        t_total = time.perf_counter_ns()
        t_alloc = time.perf_counter_ns()
        d_a = scope.enter_context(DeviceBuffer(n))   # 1. allocate
        d_b = scope.enter_context(DeviceBuffer(n))
        d_c = scope.enter_context(DeviceBuffer(n))
        ms_alloc = host_ms(t_alloc)

        def copy_in() -> None:                   # 2. copy in
            check(cu.cuMemcpyHtoD(d_a.get(), h_a.get(), nbytes), "upload a")
            check(cu.cuMemcpyHtoD(d_b.get(), h_b.get(), nbytes), "upload b")

        def kernel() -> None:                    # 3. launch
            launch(fns["add"], blocks, threads,
                   [(d_a.get(), None), (d_b.get(), None), (d_c.get(), None), (n, ctypes.c_int)])

        def copy_out() -> None:                  # 4. copy out
            check(cu.cuMemcpyDtoH(h_c.get(), d_c.get(), nbytes), "download c")

        ms_h2d = device_ms(copy_in)
        ms_kernel = device_ms(kernel)
        check_kernel("add")                      # launch accepted, no fault (already synchronized)
        ms_d2h = device_ms(copy_out)
        ms_total = host_ms(t_total)
        ok = verify_on_cpu(h_a.array(), h_b.array(), h_c.array())   # 5. verify; buffers free

        print(f"vector addition, {n} elements, {nbytes >> 20} MiB per vector, pinned host buffers, this machine:")
        print(f"  context creation   {ms_context:9.3f} ms  perf_counter_ns (paid once per process, before the stages)")
        print(f"  1. cudaMalloc x3   {ms_alloc:9.3f} ms  perf_counter_ns")
        print(f"  2. copy in  a, b   {ms_h2d:9.3f} ms  CUDA events")
        print(f"  3. kernel add      {ms_kernel:9.3f} ms  CUDA events")
        print(f"  4. copy out c      {ms_d2h:9.3f} ms  CUDA events")
        print(f"  end to end 1 to 4  {ms_total:9.3f} ms  perf_counter_ns")
        ratio = (ms_h2d + ms_d2h) / (ms_kernel if ms_kernel > 0.0 else 1e-6)
        print(f"  copies / kernel: {ratio:.1f} (a timing, never a pass condition)")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1                        # DeviceBuffer and PinnedBuffer freed themselves above


if __name__ == "__main__":
    sys.exit(main())
