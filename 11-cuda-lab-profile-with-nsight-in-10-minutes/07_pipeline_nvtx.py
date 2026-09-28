#!/usr/bin/env python3
"""Profile CUDA with NVIDIA Nsight in 10 Minutes - slide 7: name the phases with NVTX (Python version of 07_pipeline_nvtx.cu)

Run: python3 07_pipeline_nvtx.py before | python3 07_pipeline_nvtx.py after | python3 07_pipeline_nvtx.py (both)
     (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
Profile: 08_nsys_profile.sh records the .cu program; the same command works on this file:
     nsys profile --trace=cuda,nvtx python3 07_pipeline_nvtx.py after

The pipeline of 06_pipeline.py with NVTX ranges: under nsys every range is a named bar on the CPU row.
From Python the ranges come from NVIDIA's nvtx module (pip install nvtx); without it they are no-ops,
which is also what a range costs in C++ when no profiler listens.
"""
import ctypes
import sys
import time

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
# The slide shows the runtime API; the driver API names differ: cudaMallocHost is cuMemAllocHost, cudaMalloc is
# cuMemAlloc, cudaStreamCreate is cuStreamCreate and cudaMemcpyAsync is cuMemcpyHtoDAsync / cuMemcpyDtoHAsync.
K_WORK = 400
KERNELS = r"""
// x = a * x, the long way: 400 dependent steps per element so the kernel costs about as much as a copy
constexpr int kWork = 400;
extern "C" __global__ void scale(float* x, int n, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i], t = 0.0f;
    for (int k = 0; k < kWork; ++k) t = t * 0.5f + v;   // converges to 2 * v
    x[i] = t * (a * 0.5f);
}
"""

# the state the slide uses
CHUNKS = 16
N = 1 << 18                                          # 1 MiB of floats per chunk, 16 MiB in total
BYTES = N * 4                                        # sizeof(float) is 4
BLOCKS = (N + 255) // 256


class State:
    """The file scope globals of the .cu: pinned host memory, device memory, four streams, the kernel."""
    host_ptr = 0                                     # pinned: cudaMallocHost
    host: "np.ndarray"
    dev = 0
    streams: list = []
    scale = None


# nvtxRangePushA / nvtxRangePop: NVTX 3 is header only in C++; from Python the same ranges come from
# NVIDIA's nvtx module. Without it the two functions do nothing, like a C++ range with no profiler attached.
try:
    import nvtx

    NVTX_SOURCE = "nvtx module: ranges show up under nsys"

    def nvtx_range_push(name: str) -> None:
        nvtx.push_range(name)

    def nvtx_range_pop() -> None:
        nvtx.pop_range()
except ImportError:
    NVTX_SOURCE = "nvtx module missing (pip install nvtx): ranges are no-ops"

    def nvtx_range_push(name: str) -> None:
        pass

    def nvtx_range_pop() -> None:
        pass


def enqueue_chunk(c: int, s) -> None:
    """The three stages of one chunk on one stream: 2 copies, 1 launch."""
    h = State.host_ptr + c * BYTES
    d = cu.CUdeviceptr(State.dev + c * BYTES)
    check(cu.cuMemcpyHtoDAsync(d, h, BYTES, s), "upload")
    launch(State.scale, BLOCKS, 256, [(d, None), (N, ctypes.c_int), (2.0, ctypes.c_float)], stream=s)
    check(cu.cuMemcpyDtoHAsync(h, d, BYTES, s), "download")


def pipeline(after: bool) -> None:
    # named ranges on the CPU row of the timeline, near zero cost
    nvtx_range_push("after" if after else "before")
    for c in range(CHUNKS):
        nvtx_range_push("enqueue chunk")     # 2 copies + 1 launch
        enqueue_chunk(c, State.streams[c % 4 if after else 0])
        nvtx_range_pop()
    nvtx_range_push("wait")                  # the CPU idles here
    check_kernel("pipeline")                 # 1 sync point
    nvtx_range_pop()
    nvtx_range_pop()                         # closes before or after


def fill(orig: "np.ndarray") -> None:
    State.host[:] = orig


def run_mode(after: bool, orig: "np.ndarray") -> float:
    """1 warm up run, then 5 timed runs: the median end to end time in ms (perf_counter_ns)."""
    fill(orig)
    nvtx_range_push("warm up")                       # shows up on the timeline: skip it when reading
    pipeline(after)
    nvtx_range_pop()
    ms: list[float] = []
    for _ in range(5):
        fill(orig)                                   # not timed: fresh input for every run
        t0 = time.perf_counter_ns()
        pipeline(after)
        ms.append((time.perf_counter_ns() - t0) / 1e6)
    ms.sort()
    return ms[2]


def verify(orig: "np.ndarray") -> bool:
    """The CPU check: every element doubled."""
    want = np.float32(2.0) * orig
    bad = np.flatnonzero(np.abs(State.host - want) > np.float32(1e-4) * np.abs(want) + np.float32(1e-4))
    if bad.size:
        i = int(bad[0])
        print(f"mismatch at {i}: {float(State.host[i]):f}, expected {float(want[i]):f}")
        return False
    return True


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    do_before = do_after = True
    if len(sys.argv) > 1:
        do_before = sys.argv[1] == "before"
        do_after = sys.argv[1] == "after"
        if not do_before and not do_after:
            print(f"usage: {sys.argv[0]} [before|after]")
            return 1
    State.scale = compile_kernels(KERNELS, ["scale"], dev)["scale"]

    total = CHUNKS * N
    orig = ((np.arange(total) % 997) * 0.25).astype(np.float32)
    State.host_ptr = int(check(cu.cuMemAllocHost(total * 4), "cudaMallocHost"))
    State.host = np.ctypeslib.as_array((ctypes.c_float * total).from_address(State.host_ptr))
    State.dev = int(check(cu.cuMemAlloc(total * 4), "cudaMalloc"))
    State.streams = [check(cu.cuStreamCreate(0), "cudaStreamCreate") for _ in range(4)]

    print(f"{CHUNKS} chunks of {BYTES // 1024} KiB, pinned host memory, {K_WORK} dependent steps per element")
    print(f"NVTX: {NVTX_SOURCE}")
    # Python adds its own call overhead to every enqueue: these times are longer than the .cu run
    ok = True
    before_ms = after_ms = 0.0
    if do_before:
        before_ms = run_mode(False, orig)
        good = verify(orig)
        ok = ok and good
        print(f"before: 1 stream,  end to end {before_ms:7.2f} ms (median of 5, perf_counter) "
              f"{'correct' if good else 'WRONG'}")
    if do_after:
        after_ms = run_mode(True, orig)
        good = verify(orig)
        ok = ok and good
        print(f"after:  4 streams, end to end {after_ms:7.2f} ms (median of 5, perf_counter) "
              f"{'correct' if good else 'WRONG'}")
    if do_before and do_after and after_ms > 0.0:
        print(f"before / after = {before_ms / after_ms:.2f}x on this machine: same work, overlapped stages")

    for s in State.streams:
        check(cu.cuStreamDestroy(s), "cudaStreamDestroy")
    check(cu.cuMemFree(cu.CUdeviceptr(State.dev)), "cudaFree")
    del State.host                                   # drop the view before the memory goes away
    check(cu.cuMemFreeHost(State.host_ptr), "cudaFreeHost")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
