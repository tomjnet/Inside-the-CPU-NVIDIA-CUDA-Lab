#!/usr/bin/env python3
"""CUDA Streams and Asynchronous Copies in 10 Minutes - slide 8: the pipeline: one chunk, one stream (Python version of 08_pipeline.cu)

Run: python3 08_pipeline.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
     python3 08_pipeline.py            both modes, the ratio and both timelines
     python3 08_pipeline.py serial     every chunk on one stream: upload, kernel, download, one after the other
     python3 08_pipeline.py streams    chunk c on stream c % 4: uploads, kernels and downloads may overlap

Episode 11 profiles the .cu program with Nsight Systems: serial is the before, streams is the after.
"""
import ctypes
import sys
import time
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
# The slide shows the runtime API; the driver API names differ: cudaStreamCreate is cuStreamCreate, cudaMemcpyAsync
# is cuMemcpyHtoDAsync / cuMemcpyDtoHAsync, the cudaEvent calls are cuEventCreate, cuEventRecord and
# cuEventElapsedTime, and prop.asyncEngineCount is cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT).
K_CHUNKS = 8                                                 # the input is cut into 8 chunks
K_STREAMS = 4                                                # overlapped mode: 4 streams, 2 chunks each
K_CHUNK_N = 1 << 20                                          # 1 M floats = 4 MiB per chunk

KERNELS = r"""
static const int kSteps = 64;                        // Newton steps per element: the kernel has real work

// 64 Newton steps toward the square root of every element: same arithmetic as the CPU check
extern "C" __global__ void work(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;                              // guard the rounded up grid
    float x = in[i];
    float v = x;
    for (int k = 0; k < kSteps; ++k) v = 0.5f * (v + x / v);
    out[i] = v;
}
"""


@dataclass
class Stamps:
    """Four events per chunk: the text timeline."""
    up0: object
    up1: object
    k1: object
    down1: object


@dataclass
class Row:
    """Milliseconds since the first enqueue."""
    stream: int
    up0: float
    up1: float
    k1: float
    down1: float


def new_event():
    return check(cu.cuEventCreate(0), "cudaEventCreate")


def elapsed_ms(a, b) -> float:
    return float(check(cu.cuEventElapsedTime(a, b), "cudaEventElapsedTime"))


def run_pipeline(work, n_streams: int, h_in: int, h_out: int, rows: list[Row]) -> float:
    """Runs every chunk through upload, kernel, download on n_streams streams; returns end to end ms."""
    nbytes = K_CHUNK_N * 4
    stream, d_in, d_out = [], [], []
    for _ in range(n_streams):
        stream.append(check(cu.cuStreamCreate(0), "cudaStreamCreate"))
        d_in.append(check(cu.cuMemAlloc(nbytes), "cudaMalloc in"))     # one pair of buffers per stream:
        d_out.append(check(cu.cuMemAlloc(nbytes), "cudaMalloc out"))   # chunks of one stream reuse them in order
    ev = [Stamps(new_event(), new_event(), new_event(), new_event()) for _ in range(K_CHUNKS)]
    check(cu.cuCtxSynchronize(), "idle before timing")

    block = 256
    grid = (K_CHUNK_N + block - 1) // block
    # Python adds its own call overhead to the queueing time; the device timeline comes from events
    t0 = time.perf_counter_ns()
    for c in range(K_CHUNKS):               # 8 chunks of 4 MiB
        s = c % n_streams                   # n_streams = 1: serial mode
        st = stream[s]                      # one device buffer pair per s
        e = ev[c]                           # the four markers of this chunk
        check(cu.cuEventRecord(e.up0, st), "cudaEventRecord")
        check(cu.cuMemcpyHtoDAsync(d_in[s], h_in + c * nbytes, nbytes, st), "upload")
        check(cu.cuEventRecord(e.up1, st), "cudaEventRecord")
        launch(work, grid, block, [(d_in[s], None), (d_out[s], None), (K_CHUNK_N, ctypes.c_int)], stream=st)
        check(cu.cuEventRecord(e.k1, st), "cudaEventRecord")
        check(cu.cuMemcpyDtoHAsync(h_out + c * nbytes, d_out[s], nbytes, st), "download")
        check(cu.cuEventRecord(e.down1, st), "cudaEventRecord")
    tq = time.perf_counter_ns()             # 24 operations queued, 0 waits
    for st in stream:
        check(cu.cuStreamSynchronize(st), "cudaStreamSynchronize")   # 1 sync per stream, at the end
    t1 = time.perf_counter_ns()

    rows.clear()
    for c, e in enumerate(ev):              # every event is complete: read the device clock
        first = ev[0].up0
        rows.append(Row(c % n_streams, elapsed_ms(first, e.up0), elapsed_ms(first, e.up1),
                        elapsed_ms(first, e.k1), elapsed_ms(first, e.down1)))
    for e in ev:
        for marker in (e.up0, e.up1, e.k1, e.down1):
            check(cu.cuEventDestroy(marker), "cudaEventDestroy")
    for s in range(n_streams):
        check(cu.cuMemFree(d_in[s]), "cudaFree")
        check(cu.cuMemFree(d_out[s]), "cudaFree")
        check(cu.cuStreamDestroy(stream[s]), "cudaStreamDestroy")
    print(f"  host queued {K_CHUNKS} chunks on {n_streams} stream(s) in {(tq - t0) / 1e6:.2f} ms, then waited")
    return (t1 - t0) / 1e6


def print_timeline(title: str, rows: list[Row], span_ms: float) -> None:
    """One text row per chunk, time runs to the right: U upload, K kernel, D download, . waiting or idle.
    A stage starts when its stream reaches it, so U also holds the wait for a busy copy engine."""
    cols = 64
    print(f"{title} (device clock from events, {span_ms:.2f} ms across {cols} columns)")
    for c, r in enumerate(rows):
        line = []
        for x in range(cols):
            t = (x + 0.5) * span_ms / cols
            if r.up0 <= t < r.up1:
                line.append("U")
            elif r.up1 <= t < r.k1:
                line.append("K")
            elif r.k1 <= t < r.down1:
                line.append("D")
            else:
                line.append(".")
        print(f"  chunk {c} stream {r.stream} |{''.join(line)}|")


def verify(h_in: "np.ndarray", h_out: "np.ndarray") -> bool:
    """CPU check: Newton converged to the square root."""
    want = np.sqrt(h_in)
    bad = np.flatnonzero(np.abs(h_out - want) > np.float32(1e-3) * want)
    if bad.size:
        i = int(bad[0])
        print(f"mismatch at {i}: got {float(h_out[i]):f}, want {float(want[i]):f}")
        return False
    return True


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode not in ("serial", "streams", "both"):
        print(f"usage: {sys.argv[0]} [serial|streams]")
        return 1

    name = check(cu.cuDeviceGetName(128, dev), "cuDeviceGetName").split(b"\0")[0].decode()
    engines = check(cu.cuDeviceGetAttribute(cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, dev),
                    "cudaGetDeviceProperties")
    print(f"{name}, copy engines (asyncEngineCount): {engines}")
    work = compile_kernels(KERNELS, ["work"], dev)["work"]

    total = K_CHUNKS * K_CHUNK_N
    in_ptr = int(check(cu.cuMemAllocHost(total * 4), "cudaMallocHost in"))
    out_ptr = int(check(cu.cuMemAllocHost(total * 4), "cudaMallocHost out"))   # pinned: the copy engine uses DMA
    h_in = np.ctypeslib.as_array((ctypes.c_float * total).from_address(in_ptr))
    h_out = np.ctypeslib.as_array((ctypes.c_float * total).from_address(out_ptr))
    h_in[:] = 1.0 + (np.arange(total) % 1000).astype(np.float32)

    rows: list[Row] = []
    run_pipeline(work, 1, in_ptr, out_ptr, rows)   # warm up: context, kernel load, first touch
    check_kernel("warm up")                        # a fault in the warm up run surfaces here
    print(f"{K_CHUNKS} chunks of {K_CHUNK_N} floats ({(total * 4) >> 20} MiB each way), block 256")

    ok = True
    serial_ms = streams_ms = 0.0
    serial_rows: list[Row] = []
    stream_rows: list[Row] = []
    if mode != "streams":
        h_out[:] = 0.0
        serial_ms = run_pipeline(work, 1, in_ptr, out_ptr, serial_rows)
        ok = verify(h_in, h_out) and ok
        print(f"serial : 1 stream   end to end {serial_ms:.2f} ms (perf_counter_ns, this machine)")
    if mode != "serial":
        h_out[:] = 0.0
        streams_ms = run_pipeline(work, K_STREAMS, in_ptr, out_ptr, stream_rows)
        ok = verify(h_in, h_out) and ok
        print(f"streams: {K_STREAMS} streams  end to end {streams_ms:.2f} ms (perf_counter_ns, this machine)")
    if mode == "both" and streams_ms > 0.0:
        print(f"ratio serial / streams: {serial_ms / streams_ms:.2f}x (this machine, never asserted)")

    span = max([r.down1 for r in serial_rows + stream_rows], default=0.0)   # one scale for both timelines
    if serial_rows:
        print_timeline("serial timeline", serial_rows, span)
    if stream_rows:
        print_timeline("streams timeline", stream_rows, span)

    del h_in, h_out                                # drop the views before the memory goes away
    check(cu.cuMemFreeHost(in_ptr), "cudaFreeHost")
    check(cu.cuMemFreeHost(out_ptr), "cudaFreeHost")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
