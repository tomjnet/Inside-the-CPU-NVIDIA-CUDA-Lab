#!/usr/bin/env python3
"""Memory Coalescing in CUDA in 8 Minutes - slide 6: three tables: coalesced, strided, scattered (Python version of 06_index_tables.cu)

Run: python3 06_index_tables.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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


def std_shuffle_mt19937(values: list[int], seed: int) -> None:
    """std::shuffle(v.begin(), v.end(), std::mt19937(seed)) exactly as libstdc++ (g++) does it, so the
    scattered table matches the C++ run on Linux. numpy's legacy RandomState(seed) is the same MT19937
    stream; each step draws j in [0, i] with libstdc++'s 64 bit multiply (Lemire) method."""
    n = len(values)
    raw: list[int] = np.random.RandomState(seed).randint(0, 1 << 32, size=n + 1024, dtype=np.uint32).tolist()
    k = 0
    for i in range(1, n):
        span = i + 1                                        # uniform_int_distribution(0, i)
        product = raw[k] * span
        k += 1
        low = product & 0xFFFFFFFF
        if low < span:
            threshold = ((1 << 32) - span) % span           # reject the biased low values
            while low < threshold:
                product = raw[k] * span
                k += 1
                low = product & 0xFFFFFFFF
        j = product >> 32
        values[i], values[j] = values[j], values[i]


def make_tables(n: int) -> list[tuple[str, "np.ndarray"]]:
    ident = np.arange(n, dtype=np.int64)                    # neighbours: 1 transaction
    strided = (ident * 33) & (n - 1)                        # 33 apart: n is a power of 2
    scattered = list(range(n))                              # same addresses, shuffled
    std_shuffle_mt19937(scattered, 42)                      # fixed seed: same work
    # every table is a permutation: same reads, same sum, other order
    return [("coalesced", ident.astype(np.int32)),
            ("strided", strided.astype(np.int32)),
            ("scattered", np.array(scattered, dtype=np.int32))]


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["gather"], dev)

    n = 1 << 20                                             # a power of 2, the strided table needs it
    host_in = (np.arange(n) % 1000).astype(np.float32)
    host_out = np.zeros(n, dtype=np.float32)
    tables = make_tables(n)
    fbytes = host_in.nbytes
    ibytes = n * 4                                          # sizeof(int)

    d_in = check(cu.cuMemAlloc(fbytes), "cudaMalloc in")
    d_out = check(cu.cuMemAlloc(fbytes), "cudaMalloc out")
    d_map = check(cu.cuMemAlloc(ibytes), "cudaMalloc map")
    check(cu.cuMemcpyHtoD(d_in, host_in.ctypes.data, fbytes), "upload in")

    threads = 256
    blocks = (n + threads - 1) // threads
    args = [(d_in, None), (d_map, None), (d_out, None), (n, ctypes.c_int)]
    ok = True
    for name, table in tables:
        check(cu.cuMemcpyHtoD(d_map, table.ctypes.data, ibytes), "upload map")
        launch(fns["gather"], blocks, threads, args)
        check_kernel("gather")
        check(cu.cuMemcpyDtoH(host_out.ctypes.data, d_out, fbytes), "download")

        # CPU check of every element, and of the permutation claim
        expect = np.float32(2.0) * host_in[table] + np.float32(1.0)
        wrong = int(np.count_nonzero(host_out != expect))
        total = float(host_out.astype(np.float64).sum())    # integers: exact in double, any order
        seen = np.zeros(n, dtype=bool)
        seen[table] = True
        covered = int(np.count_nonzero(seen))
        t = table
        print(f"{name:<9} threads 0..3 read {t[0]:7d} {t[1]:7d} {t[2]:7d} {t[3]:7d}  sum {total:.0f}  "
              f"covered {covered} of {n}  wrong {wrong}")
        ok = ok and wrong == 0 and covered == n
    check(cu.cuMemFree(d_in), "cudaFree")
    check(cu.cuMemFree(d_out), "cudaFree")
    check(cu.cuMemFree(d_map), "cudaFree")

    print("same kernel, same reads, same sum: only the order of the addresses changes")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
