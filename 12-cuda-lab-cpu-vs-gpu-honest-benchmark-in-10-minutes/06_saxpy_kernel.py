#!/usr/bin/env python3
"""CPU vs GPU: An Honest Benchmark in 10 Minutes - slide 6: the gpu kernel (Python version of 06_saxpy_kernel.cu)

Run: python3 06_saxpy_kernel.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
# cuMemcpyHtoD / cuMemcpyDtoH, cudaFree is cuMemFree and saxpy<<<blocks, threads>>> is cuLaunchKernel.
KERNELS = r"""
// one thread per element: 2 flops, 12 bytes of global memory
extern "C" __global__ void saxpy(float a, const float* x, float* y,
                      int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];   // same math as the CPU
}
"""


def uniform_inputs(n: int, seed: int = 42) -> tuple["np.ndarray", "np.ndarray"]:
    """std::mt19937(seed) with std::uniform_real_distribution<float>(-1, 1), drawn x[i] then y0[i] as the C++ loop
    does. numpy's legacy RandomState(seed) is the same MT19937 stream; libstdc++ turns one 32 bit draw u into
    u / 2^32 in float (kept below 1), then r * 2 - 1, all in float32 as here."""
    v = np.random.RandomState(seed).randint(0, 1 << 32, size=2 * n, dtype=np.uint32).astype(np.float32)
    v /= np.float32(1 << 32)
    np.minimum(v, np.nextafter(np.float32(1), np.float32(0)), out=v)
    v *= np.float32(2)
    v += np.float32(-1)
    return v[0::2].copy(), v[1::2].copy()


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["saxpy"], dev)

    n = 1 << 20                                # 1 M floats: 4 MiB per buffer
    x, y0 = uniform_inputs(n)                  # fixed seed: the same inputs as the CPU samples
    y = np.empty_like(y0)
    nbytes = x.nbytes

    d_x = check(cu.cuMemAlloc(nbytes), "cudaMalloc x")
    d_y = check(cu.cuMemAlloc(nbytes), "cudaMalloc y")
    check(cu.cuMemcpyHtoD(d_x, x.ctypes.data, nbytes), "upload x")
    check(cu.cuMemcpyHtoD(d_y, y0.ctypes.data, nbytes), "upload y")

    # 256 threads per block, grid rounded up: 1 kernel launch
    threads = 256
    blocks = (n + threads - 1) // threads
    launch(fns["saxpy"], blocks, threads, [(2.0, ctypes.c_float), (d_x, None), (d_y, None), (n, ctypes.c_int)])
    check_kernel("saxpy")                      # launch error, then sync

    check(cu.cuMemcpyDtoH(y.ctypes.data, d_y, nbytes), "download y")
    check(cu.cuMemFree(d_x), "cudaFree x")
    check(cu.cuMemFree(d_y), "cudaFree y")

    # the CPU check: same float expression, then a tolerance instead of equality
    cpu = np.float32(2.0) * x + y0             # the CPU rounds twice: after * and after +
    diff = np.abs(y.astype(np.float64) - cpu.astype(np.float64))
    tol = 1e-6 * np.abs(cpu.astype(np.float64)) + 1e-6
    same = y == cpu
    identical = int(np.count_nonzero(same))
    last_bits = int(np.count_nonzero(~same & (diff <= tol)))   # the GPU may fuse * and + and round once
    wrong = int(np.count_nonzero(~same & (diff > tol)))
    worst = float(diff.max())

    print(f"saxpy, {n} floats, {blocks} blocks of {threads} threads, 1 launch")
    print(f"bit identical to the CPU:        {identical}")
    print(f"different only in the last bits: {last_bits} (fused multiply add rounds once)")
    print("with a = 2 the product is exact, so fused and separate usually agree; try a = 0.7f")
    print(f"outside the tolerance:           {wrong} (largest difference {worst:.3g})")
    print("PASS" if wrong == 0 else "FAIL")
    return 0 if wrong == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
