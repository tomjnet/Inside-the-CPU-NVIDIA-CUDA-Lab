#!/usr/bin/env python3
"""CUDA Grids, Blocks and Threads in 8 Minutes - slide 9: limits of a launch (Python version of 09_launch_limits.cu)

Run: python3 09_launch_limits.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
# The slide shows the runtime API; the driver API names differ: cudaGetDeviceProperties is one
# cuDeviceGetAttribute per field, cudaMalloc is cuMemAlloc, cudaMemcpy is cuMemcpyHtoD / cuMemcpyDtoH,
# cudaFree is cuMemFree and kernel<<<grid, block>>> is cuLaunchKernel, which returns the launch error itself.
KERNELS = r"""
extern "C" __global__ void invert(unsigned char* img, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h)                      // 1 thread per pixel
        img[y * w + x] = 255 - img[y * w + x];   // 1 read, 1 write
}
"""

A = cu.CUdevice_attribute


def attribute(dev, which) -> int:
    return int(check(cu.cuDeviceGetAttribute(which, dev), str(which)))


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["invert"], dev)

    # limits of this device
    name = check(cu.cuDeviceGetName(128, dev), "cuDeviceGetName").split(b"\0")[0].decode()
    major = attribute(dev, A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
    minor = attribute(dev, A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
    warp_size = attribute(dev, A.CU_DEVICE_ATTRIBUTE_WARP_SIZE)
    max_threads = attribute(dev, A.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    max_dim = [attribute(dev, a) for a in (A.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                                           A.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                                           A.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)]
    max_grid = [attribute(dev, a) for a in (A.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                                            A.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                                            A.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)]
    # typical: warpSize 32, maxThreadsPerBlock 1024
    # maxThreadsDim 1024, 1024, 64; maxGridSize 2147483647, 65535, 65535
    print(f"device               {name} (compute capability {major}.{minor})")
    print(f"warpSize             {warp_size}")
    print(f"maxThreadsPerBlock   {max_threads}")
    print(f"maxThreadsDim        {max_dim[0]}, {max_dim[1]}, {max_dim[2]}")
    print(f"maxGridSize          {max_grid[0]}, {max_grid[1]}, {max_grid[2]}\n")

    w, h = 100, 60
    pixels = w * h
    before = np.full(pixels, 40, dtype=np.uint8)
    after = np.empty_like(before)
    d_img = check(cu.cuMemAlloc(before.nbytes), "cudaMalloc")
    check(cu.cuMemcpyHtoD(d_img, before.ctypes.data, before.nbytes), "upload")
    args = [(d_img, None), (w, ctypes.c_int), (h, ctypes.c_int)]

    block = (16, 16)
    grid = ((w + 15) // 16, (h + 15) // 16)
    launch(fns["invert"], grid, block, args)     # a legal launch: 16 * 16 = 256 threads per block
    check_kernel("invert")
    print(f"block 16 x 16 = {block[0] * block[1]} threads: accepted, the image is inverted once")

    too_big = (33, 33)                           # 1089 threads in a block
    # x * y * z is over 1024. launch() would stop the program on the error, so this call is made by hand:
    # the driver API returns the launch error from cuLaunchKernel, where the runtime keeps it for cudaGetLastError.
    err, = cu.cuLaunchKernel(fns["invert"], *grid, 1, *too_big, 1, 0, 0,
                             (tuple(v for v, _ in args), tuple(t for _, t in args)), 0)
    # nothing ran: the launch error is the only place you see it
    message = cu.cuGetErrorString(err)[1]
    text = message.decode() if message else str(err)
    print(f"block 33 x 33 = {too_big[0] * too_big[1]} threads: {text}")
    check(cu.cuCtxSynchronize(), "synchronize")  # a rejected launch is not sticky: the device is fine

    check(cu.cuMemcpyDtoH(after.ctypes.data, d_img, after.nbytes), "download")
    check(cu.cuMemFree(d_img), "cudaFree")
    wrong = int(np.count_nonzero(after != 255 - before))   # inverted exactly once: the rejected launch did not run
    print(f"pixels changed by the rejected launch: {wrong}")

    ok = err != cu.CUresult.CUDA_SUCCESS and wrong == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
