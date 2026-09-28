#!/usr/bin/env python3
"""CUDA Grids, Blocks and Threads in 8 Minutes - slide 6: dim3 and 2d indexing (Python version of 06_dim3_index.cu)

Run: python3 06_dim3_index.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
# The slide shows the runtime API; the driver API names differ: cudaMalloc is cuMemAlloc, cudaMemcpy is
# cuMemcpyHtoD / cuMemcpyDtoH, cudaFree is cuMemFree and kernel<<<grid, block>>> is cuLaunchKernel.
KERNELS = r"""
extern "C" __global__ void clear_image(unsigned char* img, int w, int h) {
    // inside the kernel: one index per dimension, then the row offset
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // column
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // row
    if (x < w && y < h)                              // guard both edges
        img[y * w + x] = 0;                          // row major pixel
}
"""


@dataclass(frozen=True)
class Dim3:
    """The dim3 of CUDA C++: three sizes, y and z default to 1."""
    x: int
    y: int = 1
    z: int = 1

    def as_tuple(self) -> tuple[int, int, int]:
        return self.x, self.y, self.z


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["clear_image"], dev)

    w, h = 100, 60                               # not multiples of 16
    pixels = w * h
    padding = 64                                 # bytes after the image that nobody may touch
    host = np.full(pixels + padding, 200, dtype=np.uint8)

    img = check(cu.cuMemAlloc(host.nbytes), "cudaMalloc")
    check(cu.cuMemcpyHtoD(img, host.ctypes.data, host.nbytes), "upload")

    block = Dim3(16, 16)                         # 256 threads per block
    grid = Dim3((w + block.x - 1) // block.x,    # round up in x
                (h + block.y - 1) // block.y)    # and in y: z stays 1

    launch(fns["clear_image"], grid.as_tuple(), block.as_tuple(),
           [(img, None), (w, ctypes.c_int), (h, ctypes.c_int)])
    check_kernel("clear_image")
    check(cu.cuMemcpyDtoH(host.ctypes.data, img, host.nbytes), "download")
    check(cu.cuMemFree(img), "cudaFree")

    not_cleared = int(np.count_nonzero(host[:pixels] != 0))
    padding_touched = int(np.count_nonzero(host[pixels:] != 200))

    threads = grid.x * grid.y * grid.z * block.x * block.y * block.z
    print(f"image {w} x {h} = {pixels} pixels")
    print(f"block {block.x} x {block.y} x {block.z}, grid {grid.x} x {grid.y} x {grid.z}")
    print(f"threads launched = {threads}, stopped by the guard = {threads - pixels}")
    print(f"pixels not cleared = {not_cleared}, bytes touched after the image = {padding_touched}")
    ok = not_cleared == 0 and padding_touched == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
