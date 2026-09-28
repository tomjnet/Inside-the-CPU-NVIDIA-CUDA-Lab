#!/usr/bin/env python3
"""CUDA Host and Device Memory in 9 Minutes - slide 6: ownership: a small raii wrapper (Python version of 06_device_buffer.cu)

Run: python3 06_device_buffer.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
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
# The slide shows the runtime API; the driver API names differ: cudaMalloc is cuMemAlloc, cudaFree is cuMemFree,
# cudaMemcpy is cuMemcpyHtoD / cuMemcpyDtoH and kernel<<<grid, block>>> is cuLaunchKernel.
KERNELS = r"""
namespace std { using size_t = ::size_t; }   // NVRTC has no <cstddef>: give the kernel its std::size_t

extern "C" __global__ void twice(float* p, std::size_t n) {
    std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < n) p[i] = p[i] * 2.0f;
}
"""

FLOAT_BYTES = 4


class DeviceBuffer:
    """Owns one device allocation. Python has no destructor that runs at a known point, so the C++ RAII
    becomes a context manager: `with DeviceBuffer(n) as d:` frees on every exit path of the block."""

    frees = 0                                    # not on the slide: counts cudaFree calls for the self check

    def __init__(self, n: int) -> None:
        self._n = n
        self._p = check(cu.cuMemAlloc(n * FLOAT_BYTES), "cudaMalloc") if n else 0

    def close(self) -> None:                     # the destructor: runs once, a moved from buffer frees nothing
        if int(self._p) != 0:
            check(cu.cuMemFree(self._p), "cudaFree")
            DeviceBuffer.frees += 1
        self._p, self._n = 0, 0

    def __enter__(self) -> "DeviceBuffer":
        return self

    def __exit__(self, *exc) -> None:            # runs on return, exception or the end of the block
        self.close()

    def __copy__(self):                          # 1 owner, no copy: copy.copy(d) raises
        raise TypeError("DeviceBuffer cannot be copied, move it instead")

    __deepcopy__ = __copy__

    def get(self):                               # kernels take raw ptrs
        return self._p

    def bytes(self) -> int:
        return self._n * FLOAT_BYTES

    def move(self) -> "DeviceBuffer":
        """std::move: a new owner takes the allocation, this one is left empty."""
        new = DeviceBuffer(0)
        new._p, new._n = self._p, self._n
        self._p, self._n = 0, 0
        return new


def upload(h: "np.ndarray") -> DeviceBuffer:    # the caller becomes the owner, nothing is copied on the device
    d = DeviceBuffer(h.size)
    check(cu.cuMemcpyHtoD(d.get(), h.ctypes.data, d.bytes()), "upload")
    return d


def early_return(give_up: bool) -> bool:        # the shape that leaks with raw pointers
    with DeviceBuffer(1 << 22) as scratch:      # 16 MiB
        if give_up:
            return False                         # __exit__ still runs here
        return int(scratch.get()) != 0


def main() -> int:
    got = require_device()
    if got is None:
        return 0
    dev, _ = got
    fns = compile_kernels(KERNELS, ["twice"], dev)
    n = 1 << 20
    h_in = (np.arange(n) % 1000).astype(np.float32)
    h_out = np.zeros(n, dtype=np.float32)

    d = upload(h_in)                             # round trip through an owned buffer
    launch(fns["twice"], (n + 255) // 256, 256, [(d.get(), None), (n, ctypes.c_size_t)])   # the raw pointer
    check_kernel("twice")
    check(cu.cuMemcpyDtoH(h_out.ctypes.data, d.get(), d.bytes()), "download")
    wrong = int(np.count_nonzero(h_out != 2.0 * h_in))
    same = wrong == 0
    print(f"round trip of {d.bytes()} bytes through DeviceBuffer, doubled on the device: {wrong} wrong")

    # 64 early returns of 16 MiB each: 1 GiB if every one leaked, 16 MiB at a time with the context manager
    frees_before = DeviceBuffer.frees
    returned_early = sum(1 for _ in range(64) if not early_return(True))
    normal_path = early_return(False)
    freed = DeviceBuffer.frees - frees_before
    print(f"{returned_early} early returns of a 16 MiB buffer, every one freed by __exit__")

    # copy.copy(d) raises TypeError: the copy is forbidden, like the deleted copy constructor
    moved = d.move()
    print(f"after the move the new owner holds {moved.bytes()} bytes and the old one {d.bytes()}")

    ok = (same and returned_early == 64 and normal_path and freed == 65
          and int(moved.get()) != 0 and int(d.get()) == 0)
    moved.close()                                # the owner frees; d holds nothing
    d.close()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
