#!/usr/bin/env python3
"""Install and Verify CUDA in 7 Minutes: Driver, Toolkit and Runtime - slide 6: step 3: a device query program (Python version of 06_device_query.cu)

Run: python3 06_device_query.py   (needs pip install cuda-python numpy and an NVIDIA GPU; without them it says why)
"""
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
# The slide shows the runtime API; the driver API names differ: cudaGetDeviceCount is cuDeviceGetCount,
# cudaGetDeviceProperties is one cuDeviceGetAttribute per field (plus cuDeviceGetName and cuDeviceTotalMem),
# cudaSetDevice is the context that require_device() makes current, cudaMemGetInfo is cuMemGetInfo.
# Host calls only: like the .cu, this sample launches no kernel, so nothing is compiled.
A = cu.CUdevice_attribute


def attribute(dev, which) -> int:
    return int(check(cu.cuDeviceGetAttribute(which, dev), str(which)))


def main() -> int:
    got = require_device()                   # cuInit, the first call, loads the driver
    if got is None:
        return 0
    dev, _ = got

    err, n = cu.cuDeviceGetCount()           # checked inline today,
    if err != cu.CUresult.CUDA_SUCCESS or n == 0:
        print(f"no CUDA device: {cu.cuGetErrorName(err)[1].decode()}")
        return 0                             # check() arrives next episode
    err, raw_name = cu.cuDeviceGetName(256, dev)
    if err != cu.CUresult.CUDA_SUCCESS:
        return 1
    name = raw_name.split(b"\0")[0].decode()
    major = attribute(dev, A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
    minor = attribute(dev, A.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
    total_global_mem = int(check(cu.cuDeviceTotalMem(dev), "cuDeviceTotalMem"))
    warp_size = attribute(dev, A.CU_DEVICE_ATTRIBUTE_WARP_SIZE)
    max_threads_per_block = attribute(dev, A.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    print(f"{name}, capability {major}.{minor}")
    print(f"global memory {total_global_mem >> 20} MiB")
    print(f"warpSize {warp_size}, max threads per block {max_threads_per_block}")

    # the rest of the record: launch limits and the memory that is free right now
    max_threads_dim = [attribute(dev, a) for a in (A.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                                                   A.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                                                   A.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)]
    max_grid_size = [attribute(dev, a) for a in (A.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                                                 A.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                                                 A.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)]
    print(f"devices found: {n}")
    print(f"max block dimensions {max_threads_dim[0]} x {max_threads_dim[1]} x {max_threads_dim[2]}")
    print(f"max grid dimensions {max_grid_size[0]} x {max_grid_size[1]} x {max_grid_size[2]}")
    free_bytes, total_bytes = (int(v) for v in check(cu.cuMemGetInfo(), "cudaMemGetInfo"))
    print(f"free now {free_bytes >> 20} MiB of {total_bytes >> 20} MiB")

    # CPU check of the result: a device that answers must report sane limits
    ok = (name != "" and major >= 1 and total_global_mem > 0
          and warp_size > 0 and max_threads_per_block >= warp_size
          and max_threads_per_block % warp_size == 0
          and total_bytes > 0 and free_bytes <= total_bytes)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
