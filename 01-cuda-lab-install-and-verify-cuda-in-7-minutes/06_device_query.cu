// Install and Verify CUDA in 7 Minutes: Driver, Toolkit and Runtime - slide 6: step 3: a device query program
// Build: make 06_device_query
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// The helper block of the playlist (episode 2 explains it on a slide). This folder launches no kernel, so
// check_kernel is left out here; check and require_device are the same in all twelve folders.
static void check(cudaError_t e, const char* what = "CUDA call") {
    if (e != cudaSuccess) {                          // every runtime call returns a status
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
static bool require_device() {                       // samples exit 0 without a GPU
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n == 0) {
        std::printf("no CUDA device: %s\n", e != cudaSuccess ? cudaGetErrorString(e) : "count is 0");
        return false;
    }
    return true;
}

int main() {
    if (!require_device()) return 0;

    int n = 0;                               // host calls only: no kernel
    cudaError_t e = cudaGetDeviceCount(&n);  // first call loads the driver
    if (e != cudaSuccess || n == 0) {        // checked inline today,
        std::printf("no CUDA device: %s\n", cudaGetErrorString(e));
        return 0;                            // check() arrives next episode
    }
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 1;
    std::printf("%s, capability %d.%d\n", p.name, p.major, p.minor);
    std::printf("global memory %zu MiB\n", p.totalGlobalMem >> 20);
    std::printf("warpSize %d, max threads per block %d\n",
                p.warpSize, p.maxThreadsPerBlock);

    // the rest of the record: launch limits and the memory that is free right now
    std::printf("devices found: %d\n", n);
    std::printf("max block dimensions %d x %d x %d\n",
                p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    std::printf("max grid dimensions %d x %d x %d\n",
                p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check(cudaSetDevice(0), "cudaSetDevice");
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    std::printf("free now %zu MiB of %zu MiB\n", free_bytes >> 20, total_bytes >> 20);

    // CPU check of the result: a device that answers must report sane limits
    bool ok = p.name[0] != '\0' && p.major >= 1 && p.totalGlobalMem > 0
           && p.warpSize > 0 && p.maxThreadsPerBlock >= p.warpSize
           && p.maxThreadsPerBlock % p.warpSize == 0
           && total_bytes > 0 && free_bytes <= total_bytes;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
