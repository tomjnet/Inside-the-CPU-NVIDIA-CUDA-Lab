// Install and Verify CUDA in 7 Minutes: Driver, Toolkit and Runtime - slide 7: record the environment
// Build: make 07_environment
// Keep the output next to every benchmark: ./07_environment > environment.txt
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

    int drv = 0, rt = 0;               // major * 1000 + minor * 10
    if (cudaDriverGetVersion(&drv) != cudaSuccess) return 1;
    if (cudaRuntimeGetVersion(&rt) != cudaSuccess) return 1;
    std::printf("driver: CUDA %d.%d\n", drv / 1000, drv % 1000 / 10);
    std::printf("runtime: CUDA %d.%d\n", rt / 1000, rt % 1000 / 10);
    std::printf("built with toolkit %d.%d\n",
                CUDART_VERSION / 1000, CUDART_VERSION % 1000 / 10);
    bool ok = drv >= rt;               // the one rule: driver >= runtime
    std::printf("%s\n", ok ? "compatible" : "update the GPU driver");
    // keep these lines, plus the device query, with every benchmark

    // the device half of the record, one line per GPU
    int n = 0;
    check(cudaGetDeviceCount(&n), "cudaGetDeviceCount");
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp p;
        check(cudaGetDeviceProperties(&p, d), "cudaGetDeviceProperties");
        std::printf("gpu %d: %s, compute capability %d.%d, %zu MiB, warpSize %d\n",
                    d, p.name, p.major, p.minor, p.totalGlobalMem >> 20, p.warpSize);
    }

    // CPU check of the result: the rule holds, and the runtime has the major version of the headers
    bool pass = ok && rt / 1000 == CUDART_VERSION / 1000 && n > 0;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
