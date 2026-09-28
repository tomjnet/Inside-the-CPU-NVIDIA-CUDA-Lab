// Your First CUDA Kernel in 8 Minutes - slide 6: check and check_kernel
// Build: make 06_check
//
// The helper block below is pasted unchanged into every .cu sample of the playlist. The slide shows
// check and check_kernel with shorter comments; names and logic are the same.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static void check(cudaError_t e, const char* what = "CUDA call") {
    if (e != cudaSuccess) {                          // every runtime call returns a status
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
static void check_kernel(const char* name) {         // a launch returns nothing: ask twice
    check(cudaGetLastError(), name);                 // 1. was the launch itself accepted?
    check(cudaDeviceSynchronize(), name);            // 2. did the kernel run without a fault?
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

__global__ void store(int* out, int value) { *out = value; }

int main() {
    if (!require_device()) return 0;

    // what check would print: the status codes are an enum, cudaGetErrorString gives the readable text
    const cudaError_t samples[] = {cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation,
                                   cudaErrorIllegalAddress};
    for (cudaError_t e : samples)
        std::printf("status %3d  %-28s %s\n", static_cast<int>(e), cudaGetErrorName(e), cudaGetErrorString(e));

    // the pattern of every later sample: check around each runtime call, check_kernel after each launch
    int result = 0;
    int* d_result = nullptr;
    check(cudaMalloc(&d_result, sizeof result), "cudaMalloc");
    store<<<1, 1>>>(d_result, 2026);
    check_kernel("store");                           // launch accepted, then kernel finished
    check(cudaMemcpy(&result, d_result, sizeof result, cudaMemcpyDeviceToHost), "download");
    check(cudaFree(d_result));                       // the label is optional: "CUDA call"

    std::printf("kernel stored %d, every call checked, 1 launch and 1 synchronization point\n", result);
    bool ok = result == 2026;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
