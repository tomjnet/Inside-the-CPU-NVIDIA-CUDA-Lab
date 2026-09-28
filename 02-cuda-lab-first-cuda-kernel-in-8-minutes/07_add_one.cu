// Your First CUDA Kernel in 8 Minutes - slide 7: a one element arithmetic kernel
// Build: make 07_add_one
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

__host__ __device__ int next(int x) { return x + 1; }  // both sides
__global__ void add_one(int* x) { *x = next(*x); }     // device only

int main() {
    if (!require_device()) return 0;

    const int expected = next(41);       // the CPU runs the host copy of the same function

    int h = 41;                          // h lives in host memory
    int* d = nullptr;                    // d will point into device memory
    check(cudaMalloc(&d, sizeof h), "cudaMalloc");
    std::printf("host value %d at host address %p, device address %p (never dereference it on the host)\n",
                h, static_cast<void*>(&h), static_cast<void*>(d));
    check(cudaMemcpy(d, &h, sizeof h, cudaMemcpyHostToDevice), "upload");
    add_one<<<1, 1>>>(d);                // 1 kernel launch, 1 thread
    check_kernel("add_one");             // launch error, then 1 sync point
    check(cudaMemcpy(&h, d, sizeof h, cudaMemcpyDeviceToHost), "download");
    check(cudaFree(d), "cudaFree");      // 2 copies in total, h is now 42

    std::printf("GPU result %d, CPU result %d (1 launch, 2 copies, 1 synchronization point)\n", h, expected);
    bool ok = h == expected && h == 42;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
