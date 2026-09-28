// CUDA Unified Memory in 7 Minutes - slide 2: one pointer, two memories
// Build: make 02_managed_alloc   (nvcc -std=c++20 -O2)
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

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];   // 2 global reads, 1 global write
}

static long count_wrong(const float* c, int n, float expected) {   // CPU check of every element
    long wrong = 0;
    for (int i = 0; i < n; ++i) {
        if (c[i] != expected) ++wrong;
    }
    return wrong;
}

int main() {
    if (!require_device()) return 0;
    const int n = 1 << 22;                           // 4 M floats, 16 MiB per array, 48 MiB in total
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // one allocation, one pointer, valid on the host and on the device
    float* a = nullptr; float* b = nullptr; float* c = nullptr;
    check(cudaMallocManaged(&a, n * sizeof(float)));  // no cudaMemcpy
    check(cudaMallocManaged(&b, n * sizeof(float)));
    check(cudaMallocManaged(&c, n * sizeof(float)));
    for (int i = 0; i < n; ++i) { a[i] = 1.0f; b[i] = 2.0f; }  // CPU touch
    add<<<blocks, threads>>>(a, b, c, n);   // GPU touch: pages migrate
    check_kernel("add");                    // synchronization point
    std::printf("c[7] = %.1f\n", c[7]);     // CPU touch: pages come back

    std::printf("a = %p, b = %p, c = %p: the same addresses on the host and on the device\n",
                static_cast<void*>(a), static_cast<void*>(b), static_cast<void*>(c));
    std::printf("lines of cudaMemcpy in this program: 0\n");
    long wrong = count_wrong(c, n, 3.0f);   // the CPU reads all of c: every page comes back
    std::printf("checked %d elements on the CPU, %ld wrong\n", n, wrong);

    check(cudaFree(a)); check(cudaFree(b)); check(cudaFree(c));
    std::printf("%s\n", wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? 0 : 1;
}
