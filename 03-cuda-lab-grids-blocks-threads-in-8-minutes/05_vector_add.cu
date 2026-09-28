// CUDA Grids, Blocks and Threads in 8 Minutes - slide 5: vector addition
// Build: make 05_vector_add
// The helper block (check, check_kernel, require_device) is the one of episode 2, unchanged.
#include <cstddef>
#include <vector>
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

int main() {
    if (!require_device()) return 0;

    const int n = 1000000;                       // not a multiple of 256
    const int threads = 256;                     // a multiple of the warp
    const int blocks = (n + threads - 1) / threads;   // 3907 blocks

    const std::size_t count = static_cast<std::size_t>(n);
    const std::size_t bytes = count * sizeof(float);                    // 4 MB per array, 12 MB in total
    std::vector<float> a(count), b(count), c(count, -1.0f);
    for (std::size_t i = 0; i < count; ++i) {                           // 3 * i stays exact in a float
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    // da, db, dc: device arrays, cudaMalloc and cudaMemcpy as in episode 2
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    check(cudaMalloc(&da, bytes), "cudaMalloc a");
    check(cudaMalloc(&db, bytes), "cudaMalloc b");
    check(cudaMalloc(&dc, bytes), "cudaMalloc c");
    check(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice), "upload a");
    check(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice), "upload b");

    add<<<blocks, threads>>>(da, db, dc, n);     // 1 kernel launch
    check_kernel("add");                         // 1 synchronization point
    // then 1 device to host copy of dc and a CPU check of every element
    check(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost), "download c");
    check(cudaFree(da), "cudaFree a");
    check(cudaFree(db), "cudaFree b");
    check(cudaFree(dc), "cudaFree c");

    std::size_t wrong = 0;
    for (std::size_t i = 0; i < count; ++i)
        if (c[i] != a[i] + b[i]) ++wrong;

    std::printf("n = %d, threads per block = %d, blocks = %d\n", n, threads, blocks);
    std::printf("threads launched = %d, stopped by the guard = %d\n", blocks * threads, blocks * threads - n);
    std::printf("c[0] = %.0f, c[1] = %.0f, c[n - 1] = %.0f\n", c[0], c[1], c[count - 1]);
    std::printf("elements that differ from the CPU sum: %zu\n", wrong);
    std::printf("%s\n", wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? 0 : 1;
}
