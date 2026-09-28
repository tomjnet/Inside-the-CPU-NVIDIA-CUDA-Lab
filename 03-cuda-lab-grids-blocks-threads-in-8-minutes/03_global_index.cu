// CUDA Grids, Blocks and Threads in 8 Minutes - slide 3: the global index
// Build: make 03_global_index
// The helper block (check, check_kernel, require_device) is the one of episode 2, unchanged.
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

// every thread runs the same kernel and computes its own global index
__global__ void show_index() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // unique per thread
    printf("block %u thread %u of %u -> i = %d\n",
           blockIdx.x, threadIdx.x, blockDim.x, i);
}

// the same index, written down so the CPU can verify it: out[i] = block * 100 + thread
__global__ void record_index(int* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = static_cast<int>(blockIdx.x * 100 + threadIdx.x);
}

int main() {
    if (!require_device()) return 0;

    std::printf("the blocks may print in any order: nothing promises which block runs first\n");
    show_index<<<3, 4>>>();        // 1 kernel launch: 3 blocks of 4 threads
    check_kernel("show_index");    // launch error, then 1 sync point
    // gridDim.x = 3, blockDim.x = 4: 12 threads, i runs from 0 to 11

    const int blocks = 3, threads = 4, total = blocks * threads;
    int host[12] = {};
    int* d_out = nullptr;
    check(cudaMalloc(&d_out, sizeof host), "cudaMalloc");
    check(cudaMemset(d_out, 0xFF, sizeof host), "cudaMemset");         // every int is -1: not written yet
    record_index<<<blocks, threads>>>(d_out);
    check_kernel("record_index");
    check(cudaMemcpy(host, d_out, sizeof host, cudaMemcpyDeviceToHost), "download");
    check(cudaFree(d_out), "cudaFree");

    bool ok = true;
    std::printf("\nCPU check: element i must come from block i / 4, thread i %% 4\n");
    for (int i = 0; i < total; ++i) {
        int expected = (i / threads) * 100 + i % threads;
        if (host[i] != expected) ok = false;
        std::printf("  i = %2d  block %d thread %d%s\n", i, host[i] / 100, host[i] % 100,
                    host[i] == expected ? "" : "  (wrong)");
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
