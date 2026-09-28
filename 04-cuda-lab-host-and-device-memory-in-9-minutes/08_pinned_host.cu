// CUDA Host and Device Memory in 9 Minutes - slide 8: cudamallochost
// Build: make 08_pinned_host
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

#include <algorithm>
#include <cstddef>
#include <vector>

__global__ void add_one(float* p, std::size_t n) {
    std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < n) p[i] = p[i] + 1.0f;
}

// Device time of one host to device copy, measured with CUDA events: one warm up, then the best of five.
static double time_copy_ms(float* d_dst, const float* h_src, std::size_t bytes) {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    check(cudaMemcpy(d_dst, h_src, bytes, cudaMemcpyHostToDevice), "warm up copy");
    double best = 1e30;
    for (int r = 0; r < 5; ++r) {
        check(cudaEventRecord(start));
        check(cudaMemcpy(d_dst, h_src, bytes, cudaMemcpyHostToDevice), "timed copy");
        check(cudaEventRecord(stop));
        check(cudaEventSynchronize(stop));
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start, stop));
        best = std::min(best, double(ms));
    }
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    return best;
}

int main() {
    if (!require_device()) return 0;
    const std::size_t n = std::size_t(1) << 22;          // 4M floats: 16 MiB
    const std::size_t bytes = n * sizeof(float);
    float* d_a = nullptr;
    check(cudaMalloc(&d_a, bytes), "cudaMalloc");

    float* h_pinned = nullptr;                // page locked host memory:
    check(cudaMallocHost(&h_pinned, bytes));  // the GPU can DMA from it
    std::vector<float> h_pageable(n, 1.0f);   // ordinary new or malloc
    for (std::size_t i = 0; i < n; ++i) h_pinned[i] = 1.0f;

    // same bytes, same call; pageable pays one extra copy to a staging
    // buffer inside the driver, pinned goes straight across PCIe
    double ms_pageable = time_copy_ms(d_a, h_pageable.data(), bytes);
    double ms_pinned   = time_copy_ms(d_a, h_pinned, bytes);

    add_one<<<unsigned((n + 255) / 256), 256>>>(d_a, n);   // the device saw the pinned upload
    check_kernel("add_one");
    std::vector<float> h_back(n, 0.0f);       // CPU check: what arrived is what was sent, plus one
    check(cudaMemcpy(h_back.data(), d_a, bytes, cudaMemcpyDeviceToHost), "download");
    std::size_t wrong = 0;
    for (std::size_t i = 0; i < n; ++i) if (h_back[i] != 2.0f) ++wrong;

    const double gib = 1024.0 * 1024.0 * 1024.0;
    std::printf("host to device, %zu MiB, CUDA events (device time), best of 5, this machine:\n", bytes >> 20);
    std::printf("  pageable  %8.3f ms  %6.2f GiB/s\n", ms_pageable, double(bytes) / gib / (ms_pageable / 1000.0));
    std::printf("  pinned    %8.3f ms  %6.2f GiB/s\n", ms_pinned, double(bytes) / gib / (ms_pinned / 1000.0));
    std::printf("  ratio pageable / pinned: %.2f (a timing, never a pass condition)\n", ms_pageable / ms_pinned);

    check(cudaFreeHost(h_pinned));            // never free() or delete
    check(cudaFree(d_a));

    bool ok = wrong == 0;
    std::printf("elements that differ after the round trip: %zu\n", wrong);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
