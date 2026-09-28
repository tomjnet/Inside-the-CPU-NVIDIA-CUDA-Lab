// Profile CUDA with NVIDIA Nsight in 10 Minutes - slide 5: profiling noise and warm up
// Build: make 05_warm_up
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

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

// one read and one write per element: x = x * a + 1
__global__ void scale(float* x, int n, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * a + 1.0f;
}

int main() {
    if (!require_device()) return 0;

    const int n = 1 << 20;                           // 4 MiB of floats
    const int blocks = (n + 255) / 256;
    std::vector<float> h(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) h[static_cast<std::size_t>(i)] = static_cast<float>(i % 1000);

    float* d = nullptr;
    check(cudaMalloc(&d, h.size() * sizeof(float)), "cudaMalloc");
    check(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice), "upload");
    cudaEvent_t t0, t1;
    check(cudaEventCreate(&t0), "cudaEventCreate");
    check(cudaEventCreate(&t1), "cudaEventCreate");

    auto w0 = std::chrono::steady_clock::now();      // sample only: time the warm up to show its cost
    // launch 1 pays context and module setup: never time it
    scale<<<blocks, 256>>>(d, n, 1.0f);      // warm up: 1 launch
    check_kernel("warm up");                 // 1 sync, not timed
    auto w1 = std::chrono::steady_clock::now();      // sample only
    check(cudaEventRecord(t0));
    for (int r = 0; r < 20; ++r)             // 20 launches, 0 syncs
        scale<<<blocks, 256>>>(d, n, 1.0f);
    check(cudaEventRecord(t1));
    check_kernel("timed");                   // device time = t1 - t0

    float loop_ms = 0.0f;
    check(cudaEventElapsedTime(&loop_ms, t0, t1), "cudaEventElapsedTime");
    double warm_us = std::chrono::duration<double, std::micro>(w1 - w0).count();

    // seven more single launches, each timed with events: keep the median
    std::vector<float> one(7);
    for (float& ms : one) {
        check(cudaEventRecord(t0));
        scale<<<blocks, 256>>>(d, n, 1.0f);
        check(cudaEventRecord(t1));
        check_kernel("single");
        check(cudaEventElapsedTime(&ms, t0, t1), "cudaEventElapsedTime");
    }
    std::sort(one.begin(), one.end());

    std::printf("warm up launch, end to end (steady_clock): %9.1f us\n", warm_us);
    std::printf("20 timed launches, device time (events):   %9.1f us each\n", 1000.0 * loop_ms / 20.0);
    std::printf("single launches, device time, median of 7: %9.1f us (min %.1f, max %.1f)\n",
                1000.0 * one[3], 1000.0 * one.front(), 1000.0 * one.back());
    std::printf("this machine: the warm up is typically many times one steady launch\n");

    std::vector<float> out(h.size());
    check(cudaMemcpy(out.data(), d, out.size() * sizeof(float), cudaMemcpyDeviceToHost), "download");
    check(cudaEventDestroy(t0), "cudaEventDestroy");
    check(cudaEventDestroy(t1), "cudaEventDestroy");
    check(cudaFree(d), "cudaFree");

    bool ok = true;                                  // 1 + 20 + 7 launches, each adds 1
    for (std::size_t i = 0; i < out.size(); ++i)
        if (out[i] != h[i] + 28.0f) { ok = false; break; }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
