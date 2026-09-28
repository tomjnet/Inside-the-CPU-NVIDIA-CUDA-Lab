// CPU vs GPU: An Honest Benchmark in 10 Minutes - slide 7: kernel only, with cuda events
// Build: make 07_kernel_only
#include <algorithm>
#include <cmath>
#include <random>
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

// one thread per element: 2 flops, 12 bytes of global memory
__global__ void saxpy(float a, const float* x, float* y,
                      int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];   // same math as the CPU
}

int main() {
    if (!require_device()) return 0;

    const int n = 1 << 22;                                   // 4 M floats: 16 MiB per buffer
    const std::size_t bytes = std::size_t(n) * sizeof(float);
    std::vector<float> x(n), y0(n), y(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i) { x[i] = dist(rng); y0[i] = dist(rng); }

    float* d_x = nullptr;
    float* d_y = nullptr;
    check(cudaMalloc(&d_x, bytes), "cudaMalloc x");
    check(cudaMalloc(&d_y, bytes), "cudaMalloc y");
    check(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice), "upload x");
    check(cudaMemcpy(d_y, y0.data(), bytes, cudaMemcpyHostToDevice), "upload y");
    const int threads = 256, blocks = (n + threads - 1) / threads;

    for (int w = 0; w < 3; ++w) saxpy<<<blocks, threads>>>(2.0f, d_x, d_y, n);   // warm up: clocks, lazy loading
    check_kernel("warm up");

    std::vector<float> samples;
    for (int r = 0; r < 21; ++r) {
        cudaEvent_t t0, t1;                      // device clock, not host
        check(cudaEventCreate(&t0));  check(cudaEventCreate(&t1));
        check(cudaEventRecord(t0));              // marker in stream 0
        saxpy<<<blocks, threads>>>(2.0f, d_x, d_y, n);
        check(cudaGetLastError(), "saxpy");      // was the launch accepted?
        check(cudaEventRecord(t1));
        check(cudaEventSynchronize(t1));         // 1 synchronization point
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, t0, t1)); // kernel only: no copies
        samples.push_back(ms);
        check(cudaEventDestroy(t0));  check(cudaEventDestroy(t1));
    }
    std::sort(samples.begin(), samples.end());
    const float median_ms = samples[samples.size() / 2];

    // the check: fresh y, one more launch, every element compared on the CPU
    check(cudaMemcpy(d_y, y0.data(), bytes, cudaMemcpyHostToDevice), "reset y");
    saxpy<<<blocks, threads>>>(2.0f, d_x, d_y, n);
    check_kernel("saxpy");
    check(cudaMemcpy(y.data(), d_y, bytes, cudaMemcpyDeviceToHost), "download y");
    check(cudaFree(d_x), "cudaFree x");
    check(cudaFree(d_y), "cudaFree y");
    long long wrong = 0;
    for (int i = 0; i < n; ++i) {
        const double ref = 2.0 * double(x[i]) + double(y0[i]);
        wrong += std::fabs(double(y[i]) - ref) > 1e-6 * std::fabs(ref) + 1e-6;
    }

    const double gbs = 12.0 * double(n) / (double(median_ms) * 1e6);   // 12 bytes per element
    std::printf("saxpy, %d floats, device time from CUDA events, 21 runs after 3 warm ups\n", n);
    std::printf("kernel only: min %.3f ms, median %.3f ms, about %.0f GB/s of VRAM traffic\n",
                double(samples.front()), double(median_ms), gbs);
    std::printf("no copies inside the window: this is the number for data that already lives on the GPU\n");
    std::printf("elements outside the tolerance: %lld\n", wrong);
    std::printf("%s\n", wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? 0 : 1;
}
