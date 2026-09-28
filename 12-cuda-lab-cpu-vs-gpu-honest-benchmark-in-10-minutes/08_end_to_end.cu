// CPU vs GPU: An Honest Benchmark in 10 Minutes - slide 8: end to end, with steady_clock
// Build: make 08_end_to_end
#include <algorithm>
#include <chrono>
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

static void saxpy_cpu(float a, const float* x, float* y, int n) {
    for (int i = 0; i < n; ++i) y[i] = a * x[i] + y[i];     // the -O2 baseline of slide 3
}

static double ms_since(std::chrono::steady_clock::time_point t0, std::chrono::steady_clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    if (!require_device()) return 0;

    const int n = 1 << 22;                                   // 4 M floats: 16 MiB per buffer
    const std::size_t bytes = std::size_t(n) * sizeof(float);
    std::vector<float> hx(n), y0(n), hy(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i) { hx[i] = dist(rng); y0[i] = dist(rng); }
    const float* x = hx.data();                              // pageable host memory, as most programs have
    float* y = hy.data();
    float* d_x = nullptr;
    float* d_y = nullptr;
    const int threads = 256, blocks = (n + threads - 1) / threads;

    std::vector<double> gpu_ms, cpu_ms;
    for (int r = 0; r < 2 + 11; ++r) {                       // 2 warm ups (the first creates the context), 11 runs
        hy = y0;                                             // fresh input, outside the clock
        auto t0 = std::chrono::steady_clock::now();  // host clock
        check(cudaMalloc(&d_x, bytes));  check(cudaMalloc(&d_y, bytes));
        check(cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice));
        check(cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice));
        saxpy<<<blocks, threads>>>(2.0f, d_x, d_y, n);
        check_kernel("saxpy");                       // 1 launch, 1 sync
        check(cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost));
        check(cudaFree(d_x));  check(cudaFree(d_y));
        auto t1 = std::chrono::steady_clock::now();  // 3 PCIe crossings
        if (r >= 2) gpu_ms.push_back(ms_since(t0, t1));
    }

    // the check: the last GPU result against the double precision reference
    long long wrong = 0;
    for (int i = 0; i < n; ++i) {
        const double ref = 2.0 * double(x[i]) + double(y0[i]);
        wrong += std::fabs(double(y[i]) - ref) > 1e-6 * std::fabs(ref) + 1e-6;
    }

    std::vector<float> cy = y0;                              // the same job on the CPU, same clock
    for (int r = 0; r < 2 + 11; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        saxpy_cpu(2.0f, x, cy.data(), n);
        auto t1 = std::chrono::steady_clock::now();
        if (r >= 2) cpu_ms.push_back(ms_since(t0, t1));
    }
    std::sort(gpu_ms.begin(), gpu_ms.end());
    std::sort(cpu_ms.begin(), cpu_ms.end());
    const double g = gpu_ms[gpu_ms.size() / 2], c = cpu_ms[cpu_ms.size() / 2];

    std::printf("saxpy, %d floats, steady_clock, median of 11 runs after 2 warm ups (%.1f checksum)\n", n,
                double(cy[n / 2]));
    std::printf("GPU end to end (malloc, 2 copies in, launch, sync, 1 copy out, free): %8.3f ms\n", g);
    std::printf("CPU, one core, -O2:                                                   %8.3f ms\n", c);
    std::printf("on this machine the %s wins end to end by %.2fx; run it on yours\n", g < c ? "GPU" : "CPU",
                g < c ? c / g : g / c);
    std::printf("elements outside the tolerance: %lld\n", wrong);
    std::printf("%s\n", wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? 0 : 1;
}
