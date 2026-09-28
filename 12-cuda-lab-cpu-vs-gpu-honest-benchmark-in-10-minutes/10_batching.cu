// CPU vs GPU: An Honest Benchmark in 10 Minutes - slide 10: batching amortizes the fixed cost
// Build: make 10_batching
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

template <class F>
static double median_ms(F&& f, int runs = 7) {               // steady_clock, 1 warm up, median of runs
    f();
    std::vector<double> s;
    for (int r = 0; r < runs; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        auto t1 = std::chrono::steady_clock::now();
        s.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(s.begin(), s.end());
    return s[s.size() / 2];
}

int main() {
    if (!require_device()) return 0;

    const int jobs = 1000;
    const int total = jobs * 1024;                           // 1 M floats: 4 MiB per buffer
    const std::size_t bytes = std::size_t(total) * sizeof(float);
    std::vector<float> x(total), y0(total), y(total);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < total; ++i) { x[i] = dist(rng); y0[i] = dist(rng); }
    float* d_x = nullptr;
    float* d_y = nullptr;
    check(cudaMalloc(&d_x, bytes), "cudaMalloc x");
    check(cudaMalloc(&d_y, bytes), "cudaMalloc y");
    check(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice), "upload x");

    auto many = [&] {
        // 1000 jobs of 1 K floats, one launch each: 1000 launch costs
        for (int b = 0; b < jobs; ++b)
            saxpy<<<4, 256>>>(2.0f, d_x + b * 1024, d_y + b * 1024, 1024);
        check_kernel("1000 launches");
    };
    auto one = [&] {
        // the same 1 M floats as one batch: 1 launch, cost paid once
        int n = jobs * 1024;
        saxpy<<<(n + 255) / 256, 256>>>(2.0f, d_x, d_y, n);
        check_kernel("1 launch");      // same bytes, same flops, 1 launch
    };

    // the check first: each way once on fresh data, every element against the CPU
    long long wrong = 0;
    for (int way = 0; way < 2; ++way) {
        check(cudaMemcpy(d_y, y0.data(), bytes, cudaMemcpyHostToDevice), "reset y");
        if (way == 0) many(); else one();
        check(cudaMemcpy(y.data(), d_y, bytes, cudaMemcpyDeviceToHost), "download y");
        for (int i = 0; i < total; ++i) {
            const double ref = 2.0 * double(x[i]) + double(y0[i]);
            wrong += std::fabs(double(y[i]) - ref) > 1e-6 * std::fabs(ref) + 1e-6;
        }
    }

    // kernels only: the data stays on the device
    const double k_many = median_ms(many), k_one = median_ms(one);

    // with the copies: per job 2 small copies in and 1 out, or 3 large copies for the whole batch
    const std::size_t job_bytes = 1024 * sizeof(float);
    const double e_many = median_ms([&] {
        for (int b = 0; b < jobs; ++b) {
            const std::size_t off = std::size_t(b) * 1024;
            check(cudaMemcpy(d_x + off, x.data() + off, job_bytes, cudaMemcpyHostToDevice));
            check(cudaMemcpy(d_y + off, y0.data() + off, job_bytes, cudaMemcpyHostToDevice));
            saxpy<<<4, 256>>>(2.0f, d_x + off, d_y + off, 1024);
            check(cudaGetLastError(), "saxpy per job");
            check(cudaMemcpy(y.data() + off, d_y + off, job_bytes, cudaMemcpyDeviceToHost));
        }
    }, 3);
    const double e_one = median_ms([&] {
        check(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));
        check(cudaMemcpy(d_y, y0.data(), bytes, cudaMemcpyHostToDevice));
        saxpy<<<(total + 255) / 256, 256>>>(2.0f, d_x, d_y, total);
        check(cudaGetLastError(), "saxpy batch");
        check(cudaMemcpy(y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    }, 3);
    check(cudaFree(d_x), "cudaFree x");
    check(cudaFree(d_y), "cudaFree y");

    std::printf("saxpy on %d floats as %d jobs of 1024, steady_clock, median after 1 warm up\n", total, jobs);
    std::printf("%-26s %14s %14s %8s\n", "", "1000 jobs ms", "1 batch ms", "ratio");
    std::printf("%-26s %14.3f %14.3f %7.1fx\n", "kernels only", k_many, k_one, k_many / k_one);
    std::printf("%-26s %14.3f %14.3f %7.1fx\n", "with copies, end to end", e_many, e_one, e_many / e_one);
    std::printf("same bytes, same flops: the difference is fixed cost per launch and per copy on this machine\n");
    std::printf("elements outside the tolerance: %lld\n", wrong);
    std::printf("%s\n", wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? 0 : 1;
}
