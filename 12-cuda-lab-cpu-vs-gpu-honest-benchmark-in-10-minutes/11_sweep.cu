// CPU vs GPU: An Honest Benchmark in 10 Minutes - slide 11: the whole benchmark
// Build: make 11_sweep
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

// samples in us: median and p99 by nearest rank, as on slide 5
static double pct(std::vector<double>& s, double p) {
    std::sort(s.begin(), s.end());
    std::size_t k = std::size_t(p / 100.0 * double(s.size()));
    return s[std::min(k, s.size() - 1)];
}

struct Stats { double median; double p99; int runs; };

static Stats stats_of(std::vector<double>& s) {
    return Stats{pct(s, 50.0), pct(s, 99.0), int(s.size())};
}

static const int kWarmup = 3;
static int runs_for(int n) {                                 // many cheap runs for small sizes, few for large ones
    if (n <= (1 << 18)) return 201;                          // 201 runs: p99 is the 3rd slowest
    if (n <= (1 << 20)) return 51;
    if (n <= (1 << 22)) return 21;
    return 11;                                               // 11 runs: p99 is the max, say so
}

static std::vector<float> g_x, g_y0, g_y;                    // host inputs, pageable, sized for the largest n

static double us_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}

static Stats run_cpu(int n) {                                // -O2 loop, steady_clock
    std::vector<double> s;
    for (int r = 0; r < kWarmup + runs_for(n); ++r) {
        auto t0 = std::chrono::steady_clock::now();
        saxpy_cpu(2.0f, g_x.data(), g_y.data(), n);
        const double us = us_since(t0);
        if (r >= kWarmup) s.push_back(us);
    }
    return stats_of(s);
}

static Stats run_kernel_only(int n) {                        // CUDA events, data resident on the device
    const std::size_t bytes = std::size_t(n) * sizeof(float);
    float* d_x = nullptr;
    float* d_y = nullptr;
    check(cudaMalloc(&d_x, bytes), "cudaMalloc x");
    check(cudaMalloc(&d_y, bytes), "cudaMalloc y");
    check(cudaMemcpy(d_x, g_x.data(), bytes, cudaMemcpyHostToDevice), "upload x");
    check(cudaMemcpy(d_y, g_y0.data(), bytes, cudaMemcpyHostToDevice), "upload y");
    const int threads = 256, blocks = (n + threads - 1) / threads;
    cudaEvent_t t0, t1;
    check(cudaEventCreate(&t0));
    check(cudaEventCreate(&t1));
    std::vector<double> s;
    for (int r = 0; r < kWarmup + runs_for(n); ++r) {
        check(cudaEventRecord(t0));
        saxpy<<<blocks, threads>>>(2.0f, d_x, d_y, n);
        check(cudaGetLastError(), "saxpy");
        check(cudaEventRecord(t1));
        check(cudaEventSynchronize(t1));
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, t0, t1));
        if (r >= kWarmup) s.push_back(double(ms) * 1000.0);
    }
    check_kernel("saxpy kernel only");
    check(cudaEventDestroy(t0));
    check(cudaEventDestroy(t1));
    check(cudaFree(d_x), "cudaFree x");
    check(cudaFree(d_y), "cudaFree y");
    return stats_of(s);
}

// one complete job: allocate, 2 copies in, launch, sync, 1 copy out, free
static void gpu_job(int n, const float* x, float* y) {
    const std::size_t bytes = std::size_t(n) * sizeof(float);
    float* d_x = nullptr;
    float* d_y = nullptr;
    check(cudaMalloc(&d_x, bytes));  check(cudaMalloc(&d_y, bytes));
    check(cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice));
    saxpy<<<(n + 255) / 256, 256>>>(2.0f, d_x, d_y, n);
    check_kernel("saxpy");
    check(cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost));
    check(cudaFree(d_x));  check(cudaFree(d_y));
}

static Stats run_end_to_end(int n) {                         // steady_clock around the whole job
    std::vector<double> s;
    for (int r = 0; r < kWarmup + runs_for(n); ++r) {
        auto t0 = std::chrono::steady_clock::now();
        gpu_job(n, g_x.data(), g_y.data());
        const double us = us_since(t0);
        if (r >= kWarmup) s.push_back(us);
    }
    return stats_of(s);
}

static bool verify(int n) {                                  // fresh inputs, one GPU job, every element checked
    std::copy(g_y0.begin(), g_y0.begin() + n, g_y.begin());
    gpu_job(n, g_x.data(), g_y.data());
    long long wrong = 0;
    for (int i = 0; i < n; ++i) {
        const double ref = 2.0 * double(g_x[i]) + double(g_y0[i]);
        wrong += std::fabs(double(g_y[i]) - ref) > 1e-6 * std::fabs(ref) + 1e-6;
    }
    if (wrong != 0) {
        std::printf("n = %d: %lld elements outside the tolerance\nFAIL\n", n, wrong);
        return false;
    }
    return true;
}

static void size_label(int n, char* out, std::size_t len) {
    if (n >= (1 << 20)) std::snprintf(out, len, "%d M", n >> 20);
    else std::snprintf(out, len, "%d K", n >> 10);
}

static void print_row(int n, const Stats& cpu, const Stats& gk, const Stats& ge) {
    char label[16];
    size_label(n, label, sizeof label);
    std::printf("%-6s %5d | %9.1f %9.1f | %9.1f %9.1f | %9.1f %9.1f | %-8s %-8s\n", label, cpu.runs,
                cpu.median, cpu.p99, gk.median, gk.p99, ge.median, ge.p99,
                gk.median < cpu.median ? "GPU" : "CPU", ge.median < cpu.median ? "GPU" : "CPU");
}

int main() {
    if (!require_device()) return 0;

    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    const int max_n = 1 << 24;                               // 16 M floats: 64 MiB per buffer, 128 MiB on the device
    g_x.resize(max_n);
    g_y0.resize(max_n);
    g_y.resize(max_n);
    std::mt19937 rng(42);                                    // fixed seed: same work on every run
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < max_n; ++i) { g_x[i] = dist(rng); g_y0[i] = dist(rng); }
    g_y = g_y0;

    std::printf("saxpy y = 2x + y on %s, CPU one core -O2; times in microseconds, median and p99\n", prop.name);
    std::printf("kernel only: CUDA events, data resident. end to end: steady_clock around malloc, copies, launch, free\n");
    std::printf("%-6s %5s | %9s %9s | %9s %9s | %9s %9s | %-8s %-8s\n", "size", "runs", "cpu med", "cpu p99",
                "krn med", "krn p99", "e2e med", "e2e p99", "resident", "e2e");

    // 1 K to 16 M floats, x4 per step: find the break even size
    for (int n = 1 << 10; n <= 1 << 24; n <<= 2) {
        Stats cpu = run_cpu(n);           // -O2 loop, steady_clock
        Stats gk  = run_kernel_only(n);   // events, data resident
        Stats ge  = run_end_to_end(n);    // malloc, 3 copies, sync
        if (!verify(n)) return 1;         // CPU check before any number
        print_row(n, cpu, gk, ge);        // median and p99 each
    }                                     // winner per size, 2 ways

    std::printf("with 11 runs the p99 is the slowest run; every number is for this machine only\n");
    std::printf("resident: GPU kernel only against the CPU. e2e: the whole GPU job against the CPU\n");
    std::printf("PASS\n");                                   // every size passed the CPU check
    return 0;
}
