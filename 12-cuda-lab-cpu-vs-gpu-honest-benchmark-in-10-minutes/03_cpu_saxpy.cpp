// CPU vs GPU: An Honest Benchmark in 10 Minutes - slide 3: the cpu baseline, no strawman
// Build: make 03_cpu_saxpy
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

// the harness of the Low Latency C++ Lab
static volatile std::uint64_t g_sink = 0;
static void sink(std::uint64_t x) { g_sink = g_sink + x; }   // keeps the result alive: the loop cannot be deleted

struct BenchResult { double min_ns; double median_ns; };

template <class F>
static BenchResult bench_ns(F&& f, int warmup = 3, int repeats = 21) {
    for (int i = 0; i < warmup; ++i) f();                     // warm up: caches, branch predictor, page faults
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return BenchResult{samples.front(), samples[samples.size() / 2]};
}

// y = a * x + y: 2 flops and 12 bytes of memory per element
void saxpy_cpu(float a, const float* x, float* y,
               std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        y[i] = a * x[i] + y[i];      // same math as the kernel
}
// built with -O2, the normal build: no -O0 strawman
// same inputs, same result check, same clock as the GPU side
// one core here; split it over threads for the fair fight

static void saxpy_threads(float a, const float* x, float* y, std::size_t n, unsigned threads) {
    std::vector<std::thread> pool;
    const std::size_t chunk = (n + threads - 1) / threads;   // one contiguous slice per thread
    for (unsigned t = 0; t < threads; ++t) {
        const std::size_t begin = std::min(n, std::size_t(t) * chunk);
        const std::size_t end = std::min(n, begin + chunk);
        pool.emplace_back([=] { saxpy_cpu(a, x + begin, y + begin, end - begin); });
    }
    for (std::thread& th : pool) th.join();                   // thread start and join are on the clock too
}

int main() {
    const std::size_t n = std::size_t(1) << 22;               // 4 M floats: 16 MiB per vector
    std::vector<float> x(n), y0(n), y(n);
    std::mt19937 rng(42);                                     // fixed seed: same inputs on every toolchain
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) { x[i] = dist(rng); y0[i] = dist(rng); }
    y = y0;

    const BenchResult one = bench_ns([&] {
        saxpy_cpu(2.0f, x.data(), y.data(), n);
        sink(std::uint64_t(y[n / 2] > 0.0f));
    });
    const BenchResult four = bench_ns([&] {
        saxpy_threads(2.0f, x.data(), y.data(), n, 4);
        sink(std::uint64_t(y[n / 3] > 0.0f));
    });

    // the check: fresh inputs, one run each way, compared with a double precision reference
    std::vector<float> y1 = y0, y4 = y0;
    saxpy_cpu(2.0f, x.data(), y1.data(), n);
    saxpy_threads(2.0f, x.data(), y4.data(), n, 4);
    std::size_t wrong = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double ref = 2.0 * double(x[i]) + double(y0[i]);
        const double tol = 1e-6 * std::fabs(ref) + 1e-6;
        if (std::fabs(double(y1[i]) - ref) > tol || y4[i] != y1[i]) ++wrong;
    }

    const double bytes = 12.0 * double(n);                    // read x, read y, write y
    std::printf("saxpy on the CPU, %llu floats, steady_clock, minimum and median of 21 runs after 3 warm ups\n",
                static_cast<unsigned long long>(n));
    std::printf("%-10s %12s %12s %10s\n", "threads", "min us", "median us", "GB/s");
    std::printf("%-10d %12.1f %12.1f %10.1f\n", 1, one.min_ns / 1e3, one.median_ns / 1e3, bytes / one.median_ns);
    std::printf("%-10d %12.1f %12.1f %10.1f\n", 4, four.min_ns / 1e3, four.median_ns / 1e3, bytes / four.median_ns);
    std::printf("4 threads vs 1: %.2fx on this machine; memory bandwidth, not cores, sets the ceiling\n",
                one.median_ns / four.median_ns);
    std::printf("elements outside the tolerance: %llu of %llu\n", static_cast<unsigned long long>(wrong),
                static_cast<unsigned long long>(n));
    std::printf("%s\n", wrong == 0 ? "PASS" : "FAIL");        // the verdict is the check, never a timing
    return wrong == 0 ? 0 : 1;
}
