// CPU vs GPU: An Honest Benchmark in 10 Minutes - slide 5: warm up, repeat, median and p99
// Build: make 05_percentiles
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

static volatile std::uint64_t g_sink = 0;
static void sink(std::uint64_t x) { g_sink = g_sink + x; }   // keeps the result alive: the loop cannot be deleted

// samples in ns: median and p99 by nearest rank
double pct(std::vector<double>& s, double p) {
    std::sort(s.begin(), s.end());             // O(n log n)
    std::size_t k = std::size_t(p / 100.0 * double(s.size()));
    return s[std::min(k, s.size() - 1)];
}

int main() {
    // 1. the rank rule on known data: 1, 2, ..., 201 in reverse order
    std::vector<double> known;
    for (int i = 201; i >= 1; --i) known.push_back(double(i));   // any order: pct sorts
    const double m = pct(known, 50.0), q = pct(known, 99.0), top = pct(known, 100.0);
    std::printf("known data 1 to 201: median %.0f, p99 %.0f, max %.0f\n", m, q, top);
    std::printf("p99 is the 3rd slowest of 201 runs (199), not the max (201)\n\n");
    const bool ok = m == 101.0 && q == 199.0 && top == 201.0;

    // 2. real samples: sum 64 K integers, 5 warm ups thrown away, then 201 timed runs
    std::vector<std::uint32_t> data(std::size_t(1) << 16);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = std::uint32_t(i * 2654435761u);
    auto work = [&] {
        std::uint64_t s = 0;
        for (std::uint32_t v : data) s += v;
        sink(s);
    };
    for (int i = 0; i < 5; ++i) work();        // warm up: caches, page faults, clocks
    std::vector<double> samples;
    for (int i = 0; i < 201; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        work();
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    // 201 runs: p99 is the 3rd slowest, not the max
    double median = pct(samples, 50.0);
    double p99 = pct(samples, 99.0);               // the tail you ship
    std::printf("sum of 64 K integers, 201 runs on this machine, steady_clock\n");
    std::printf("  min    %10.1f ns\n", samples.front());
    std::printf("  median %10.1f ns\n", median);
    std::printf("  p99    %10.1f ns   (%.2fx the median)\n", p99, p99 / median);
    std::printf("  max    %10.1f ns   (one run: a page fault, an interrupt, another process)\n", samples.back());
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
