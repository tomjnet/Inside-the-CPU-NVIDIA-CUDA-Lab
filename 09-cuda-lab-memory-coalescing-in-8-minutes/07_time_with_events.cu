// Memory Coalescing in CUDA in 8 Minutes - slide 7: timing with events
// Build: make 07_time_with_events
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
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

// One thread per output element; the table decides the read address.
__global__ void gather(const float* in, const int* map, float* out,
                       int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;                 // guard the rounded up grid
    out[i] = 2.0f * in[map[i]] + 1.0f;  // map[i], out[i]: coalesced
}                                       // in[map[i]]: up to the table

template <class Launch>
static float device_ms(Launch&& launch) {  // device time only
    cudaEvent_t t0, t1;
    check(cudaEventCreate(&t0));  check(cudaEventCreate(&t1));
    check(cudaEventRecord(t0, 0));      // marker into stream 0
    launch();                           // asynchronous: returns at once
    check(cudaEventRecord(t1, 0));
    check(cudaEventSynchronize(t1));    // 1 synchronization point
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, t0, t1));
    check(cudaEventDestroy(t0));  check(cudaEventDestroy(t1));
    return ms;
}

struct Stat { float min_ms; float median_ms; };
template <class Launch>
static Stat measure(Launch&& launch, int warmup = 3, int repeats = 21) {
    for (int i = 0; i < warmup; ++i) launch();       // warm up: clocks, caches, lazy setup
    check(cudaDeviceSynchronize(), "warm up");
    std::vector<float> s;
    for (int i = 0; i < repeats; ++i) s.push_back(device_ms(launch));
    std::sort(s.begin(), s.end());                   // keep minimum and median, as the Low Latency C++ Lab does
    return Stat{s.front(), s[s.size() / 2]};
}

int main() {
    if (!require_device()) return 0;

    const int n = 1 << 22;                                   // 4 M floats: 16 MiB per buffer, 48 MiB in total
    const std::size_t fbytes = std::size_t(n) * sizeof(float);
    const std::size_t ibytes = std::size_t(n) * sizeof(int);
    std::vector<float> in(n), out(n);
    for (int i = 0; i < n; ++i) in[i] = float(i % 1000);

    std::vector<int> ident(n), strided(n), scattered(n);
    for (int i = 0; i < n; ++i) {
        ident[i] = i;                       // neighbours: 1 transaction
        strided[i] = int((unsigned(i) * 33u) & unsigned(n - 1));
    }                                       // 33 apart: n is a power of 2
    scattered = ident;                      // same addresses, shuffled
    std::mt19937 rng(42);                   // fixed seed: same work
    std::shuffle(scattered.begin(), scattered.end(), rng);
    // every table is a permutation: same reads, same sum, other order

    float* d_in = nullptr;
    float* d_out = nullptr;
    int* d_map = nullptr;
    check(cudaMalloc(&d_in, fbytes), "cudaMalloc in");
    check(cudaMalloc(&d_out, fbytes), "cudaMalloc out");
    check(cudaMalloc(&d_map, ibytes), "cudaMalloc map");
    check(cudaMemcpy(d_in, in.data(), fbytes, cudaMemcpyHostToDevice), "upload in");

    const int threads = 256, blocks = (n + threads - 1) / threads;
    const std::vector<int>* tables[] = {&ident, &strided, &scattered};
    const char* names[] = {"coalesced", "strided", "scattered"};
    Stat stats[3] = {};
    bool ok = true;
    for (int t = 0; t < 3; ++t) {
        const std::vector<int>& map = *tables[t];
        check(cudaMemcpy(d_map, map.data(), ibytes, cudaMemcpyHostToDevice), "upload map");
        stats[t] = measure([&] { gather<<<blocks, threads>>>(d_in, d_map, d_out, n); });
        check_kernel("gather");
        check(cudaMemcpy(out.data(), d_out, fbytes, cudaMemcpyDeviceToHost), "download");
        long long wrong = 0;                                 // CPU check of every element
        for (int i = 0; i < n; ++i) wrong += out[i] != 2.0f * in[map[i]] + 1.0f;
        ok = ok && wrong == 0;
    }
    check(cudaFree(d_in), "cudaFree");
    check(cudaFree(d_out), "cudaFree");
    check(cudaFree(d_map), "cudaFree");

    std::printf("gather, %d elements, device time from events (kernel only, no copies), 21 runs after 3 warm ups\n", n);
    std::printf("%-10s %10s %12s %8s\n", "table", "min ms", "median ms", "ratio");
    for (int t = 0; t < 3; ++t)
        std::printf("%-10s %10.3f %12.3f %7.2fx\n", names[t], stats[t].min_ms, stats[t].median_ms,
                    stats[t].median_ms / stats[0].median_ms);
    std::printf("ratios are for this GPU only: run it on yours, and never trust a single run\n");
    std::printf("%s\n", ok ? "PASS" : "FAIL");                // the verdict is the CPU check, never a timing
    return ok ? 0 : 1;
}
