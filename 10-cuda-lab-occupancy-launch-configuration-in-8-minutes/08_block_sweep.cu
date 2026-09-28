// CUDA Occupancy and Launch Configuration in 8 Minutes - slide 8: the block size sweep
// Build: make 08_block_sweep
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

__global__ void scale_add(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global index
    if (i < n) y[i] = a * x[i] + y[i];               // 2 global reads, 1 write
}

#include <algorithm>
#include <vector>

// device time of one launch with CUDA events: 1 warm up, then the median of 7 runs
template <class F>
static float time_ms(F&& launch) {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");
    launch();                                        // warm up: first launch pays setup
    check(cudaGetLastError(), "warm up");
    std::vector<float> runs;
    for (int r = 0; r < 7; ++r) {
        check(cudaEventRecord(start), "record");
        launch();
        check(cudaEventRecord(stop), "record");
        check(cudaEventSynchronize(stop), "sync");   // 1 synchronization point per run
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        runs.push_back(ms);
    }
    check(cudaEventDestroy(start), "event");
    check(cudaEventDestroy(stop), "event");
    std::sort(runs.begin(), runs.end());
    return runs[runs.size() / 2];
}

struct Row { int threads; int per_sm; float occupancy; const char* limit; float ms; };
static std::vector<Row> table;
static cudaDeviceProp p;
static cudaFuncAttributes attr;

// collects one row: occupancy from blocks per SM, and the ceiling that matches it
static void print_row(int threads, int per_sm, float ms) {
    int by_threads = p.maxThreadsPerMultiProcessor / threads;
    int by_slots = p.maxBlocksPerMultiProcessor;
    int by_regs = p.regsPerMultiprocessor / (attr.numRegs * threads);
    const char* limit = per_sm == by_threads ? "threads"
                      : per_sm == by_slots   ? "block slots"
                      : per_sm == by_regs    ? "registers"
                                             : "other";
    float occ = float(per_sm * threads) / float(p.maxThreadsPerMultiProcessor);
    table.push_back(Row{threads, per_sm, occ, limit, ms});
}

int main() {
    if (!require_device()) return 0;
    check(cudaGetDeviceProperties(&p, 0), "properties");
    check(cudaFuncGetAttributes(&attr, scale_add), "attributes");

    const int n = 1 << 22;                      // 4 M floats, 16 MiB per array
    static float h_x[n], h_y[n];
    for (int i = 0; i < n; ++i) h_x[i] = float(i % 7);
    float *d_x = nullptr, *d_y = nullptr;
    check(cudaMalloc(&d_x, n * sizeof(float)), "cudaMalloc x");
    check(cudaMalloc(&d_y, n * sizeof(float)), "cudaMalloc y");
    check(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice), "upload");
    check(cudaMemset(d_y, 0, n * sizeof(float)), "memset");
    int per_sm = 0;

    for (int threads = 32; threads <= 1024; threads *= 2) {
        int blocks = (n + threads - 1) / threads;   // round up
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &per_sm, scale_add, threads, 0), "occupancy");
        float ms = time_ms([&] {                // CUDA events, median
            scale_add<<<blocks, threads>>>(n, 2.0f, d_x, d_y); });
        check_kernel("scale_add");
        print_row(threads, per_sm, ms);         // occupancy, limit, time
    }

    std::size_t best = 0;
    for (std::size_t r = 1; r < table.size(); ++r)
        if (table[r].ms < table[best].ms) best = r;
    std::printf("%s, %d SMs, scale_add uses %d registers, n = %d\n", p.name, p.multiProcessorCount,
                attr.numRegs, n);
    std::printf("threads  blocks/SM  occupancy  limit        median ms  (device time, this machine)\n");
    for (std::size_t r = 0; r < table.size(); ++r) {
        const Row& w = table[r];
        const char* mark = r == best ? "  fastest"
                         : w.ms <= table[best].ms * 1.05f ? "  within 5 percent" : "";
        std::printf("%7d  %9d  %8.0f%%  %-11s  %9.3f%s\n", w.threads, w.per_sm, w.occupancy * 100.0f,
                    w.limit, w.ms, mark);
    }
    std::printf("on a plateau the fastest row can change from run to run: tune by measurement\n");

    // one clean run with the fastest configuration, checked on the CPU
    const int t = table[best].threads;
    check(cudaMemset(d_y, 0, n * sizeof(float)), "memset");
    scale_add<<<(n + t - 1) / t, t>>>(n, 2.0f, d_x, d_y);
    check_kernel("scale_add best");
    check(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost), "download");
    check(cudaFree(d_x), "cudaFree x");
    check(cudaFree(d_y), "cudaFree y");
    bool ok = table.size() == 6;
    for (int i = 0; i < n; ++i) ok = ok && h_y[i] == 2.0f * h_x[i];
    std::printf("%d threads per block, y = 2 x checked on the CPU for %d elements\n", t, n);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
