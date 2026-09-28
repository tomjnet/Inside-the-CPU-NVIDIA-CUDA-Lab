// CUDA Occupancy and Launch Configuration in 8 Minutes - slide 9: the toolkit's suggestion
// Build: make 09_suggest
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

int main() {
    if (!require_device()) return 0;
    const int n = 1 << 22;                      // 4 M floats, 16 MiB per array
    static float h_x[n], h_y[n];
    for (int i = 0; i < n; ++i) h_x[i] = float(i % 7);
    float *d_x = nullptr, *d_y = nullptr;
    check(cudaMalloc(&d_x, n * sizeof(float)), "cudaMalloc x");
    check(cudaMalloc(&d_y, n * sizeof(float)), "cudaMalloc y");
    check(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice), "upload");
    check(cudaMemset(d_y, 0, n * sizeof(float)), "memset");

    int min_grid = 0, suggested = 0;            // the toolkit's advice
    check(cudaOccupancyMaxPotentialBlockSize(
              &min_grid, &suggested, scale_add), "suggest");
    // suggested: highest occupancy, then the fewest blocks
    // min_grid: blocks that fill every SM once
    // typical GTX 1650: suggested 1024, min_grid 16
    int blocks = (n + suggested - 1) / suggested;   // still cover n
    scale_add<<<blocks, suggested>>>(n, 2.0f, d_x, d_y);
    check_kernel("scale_add");                  // then time it: sweep

    check(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost), "download");
    bool ok = suggested > 0 && min_grid > 0;
    for (int i = 0; i < n; ++i) ok = ok && h_y[i] == 2.0f * h_x[i];

    std::printf("suggested block size %d, minimum grid %d blocks, our grid %d blocks\n", suggested,
                min_grid, blocks);
    float ts = time_ms([&] { scale_add<<<blocks, suggested>>>(n, 2.0f, d_x, d_y); });
    check_kernel("scale_add suggested");
    float t256 = time_ms([&] { scale_add<<<(n + 255) / 256, 256>>>(n, 2.0f, d_x, d_y); });
    check_kernel("scale_add 256");
    std::printf("median device time: suggested %.3f ms, 256 threads %.3f ms (this machine)\n", ts, t256);
    std::printf("the suggestion knows the resource tables, not the memory traffic: measure\n");

    check(cudaFree(d_x), "cudaFree x");
    check(cudaFree(d_y), "cudaFree y");
    std::printf("y = 2 x checked on the CPU for %d elements\n", n);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
