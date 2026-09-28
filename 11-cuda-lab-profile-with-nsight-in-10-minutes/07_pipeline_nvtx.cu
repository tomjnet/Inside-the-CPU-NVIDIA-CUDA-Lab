// Profile CUDA with NVIDIA Nsight in 10 Minutes - slide 7: name the phases with NVTX
// Build: make 07_pipeline_nvtx
// Run:   ./07_pipeline_nvtx before | ./07_pipeline_nvtx after | ./07_pipeline_nvtx (both)
// Profile: 08_nsys_profile.sh records one Nsight Systems report per mode.
// The pipeline of 06_pipeline.cu with NVTX ranges: under nsys every range is a named bar on the CPU row.
// NVTX 3 is header only (CUDA 13): nothing extra to link, and a range costs close to nothing without a profiler.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
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

// x = a * x, the long way: 400 dependent steps per element so the kernel costs about as much as a copy
constexpr int kWork = 400;
__global__ void scale(float* x, int n, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i], t = 0.0f;
    for (int k = 0; k < kWork; ++k) t = t * 0.5f + v;   // converges to 2 * v
    x[i] = t * (a * 0.5f);
}

// the state the slide uses
constexpr int chunks = 16;
constexpr int n = 1 << 18;                           // 1 MiB of floats per chunk, 16 MiB in total
constexpr std::size_t bytes = n * sizeof(float);
constexpr int blocks = (n + 255) / 256;
static float* host = nullptr;                        // pinned: cudaMallocHost
static float* dev = nullptr;
static cudaStream_t streams[4];

// the three stages of one chunk on one stream: 2 copies, 1 launch
static void enqueue_chunk(int c, cudaStream_t s) {
    float* h = host + c * n;
    float* d = dev + c * n;
    check(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s), "upload");
    scale<<<blocks, 256, 0, s>>>(d, n, 2.0f);
    check(cudaGetLastError(), "scale");
    check(cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, s), "download");
}

static void pipeline(bool after) {
    // named ranges on the CPU row of the timeline, near zero cost
    nvtxRangePushA(after ? "after" : "before");
    for (int c = 0; c < chunks; ++c) {
        nvtxRangePushA("enqueue chunk");     // 2 copies + 1 launch
        enqueue_chunk(c, streams[after ? c % 4 : 0]);
        nvtxRangePop();
    }
    nvtxRangePushA("wait");                  // the CPU idles here
    check_kernel("pipeline");                // 1 sync point
    nvtxRangePop();
    nvtxRangePop();                          // closes before or after
}

static void fill(const std::vector<float>& orig) { std::memcpy(host, orig.data(), orig.size() * sizeof(float)); }

// 1 warm up run, then 5 timed runs: the median end to end time in ms (steady_clock)
static double run_mode(bool after, const std::vector<float>& orig) {
    fill(orig);
    nvtxRangePushA("warm up");                      // shows up on the timeline: skip it when reading
    pipeline(after);
    nvtxRangePop();
    std::vector<double> ms;
    for (int r = 0; r < 5; ++r) {
        fill(orig);                                  // not timed: fresh input for every run
        auto t0 = std::chrono::steady_clock::now();
        pipeline(after);
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    return ms[2];
}

static bool verify(const std::vector<float>& orig) {  // the CPU check: every element doubled
    for (std::size_t i = 0; i < orig.size(); ++i) {
        float want = 2.0f * orig[i];
        if (std::fabs(host[i] - want) > 1e-4f * std::fabs(want) + 1e-4f) {
            std::printf("mismatch at %zu: %f, expected %f\n", i, static_cast<double>(host[i]), static_cast<double>(want));
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (!require_device()) return 0;
    bool do_before = true, do_after = true;
    if (argc > 1) {
        do_before = std::strcmp(argv[1], "before") == 0;
        do_after = std::strcmp(argv[1], "after") == 0;
        if (!do_before && !do_after) { std::printf("usage: %s [before|after]\n", argv[0]); return 1; }
    }

    const std::size_t total = static_cast<std::size_t>(chunks) * n;
    std::vector<float> orig(total);
    for (std::size_t i = 0; i < total; ++i) orig[i] = static_cast<float>(i % 997) * 0.25f;
    check(cudaMallocHost(&host, total * sizeof(float)), "cudaMallocHost");
    check(cudaMalloc(&dev, total * sizeof(float)), "cudaMalloc");
    for (cudaStream_t& s : streams) check(cudaStreamCreate(&s), "cudaStreamCreate");

    std::printf("%d chunks of %zu KiB, pinned host memory, %d dependent steps per element\n",
                chunks, bytes / 1024, kWork);
    bool ok = true;
    double before_ms = 0.0, after_ms = 0.0;
    if (do_before) {
        before_ms = run_mode(false, orig);
        bool good = verify(orig);
        ok = ok && good;
        std::printf("before: 1 stream,  end to end %7.2f ms (median of 5, steady_clock) %s\n", before_ms, good ? "correct" : "WRONG");
    }
    if (do_after) {
        after_ms = run_mode(true, orig);
        bool good = verify(orig);
        ok = ok && good;
        std::printf("after:  4 streams, end to end %7.2f ms (median of 5, steady_clock) %s\n", after_ms, good ? "correct" : "WRONG");
    }
    if (do_before && do_after && after_ms > 0.0)
        std::printf("before / after = %.2fx on this machine: same work, overlapped stages\n", before_ms / after_ms);

    for (cudaStream_t s : streams) check(cudaStreamDestroy(s), "cudaStreamDestroy");
    check(cudaFree(dev), "cudaFree");
    check(cudaFreeHost(host), "cudaFreeHost");
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
