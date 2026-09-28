// Memory Coalescing in CUDA in 8 Minutes - slide 9: two dimensions: row major or stride
// Build: make 09_row_major
#include <algorithm>
#include <cstdio>
#include <cstdlib>
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

__global__ void invert(const float* in, float* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // fast index
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int good = y * w + x;   // x neighbours are 1 apart: coalesced
    // int bad = x * w + y; // x neighbours are w apart: strided
    out[good] = 1.0f - in[good];
}

// The commented line of the slide as a second kernel: same arithmetic, the two indices swapped.
// On a square image it still visits every pixel exactly once, so the result is identical.
__global__ void invert_bad(const float* in, float* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int bad = x * w + y;    // x neighbours are w apart: strided
    out[bad] = 1.0f - in[bad];
}

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

    const int w = 2048, h = 2048;                            // square, 4 M pixels: 16 MiB per buffer
    const int n = w * h;
    const std::size_t bytes = std::size_t(n) * sizeof(float);
    std::vector<float> in(n), out_good(n), out_bad(n);
    for (int i = 0; i < n; ++i) in[i] = float(i % 256) / 256.0f;

    float* d_in = nullptr;
    float* d_out = nullptr;
    check(cudaMalloc(&d_in, bytes), "cudaMalloc in");
    check(cudaMalloc(&d_out, bytes), "cudaMalloc out");
    check(cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice), "upload");

    const dim3 threads(16, 16);                              // threadIdx.x is the index that changes fastest
    const dim3 blocks((w + 15) / 16, (h + 15) / 16);

    Stat g = measure([&] { invert<<<blocks, threads>>>(d_in, d_out, w, h); });
    check_kernel("invert");
    check(cudaMemcpy(out_good.data(), d_out, bytes, cudaMemcpyDeviceToHost), "download");

    check(cudaMemset(d_out, 0, bytes), "cudaMemset");
    Stat b = measure([&] { invert_bad<<<blocks, threads>>>(d_in, d_out, w, h); });
    check_kernel("invert_bad");
    check(cudaMemcpy(out_bad.data(), d_out, bytes, cudaMemcpyDeviceToHost), "download");

    check(cudaFree(d_in), "cudaFree");
    check(cudaFree(d_out), "cudaFree");

    long long wrong = 0;                                     // CPU check of both results
    for (int i = 0; i < n; ++i) wrong += out_good[i] != 1.0f - in[i] || out_bad[i] != 1.0f - in[i];

    std::printf("invert %d x %d, blocks of 16 x 16, device time from events\n", w, h);
    std::printf("%-22s %10s %12s\n", "index", "min ms", "median ms");
    std::printf("%-22s %10.3f %12.3f\n", "y * w + x (row major)", g.min_ms, g.median_ms);
    std::printf("%-22s %10.3f %12.3f   bad / good = %.2fx on this GPU\n", "x * w + y (strided)", b.min_ms,
                b.median_ms, b.median_ms / g.median_ms);
    std::printf("wrong pixels: %lld\n", wrong);
    bool ok = wrong == 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
