// CUDA Occupancy and Launch Configuration in 8 Minutes - slide 6: register pressure and launch bounds
// Build: make 06_launch_bounds
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

// 64 live values per thread: the compiler keeps them in registers if it may
__host__ __device__ inline float heavy_body(int i, int n, const float* x) {
    float v[64];
    for (int k = 0; k < 64; ++k) v[k] = x[(i + 97 * k) % n];
    float s = 0.0f;
    for (int r = 0; r < 8; ++r)
        for (int k = 0; k < 64; ++k) s = s * 0.999f + v[k] * v[(k + r) % 64];
    return s;
}
__global__ void heavy(int n, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = heavy_body(i, n, x);
}

__global__ void __launch_bounds__(256, 4)   // 256 threads, 4 blocks
heavy_lb(int n, const float* x, float* y);  // same body as heavy

__global__ void __launch_bounds__(256, 4) heavy_lb(int n, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = heavy_body(i, n, x);
}

static bool close_to(float got, float want) {        // the GPU may fuse multiply and add
    float d = got > want ? got - want : want - got;
    float m = want > 0 ? want : -want;
    return d <= 1e-3f * (m > 1.0f ? m : 1.0f);
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
    cudaDeviceProp p;
    check(cudaGetDeviceProperties(&p, 0), "properties");
    const int n = 1 << 18;                      // 1 MiB per array
    static float h_x[n], h_y[n], h_z[n];
    for (int i = 0; i < n; ++i) h_x[i] = float(i % 13) / 13.0f;
    float *d_x = nullptr, *d_y = nullptr;
    check(cudaMalloc(&d_x, n * sizeof(float)), "cudaMalloc x");
    check(cudaMalloc(&d_y, n * sizeof(float)), "cudaMalloc y");
    check(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice), "upload");

    cudaFuncAttributes a, b;
    check(cudaFuncGetAttributes(&a, heavy), "heavy");
    // typical: numRegs 128, 2 blocks of 256 per SM: 50 percent
    check(cudaFuncGetAttributes(&b, heavy_lb), "heavy_lb");
    // typical: numRegs 64, localSizeBytes grows: values spill
    // 4 blocks per SM: 100 percent, and often several times slower

    int max_warps = p.maxThreadsPerMultiProcessor / p.warpSize;
    int ba = 0, bb = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ba, heavy, 256, 0), "occupancy");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bb, heavy_lb, 256, 0), "occupancy");
    const int blocks = (n + 255) / 256;
    float ta = time_ms([&] { heavy<<<blocks, 256>>>(n, d_x, d_y); });
    check_kernel("heavy");
    check(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost), "download heavy");
    float tb = time_ms([&] { heavy_lb<<<blocks, 256>>>(n, d_x, d_y); });
    check_kernel("heavy_lb");
    check(cudaMemcpy(h_z, d_y, n * sizeof(float), cudaMemcpyDeviceToHost), "download heavy_lb");

    std::printf("kernel     registers  local bytes  blocks/SM  occupancy  median ms (device, this machine)\n");
    std::printf("heavy      %9d  %11zu  %9d  %8.0f%%  %9.3f\n", a.numRegs, a.localSizeBytes, ba,
                100.0 * ba * 256 / p.warpSize / max_warps, ta);
    std::printf("heavy_lb   %9d  %11zu  %9d  %8.0f%%  %9.3f\n", b.numRegs, b.localSizeBytes, bb,
                100.0 * bb * 256 / p.warpSize / max_warps, tb);
    std::printf("heavy_lb / heavy time: %.2f (above 1: more occupancy, slower kernel)\n", tb / ta);

    heavy<<<1, 1024>>>(n, d_x, d_y);            // 1024 x 128 registers
    // cudaGetLastError: too many resources requested for launch
    cudaError_t e = cudaGetLastError();         // a launch error is not sticky
    std::printf("heavy with 1024 threads per block: %s\n", cudaGetErrorString(e));
    check(cudaDeviceSynchronize(), "after the refused launch");

    check(cudaFree(d_x), "cudaFree x");
    check(cudaFree(d_y), "cudaFree y");
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        float want = heavy_body(i, n, h_x);
        ok = ok && close_to(h_y[i], want) && close_to(h_z[i], want);
    }
    std::printf("both kernels checked on the CPU for %d elements\n", n);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
