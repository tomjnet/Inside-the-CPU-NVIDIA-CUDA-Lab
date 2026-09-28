// CUDA Occupancy and Launch Configuration in 8 Minutes - slide 5: the occupancy calculator
// Build: make 05_occupancy_api
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

static bool close_to(float got, float want) {        // the GPU may fuse multiply and add
    float d = got > want ? got - want : want - got;
    float m = want > 0 ? want : -want;
    return d <= 1e-3f * (m > 1.0f ? m : 1.0f);
}

int main() {
    if (!require_device()) return 0;
    cudaDeviceProp p;
    check(cudaGetDeviceProperties(&p, 0), "properties");
    cudaFuncAttributes a;
    check(cudaFuncGetAttributes(&a, heavy), "attributes");
    std::printf("heavy: %d registers per thread, %zu bytes local memory, blocks up to %d threads\n",
                a.numRegs, a.localSizeBytes, a.maxThreadsPerBlock);

    int per_sm = 0;                             // resident blocks per SM
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &per_sm, heavy, 256, 0), "occupancy");  // 0: dynamic smem
    int warps = per_sm * 256 / p.warpSize;      // resident warps
    int max_warps = p.maxThreadsPerMultiProcessor / p.warpSize;
    float occ = float(warps) / max_warps;       // occupancy, 0 to 1
    // heavy, 128 registers per thread: 2 blocks, 16 of 32 warps
    // no launch, no timing: computed from the kernel and the device

    std::printf("256 threads per block: %d blocks per SM, %d of %d warps, occupancy %.0f percent\n",
                per_sm, warps, max_warps, occ * 100.0f);
    std::printf("\nthreads  blocks/SM  warps  occupancy   ceilings: threads  blocks  registers\n");
    for (int t = 32; t <= a.maxThreadsPerBlock; t *= 2) {
        int b = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, heavy, t, 0), "occupancy");
        int w = b * t / p.warpSize;
        std::printf("%7d  %9d  %5d  %8.0f%%   %17d  %6d  %9d\n", t, b, w, 100.0 * w / max_warps,
                    p.maxThreadsPerMultiProcessor / t, p.maxBlocksPerMultiProcessor,
                    p.regsPerMultiprocessor / (a.numRegs * t));
    }
    std::printf("the smallest ceiling wins; the register column is before allocation granularity\n");

    const int n = 1 << 18;                      // 1 MiB per array
    static float h_x[n], h_y[n];
    for (int i = 0; i < n; ++i) h_x[i] = float(i % 13) / 13.0f;
    float *d_x = nullptr, *d_y = nullptr;
    check(cudaMalloc(&d_x, n * sizeof(float)), "cudaMalloc x");
    check(cudaMalloc(&d_y, n * sizeof(float)), "cudaMalloc y");
    check(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice), "upload");
    heavy<<<(n + 255) / 256, 256>>>(n, d_x, d_y);  // the configuration we just rated
    check_kernel("heavy");
    check(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost), "download");
    check(cudaFree(d_x), "cudaFree x");
    check(cudaFree(d_y), "cudaFree y");

    bool ok = per_sm > 0;
    for (int i = 0; i < n; ++i) ok = ok && close_to(h_y[i], heavy_body(i, n, h_x));
    std::printf("heavy checked on the CPU for %d elements\n", n);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
