// CUDA Occupancy and Launch Configuration in 8 Minutes - slide 7: shared memory per block
// Build: make 07_shared_limit
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

__global__ void reverse_tile(const float* x, float* y) {
    extern __shared__ float tile[];         // 4096 floats: 16 KiB
    int base = blockIdx.x * 4096;           // one tile per block
    for (int k = threadIdx.x; k < 4096; k += blockDim.x)
        tile[k] = x[base + k];              // coalesced global reads
    __syncthreads();                        // the tile is complete
    for (int k = threadIdx.x; k < 4096; k += blockDim.x)
        y[base + k] = tile[4095 - k];       // another thread's load
}

int main() {
    if (!require_device()) return 0;
    cudaDeviceProp p;
    check(cudaGetDeviceProperties(&p, 0), "properties");
    const int tiles = 256;                  // 256 tiles of 4096 floats: 4 MiB
    const int n = tiles * 4096;
    static float h_x[n], h_y[n];
    for (int i = 0; i < n; ++i) h_x[i] = float(i);
    float *d_x = nullptr, *d_y = nullptr;
    check(cudaMalloc(&d_x, n * sizeof(float)), "cudaMalloc x");
    check(cudaMalloc(&d_y, n * sizeof(float)), "cudaMalloc y");
    check(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice), "upload");

    size_t smem = 4096 * sizeof(float);         // 3rd launch parameter
    reverse_tile<<<tiles, 128, smem>>>(d_x, d_y);
    check_kernel("reverse_tile");               // 64 / 16 KiB: 4 blocks

    int max_warps = p.maxThreadsPerMultiProcessor / p.warpSize;
    std::printf("shared memory: %zu KiB per SM, tile %zu KiB per block, default cap %zu KiB per block\n",
                p.sharedMemPerMultiprocessor / 1024, smem / 1024, p.sharedMemPerBlock / 1024);
    std::printf("threads  blocks/SM  warps  occupancy\n");
    for (int t = 64; t <= 1024; t *= 2) {
        int b = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, reverse_tile, t, smem), "occupancy");
        int w = b * t / p.warpSize;
        std::printf("%7d  %9d  %5d  %8.0f%%\n", t, b, w, 100.0 * w / max_warps);
    }
    std::printf("the tile does not shrink with the block: small blocks leave warp slots empty\n");

    check(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost), "download");
    check(cudaFree(d_x), "cudaFree x");
    check(cudaFree(d_y), "cudaFree y");
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        int base = i / 4096 * 4096;
        ok = ok && h_y[i] == h_x[base + 4095 - (i - base)];
    }
    std::printf("every tile reversed, checked on the CPU for %d elements\n", n);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
