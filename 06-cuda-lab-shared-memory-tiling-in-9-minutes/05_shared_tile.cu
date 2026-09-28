// CUDA Shared Memory and Tiling in 9 Minutes - slide 5: declaring shared memory
// Build: make 05_shared_tile   (nvcc -std=c++20 -O2)
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
#include <vector>

#define TILE 16                           // tile width = block width

__global__ void mirror_tiles(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE];    // one copy per block, on chip
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * TILE + tx;     // n is a multiple of TILE
    int row = blockIdx.y * TILE + ty;
    tile[ty][tx] = in[row * n + col];     // 1 global read per thread
    __syncthreads();                      // barrier: tile is complete
    out[row * n + col] = tile[ty][TILE - 1 - tx]; // a neighbour
}

int main() {
    if (!require_device()) return 0;
    const int n = 256;                                            // 16 by 16 tiles, 256 blocks
    const std::size_t count = static_cast<std::size_t>(n) * n, bytes = count * sizeof(float);
    std::vector<float> in(count), out(count);
    for (std::size_t i = 0; i < count; ++i) in[i] = static_cast<float>(i);

    float *din = nullptr, *dout = nullptr;
    check(cudaMalloc(&din, bytes), "cudaMalloc in");
    check(cudaMalloc(&dout, bytes), "cudaMalloc out");
    check(cudaMemcpy(din, in.data(), bytes, cudaMemcpyHostToDevice), "copy in");

    dim3 block(TILE, TILE), grid(n / TILE, n / TILE);            // one block per tile
    mirror_tiles<<<grid, block>>>(din, dout, n);
    check_kernel("mirror_tiles");
    check(cudaMemcpy(out.data(), dout, bytes, cudaMemcpyDeviceToHost), "copy out");

    // the first row of the first tile: the input and what the neighbours' slots gave back
    std::printf("in  row 0, tile 0:");
    for (int q = 0; q < TILE; ++q) std::printf(" %3.0f", in[static_cast<std::size_t>(q)]);
    std::printf("\nout row 0, tile 0:");
    for (int q = 0; q < TILE; ++q) std::printf(" %3.0f", out[static_cast<std::size_t>(q)]);
    std::printf("\n");

    // CPU check: every tile is mirrored left to right, nothing crosses a tile border
    int bad = 0;
    for (int r = 0; r < n; ++r)
        for (int q = 0; q < n; ++q) {
            int mirrored = (q / TILE) * TILE + (TILE - 1 - q % TILE);
            if (out[static_cast<std::size_t>(r * n + q)] != in[static_cast<std::size_t>(r * n + mirrored)]) ++bad;
        }
    std::printf("CPU check: %d of %zu elements wrong\n", bad, count);
    std::printf("shared memory per block: %zu bytes, %d threads loaded one element each\n",
                sizeof(float) * TILE * TILE, TILE * TILE);

    check(cudaFree(din)); check(cudaFree(dout));
    std::printf(bad == 0 ? "PASS\n" : "FAIL\n");
    return bad == 0 ? 0 : 1;
}
