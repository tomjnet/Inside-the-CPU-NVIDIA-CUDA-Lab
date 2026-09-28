// CUDA Shared Memory and Tiling in 9 Minutes - slide 6: tiled multiply, part one: the cooperative load
// Build: make 06_cooperative_load   (nvcc -std=c++20 -O2)
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
#include <cmath>
#include <vector>

#define TILE 16                           // tile width = block width

__global__ void matmul_tiled(const float* a, const float* b,
                             float* c, int n) {
    __shared__ float ta[TILE][TILE];      // tile of a, one per block
    __shared__ float tb[TILE][TILE];      // tile of b
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float sum = 0.0f;
    for (int t = 0; t < n / TILE; ++t) {  // walk the row and the column
        ta[ty][tx] = a[row * n + t * TILE + tx];   // 2 global reads
        tb[ty][tx] = b[(t * TILE + ty) * n + col]; // per thread, tile
        __syncthreads();                  // barrier 1: tiles loaded
        for (int k = 0; k < TILE; ++k)    // 32 reads, all shared memory
            sum += ta[ty][k] * tb[k][tx];
        __syncthreads();                  // barrier 2: all have read
    }                                     // before the next load
    c[row * n + col] = sum;               // 1 global write per thread
}

// Small exact values: every product and every sum is an integer below 2^24, so the GPU result must
// match the CPU result bit for bit, whatever the order of the additions.
static void fill(std::vector<float>& a, std::vector<float>& b, int n) {
    for (int i = 0; i < n * n; ++i) {
        a[static_cast<std::size_t>(i)] = static_cast<float>(i % 7);
        b[static_cast<std::size_t>(i)] = static_cast<float>(i % 5) - 2.0f;
    }
}

int main() {
    if (!require_device()) return 0;
    const int n = 128;                                            // small: the CPU checks every element
    const std::size_t count = static_cast<std::size_t>(n) * n, bytes = count * sizeof(float);
    std::vector<float> a(count), b(count), c(count);
    fill(a, b, n);

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    check(cudaMalloc(&da, bytes), "cudaMalloc a");
    check(cudaMalloc(&db, bytes), "cudaMalloc b");
    check(cudaMalloc(&dc, bytes), "cudaMalloc c");
    check(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice), "copy a");
    check(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice), "copy b");

    dim3 block(TILE, TILE), grid(n / TILE, n / TILE);            // one block per tile of c
    matmul_tiled<<<grid, block>>>(da, db, dc, n);
    check_kernel("matmul_tiled");
    check(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost), "copy c");

    // the cooperative load, counted: each thread loads one cell of each tile per step
    const int steps = n / TILE;
    std::printf("n = %d, TILE = %d: %d steps, %d global reads per thread (2 per step)\n",
                n, TILE, steps, 2 * steps);
    std::printf("per step a block loads %d + %d cells with %d threads: one cell each, then barrier 1\n",
                TILE * TILE, TILE * TILE, TILE * TILE);
    std::printf("shared memory per block: %zu bytes for the two tiles\n", 2 * sizeof(float) * TILE * TILE);

    // CPU check of every element (n^3 = 2 M multiply adds)
    int bad = 0;
    for (int r = 0; r < n; ++r)
        for (int q = 0; q < n; ++q) {
            double s = 0.0;
            for (int k = 0; k < n; ++k)
                s += static_cast<double>(a[static_cast<std::size_t>(r * n + k)]) * b[static_cast<std::size_t>(k * n + q)];
            if (std::fabs(s - c[static_cast<std::size_t>(r * n + q)]) > 1e-3) ++bad;
        }
    std::printf("CPU check: %zu elements compared, %d wrong, c[0] = %g\n", count, bad, c[0]);

    check(cudaFree(da)); check(cudaFree(db)); check(cudaFree(dc));
    std::printf(bad == 0 ? "PASS\n" : "FAIL\n");
    return bad == 0 ? 0 : 1;
}
