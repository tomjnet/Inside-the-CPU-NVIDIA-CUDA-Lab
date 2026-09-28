// CUDA Shared Memory and Tiling in 9 Minutes - slide 8: time both kernels with events
// Build: make 08_time_both   (nvcc -std=c++20 -O2)
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

// naive: every thread reads 2 * n values from global memory
__global__ void matmul_naive(const float* a, const float* b,
                             float* c, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;     // guard the extra threads
    float sum = 0.0f;                     // lives in a register
    for (int k = 0; k < n; ++k)           // n = 512: 1024 global reads
        sum += a[row * n + k] * b[k * n + col];
    c[row * n + col] = sum;               // 1 global write
}

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
    const int n = 512;                                            // 4 matrices of 1 MiB each
    const std::size_t count = static_cast<std::size_t>(n) * n, bytes = count * sizeof(float);
    std::vector<float> a(count), b(count), c_naive(count), c_tiled(count);
    fill(a, b, n);

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    check(cudaMalloc(&da, bytes), "cudaMalloc a");
    check(cudaMalloc(&db, bytes), "cudaMalloc b");
    check(cudaMalloc(&dc, bytes), "cudaMalloc c");
    check(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice), "copy a");
    check(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice), "copy b");
    dim3 block(TILE, TILE), grid(n / TILE, n / TILE);

    // warm up: the first launch of each kernel pays one time setup costs
    matmul_naive<<<grid, block>>>(da, db, dc, n);
    check_kernel("matmul_naive warm up");
    matmul_tiled<<<grid, block>>>(da, db, dc, n);
    check_kernel("matmul_tiled warm up");

    cudaEvent_t t0, t1;                       // device time, not host time
    check(cudaEventCreate(&t0));
    check(cudaEventCreate(&t1));
    check(cudaEventRecord(t0));               // queued before the kernel
    matmul_naive<<<grid, block>>>(da, db, dc, n);
    check(cudaEventRecord(t1));               // queued after the kernel
    check(cudaEventSynchronize(t1));          // 1 synchronization point
    float naive_ms = 0.0f;
    check(cudaEventElapsedTime(&naive_ms, t0, t1));
    check_kernel("matmul_naive");             // then the same for the tiled
    check(cudaMemcpy(c_naive.data(), dc, bytes, cudaMemcpyDeviceToHost), "copy c naive");

    check(cudaEventRecord(t0));
    matmul_tiled<<<grid, block>>>(da, db, dc, n);
    check(cudaEventRecord(t1));
    check(cudaEventSynchronize(t1));
    float tiled_ms = 0.0f;
    check(cudaEventElapsedTime(&tiled_ms, t0, t1));
    check_kernel("matmul_tiled");
    check(cudaMemcpy(c_tiled.data(), dc, bytes, cudaMemcpyDeviceToHost), "copy c tiled");

    // same c on the CPU for both kernels, or the timing means nothing
    int bad = 0, checked = 0;
    for (int r = 0; r < n; r += 37)
        for (int q = 0; q < n; q += 41) {
            double s = 0.0;
            for (int k = 0; k < n; ++k)
                s += static_cast<double>(a[static_cast<std::size_t>(r * n + k)]) * b[static_cast<std::size_t>(k * n + q)];
            ++checked;
            const std::size_t i = static_cast<std::size_t>(r * n + q);
            if (std::fabs(s - c_naive[i]) > 1e-3 || std::fabs(s - c_tiled[i]) > 1e-3) ++bad;
        }
    std::size_t differ = 0;
    for (std::size_t i = 0; i < count; ++i)
        if (c_naive[i] != c_tiled[i]) ++differ;

    std::printf("n = %d, TILE = %d, kernel time by CUDA events (this machine, not a claim):\n", n, TILE);
    std::printf("  naive %.3f ms   tiled %.3f ms   ratio %.2fx\n", naive_ms, tiled_ms,
                tiled_ms > 0.0f ? naive_ms / tiled_ms : 0.0f);
    std::printf("  global reads per thread: naive %d, tiled %d (%d times fewer)\n", 2 * n, 2 * n / TILE, TILE);
    std::printf("CPU check: %d elements compared against both kernels, %d wrong; naive and tiled differ in %zu of %zu\n",
                checked, bad, differ, count);

    check(cudaEventDestroy(t0)); check(cudaEventDestroy(t1));
    check(cudaFree(da)); check(cudaFree(db)); check(cudaFree(dc));
    const bool ok = bad == 0 && differ == 0;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
