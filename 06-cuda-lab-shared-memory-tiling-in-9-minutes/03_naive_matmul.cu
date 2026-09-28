// CUDA Shared Memory and Tiling in 9 Minutes - slide 3: the naive matrix multiply
// Build: make 03_naive_matmul   (nvcc -std=c++20 -O2)
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
    const int n = 512;                                            // 3 matrices of 1 MiB each
    const std::size_t count = static_cast<std::size_t>(n) * n, bytes = count * sizeof(float);
    std::vector<float> a(count), b(count), c(count);
    fill(a, b, n);

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    check(cudaMalloc(&da, bytes), "cudaMalloc a");
    check(cudaMalloc(&db, bytes), "cudaMalloc b");
    check(cudaMalloc(&dc, bytes), "cudaMalloc c");
    check(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice), "copy a");
    check(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice), "copy b");

    dim3 block(16, 16), grid((n + 15) / 16, (n + 15) / 16);     // one thread per element of c
    matmul_naive<<<grid, block>>>(da, db, dc, n);
    check_kernel("matmul_naive");
    check(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost), "copy c");

    // the traffic of the slide, counted rather than guessed
    const long long threads = static_cast<long long>(n) * n;
    std::printf("n = %d: %lld threads, %d global reads per thread, %lld global reads in total\n",
                n, threads, 2 * n, threads * 2 * n);
    std::printf("every element of a is fetched by %d different threads\n", n);

    // CPU check of a sample of the result (the full product costs n^3 = 134 M multiply adds)
    int bad = 0, checked = 0;
    for (int r = 0; r < n; r += 37)
        for (int q = 0; q < n; q += 41) {
            double s = 0.0;
            for (int k = 0; k < n; ++k)
                s += static_cast<double>(a[static_cast<std::size_t>(r * n + k)]) * b[static_cast<std::size_t>(k * n + q)];
            ++checked;
            if (std::fabs(s - c[static_cast<std::size_t>(r * n + q)]) > 1e-3) ++bad;
        }
    std::printf("CPU check: %d elements compared, %d wrong, c[0] = %g\n", checked, bad, c[0]);

    check(cudaFree(da)); check(cudaFree(db)); check(cudaFree(dc));
    std::printf(bad == 0 ? "PASS\n" : "FAIL\n");
    return bad == 0 ? 0 : 1;
}
