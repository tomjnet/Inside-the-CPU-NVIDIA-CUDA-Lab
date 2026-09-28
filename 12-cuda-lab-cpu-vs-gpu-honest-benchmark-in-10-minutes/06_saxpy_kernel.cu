// CPU vs GPU: An Honest Benchmark in 10 Minutes - slide 6: the gpu kernel
// Build: make 06_saxpy_kernel
#include <cmath>
#include <random>
#include <vector>
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

// one thread per element: 2 flops, 12 bytes of global memory
__global__ void saxpy(float a, const float* x, float* y,
                      int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];   // same math as the CPU
}

int main() {
    if (!require_device()) return 0;

    const int n = 1 << 20;                                   // 1 M floats: 4 MiB per buffer
    const std::size_t bytes = std::size_t(n) * sizeof(float);
    std::vector<float> x(n), y0(n), y(n);
    std::mt19937 rng(42);                                    // fixed seed: the same inputs as the CPU samples
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i) { x[i] = dist(rng); y0[i] = dist(rng); }

    float* d_x = nullptr;
    float* d_y = nullptr;
    check(cudaMalloc(&d_x, bytes), "cudaMalloc x");
    check(cudaMalloc(&d_y, bytes), "cudaMalloc y");
    check(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice), "upload x");
    check(cudaMemcpy(d_y, y0.data(), bytes, cudaMemcpyHostToDevice), "upload y");

    // 256 threads per block, grid rounded up: 1 kernel launch
    int threads = 256, blocks = (n + threads - 1) / threads;
    saxpy<<<blocks, threads>>>(2.0f, d_x, d_y, n);
    check_kernel("saxpy");                   // launch error, then sync

    check(cudaMemcpy(y.data(), d_y, bytes, cudaMemcpyDeviceToHost), "download y");
    check(cudaFree(d_x), "cudaFree x");
    check(cudaFree(d_y), "cudaFree y");

    // the CPU check: same float expression, then a tolerance instead of equality
    long long identical = 0, last_bits = 0, wrong = 0;
    double worst = 0.0;
    for (int i = 0; i < n; ++i) {
        const float cpu = 2.0f * x[i] + y0[i];               // the CPU rounds twice: after * and after +
        const double diff = std::fabs(double(y[i]) - double(cpu));
        const double tol = 1e-6 * std::fabs(double(cpu)) + 1e-6;
        if (y[i] == cpu) ++identical;
        else if (diff <= tol) ++last_bits;                   // the GPU may fuse * and + and round once
        else ++wrong;
        if (diff > worst) worst = diff;
    }
    std::printf("saxpy, %d floats, %d blocks of %d threads, 1 launch\n", n, blocks, threads);
    std::printf("bit identical to the CPU:        %lld\n", identical);
    std::printf("different only in the last bits: %lld (fused multiply add rounds once)\n", last_bits);
    std::printf("with a = 2 the product is exact, so fused and separate usually agree; try a = 0.7f\n");
    std::printf("outside the tolerance:           %lld (largest difference %.3g)\n", wrong, worst);
    std::printf("%s\n", wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? 0 : 1;
}
