// CUDA Host and Device Memory in 9 Minutes - slide 9: timing each stage separately
// Build: make 09_time_stages
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

#include <chrono>
#include <cstddef>
#include <vector>

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global index, as in episode 3
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    if (!require_device()) return 0;
    const int n = 1 << 22;                            // 4M floats: 16 MiB per vector
    const std::size_t bytes = std::size_t(n) * sizeof(float);
    const int threads = 256, blocks = (n + threads - 1) / threads;
    std::vector<float> h_a(std::size_t(n), 1.0f), h_b(std::size_t(n), 2.0f), h_c(std::size_t(n), 0.0f);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    check(cudaMalloc(&d_a, bytes)); check(cudaMalloc(&d_b, bytes)); check(cudaMalloc(&d_c, bytes));
    check(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    add<<<blocks, threads>>>(d_a, d_b, d_c, n);       // warm up: the first launch loads the kernel
    check_kernel("add warm up");

    // the wrong way: a host clock around the launch measures the launch, not the kernel
    auto h0 = std::chrono::steady_clock::now();
    add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    auto h1 = std::chrono::steady_clock::now();
    check_kernel("add, host clock");
    double launch_ms = std::chrono::duration<double, std::milli>(h1 - h0).count();

    cudaEvent_t start, stop;                  // markers in the GPU queue
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    check(cudaEventRecord(start));            // device time, not host time
    add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));        // 1 synchronization point

    float kernel_ms = 0.0f;                   // resolution about 0.5 us
    check(cudaEventElapsedTime(&kernel_ms, start, stop));
    check(cudaGetLastError(), "add");         // was the launch accepted?

    check(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    std::size_t wrong = 0;
    for (std::size_t i = 0; i < h_c.size(); ++i) if (h_c[i] != 3.0f) ++wrong;

    std::printf("add over %d elements, this machine:\n", n);
    std::printf("  steady_clock around the launch only: %8.4f ms  (host time: the call returned, the GPU had not finished)\n", launch_ms);
    std::printf("  CUDA events around the kernel:       %8.4f ms  (device time: what the kernel really took)\n", double(kernel_ms));
    std::printf("elements that differ from 3: %zu\n", wrong);

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaFree(d_a)); check(cudaFree(d_b)); check(cudaFree(d_c));
    bool ok = wrong == 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
