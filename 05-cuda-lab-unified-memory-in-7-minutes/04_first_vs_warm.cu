// CUDA Unified Memory in 7 Minutes - slide 4: the first run pays the migration
// Build: make 04_first_vs_warm   (nvcc -std=c++20 -O2)
#include <chrono>
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

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];   // 2 global reads, 1 global write
}

static long count_wrong(const float* c, int n, float expected) {   // CPU check of every element
    long wrong = 0;
    for (int i = 0; i < n; ++i) {
        if (c[i] != expected) ++wrong;
    }
    return wrong;
}

static const int threads = 256;
static int blocks_for(int n) { return (n + threads - 1) / threads; }

// same kernel launch twice: only the first one pays the migration
double run_ms(const float* a, const float* b, float* c, int n) {
    auto t0 = std::chrono::steady_clock::now();      // end to end
    add<<<blocks_for(n), threads>>>(a, b, c, n);     // 1 kernel launch
    check_kernel("add");                             // 1 sync point
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    if (!require_device()) return 0;
    const int n = 1 << 22;                           // 4 M floats, 16 MiB per array, 48 MiB in total
    float* a = nullptr; float* b = nullptr; float* c = nullptr;
    check(cudaMallocManaged(&a, n * sizeof(float)));
    check(cudaMallocManaged(&b, n * sizeof(float)));
    check(cudaMallocManaged(&c, n * sizeof(float)));
    for (int i = 0; i < n; ++i) { a[i] = 1.0f; b[i] = 2.0f; c[i] = 0.0f; }   // CPU touch: pages in host RAM

    cudaDeviceProp p;
    check(cudaGetDeviceProperties(&p, 0));
    if (!p.concurrentManagedAccess) {
        std::printf("concurrentManagedAccess is 0 (Windows hosts, WSL): every launch moves the whole allocations,\n"
                    "so the warmed run pays too and the gap is small; on native Linux only the first run pays\n");
    }
    double first = run_ms(a, b, c, n);   // page faults + host to device
    double warm = run_ms(a, b, c, n);    // pages already in device memory
    std::printf("first %.2f ms, warm %.2f ms\n", first, warm);

    std::printf("end to end times from steady_clock on this machine, never asserted\n");
    if (warm > 0.0) {
        std::printf("first / warm = %.1fx: the difference is the migration, not the kernel\n", first / warm);
    }

    auto t0 = std::chrono::steady_clock::now();
    long wrong = count_wrong(c, n, 3.0f);            // the CPU touches c: the return trip
    double back = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("CPU check of %d elements: %.2f ms, it includes the migration of c back to the host\n", n, back);
    std::printf("%ld wrong\n", wrong);

    check(cudaFree(a)); check(cudaFree(b)); check(cudaFree(c));
    std::printf("%s\n", wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? 0 : 1;
}
