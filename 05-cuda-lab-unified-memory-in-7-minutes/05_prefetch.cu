// CUDA Unified Memory in 7 Minutes - slide 5: prefetch: move the pages ahead of time
// Build: make 05_prefetch   (nvcc -std=c++20 -O2)
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

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
    if (!require_device()) return 0;
    const int n = 1 << 22;                           // 4 M floats, 16 MiB per array, 48 MiB in total
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const size_t bytes = n * sizeof(float);
    float* a = nullptr; float* b = nullptr; float* c = nullptr;
    check(cudaMallocManaged(&a, bytes));
    check(cudaMallocManaged(&b, bytes));
    check(cudaMallocManaged(&c, bytes));
    for (int i = 0; i < n; ++i) { a[i] = 1.0f; b[i] = 2.0f; c[i] = 0.0f; }   // CPU touch: pages in host RAM

    // move the pages before the launch: one bulk transfer, no faults
    int dev = 0;
    check(cudaGetDevice(&dev));
    cudaMemLocation gpu{cudaMemLocationTypeDevice, dev};
    cudaMemLocation host{cudaMemLocationTypeHost, 0};

    cudaDeviceProp p;
    check(cudaGetDeviceProperties(&p, dev));
    auto t0 = std::chrono::steady_clock::now();
    if (p.concurrentManagedAccess) {                 // the driver accepts prefetch hints here
        check(cudaMemPrefetchAsync(a, bytes, gpu, 0, 0));   // queued, stream 0
        check(cudaMemPrefetchAsync(b, bytes, gpu, 0, 0));
        add<<<blocks, threads>>>(a, b, c, n);               // runs warm
        check(cudaMemPrefetchAsync(c, bytes, host, 0, 0));  // result goes home
        check_kernel("add");                         // wait for the prefetches and the kernel
        std::printf("prefetch, kernel, prefetch back: %.2f ms end to end (steady_clock, this machine)\n",
                    ms_since(t0));
    } else {                                         // Windows hosts and WSL: see 06_refused_hint
        std::printf("concurrentManagedAccess is 0: this driver refuses prefetch hints (typical on Windows, WSL)\n");
        std::printf("on native Linux the three cudaMemPrefetchAsync calls of the slide run here\n");
        std::printf("locations that would be used: type %d id %d (device), type %d (host)\n",
                    static_cast<int>(gpu.type), gpu.id, static_cast<int>(host.type));
        add<<<blocks, threads>>>(a, b, c, n);        // the pages migrate at the launch instead
        check_kernel("add");
        std::printf("kernel without prefetch: %.2f ms end to end (steady_clock, this machine)\n", ms_since(t0));
    }

    t0 = std::chrono::steady_clock::now();
    long wrong = count_wrong(c, n, 3.0f);
    std::printf("CPU check of %d elements: %.2f ms, %ld wrong\n", n, ms_since(t0), wrong);
    std::printf("compare with 04_first_vs_warm: a prefetched run sits close to the warmed run\n");

    check(cudaFree(a)); check(cudaFree(b)); check(cudaFree(c));
    std::printf("%s\n", wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? 0 : 1;
}
