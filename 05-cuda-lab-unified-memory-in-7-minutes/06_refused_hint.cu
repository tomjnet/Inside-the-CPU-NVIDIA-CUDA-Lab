// CUDA Unified Memory in 7 Minutes - slide 6: when the driver refuses the hint
// Build: make 06_refused_hint   (nvcc -std=c++20 -O2)
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
    int dev = 0;
    check(cudaGetDevice(&dev));
    cudaMemLocation gpu{cudaMemLocationTypeDevice, dev};

    // a hint may be refused: report it, fall back, never abort
    cudaDeviceProp p;
    check(cudaGetDeviceProperties(&p, dev));
    std::printf("managed %d, concurrent access %d\n",
                p.managedMemory, p.concurrentManagedAccess);
    cudaError_t e = cudaMemPrefetchAsync(a, bytes, gpu, 0, 0);
    if (e != cudaSuccess) {                    // typical on Windows and WSL
        std::printf("prefetch refused: %s\n", cudaGetErrorString(e));
        (void)cudaGetLastError();              // clear it, keep going
        add<<<blocks, threads>>>(a, b, c, n);  // fallback: a warm up launch
        check_kernel("warm up");
    } else {                                   // typical on native Linux
        std::printf("prefetch accepted: a moves now, b and c follow\n");
        check(cudaMemPrefetchAsync(b, bytes, gpu, 0, 0));
        check(cudaMemPrefetchAsync(c, bytes, gpu, 0, 0));
        check(cudaDeviceSynchronize());
    }

    auto t0 = std::chrono::steady_clock::now();      // the timed region starts warm on both paths
    add<<<blocks, threads>>>(a, b, c, n);
    check_kernel("add");
    std::printf("timed launch after the hint or the fallback: %.2f ms end to end (steady_clock, this machine)\n",
                ms_since(t0));

    long wrong = count_wrong(c, n, 3.0f);
    std::printf("checked %d elements on the CPU, %ld wrong: correct on both paths, only the speed differs\n",
                n, wrong);

    check(cudaFree(a)); check(cudaFree(b)); check(cudaFree(c));
    std::printf("%s\n", wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? 0 : 1;
}
