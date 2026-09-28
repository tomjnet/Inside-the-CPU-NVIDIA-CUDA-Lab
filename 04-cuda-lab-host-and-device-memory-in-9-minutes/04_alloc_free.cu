// CUDA Host and Device Memory in 9 Minutes - slide 4: cudamalloc and cudafree
// Build: make 04_alloc_free
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

#include <cstddef>

__global__ void fill(float* p, float v, std::size_t n) {   // a kernel may use the device address
    std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < n) p[i] = v;
}

int main() {
    if (!require_device()) return 0;
    const std::size_t n = std::size_t(1) << 20;
    const double mib = 1024.0 * 1024.0;

    std::size_t bytes = n * sizeof(float);    // 1M floats: 4 MiB of VRAM
    float* d_a = nullptr;                     // d_ prefix: device address,
    check(cudaMalloc(&d_a, bytes));           // never dereferenced on host

    std::size_t free_b = 0, total_b = 0;      // how much VRAM is left?
    check(cudaMemGetInfo(&free_b, &total_b));

    float* d_huge = nullptr;                  // a refused allocation is
    cudaError_t e = cudaMalloc(&d_huge, std::size_t(1) << 40);
    std::printf("1 TiB: %s\n", cudaGetErrorString(e));   // a status
    cudaGetLastError();                       // read once more to clear the recorded error

    std::printf("allocated %zu bytes (%.1f MiB) at device address %p\n", bytes, double(bytes) / mib,
                static_cast<void*>(d_a));
    std::printf("VRAM now: %.0f MiB free of %.0f MiB (other programs share this number)\n",
                double(free_b) / mib, double(total_b) / mib);
    // *d_a = 1.0f;   // invalid on the host: d_a is a device address, this line would crash
    fill<<<unsigned((n + 255) / 256), 256>>>(d_a, 1.0f, n);   // valid on the device: the same address
    check_kernel("fill");
    float first = 0.0f;                       // read it back the only legal way: a copy
    check(cudaMemcpy(&first, d_a, sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("a kernel wrote 1.0 through that address; copied back: %g\n", first);

    check(cudaFree(d_a));                     // one cudaFree per cudaMalloc
    check(cudaFree(nullptr));                 // freeing a null pointer is allowed and does nothing

    bool ok = e != cudaSuccess && d_huge == nullptr && total_b >= free_b && first == 1.0f;
    std::printf("refused allocation left the pointer null and the program alive: %s\n", ok ? "yes" : "no");
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
