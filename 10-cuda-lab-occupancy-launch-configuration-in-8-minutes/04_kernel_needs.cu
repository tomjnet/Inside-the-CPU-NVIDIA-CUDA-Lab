// CUDA Occupancy and Launch Configuration in 8 Minutes - slide 4: what a kernel asks for
// Build: make 04_kernel_needs
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

__global__ void scale_add(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global index
    if (i < n) y[i] = a * x[i] + y[i];               // 2 global reads, 1 write
}

int main() {
    if (!require_device()) return 0;

    cudaFuncAttributes a;                       // what one kernel asks for
    check(cudaFuncGetAttributes(&a, scale_add), "attributes");
    // a.numRegs             registers per thread
    // a.sharedSizeBytes     static shared memory per block
    // a.localSizeBytes      local memory per thread: stack and spills
    // a.maxThreadsPerBlock  largest block this kernel can launch with

    // the same numbers at compile time, from the assembler:
    //   nvcc -O2 -Xptxas -v 04_kernel_needs.cu
    //   typical: ptxas info: Used 10 registers, 0 bytes spill stores

    std::printf("scale_add: numRegs %d, sharedSizeBytes %zu, localSizeBytes %zu, maxThreadsPerBlock %d\n",
                a.numRegs, a.sharedSizeBytes, a.localSizeBytes, a.maxThreadsPerBlock);
    std::printf("a block of 256 threads asks for %d registers and %zu bytes of shared memory\n",
                a.numRegs * 256, a.sharedSizeBytes);

    const int n = 1 << 20;                      // 4 MiB per array
    static float h_x[n], h_y[n];
    for (int i = 0; i < n; ++i) { h_x[i] = float(i % 7); h_y[i] = 1.0f; }
    float *d_x = nullptr, *d_y = nullptr;
    check(cudaMalloc(&d_x, n * sizeof(float)), "cudaMalloc x");
    check(cudaMalloc(&d_y, n * sizeof(float)), "cudaMalloc y");
    check(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice), "upload x");
    check(cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice), "upload y");
    scale_add<<<(n + 255) / 256, 256>>>(n, 2.0f, d_x, d_y);  // 1 launch
    check_kernel("scale_add");
    check(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost), "download");
    check(cudaFree(d_x), "cudaFree x");
    check(cudaFree(d_y), "cudaFree y");

    bool ok = a.numRegs > 0;
    for (int i = 0; i < n; ++i) ok = ok && h_y[i] == 2.0f * float(i % 7) + 1.0f;
    std::printf("y = 2 x + y checked on the CPU for %d elements\n", n);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
