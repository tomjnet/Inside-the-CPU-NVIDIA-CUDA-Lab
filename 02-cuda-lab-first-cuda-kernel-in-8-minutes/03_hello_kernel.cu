// Your First CUDA Kernel in 8 Minutes - slide 3: hello gpu: a kernel and a launch
// Build: make 03_hello_kernel
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

// device code: the kernel, run once by every thread of the launch
__global__ void hello() {
    printf("Hello GPU from thread %d\n", threadIdx.x);
}

// the same launch shape, but every thread leaves a mark the CPU can verify
__global__ void sign_in(int* seen) {
    seen[threadIdx.x] = static_cast<int>(threadIdx.x) + 1;
}

int main() {
    if (!require_device()) return 0;

    std::printf("host: before the launch\n");
    // host code: the launch, <<<blocks in the grid, threads per block>>>
    hello<<<1, 4>>>();         // 1 kernel launch, 4 threads, async
    std::printf("host: the launch returned, the kernel output is still on the device\n");
    cudaDeviceSynchronize();   // 1 synchronization point, flushes printf
    std::printf("host: after the synchronization (comment it out and the four lines may never appear)\n");

    // CPU verification: did all four threads really run? Each one writes its own slot.
    int seen[4] = {0, 0, 0, 0};
    int* d_seen = nullptr;
    check(cudaMalloc(&d_seen, sizeof seen), "cudaMalloc");
    check(cudaMemcpy(d_seen, seen, sizeof seen, cudaMemcpyHostToDevice), "upload");
    sign_in<<<1, 4>>>(d_seen);
    check_kernel("sign_in");
    check(cudaMemcpy(seen, d_seen, sizeof seen, cudaMemcpyDeviceToHost), "download");
    check(cudaFree(d_seen), "cudaFree");

    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        std::printf("thread %d signed in with %d\n", i, seen[i]);
        if (seen[i] != i + 1) ok = false;
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
