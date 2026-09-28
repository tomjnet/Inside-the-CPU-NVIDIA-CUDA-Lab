// CUDA Host and Device Memory in 9 Minutes - slide 5: cudamemcpy in both directions
// Build: make 05_copy_both_ways
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
#include <vector>

__global__ void scale(float* a, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global index, as in episode 3
    if (i < n) a[i] = a[i] * factor;                 // guard the threads past the end
}

int main() {
    if (!require_device()) return 0;
    const int n = 1 << 20;
    const int threads = 256, blocks = (n + threads - 1) / threads;
    float* d_a = nullptr;
    check(cudaMalloc(&d_a, n * sizeof(float)), "cudaMalloc");

    std::vector<float> h_a(n, 1.5f), h_back(n);   // h_ prefix: host RAM
    std::size_t bytes = n * sizeof(float);

    // destination first, like memcpy; the last argument is the direction
    check(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));

    scale<<<blocks, threads>>>(d_a, 2.0f, n);     // 1 kernel launch
    check_kernel("scale");

    // 1 PCIe crossing each way; the call blocks until the bytes arrived
    check(cudaMemcpy(h_back.data(), d_a, bytes, cudaMemcpyDeviceToHost));
    bool ok = h_back[0] == 3.0f && h_back[n - 1] == 3.0f;   // CPU check

    std::size_t wrong = 0;                        // the full check: every element, on the CPU
    for (int i = 0; i < n; ++i) if (h_back[std::size_t(i)] != 3.0f) ++wrong;
    std::printf("uploaded %zu bytes, scaled by 2 on the device, downloaded %zu bytes\n", bytes, bytes);
    std::printf("h_a[0] = %g stayed on the host, h_back[0] = %g came from the device\n", h_a[0], h_back[0]);
    std::printf("elements that differ from 3: %zu of %d\n", wrong, n);

    // the wrong direction is an error status, not silent corruption
    cudaError_t e = cudaMemcpy(h_back.data(), h_a.data(), bytes, cudaMemcpyDeviceToDevice);
    std::printf("host pointers with cudaMemcpyDeviceToDevice: %s\n", cudaGetErrorString(e));
    cudaGetLastError();                           // clear the recorded error

    check(cudaFree(d_a));
    ok = ok && wrong == 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
