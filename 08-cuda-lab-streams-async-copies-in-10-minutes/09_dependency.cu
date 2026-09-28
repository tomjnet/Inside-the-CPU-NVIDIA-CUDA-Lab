// CUDA Streams and Asynchronous Copies in 10 Minutes - slide 9: dependencies between streams
// Build: make 09_dependency
//
// Stream a uploads a lookup table, stream b runs the kernel that reads it. Two streams promise no order
// between them; cudaStreamWaitEvent makes b wait for the marker on a, on the device, without a host wait.
#include <cstddef>
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

__global__ void lookup(const int* table, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = table[(n - 1) - i] + 1;              // reads the far end of the table first
}

int main() {
    if (!require_device()) return 0;
    const int n = 1 << 22;                                   // 16 MiB table
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(int);
    const int block = 256;
    const int grid = (n + block - 1) / block;

    cudaStream_t a, b;
    check(cudaStreamCreate(&a), "cudaStreamCreate a");
    check(cudaStreamCreate(&b), "cudaStreamCreate b");
    cudaEvent_t ready;
    check(cudaEventCreate(&ready), "cudaEventCreate");
    int* h_table = nullptr;
    int* h_out = nullptr;
    check(cudaMallocHost(&h_table, bytes), "cudaMallocHost table");
    check(cudaMallocHost(&h_out, bytes), "cudaMallocHost out");
    int* d_table = nullptr;
    int* d_out = nullptr;
    check(cudaMalloc(&d_table, bytes), "cudaMalloc table");
    check(cudaMalloc(&d_out, bytes), "cudaMalloc out");
    check(cudaMemset(d_table, 0, bytes), "cudaMemset");      // a stale table: all zeros
    for (int i = 0; i < n; ++i) h_table[i] = i;
    check(cudaDeviceSynchronize(), "idle");

    // stream a uploads the table, stream b must not start before it
    check(cudaMemcpyAsync(d_table, h_table, bytes,
                          cudaMemcpyHostToDevice, a));
    check(cudaEventRecord(ready, a));        // marker after the upload
    check(cudaStreamWaitEvent(b, ready, 0)); // b waits on the device,
    lookup<<<grid, block, 0, b>>>(d_table, d_out, n);  // the host does not
    check(cudaGetLastError());
    check(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, b), "download");
    check(cudaStreamSynchronize(b));         // 1 synchronization point

    long stale = 0;
    for (int i = 0; i < n; ++i) stale += h_out[i] != (n - 1 - i) + 1;
    std::printf("with cudaStreamWaitEvent: %ld of %d results read a stale table\n", stale, n);
    std::printf("the dependency cost 1 event and 0 host waits: stream b waited on the device\n");
    std::printf("without the wait, the kernel on b may race the upload on a: two streams promise no order\n");
    bool ok = stale == 0;

    check(cudaStreamSynchronize(a), "cudaStreamSynchronize a");
    check(cudaEventDestroy(ready));
    check(cudaFree(d_table));
    check(cudaFree(d_out));
    check(cudaFreeHost(h_table));
    check(cudaFreeHost(h_out));
    check(cudaStreamDestroy(a));
    check(cudaStreamDestroy(b));
    check_kernel("final check");
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
