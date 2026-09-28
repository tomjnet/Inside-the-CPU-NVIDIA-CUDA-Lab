// CUDA Streams and Asynchronous Copies in 10 Minutes - slide 5: create streams and pinned buffers
// Build: make 05_create_streams
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

__global__ void fill(float* out, int n, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = value;
}

int main() {
    if (!require_device()) return 0;
    const int n = 1 << 20;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

    const int S = 4;                          // 4 ordered queues
    cudaStream_t stream[S];
    for (int s = 0; s < S; ++s)
        check(cudaStreamCreate(&stream[s]));  // cheap: create once, reuse

    float* h_in = nullptr;                    // pinned host memory:
    check(cudaMallocHost(&h_in, bytes));      // page locked, DMA ready

    // four fills of the same buffer, all on stream 0: one queue, so they run in the order they were queued
    float* d_buf = nullptr;
    check(cudaMalloc(&d_buf, bytes), "cudaMalloc");
    const int block = 256;
    const int grid = (n + block - 1) / block;
    for (int k = 1; k <= 4; ++k) {
        fill<<<grid, block, 0, stream[0]>>>(d_buf, n, static_cast<float>(k));
        check(cudaGetLastError(), "fill launch");
    }
    check(cudaMemcpyAsync(h_in, d_buf, bytes, cudaMemcpyDeviceToHost, stream[0]), "download");

    // a stream is a queue you can ask without waiting: cudaStreamQuery
    cudaError_t busy = cudaStreamQuery(stream[0]);
    cudaError_t idle = cudaStreamQuery(stream[1]);
    std::printf("stream 0 right after queueing 5 operations: %s\n", cudaGetErrorName(busy));
    std::printf("stream 1, which got no work:                %s\n", cudaGetErrorName(idle));
    check(cudaStreamSynchronize(stream[0]), "cudaStreamSynchronize");
    std::printf("stream 0 after its synchronization point:   %s\n", cudaGetErrorName(cudaStreamQuery(stream[0])));

    bool ok = (busy == cudaSuccess || busy == cudaErrorNotReady) && idle == cudaSuccess;
    for (int i = 0; i < n; ++i) ok = ok && h_in[i] == 4.0f;  // queue order: the last fill wins
    std::printf("4 fills on one stream ran in order: every element is %.0f\n", static_cast<double>(h_in[0]));

    check(cudaFree(d_buf));
    for (int s = 0; s < S; ++s)
        check(cudaStreamDestroy(stream[s])); // queued work still finishes
    check(cudaFreeHost(h_in));
    check_kernel("final check");
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
