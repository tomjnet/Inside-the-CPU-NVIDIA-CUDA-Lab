// CUDA Streams and Asynchronous Copies in 10 Minutes - slide 6: asynchronous copies on a stream
// Build: make 06_async_copy
//
// The same three operations twice. From pinned memory the calls return at once and the host keeps working;
// from pageable memory the runtime has to stage the data first, so the calls themselves do the waiting.
#include <chrono>
#include <cmath>
#include <cstddef>
#include <vector>
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

__global__ void work(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 2.0f * in[i] + 1.0f;
}

static long g_prepared = 0;
static void prepare_next_chunk() { ++g_prepared; }           // stands for useful host work
static double use(const float* h_out) { return static_cast<double>(h_out[0]); }

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
    if (!require_device()) return 0;
    const int n = 1 << 22;                                   // 16 MiB each way
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    const int block = 256;
    const int grid = (n + block - 1) / block;

    cudaStream_t stream[1];
    check(cudaStreamCreate(&stream[0]), "cudaStreamCreate");
    float* h_in = nullptr;
    float* h_out = nullptr;
    check(cudaMallocHost(&h_in, bytes), "cudaMallocHost in");
    check(cudaMallocHost(&h_out, bytes), "cudaMallocHost out");
    float* d_in = nullptr;
    float* d_out = nullptr;
    check(cudaMalloc(&d_in, bytes), "cudaMalloc in");
    check(cudaMalloc(&d_out, bytes), "cudaMalloc out");
    for (int i = 0; i < n; ++i) h_in[i] = static_cast<float>(i % 100);
    work<<<grid, block>>>(d_in, d_out, n);                   // warm up: load the kernel once
    check_kernel("warm up");

    auto t0 = std::chrono::steady_clock::now();
    cudaStream_t st = stream[0];
    // 3 calls, 0 waits: each one only appends to the queue of st
    check(cudaMemcpyAsync(d_in, h_in, bytes,
                          cudaMemcpyHostToDevice, st));    // 1 upload
    work<<<grid, block, 0, st>>>(d_in, d_out, n);          // 1 launch
    check(cudaGetLastError());                             // launch ok?
    check(cudaMemcpyAsync(h_out, d_out, bytes,
                          cudaMemcpyDeviceToHost, st));    // 1 download
    double queued_ms = ms_since(t0);
    while (cudaStreamQuery(st) == cudaErrorNotReady)         // the slide calls it once; here it fills the wait
        prepare_next_chunk();               // the host is free meanwhile
    check(cudaStreamSynchronize(st));       // 1 synchronization point
    double done_ms = ms_since(t0);
    double first = use(h_out);              // only now is h_out valid

    bool ok = true;
    for (int i = 0; i < n; ++i) ok = ok && std::fabs(h_out[i] - (2.0f * h_in[i] + 1.0f)) < 1e-4f;
    std::printf("pinned  : 3 calls returned after %.3f ms, all done after %.3f ms (this machine)\n", queued_ms,
                done_ms);
    std::printf("          the host prepared %ld chunks while the device worked, h_out[0] = %.1f\n", g_prepared,
                first);

    // the same with pageable memory: legal, but the runtime stages the data through its own pinned buffer
    std::vector<float> p_in(h_in, h_in + n), p_out(static_cast<std::size_t>(n), 0.0f);
    t0 = std::chrono::steady_clock::now();
    check(cudaMemcpyAsync(d_in, p_in.data(), bytes, cudaMemcpyHostToDevice, st), "pageable upload");
    work<<<grid, block, 0, st>>>(d_in, d_out, n);
    check(cudaGetLastError(), "work launch");
    check(cudaMemcpyAsync(p_out.data(), d_out, bytes, cudaMemcpyDeviceToHost, st), "pageable download");
    double p_queued_ms = ms_since(t0);
    check(cudaStreamSynchronize(st), "cudaStreamSynchronize");
    double p_done_ms = ms_since(t0);
    for (int i = 0; i < n; ++i) ok = ok && p_out[static_cast<std::size_t>(i)] == h_out[i];
    std::printf("pageable: 3 calls returned after %.3f ms, all done after %.3f ms (this machine)\n", p_queued_ms,
                p_done_ms);
    std::printf("          when the calls return late, the copy was asynchronous in name only\n");

    check(cudaFree(d_in));
    check(cudaFree(d_out));
    check(cudaFreeHost(h_in));
    check(cudaFreeHost(h_out));
    check(cudaStreamDestroy(st));
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
