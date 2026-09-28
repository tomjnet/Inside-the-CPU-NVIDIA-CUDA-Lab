// CUDA Streams and Asynchronous Copies in 10 Minutes - slide 7: events: markers in the queue
// Build: make 07_events
//
// Device time from events against end to end time from steady_clock, for one upload, kernel and download.
#include <chrono>
#include <cmath>
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

__global__ void work(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];
    float v = x;
    for (int k = 0; k < 32; ++k) v = 0.5f * (v + x / v);     // Newton steps toward sqrt(x)
    out[i] = v;
}

int main() {
    if (!require_device()) return 0;
    const int n = 1 << 21;                                   // 8 MiB each way
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    const int block = 256;
    const int grid = (n + block - 1) / block;

    cudaStream_t st;
    check(cudaStreamCreate(&st), "cudaStreamCreate");
    float* h_in = nullptr;
    float* h_out = nullptr;
    check(cudaMallocHost(&h_in, bytes), "cudaMallocHost in");
    check(cudaMallocHost(&h_out, bytes), "cudaMallocHost out");
    float* d_in = nullptr;
    float* d_out = nullptr;
    check(cudaMalloc(&d_in, bytes), "cudaMalloc in");
    check(cudaMalloc(&d_out, bytes), "cudaMalloc out");
    for (int i = 0; i < n; ++i) h_in[i] = 1.0f + static_cast<float>(i % 500);
    check(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, st), "warm up upload");
    work<<<grid, block, 0, st>>>(d_in, d_out, n);            // warm up
    check_kernel("warm up");

    cudaEvent_t up0, down1;                                  // two more markers, around the copies
    check(cudaEventCreate(&up0));
    check(cudaEventCreate(&down1));
    auto h0 = std::chrono::steady_clock::now();
    check(cudaEventRecord(up0, st));
    check(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, st), "upload");

    cudaEvent_t t0, t1;
    check(cudaEventCreate(&t0));
    check(cudaEventCreate(&t1));
    check(cudaEventRecord(t0, st));          // a marker in the queue of st
    work<<<grid, block, 0, st>>>(d_in, d_out, n);
    check(cudaGetLastError());
    check(cudaEventRecord(t1, st));          // stamped when st gets here
    check(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, st), "download");
    check(cudaEventRecord(down1, st));
    check(cudaEventSynchronize(t1));         // wait for this marker only
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, t0, t1)); // device time, not host
    std::printf("kernel: %.3f ms on the device clock\n", static_cast<double>(ms));

    // t1 is complete, the download behind it may still be running: an event waits for a point, not a stream
    std::printf("the marker after the download says right now: %s\n", cudaGetErrorName(cudaEventQuery(down1)));
    check(cudaStreamSynchronize(st), "cudaStreamSynchronize");
    double host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h0).count();
    float up_ms = 0.0f, down_ms = 0.0f, all_ms = 0.0f;
    check(cudaEventElapsedTime(&up_ms, up0, t0));
    check(cudaEventElapsedTime(&down_ms, t1, down1));
    check(cudaEventElapsedTime(&all_ms, up0, down1));
    std::printf("upload %.3f ms + kernel %.3f ms + download %.3f ms = %.3f ms on the device clock\n",
                static_cast<double>(up_ms), static_cast<double>(ms), static_cast<double>(down_ms),
                static_cast<double>(all_ms));
    std::printf("end to end %.3f ms with steady_clock: queueing and the final wait included (this machine)\n",
                host_ms);

    bool ok = true;
    for (int i = 0; i < n; ++i) ok = ok && std::fabs(h_out[i] - std::sqrt(h_in[i])) < 1e-3f * h_out[i];
    check(cudaEventDestroy(up0));
    check(cudaEventDestroy(t0));
    check(cudaEventDestroy(t1));
    check(cudaEventDestroy(down1));
    check(cudaFree(d_in));
    check(cudaFree(d_out));
    check(cudaFreeHost(h_in));
    check(cudaFreeHost(h_out));
    check(cudaStreamDestroy(st));
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
