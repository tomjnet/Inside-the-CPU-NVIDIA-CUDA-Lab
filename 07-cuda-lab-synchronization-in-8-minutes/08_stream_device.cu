// CUDA Synchronization Without Deadlocks in 8 Minutes - slide 8: stream and device synchronization
// Build: make 08_stream_device
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
#include <vector>

__global__ void block_sum(const int* x, int* out, int n) {
    __shared__ int cell[256];              // 1 cell per thread
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    cell[t] = i < n ? x[i] : 0;            // no early return: stay in
    __syncthreads();                       // barrier 1: all cells ready
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (t < s) cell[t] += cell[t + s]; // work inside the branch
        __syncthreads();                   // barrier outside: s done
    }
    if (t == 0) out[blockIdx.x] = cell[0]; // 9 barriers, 1 global write
}

static long long sum_on_cpu(const int* x, int n) {   // the host works while the GPU runs
    long long s = 0;
    for (int i = 0; i < n; ++i) s += x[i];
    return s;
}

int main() {
    if (!require_device()) return 0;

    const int n = 1 << 20, blocks = (n + 255) / 256;
    const size_t bytes = blocks * sizeof(int);
    std::vector<int> xs(n), outs(blocks);
    for (int i = 0; i < n; ++i) xs[i] = i % 9 + 1;
    const int* x = xs.data();
    int* out = outs.data();
    int* dx = nullptr;
    int* dout = nullptr;
    check(cudaMalloc(&dx, n * sizeof(int)), "cudaMalloc x");
    check(cudaMalloc(&dout, bytes), "cudaMalloc out");
    check(cudaMemcpy(dx, x, n * sizeof(int), cudaMemcpyHostToDevice), "upload");
    block_sum<<<blocks, 256>>>(dx, dout, n);    // warm up on the default stream: first launch costs more
    check_kernel("warm up");

    // the slide, verbatim:
    cudaStream_t st = nullptr;
    check(cudaStreamCreate(&st), "create");      // an ordered queue
    block_sum<<<blocks, 256, 0, st>>>(dx, dout, n);  // 1 kernel launch
    check(cudaGetLastError(), "block_sum");      // launch accepted?
    long long cpu_sum = sum_on_cpu(x, n);        // the host did not wait
    check(cudaStreamSynchronize(st), "stream");  // waits for st only
    check(cudaMemcpy(out, dout, bytes, cudaMemcpyDeviceToHost), "copy");
    check(cudaDeviceSynchronize(), "device");    // waits for every stream
    check(cudaStreamDestroy(st), "destroy");     // 2 sync points, 1 copy

    check(cudaFree(dx), "cudaFree x");
    check(cudaFree(dout), "cudaFree out");
    long long gpu_sum = 0;
    for (int b = 0; b < blocks; ++b) gpu_sum += out[b];
    std::printf("CPU sum, computed while the kernel ran on the stream: %lld\n", cpu_sum);
    std::printf("GPU block sums after cudaStreamSynchronize and one copy: %lld\n", gpu_sum);
    bool ok = gpu_sum == cpu_sum;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
