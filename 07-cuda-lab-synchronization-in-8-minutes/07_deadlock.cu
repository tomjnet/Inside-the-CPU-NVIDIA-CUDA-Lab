// CUDA Synchronization Without Deadlocks in 8 Minutes - slide 7: the deadlock: a barrier inside a branch
// Build: make 07_deadlock
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

// The slide's loop inside a whole kernel. It compiles without a warning and would hang, so it is
// never launched: the next synchronization would block until a watchdog, if there is one, kills us.
__global__ void block_sum_deadlock(const int* x, int* out, int n) {
    __shared__ int cell[256];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    cell[t] = i < n ? x[i] : 0;
    __syncthreads();
    // the slide, verbatim:
for (int s = blockDim.x / 2; s > 0; s /= 2) {
    if (t < s) {                           // divergent: s threads enter
        cell[t] += cell[t + s];
        __syncthreads();                   // DEADLOCK: the others
    }                                      // never reach this barrier
}
// the same bug with an early return above a barrier:
//     if (i >= n) return;                 // last block never leaves
// nvcc compiles both without a warning
    if (t == 0) out[blockIdx.x] = cell[0];
}

// The fix of slide 6: the barrier outside the branch, no early return
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

int main() {
    if (!require_device()) return 0;

    // On the CPU: how many threads of a block of 256 reach the barrier inside if (t < s)?
    std::printf("barrier inside if (t < s), block of 256 threads:\n");
    for (int s = 128; s > 0; s /= 2)
        std::printf("  s = %3d: %3d arrive, %3d never do -> the block waits forever\n", s, s, 256 - s);
    std::printf("block_sum_deadlock is compiled but never launched (host stub at %p)\n",
                reinterpret_cast<void*>(&block_sum_deadlock));

    // The fixed kernel on a size that is not a multiple of 256, where an early return would bite.
    const int n = 1000 * 256 + 17, blocks = (n + 255) / 256;
    std::vector<int> x(n), out(blocks);
    long long cpu_sum = 0;
    for (int i = 0; i < n; ++i) { x[i] = i % 5 + 1; cpu_sum += x[i]; }
    int* dx = nullptr;
    int* dout = nullptr;
    check(cudaMalloc(&dx, n * sizeof(int)), "cudaMalloc x");
    check(cudaMalloc(&dout, blocks * sizeof(int)), "cudaMalloc out");
    check(cudaMemcpy(dx, x.data(), n * sizeof(int), cudaMemcpyHostToDevice), "upload");
    block_sum<<<blocks, 256>>>(dx, dout, n);
    check_kernel("block_sum");              // returns: every thread reached every barrier
    check(cudaMemcpy(out.data(), dout, blocks * sizeof(int), cudaMemcpyDeviceToHost), "download");
    check(cudaFree(dx), "cudaFree x");
    check(cudaFree(dout), "cudaFree out");

    long long gpu_sum = 0;
    for (int b = 0; b < blocks; ++b) gpu_sum += out[b];
    std::printf("fixed kernel, %d elements (last block holds %d): GPU %lld, CPU %lld\n",
                n, n - (blocks - 1) * 256, gpu_sum, cpu_sum);
    bool ok = gpu_sum == cpu_sum;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
