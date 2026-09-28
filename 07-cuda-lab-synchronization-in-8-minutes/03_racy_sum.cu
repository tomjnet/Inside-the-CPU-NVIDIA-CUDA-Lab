// CUDA Synchronization Without Deadlocks in 8 Minutes - slide 3: a race on one shared cell
// Build: make 03_racy_sum
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

__global__ void racy_sum(const int* x, int* out, int n) {
    __shared__ int total;                  // 1 cell for the whole block
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0) total = 0;
    __syncthreads();                       // barrier: total is 0 now
    if (i < n) total += x[i];              // RACE: read, add, write
    __syncthreads();                       // barrier: every add is over
    if (threadIdx.x == 0) out[blockIdx.x] = total;
}
// 256 threads read the same old total, 255 updates are lost
// a barrier orders phases, it does not make += atomic

int main() {
    if (!require_device()) return 0;

    const int n = 1 << 20, threads = 256, blocks = (n + threads - 1) / threads;
    std::vector<int> x(n), out(blocks), expected(blocks, 0);
    for (int i = 0; i < n; ++i) x[i] = i % 7 + 1;          // every element is 1 to 7
    for (int i = 0; i < n; ++i) expected[i / threads] += x[i];
    int* dx = nullptr;
    int* dout = nullptr;
    check(cudaMalloc(&dx, n * sizeof(int)), "cudaMalloc x");
    check(cudaMalloc(&dout, blocks * sizeof(int)), "cudaMalloc out");
    check(cudaMemcpy(dx, x.data(), n * sizeof(int), cudaMemcpyHostToDevice), "upload");

    int wrong_runs = 0;
    long long lost = 0;
    bool only_lost = true;                  // a lost update can only make a sum smaller
    for (int run = 0; run < 20; ++run) {
        racy_sum<<<blocks, threads>>>(dx, dout, n);
        check_kernel("racy_sum");
        check(cudaMemcpy(out.data(), dout, blocks * sizeof(int), cudaMemcpyDeviceToHost), "download");
        bool wrong = false;
        for (int b = 0; b < blocks; ++b) {
            if (out[b] != expected[b]) wrong = true;
            if (out[b] > expected[b]) only_lost = false;
            lost += expected[b] - out[b];
        }
        if (wrong) ++wrong_runs;
    }
    check(cudaFree(dx), "cudaFree x");
    check(cudaFree(dout), "cudaFree out");

    std::printf("block 0: GPU %d, CPU %d\n", out[0], expected[0]);
    long long cpu_total = 0;
    for (int b = 0; b < blocks; ++b) cpu_total += expected[b];
    std::printf("%d of 20 runs wrong; per run the GPU total misses %lld of %lld on average (this GPU)\n",
                wrong_runs, lost / 20, cpu_total);
    std::printf("the barriers are in place and the sum is still wrong: += is read, add, write\n");
    std::printf("every GPU sum is at most the CPU sum (updates lost, never invented): %s\n",
                only_lost ? "yes" : "no");
    std::printf("%s\n", only_lost ? "PASS" : "FAIL");
    return only_lost ? 0 : 1;
}
