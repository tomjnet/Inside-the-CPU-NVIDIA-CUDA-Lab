// CUDA Synchronization Without Deadlocks in 8 Minutes - slide 6: the fix: one barrier per step
// Build: make 06_reduce_barrier
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

// Launch one reduction kernel `runs` times; count the runs whose total differs from the CPU sum.
template <class Kernel>
static int wrong_runs(Kernel kernel, const char* name, const int* dx, int* dout, int n, int blocks,
                      long long cpu_sum, int runs) {
    std::vector<int> out(blocks);
    int wrong = 0;
    for (int run = 0; run < runs; ++run) {
        kernel<<<blocks, 256>>>(dx, dout, n);
        check_kernel(name);
        check(cudaMemcpy(out.data(), dout, blocks * sizeof(int), cudaMemcpyDeviceToHost), "download");
        long long sum = 0;
        for (int b = 0; b < blocks; ++b) sum += out[b];
        if (sum != cpu_sum) ++wrong;
    }
    return wrong;
}

int main() {
    if (!require_device()) return 0;

    const int n = 1000000;                  // not a multiple of 256: the last block is partial
    const int blocks = (n + 255) / 256;
    std::vector<int> x(n);
    long long cpu_sum = 0;
    for (int i = 0; i < n; ++i) { x[i] = i % 7 + 1; cpu_sum += x[i]; }
    int* dx = nullptr;
    int* dout = nullptr;
    check(cudaMalloc(&dx, n * sizeof(int)), "cudaMalloc x");
    check(cudaMalloc(&dout, blocks * sizeof(int)), "cudaMalloc out");
    check(cudaMemcpy(dx, x.data(), n * sizeof(int), cudaMemcpyHostToDevice), "upload");

    int wrong = wrong_runs(block_sum, "block_sum", dx, dout, n, blocks, cpu_sum, 100);

    std::vector<int> out(blocks);           // one more run to show the last, partial block
    block_sum<<<blocks, 256>>>(dx, dout, n);
    check_kernel("block_sum");
    check(cudaMemcpy(out.data(), dout, blocks * sizeof(int), cudaMemcpyDeviceToHost), "download");
    check(cudaFree(dx), "cudaFree x");
    check(cudaFree(dout), "cudaFree out");

    long long last_cpu = 0;
    for (int i = (blocks - 1) * 256; i < n; ++i) last_cpu += x[i];
    std::printf("%d blocks, the last one holds %d elements: GPU %d, CPU %lld\n",
                blocks, n - (blocks - 1) * 256, out[blocks - 1], last_cpu);
    std::printf("per block: 1 load barrier + 8 step barriers = 9 synchronization points\n");
    std::printf("block_sum: %d of 100 runs wrong\n", wrong);
    bool ok = wrong == 0 && out[blocks - 1] == last_cpu;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
