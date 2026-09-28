// CUDA Synchronization Without Deadlocks in 8 Minutes - slide 5: the racy reduction
// Build: make 05_reduce_racy
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

__global__ void reduce_racy(const int* x, int* out, int n) {
    __shared__ int cell[256];              // 1 cell per thread
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    cell[t] = i < n ? x[i] : 0;            // 1 global read per thread
    for (int s = blockDim.x / 2; s > 0; s /= 2)
        if (t < s) cell[t] += cell[t + s]; // RACE: cell[t + s] ready?
    if (t == 0) out[blockIdx.x] = cell[0]; // 1 global write per block
}
// 0 synchronization points: thread 0 can finish all 8 steps
// before thread 128 has even loaded its cell

// the fixed kernel of slide 6, for comparison
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

    const int n = 1 << 20, blocks = (n + 255) / 256;
    std::vector<int> x(n);
    long long cpu_sum = 0;
    for (int i = 0; i < n; ++i) { x[i] = i % 7 + 1; cpu_sum += x[i]; }
    int* dx = nullptr;
    int* dout = nullptr;
    check(cudaMalloc(&dx, n * sizeof(int)), "cudaMalloc x");
    check(cudaMalloc(&dout, blocks * sizeof(int)), "cudaMalloc out");
    check(cudaMemcpy(dx, x.data(), n * sizeof(int), cudaMemcpyHostToDevice), "upload");

    int racy = wrong_runs(reduce_racy, "reduce_racy", dx, dout, n, blocks, cpu_sum, 50);
    int fixed = wrong_runs(block_sum, "block_sum", dx, dout, n, blocks, cpu_sum, 50);
    check(cudaFree(dx), "cudaFree x");
    check(cudaFree(dout), "cudaFree out");

    std::printf("CPU sum %lld over %d elements, %d blocks of 256 threads\n", cpu_sum, n, blocks);
    std::printf("reduce_racy (0 barriers): %2d of 50 runs wrong on this GPU\n", racy);
    std::printf("block_sum   (9 barriers): %2d of 50 runs wrong\n", fixed);
    if (racy == 0) std::printf("the race did not show this time: that proves nothing, the kernel is still wrong\n");
    bool ok = fixed == 0;                   // only the fixed kernel must be right every time
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
