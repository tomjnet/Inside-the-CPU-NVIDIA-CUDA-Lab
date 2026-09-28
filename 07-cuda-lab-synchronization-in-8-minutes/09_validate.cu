// CUDA Synchronization Without Deadlocks in 8 Minutes - slide 9: validate over repeated runs
// Build: make 09_validate
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

__global__ void racy_sum(const int* x, int* out, int n) {     // slide 3: one shared cell
    __shared__ int total;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0) total = 0;
    __syncthreads();
    if (i < n) total += x[i];
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = total;
}
__global__ void reduce_racy(const int* x, int* out, int n) {  // slide 5: no barriers
    __shared__ int cell[256];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    cell[t] = i < n ? x[i] : 0;
    for (int s = blockDim.x / 2; s > 0; s /= 2)
        if (t < s) cell[t] += cell[t + s];
    if (t == 0) out[blockIdx.x] = cell[0];
}
// slide 6: one barrier per step
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
    const size_t bytes = blocks * sizeof(int);
    std::vector<int> x(n), outs(blocks);
    long long cpu_sum = 0;
    for (int i = 0; i < n; ++i) { x[i] = i % 7 + 1; cpu_sum += x[i]; }
    int* out = outs.data();
    int* dx = nullptr;
    int* dout = nullptr;
    check(cudaMalloc(&dx, n * sizeof(int)), "cudaMalloc x");
    check(cudaMalloc(&dout, blocks * sizeof(int)), "cudaMalloc out");
    check(cudaMemcpy(dx, x.data(), n * sizeof(int), cudaMemcpyHostToDevice), "upload");

    // the slide, verbatim:
    int wrong = 0;                               // runs that differ
    for (int run = 0; run < 100; ++run) {        // 100 kernel launches
        block_sum<<<blocks, 256>>>(dx, dout, n);
        check_kernel("block_sum");               // 1 sync point per run
        check(cudaMemcpy(out, dout, bytes, cudaMemcpyDeviceToHost), "copy");
        long long sum = 0;
        for (int b = 0; b < blocks; ++b) sum += out[b];
        if (sum != cpu_sum) ++wrong;             // the CPU sum is the truth
    }
    std::printf("%d of 100 runs wrong\n", wrong);
    // a race may pass one run: only 0 of 100, on every GPU, is a fix

    int w_cell = wrong_runs(racy_sum, "racy_sum", dx, dout, n, blocks, cpu_sum, 100);
    int w_racy = wrong_runs(reduce_racy, "reduce_racy", dx, dout, n, blocks, cpu_sum, 100);
    check(cudaFree(dx), "cudaFree x");
    check(cudaFree(dout), "cudaFree out");

    std::printf("\nkernel                    barriers  wrong runs of 100 (this GPU)\n");
    std::printf("racy_sum (one cell)       2         %3d\n", w_cell);
    std::printf("reduce_racy (no barrier)  0         %3d\n", w_racy);
    std::printf("block_sum (fixed)         9         %3d\n", wrong);
    bool ok = wrong == 0;                   // the racy counts are shown, never asserted
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
