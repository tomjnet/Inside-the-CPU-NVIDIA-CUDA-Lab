// CUDA Grids, Blocks and Threads in 8 Minutes - slide 4: round the grid up, guard the extra threads
// Build: make 04_round_up_guard
// The helper block (check, check_kernel, require_device) is the one of episode 2, unchanged.
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

__global__ void fill(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i;        // the guard: extra threads do nothing
}

// the same kernel without the guard, to watch where the extra threads write
__global__ void fill_no_guard(int* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = i;
}

static void show(const char* label, const int* v, int count) {
    std::printf("%-20s", label);
    for (int i = 0; i < count; ++i) std::printf(" %3d", v[i]);
    std::printf("\n");
}

int main() {
    if (!require_device()) return 0;

    // The buffer has room for 12 ints on purpose: the 2 slots after the 10 elements play the role of
    // the memory of somebody else, so the unguarded kernel below can be observed without a stray write.
    const int capacity = 12;
    int host[capacity] = {};
    int* d_out = nullptr;
    check(cudaMalloc(&d_out, sizeof host), "cudaMalloc");
    check(cudaMemset(d_out, 0xFF, sizeof host), "cudaMemset");         // every int is -1

    int n = 10, threads = 4;                     // 10: not a multiple of 4
    int blocks = (n + threads - 1) / threads;    // round up: 3, not 2
    fill<<<blocks, threads>>>(d_out, n);         // 12 threads, 2 extra
    check_kernel("fill");                        // no guard: 2 bad writes

    std::printf("n = %d, threads per block = %d\n", n, threads);
    std::printf("n / threads                 = %d blocks: %d threads, elements 8 and 9 never computed\n",
                n / threads, n / threads * threads);
    std::printf("(n + threads - 1) / threads = %d blocks: %d threads, %d extra\n\n",
                blocks, blocks * threads, blocks * threads - n);

    check(cudaMemcpy(host, d_out, sizeof host, cudaMemcpyDeviceToHost), "download");
    show("with the guard:", host, capacity);
    bool ok = true;
    for (int i = 0; i < n; ++i) ok = ok && host[i] == i;
    for (int i = n; i < capacity; ++i) ok = ok && host[i] == -1;        // the neighbours are untouched

    fill_no_guard<<<blocks, threads>>>(d_out);
    check_kernel("fill_no_guard");
    check(cudaMemcpy(host, d_out, sizeof host, cudaMemcpyDeviceToHost), "download");
    show("without the guard:", host, capacity);
    ok = ok && host[10] == 10 && host[11] == 11;                        // threads 10 and 11 did write
    std::printf("slots 10 and 11 are past the 10 elements: in a real program that memory belongs to\n"
                "another allocation, or to nobody, and check_kernel may or may not report it\n");
    check(cudaFree(d_out), "cudaFree");

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
