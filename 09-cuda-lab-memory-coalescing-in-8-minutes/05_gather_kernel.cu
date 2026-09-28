// Memory Coalescing in CUDA in 8 Minutes - slide 5: one kernel, one index table
// Build: make 05_gather_kernel
#include <cstdio>
#include <cstdlib>
#include <vector>
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

// One thread per output element; the table decides the read address.
__global__ void gather(const float* in, const int* map, float* out,
                       int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;                 // guard the rounded up grid
    out[i] = 2.0f * in[map[i]] + 1.0f;  // map[i], out[i]: coalesced
}                                       // in[map[i]]: up to the table

int main() {
    if (!require_device()) return 0;

    const int n = 1 << 20;                                   // 1 M floats: 4 MiB per buffer
    const std::size_t fbytes = std::size_t(n) * sizeof(float);
    const std::size_t ibytes = std::size_t(n) * sizeof(int);
    std::vector<float> in(n), out(n, 0.0f);
    std::vector<int> map(n);
    for (int i = 0; i < n; ++i) {
        in[i] = float(i % 1000);
        map[i] = i;                                          // identity table: thread i reads address i
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    int* d_map = nullptr;
    check(cudaMalloc(&d_in, fbytes), "cudaMalloc in");
    check(cudaMalloc(&d_out, fbytes), "cudaMalloc out");
    check(cudaMalloc(&d_map, ibytes), "cudaMalloc map");
    check(cudaMemcpy(d_in, in.data(), fbytes, cudaMemcpyHostToDevice), "upload in");
    check(cudaMemcpy(d_map, map.data(), ibytes, cudaMemcpyHostToDevice), "upload map");

    int threads = 256, blocks = (n + threads - 1) / threads;
    gather<<<blocks, threads>>>(d_in, d_map, d_out, n);  // 1 launch
    check_kernel("gather");                 // the helpers from episode 2

    check(cudaMemcpy(out.data(), d_out, fbytes, cudaMemcpyDeviceToHost), "download");
    check(cudaFree(d_in), "cudaFree");
    check(cudaFree(d_out), "cudaFree");
    check(cudaFree(d_map), "cudaFree");

    long long wrong = 0;                                     // CPU check of every element
    for (int i = 0; i < n; ++i) wrong += out[i] != 2.0f * in[map[i]] + 1.0f;
    std::printf("gather over %d elements in %d blocks of %d threads, identity table\n", n, blocks, threads);
    std::printf("threads 0..3 read addresses %d %d %d %d: neighbours, so their reads can coalesce\n",
                map[0], map[1], map[2], map[3]);
    std::printf("out[0..3] = %.0f %.0f %.0f %.0f, wrong elements: %lld\n", out[0], out[1], out[2], out[3], wrong);
    bool ok = wrong == 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
