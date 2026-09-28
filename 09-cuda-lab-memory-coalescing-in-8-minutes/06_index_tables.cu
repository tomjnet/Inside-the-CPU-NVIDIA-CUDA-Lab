// Memory Coalescing in CUDA in 8 Minutes - slide 6: three tables: coalesced, strided, scattered
// Build: make 06_index_tables
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
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

    const int n = 1 << 20;                                   // a power of 2, the strided table needs it
    const std::size_t fbytes = std::size_t(n) * sizeof(float);
    const std::size_t ibytes = std::size_t(n) * sizeof(int);
    std::vector<float> in(n), out(n);
    for (int i = 0; i < n; ++i) in[i] = float(i % 1000);

    std::vector<int> ident(n), strided(n), scattered(n);
    for (int i = 0; i < n; ++i) {
        ident[i] = i;                       // neighbours: 1 transaction
        strided[i] = int((unsigned(i) * 33u) & unsigned(n - 1));
    }                                       // 33 apart: n is a power of 2
    scattered = ident;                      // same addresses, shuffled
    std::mt19937 rng(42);                   // fixed seed: same work
    std::shuffle(scattered.begin(), scattered.end(), rng);
    // every table is a permutation: same reads, same sum, other order

    float* d_in = nullptr;
    float* d_out = nullptr;
    int* d_map = nullptr;
    check(cudaMalloc(&d_in, fbytes), "cudaMalloc in");
    check(cudaMalloc(&d_out, fbytes), "cudaMalloc out");
    check(cudaMalloc(&d_map, ibytes), "cudaMalloc map");
    check(cudaMemcpy(d_in, in.data(), fbytes, cudaMemcpyHostToDevice), "upload in");

    const int threads = 256, blocks = (n + threads - 1) / threads;
    const std::vector<int>* tables[] = {&ident, &strided, &scattered};
    const char* names[] = {"coalesced", "strided", "scattered"};
    bool ok = true;
    for (int t = 0; t < 3; ++t) {
        const std::vector<int>& map = *tables[t];
        check(cudaMemcpy(d_map, map.data(), ibytes, cudaMemcpyHostToDevice), "upload map");
        gather<<<blocks, threads>>>(d_in, d_map, d_out, n);
        check_kernel("gather");
        check(cudaMemcpy(out.data(), d_out, fbytes, cudaMemcpyDeviceToHost), "download");

        long long wrong = 0;                                 // CPU check of every element
        double sum = 0.0;
        std::vector<char> seen(n, 0);                        // and of the permutation claim
        for (int i = 0; i < n; ++i) {
            wrong += out[i] != 2.0f * in[map[i]] + 1.0f;
            sum += out[i];
            seen[map[i]] = 1;
        }
        long long covered = 0;
        for (int i = 0; i < n; ++i) covered += seen[i];
        std::printf("%-9s threads 0..3 read %7d %7d %7d %7d  sum %.0f  covered %lld of %d  wrong %lld\n",
                    names[t], map[0], map[1], map[2], map[3], sum, covered, n, wrong);
        ok = ok && wrong == 0 && covered == n;
    }
    check(cudaFree(d_in), "cudaFree");
    check(cudaFree(d_out), "cudaFree");
    check(cudaFree(d_map), "cudaFree");

    std::printf("same kernel, same reads, same sum: only the order of the addresses changes\n");
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
