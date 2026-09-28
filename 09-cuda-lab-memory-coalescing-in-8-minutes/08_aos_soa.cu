// Memory Coalescing in CUDA in 8 Minutes - slide 8: array of structures against structure of arrays
// Build: make 08_aos_soa
#include <algorithm>
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

struct Particle { float x, y, z, w; };  // AoS: x values 16 B apart
__global__ void scale_aos(Particle* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i].x *= 2.0f;          // stride 4: 1 float in 4 used
}
__global__ void scale_soa(float* x, int n) {  // SoA: x adjacent
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 2.0f;            // 4 times fewer bytes moved
}

template <class Launch>
static float device_ms(Launch&& launch) {  // device time only
    cudaEvent_t t0, t1;
    check(cudaEventCreate(&t0));  check(cudaEventCreate(&t1));
    check(cudaEventRecord(t0, 0));      // marker into stream 0
    launch();                           // asynchronous: returns at once
    check(cudaEventRecord(t1, 0));
    check(cudaEventSynchronize(t1));    // 1 synchronization point
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, t0, t1));
    check(cudaEventDestroy(t0));  check(cudaEventDestroy(t1));
    return ms;
}

struct Stat { float min_ms; float median_ms; };
template <class Launch>
static Stat measure(Launch&& launch, int warmup = 3, int repeats = 21) {
    for (int i = 0; i < warmup; ++i) launch();       // warm up: clocks, caches, lazy setup
    check(cudaDeviceSynchronize(), "warm up");
    std::vector<float> s;
    for (int i = 0; i < repeats; ++i) s.push_back(device_ms(launch));
    std::sort(s.begin(), s.end());                   // keep minimum and median, as the Low Latency C++ Lab does
    return Stat{s.front(), s[s.size() / 2]};
}

int main() {
    if (!require_device()) return 0;

    const int n = 1 << 22;                                   // 4 M particles: 64 MiB as AoS, 16 MiB for x alone
    std::vector<Particle> aos(n);
    std::vector<float> soa_x(n);
    for (int i = 0; i < n; ++i) {
        aos[i] = Particle{float(i % 1000), 1.0f, 2.0f, 3.0f};
        soa_x[i] = aos[i].x;
    }
    const std::size_t abytes = std::size_t(n) * sizeof(Particle);
    const std::size_t xbytes = std::size_t(n) * sizeof(float);

    Particle* d_aos = nullptr;
    float* d_x = nullptr;
    check(cudaMalloc(&d_aos, abytes), "cudaMalloc aos");
    check(cudaMalloc(&d_x, xbytes), "cudaMalloc x");

    const int threads = 256, blocks = (n + threads - 1) / threads;
    const int runs = 3 + 21;                                 // warm ups plus repeats: every run doubles x again

    check(cudaMemcpy(d_aos, aos.data(), abytes, cudaMemcpyHostToDevice), "upload aos");
    Stat a = measure([&] { scale_aos<<<blocks, threads>>>(d_aos, n); });
    check_kernel("scale_aos");
    check(cudaMemcpy(aos.data(), d_aos, abytes, cudaMemcpyDeviceToHost), "download aos");

    check(cudaMemcpy(d_x, soa_x.data(), xbytes, cudaMemcpyHostToDevice), "upload x");
    Stat s = measure([&] { scale_soa<<<blocks, threads>>>(d_x, n); });
    check_kernel("scale_soa");
    check(cudaMemcpy(soa_x.data(), d_x, xbytes, cudaMemcpyDeviceToHost), "download x");

    check(cudaFree(d_aos), "cudaFree");
    check(cudaFree(d_x), "cudaFree");

    long long wrong = 0;                                     // CPU check: x doubled `runs` times, y z w untouched
    const float factor = float(1 << runs);                   // a power of two: exact in float
    for (int i = 0; i < n; ++i) {
        float expect = float(i % 1000) * factor;
        wrong += aos[i].x != expect || soa_x[i] != expect;
        wrong += aos[i].y != 1.0f || aos[i].z != 2.0f || aos[i].w != 3.0f;
    }
    std::printf("sizeof(Particle) = %zu bytes: neighbouring threads touch x values %zu bytes apart in AoS, %zu in SoA\n",
                sizeof(Particle), sizeof(Particle), sizeof(float));
    std::printf("%-6s %10s %12s\n", "layout", "min ms", "median ms");
    std::printf("%-6s %10.3f %12.3f\n", "AoS", a.min_ms, a.median_ms);
    std::printf("%-6s %10.3f %12.3f   AoS / SoA = %.2fx on this GPU\n", "SoA", s.min_ms, s.median_ms,
                a.median_ms / s.median_ms);
    std::printf("wrong elements: %lld\n", wrong);
    bool ok = wrong == 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
