// CUDA Host and Device Memory in 9 Minutes - slide 10: complete vector addition
// Build: make 10_vector_add
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

#include <chrono>
#include <cstddef>

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global index, as in episode 3
    if (i < n) c[i] = a[i] + b[i];
}

template <class T> class DeviceBuffer {              // slide 6: owns one device allocation
    T* p_ = nullptr;  std::size_t n_ = 0;
public:
    explicit DeviceBuffer(std::size_t n) : n_(n) { check(cudaMalloc(&p_, n * sizeof(T)), "cudaMalloc"); }
    ~DeviceBuffer() { cudaFree(p_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return p_; }
    std::size_t bytes() const { return n_ * sizeof(T); }
};

template <class T> class PinnedBuffer {              // the same idea for cudaMallocHost and cudaFreeHost
    T* p_ = nullptr;
public:
    explicit PinnedBuffer(std::size_t n) { check(cudaMallocHost(&p_, n * sizeof(T)), "cudaMallocHost"); }
    ~PinnedBuffer() { cudaFreeHost(p_); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    T* get() const { return p_; }
};

template <class F>
static double device_ms(F&& f) {                     // slide 9: one event pair around a stage, device time
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    check(cudaEventRecord(start));
    f();
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop));
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    return double(ms);
}

static bool verify_on_cpu(const float* a, const float* b, const float* c, std::size_t n) {
    std::size_t wrong = 0;
    for (std::size_t i = 0; i < n; ++i) if (c[i] != a[i] + b[i]) ++wrong;
    std::printf("CPU verification: %zu of %zu elements differ\n", wrong, n);
    return wrong == 0;
}

static double host_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
    if (!require_device()) return 0;
    auto t_context = std::chrono::steady_clock::now();
    check(cudaFree(nullptr), "context warm up");     // the first runtime call creates the context
    double ms_context = host_ms(t_context);

    const std::size_t n = std::size_t(1) << 22;      // 4M floats: 16 MiB per vector
    const std::size_t bytes = n * sizeof(float);
    const int threads = 256, blocks = int((n + threads - 1) / threads);
    PinnedBuffer<float> pin_a(n), pin_b(n), pin_c(n);
    float* h_a = pin_a.get();
    float* h_b = pin_b.get();
    float* h_c = pin_c.get();
    for (std::size_t i = 0; i < n; ++i) { h_a[i] = float(i % 1024); h_b[i] = 0.5f; h_c[i] = 0.0f; }

    auto t_total = std::chrono::steady_clock::now();
    auto t_alloc = std::chrono::steady_clock::now();
    DeviceBuffer<float> d_a(n), d_b(n), d_c(n);           // 1. allocate
    double ms_alloc = host_ms(t_alloc);
    double ms_h2d = device_ms([&] {                       // 2. copy in
        check(cudaMemcpy(d_a.get(), h_a, bytes, cudaMemcpyHostToDevice));
        check(cudaMemcpy(d_b.get(), h_b, bytes, cudaMemcpyHostToDevice));
    });
    double ms_kernel = device_ms([&] {                    // 3. launch
        add<<<blocks, threads>>>(d_a.get(), d_b.get(), d_c.get(), int(n));
    });
    check_kernel("add");                                 // launch accepted, no fault (already synchronized)
    double ms_d2h = device_ms([&] {                       // 4. copy out
        check(cudaMemcpy(h_c, d_c.get(), bytes, cudaMemcpyDeviceToHost));
    });
    double ms_total = host_ms(t_total);
    bool ok = verify_on_cpu(h_a, h_b, h_c, n);  // 5. verify; buffers free

    std::printf("vector addition, %zu elements, %zu MiB per vector, pinned host buffers, this machine:\n", n, bytes >> 20);
    std::printf("  context creation   %9.3f ms  steady_clock (paid once per process, before the stages)\n", ms_context);
    std::printf("  1. cudaMalloc x3   %9.3f ms  steady_clock\n", ms_alloc);
    std::printf("  2. copy in  a, b   %9.3f ms  CUDA events\n", ms_h2d);
    std::printf("  3. kernel add      %9.3f ms  CUDA events\n", ms_kernel);
    std::printf("  4. copy out c      %9.3f ms  CUDA events\n", ms_d2h);
    std::printf("  end to end 1 to 4  %9.3f ms  steady_clock\n", ms_total);
    std::printf("  copies / kernel: %.1f (a timing, never a pass condition)\n", (ms_h2d + ms_d2h) / (ms_kernel > 0.0 ? ms_kernel : 1e-6));
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;                                    // DeviceBuffer and PinnedBuffer free themselves here
}
