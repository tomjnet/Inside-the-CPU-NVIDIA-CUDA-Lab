// CUDA Host and Device Memory in 9 Minutes - slide 6: ownership: a small raii wrapper
// Build: make 06_device_buffer
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

#include <cstddef>
#include <utility>
#include <vector>

__global__ void twice(float* p, std::size_t n) {
    std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < n) p[i] = p[i] * 2.0f;
}

template <class T> class DeviceBuffer {       // owns one allocation
    T* p_ = nullptr;  std::size_t n_ = 0;
public:
    explicit DeviceBuffer(std::size_t n) : n_(n) {
        check(cudaMalloc(&p_, n * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(p_); }         // runs on every exit path
    DeviceBuffer(const DeviceBuffer&) = delete;             // 1 owner,
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;  // no copy
    T* get() const { return p_; }             // kernels take raw ptrs
    std::size_t bytes() const { return n_ * sizeof(T); }

    // not on the slide: moving hands the allocation over, so a function can return a buffer
    DeviceBuffer(DeviceBuffer&& o) noexcept : p_(std::exchange(o.p_, nullptr)), n_(std::exchange(o.n_, 0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            cudaFree(p_);
            p_ = std::exchange(o.p_, nullptr);
            n_ = std::exchange(o.n_, 0);
        }
        return *this;
    }
};

static DeviceBuffer<float> upload(const std::vector<float>& h) {   // returned by move, never copied
    DeviceBuffer<float> d(h.size());
    check(cudaMemcpy(d.get(), h.data(), d.bytes(), cudaMemcpyHostToDevice), "upload");
    return d;
}

static bool early_return(bool give_up) {      // the shape that leaks with raw pointers
    DeviceBuffer<float> scratch(std::size_t(1) << 22);   // 16 MiB
    if (give_up) return false;                // the destructor still runs here
    return scratch.get() != nullptr;
}

int main() {
    if (!require_device()) return 0;
    const std::size_t n = std::size_t(1) << 20;
    std::vector<float> h_in(n), h_out(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) h_in[i] = float(i % 1000);

    DeviceBuffer<float> d = upload(h_in);     // round trip through an owned buffer
    twice<<<unsigned((n + 255) / 256), 256>>>(d.get(), n);   // kernels take the raw pointer
    check_kernel("twice");
    check(cudaMemcpy(h_out.data(), d.get(), d.bytes(), cudaMemcpyDeviceToHost), "download");
    std::size_t wrong = 0;
    for (std::size_t i = 0; i < n; ++i) if (h_out[i] != 2.0f * h_in[i]) ++wrong;
    bool same = wrong == 0;
    std::printf("round trip of %zu bytes through DeviceBuffer, doubled on the device: %zu wrong\n", d.bytes(), wrong);

    // 64 early returns of 16 MiB each: 1 GiB if every one leaked, 16 MiB at a time with RAII
    int returned_early = 0;
    for (int i = 0; i < 64; ++i) if (!early_return(true)) ++returned_early;
    bool normal_path = early_return(false);
    std::printf("%d early returns of a 16 MiB buffer, every one freed by the destructor\n", returned_early);

    // DeviceBuffer<float> copy = d;   // does not compile: the copy constructor is deleted
    DeviceBuffer<float> moved = std::move(d);
    std::printf("after the move the new owner holds %zu bytes and the old one %zu\n", moved.bytes(), d.bytes());

    bool ok = same && returned_early == 64 && normal_path && moved.get() != nullptr && d.get() == nullptr;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
