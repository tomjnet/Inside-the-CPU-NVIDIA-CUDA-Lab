// Your First CUDA Kernel in 8 Minutes - slide 8: two mistakes, two places
// Build: make 08_two_errors
//
// This sample breaks a kernel on purpose. The second mistake leaves a sticky error in the CUDA context of
// this process only; the process ends right after, so nothing else is affected.
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

__global__ void hello() { printf("Hello GPU from thread %d\n", threadIdx.x); }
__global__ void add_one(int* x) { *x = *x + 1; }

static void report(const char* label, cudaError_t e) {
    std::printf("%-44s %-24s %s\n", label, cudaGetErrorName(e), cudaGetErrorString(e));
}

int main() {
    if (!require_device()) return 0;

    // a healthy launch first, with the helpers: the device and the kernels are fine
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::printf("%s: at most %d threads per block\n", prop.name, prop.maxThreadsPerBlock);
    hello<<<1, 2>>>();
    check_kernel("hello");

    // from here on without the helpers, to see where each error surfaces
    hello<<<1, 4096>>>();                // too many threads per block
    cudaError_t e1 = cudaGetLastError(); // invalid argument, right away

    add_one<<<1, 1>>>(nullptr);          // a legal launch, a bad pointer
    cudaError_t e2 = cudaGetLastError(); // cudaSuccess: launch accepted
    cudaError_t e3 = cudaDeviceSynchronize();  // illegal memory access

    int* d = nullptr;                    // the fault is sticky:
    cudaError_t e4 = cudaMalloc(&d, 4);  // same error until the exit

    report("e1 cudaGetLastError after <<<1, 4096>>>", e1);
    report("e2 cudaGetLastError after the bad pointer", e2);
    report("e3 cudaDeviceSynchronize", e3);
    report("e4 cudaMalloc after the fault", e4);
    std::printf("d is still %p: nothing was allocated\n", static_cast<void*>(d));

    // CPU verification of the lesson: launch error at once, execution error at the sync, then sticky
    bool ok = e1 != cudaSuccess && e2 == cudaSuccess && e3 != cudaSuccess && e4 == e3 && d == nullptr;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
