// CUDA Occupancy and Launch Configuration in 8 Minutes - slide 3: the budget of one sm
// Build: make 03_sm_limits
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

__global__ void read_warp_size(int* out) { *out = warpSize; }  // the device's own view

int main() {
    if (!require_device()) return 0;

    cudaDeviceProp p;                               // the budget of one SM
    check(cudaGetDeviceProperties(&p, 0), "properties");
    int warps  = p.maxThreadsPerMultiProcessor / p.warpSize;
    int blocks = p.maxBlocksPerMultiProcessor;      // block slots
    int regs   = p.regsPerMultiprocessor;           // 32 bit registers
    size_t shm = p.sharedMemPerMultiprocessor;      // bytes
    // typical (Turing, GTX 1650): 32 warps, 16 blocks,
    // 65536 registers, 64 KiB shared memory, 16 SMs
    // occupancy = resident warps / warps, per kernel and launch

    std::printf("device %s, compute capability %d.%d, %d SMs\n", p.name, p.major, p.minor,
                p.multiProcessorCount);
    std::printf("per SM: %d threads = %d warp slots of %d, %d block slots\n",
                p.maxThreadsPerMultiProcessor, warps, p.warpSize, blocks);
    std::printf("per SM: %d registers (%d per thread if every slot is full), %zu KiB shared memory\n",
                regs, regs / p.maxThreadsPerMultiProcessor, shm / 1024);
    std::printf("per block: at most %d threads, %d registers, %zu KiB shared memory by default\n",
                p.maxThreadsPerBlock, p.regsPerBlock, p.sharedMemPerBlock / 1024);

    int* d = nullptr;
    check(cudaMalloc(&d, sizeof(int)), "cudaMalloc");
    read_warp_size<<<1, 1>>>(d);                    // 1 launch, 1 thread
    check_kernel("read_warp_size");
    int device_warp = 0;
    check(cudaMemcpy(&device_warp, d, sizeof(int), cudaMemcpyDeviceToHost), "download");
    check(cudaFree(d), "cudaFree");
    std::printf("warpSize inside a kernel: %d, from the properties: %d\n", device_warp, p.warpSize);

    bool ok = device_warp == p.warpSize && warps > 0 && blocks > 0 && regs > 0 && shm > 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
