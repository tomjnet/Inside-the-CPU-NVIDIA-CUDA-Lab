// CUDA Grids, Blocks and Threads in 8 Minutes - slide 9: limits of a launch
// Build: make 09_launch_limits
// The helper block (check, check_kernel, require_device) is the one of episode 2, unchanged.
#include <cstddef>
#include <vector>
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

__global__ void invert(unsigned char* img, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h)                      // 1 thread per pixel
        img[y * w + x] = 255 - img[y * w + x];   // 1 read, 1 write
}

int main() {
    if (!require_device()) return 0;

    cudaDeviceProp p;                            // limits of this device
    check(cudaGetDeviceProperties(&p, 0), "properties");
    // typical: warpSize 32, maxThreadsPerBlock 1024
    // maxThreadsDim 1024, 1024, 64; maxGridSize 2147483647, 65535, 65535
    std::printf("device               %s (compute capability %d.%d)\n", p.name, p.major, p.minor);
    std::printf("warpSize             %d\n", p.warpSize);
    std::printf("maxThreadsPerBlock   %d\n", p.maxThreadsPerBlock);
    std::printf("maxThreadsDim        %d, %d, %d\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    std::printf("maxGridSize          %d, %d, %d\n\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);

    const int w = 100, h = 60;
    const std::size_t pixels = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    std::vector<unsigned char> before(pixels, 40), after(pixels);
    unsigned char* d_img = nullptr;
    check(cudaMalloc(&d_img, pixels), "cudaMalloc");
    check(cudaMemcpy(d_img, before.data(), pixels, cudaMemcpyHostToDevice), "upload");

    dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
    invert<<<grid, block>>>(d_img, w, h);        // a legal launch: 16 * 16 = 256 threads per block
    check_kernel("invert");
    std::printf("block 16 x 16 = %u threads: accepted, the image is inverted once\n", block.x * block.y);

    dim3 too_big(33, 33);                        // 1089 threads in a block
    invert<<<grid, too_big>>>(d_img, w, h);      // x * y * z is over 1024
    cudaError_t e = cudaGetLastError();          // invalid argument
    // nothing ran: the launch error query is the only place you see it
    std::printf("block 33 x 33 = %u threads: %s\n", too_big.x * too_big.y, cudaGetErrorString(e));
    check(cudaDeviceSynchronize(), "synchronize");   // a rejected launch is not sticky: the device is fine

    check(cudaMemcpy(after.data(), d_img, pixels, cudaMemcpyDeviceToHost), "download");
    check(cudaFree(d_img), "cudaFree");
    std::size_t wrong = 0;                       // inverted exactly once: the rejected launch did not run
    for (std::size_t i = 0; i < pixels; ++i)
        if (after[i] != 255 - before[i]) ++wrong;
    std::printf("pixels changed by the rejected launch: %zu\n", wrong);

    bool ok = e != cudaSuccess && wrong == 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
