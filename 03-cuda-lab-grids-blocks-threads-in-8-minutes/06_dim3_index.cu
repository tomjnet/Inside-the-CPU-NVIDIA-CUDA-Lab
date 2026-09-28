// CUDA Grids, Blocks and Threads in 8 Minutes - slide 6: dim3 and 2d indexing
// Build: make 06_dim3_index
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

__global__ void clear_image(unsigned char* img, int w, int h) {
    // inside the kernel: one index per dimension, then the row offset
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // column
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // row
    if (x < w && y < h)                              // guard both edges
        img[y * w + x] = 0;                          // row major pixel
}

int main() {
    if (!require_device()) return 0;

    const int w = 100, h = 60;                       // not multiples of 16
    const std::size_t pixels = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    const std::size_t padding = 64;                  // bytes after the image that nobody may touch
    std::vector<unsigned char> host(pixels + padding, 200);

    unsigned char* img = nullptr;
    check(cudaMalloc(&img, host.size()), "cudaMalloc");
    check(cudaMemcpy(img, host.data(), host.size(), cudaMemcpyHostToDevice), "upload");

    dim3 block(16, 16);                          // 256 threads per block
    dim3 grid((w + block.x - 1) / block.x,       // round up in x
              (h + block.y - 1) / block.y);      // and in y: z stays 1

    clear_image<<<grid, block>>>(img, w, h);
    check_kernel("clear_image");
    check(cudaMemcpy(host.data(), img, host.size(), cudaMemcpyDeviceToHost), "download");
    check(cudaFree(img), "cudaFree");

    std::size_t not_cleared = 0, padding_touched = 0;
    for (std::size_t i = 0; i < pixels; ++i) if (host[i] != 0) ++not_cleared;
    for (std::size_t i = pixels; i < host.size(); ++i) if (host[i] != 200) ++padding_touched;

    const unsigned threads = grid.x * grid.y * grid.z * block.x * block.y * block.z;
    std::printf("image %d x %d = %zu pixels\n", w, h, pixels);
    std::printf("block %u x %u x %u, grid %u x %u x %u\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);
    std::printf("threads launched = %u, stopped by the guard = %u\n", threads, threads - static_cast<unsigned>(pixels));
    std::printf("pixels not cleared = %zu, bytes touched after the image = %zu\n", not_cleared, padding_touched);
    bool ok = not_cleared == 0 && padding_touched == 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
