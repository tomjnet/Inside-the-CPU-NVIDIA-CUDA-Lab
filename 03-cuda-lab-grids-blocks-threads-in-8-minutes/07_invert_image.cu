// CUDA Grids, Blocks and Threads in 8 Minutes - slide 7: invert a grayscale image
// Build: make 07_invert_image
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

// one text row of the image: every second pixel as one of ten characters, dark to bright
static void print_row(const std::vector<unsigned char>& img, int w, int y) {
    static const char ramp[] = " .:-=+*#%@";
    std::printf("  row %2d |", y);
    for (int x = 0; x < w; x += 2) {
        std::size_t i = static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x);
        std::printf("%c", ramp[img[i] * 10 / 256]);
    }
    std::printf("|\n");
}

int main() {
    if (!require_device()) return 0;

    const int w = 100, h = 60;                   // not multiples of 16
    const std::size_t pixels = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    std::vector<unsigned char> before(pixels), after(pixels);
    for (int y = 0; y < h; ++y)                  // a horizontal gradient: black on the left, white on the right
        for (int x = 0; x < w; ++x)
            before[static_cast<std::size_t>(y * w + x)] = static_cast<unsigned char>(x * 255 / (w - 1));

    unsigned char* d_img = nullptr;
    check(cudaMalloc(&d_img, pixels), "cudaMalloc");
    check(cudaMemcpy(d_img, before.data(), pixels, cudaMemcpyHostToDevice), "upload");

    dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);   // 7 by 4
    invert<<<grid, block>>>(d_img, w, h);        // 28 blocks, 7168 threads
    check_kernel("invert");                      // 6000 pixels, 1168 idle

    check(cudaMemcpy(after.data(), d_img, pixels, cudaMemcpyDeviceToHost), "download");
    check(cudaFree(d_img), "cudaFree");

    std::size_t wrong = 0;
    for (std::size_t i = 0; i < pixels; ++i)
        if (after[i] != 255 - before[i]) ++wrong;

    const unsigned threads = grid.x * grid.y * block.x * block.y;
    std::printf("grid %u by %u blocks of %u by %u threads = %u threads for %zu pixels, %u idle\n",
                grid.x, grid.y, block.x, block.y, threads, pixels, threads - static_cast<unsigned>(pixels));
    std::printf("before:\n");
    for (int y = 0; y < h; y += 29) print_row(before, w, y);
    std::printf("after:\n");
    for (int y = 0; y < h; y += 29) print_row(after, w, y);
    std::printf("pixels that differ from 255 - original: %zu\n", wrong);
    std::printf("%s\n", wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? 0 : 1;
}
