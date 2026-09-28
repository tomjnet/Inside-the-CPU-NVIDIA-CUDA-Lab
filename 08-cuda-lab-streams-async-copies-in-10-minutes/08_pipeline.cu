// CUDA Streams and Asynchronous Copies in 10 Minutes - slide 8: the pipeline: one chunk, one stream
// Build: make 08_pipeline
// Run:   ./08_pipeline            both modes, the ratio and both timelines
//        ./08_pipeline serial     every chunk on one stream: upload, kernel, download, one after the other
//        ./08_pipeline streams    chunk c on stream c % 4: uploads, kernels and downloads may overlap
//
// Episode 11 profiles this program with Nsight Systems: serial is the before, streams is the after.
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
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

static const int kChunks = 8;                        // the input is cut into 8 chunks
static const int kStreams = 4;                       // overlapped mode: 4 streams, 2 chunks each
static const int kChunkN = 1 << 20;                  // 1 M floats = 4 MiB per chunk
static const int kSteps = 64;                        // Newton steps per element: the kernel has real work

// 64 Newton steps toward the square root of every element: same arithmetic as the CPU check
__global__ void work(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;                              // guard the rounded up grid
    float x = in[i];
    float v = x;
    for (int k = 0; k < kSteps; ++k) v = 0.5f * (v + x / v);
    out[i] = v;
}

struct Stamps {                                      // four events per chunk: the text timeline
    cudaEvent_t up0, up1, k1, down1;
};
struct Row {                                         // milliseconds since the first enqueue
    int stream;
    float up0, up1, k1, down1;
};

// Runs every chunk through upload, kernel, download on n_streams streams; returns end to end ms.
static double run_pipeline(int n_streams, const float* h_in, float* h_out, std::vector<Row>& rows) {
    const std::size_t bytes = static_cast<std::size_t>(kChunkN) * sizeof(float);
    std::vector<cudaStream_t> stream(static_cast<std::size_t>(n_streams));
    std::vector<float*> d_in(stream.size(), nullptr), d_out(stream.size(), nullptr);
    for (int s = 0; s < n_streams; ++s) {
        check(cudaStreamCreate(&stream[s]), "cudaStreamCreate");
        check(cudaMalloc(&d_in[s], bytes), "cudaMalloc in");      // one pair of buffers per stream:
        check(cudaMalloc(&d_out[s], bytes), "cudaMalloc out");    // chunks of one stream reuse them in order
    }
    std::vector<Stamps> ev(static_cast<std::size_t>(kChunks));
    for (Stamps& e : ev) {
        check(cudaEventCreate(&e.up0));
        check(cudaEventCreate(&e.up1));
        check(cudaEventCreate(&e.k1));
        check(cudaEventCreate(&e.down1));
    }
    check(cudaDeviceSynchronize(), "idle before timing");

    const int block = 256;
    const int grid = (kChunkN + block - 1) / block;
    auto t0 = std::chrono::steady_clock::now();
    for (int c = 0; c < kChunks; ++c) {     // 8 chunks of 4 MiB
        int s = c % n_streams;              // n_streams = 1: serial mode
        cudaStream_t st = stream[s];        // one device buffer pair per s
        Stamps& e = ev[c];                  // the four markers of this chunk
        check(cudaEventRecord(e.up0, st));
        check(cudaMemcpyAsync(d_in[s], h_in + c * kChunkN, bytes,
                              cudaMemcpyHostToDevice, st));
        check(cudaEventRecord(e.up1, st));
        work<<<grid, block, 0, st>>>(d_in[s], d_out[s], kChunkN);
        check(cudaGetLastError());
        check(cudaEventRecord(e.k1, st));
        check(cudaMemcpyAsync(h_out + c * kChunkN, d_out[s], bytes,
                              cudaMemcpyDeviceToHost, st));
        check(cudaEventRecord(e.down1, st));
    }                                       // 24 operations queued, 0 waits
    auto tq = std::chrono::steady_clock::now();
    for (cudaStream_t st : stream)
        check(cudaStreamSynchronize(st));   // 1 sync per stream, at the end
    auto t1 = std::chrono::steady_clock::now();

    rows.clear();
    for (int c = 0; c < kChunks; ++c) {              // every event is complete: read the device clock
        const Stamps& e = ev[c];
        Row r{c % n_streams, 0.0f, 0.0f, 0.0f, 0.0f};
        check(cudaEventElapsedTime(&r.up0, ev[0].up0, e.up0));
        check(cudaEventElapsedTime(&r.up1, ev[0].up0, e.up1));
        check(cudaEventElapsedTime(&r.k1, ev[0].up0, e.k1));
        check(cudaEventElapsedTime(&r.down1, ev[0].up0, e.down1));
        rows.push_back(r);
    }
    for (Stamps& e : ev) {
        check(cudaEventDestroy(e.up0));
        check(cudaEventDestroy(e.up1));
        check(cudaEventDestroy(e.k1));
        check(cudaEventDestroy(e.down1));
    }
    for (int s = 0; s < n_streams; ++s) {
        check(cudaFree(d_in[s]));
        check(cudaFree(d_out[s]));
        check(cudaStreamDestroy(stream[s]), "cudaStreamDestroy");
    }
    std::printf("  host queued %d chunks on %d stream(s) in %.2f ms, then waited\n", kChunks, n_streams,
                std::chrono::duration<double, std::milli>(tq - t0).count());
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// One text row per chunk, time runs to the right: U upload, K kernel, D download, . waiting or idle.
// A stage starts when its stream reaches it, so U also holds the wait for a busy copy engine.
static void print_timeline(const char* title, const std::vector<Row>& rows, float span_ms) {
    const int cols = 64;
    std::printf("%s (device clock from events, %.2f ms across %d columns)\n", title,
                static_cast<double>(span_ms), cols);
    for (std::size_t c = 0; c < rows.size(); ++c) {
        const Row& r = rows[c];
        std::string line(static_cast<std::size_t>(cols), '.');
        for (int x = 0; x < cols; ++x) {
            float t = (static_cast<float>(x) + 0.5f) * span_ms / static_cast<float>(cols);
            if (t >= r.up0 && t < r.up1) line[static_cast<std::size_t>(x)] = 'U';
            else if (t >= r.up1 && t < r.k1) line[static_cast<std::size_t>(x)] = 'K';
            else if (t >= r.k1 && t < r.down1) line[static_cast<std::size_t>(x)] = 'D';
        }
        std::printf("  chunk %d stream %d |%s|\n", static_cast<int>(c), r.stream, line.c_str());
    }
}

static bool verify(const float* h_in, const float* h_out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {            // CPU check: Newton converged to the square root
        float want = std::sqrt(h_in[i]);
        if (std::fabs(h_out[i] - want) > 1e-3f * want) {
            std::printf("mismatch at %zu: got %f, want %f\n", i, static_cast<double>(h_out[i]),
                        static_cast<double>(want));
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (!require_device()) return 0;
    std::string mode = argc > 1 ? argv[1] : "both";
    if (mode != "serial" && mode != "streams" && mode != "both") {
        std::printf("usage: %s [serial|streams]\n", argv[0]);
        return 1;
    }

    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::printf("%s, copy engines (asyncEngineCount): %d\n", prop.name, prop.asyncEngineCount);

    const std::size_t total = static_cast<std::size_t>(kChunks) * kChunkN;
    float* h_in = nullptr;
    float* h_out = nullptr;                          // pinned: the copy engine reads and writes it by DMA
    check(cudaMallocHost(&h_in, total * sizeof(float)), "cudaMallocHost in");
    check(cudaMallocHost(&h_out, total * sizeof(float)), "cudaMallocHost out");
    for (std::size_t i = 0; i < total; ++i) h_in[i] = 1.0f + static_cast<float>(i % 1000);

    std::vector<Row> rows;
    run_pipeline(1, h_in, h_out, rows);              // warm up: context, kernel load, first touch
    check_kernel("warm up");                         // a fault in the warm up run surfaces here
    std::printf("%d chunks of %d floats (%zu MiB each way), block 256\n", kChunks, kChunkN,
                total * sizeof(float) >> 20);

    bool ok = true;
    double serial_ms = 0.0, streams_ms = 0.0;
    std::vector<Row> serial_rows, stream_rows;
    if (mode != "streams") {
        std::memset(h_out, 0, total * sizeof(float));
        serial_ms = run_pipeline(1, h_in, h_out, serial_rows);
        ok = verify(h_in, h_out, total) && ok;
        std::printf("serial : 1 stream   end to end %.2f ms (steady_clock, this machine)\n", serial_ms);
    }
    if (mode != "serial") {
        std::memset(h_out, 0, total * sizeof(float));
        streams_ms = run_pipeline(kStreams, h_in, h_out, stream_rows);
        ok = verify(h_in, h_out, total) && ok;
        std::printf("streams: %d streams  end to end %.2f ms (steady_clock, this machine)\n", kStreams,
                    streams_ms);
    }
    if (mode == "both" && streams_ms > 0.0)
        std::printf("ratio serial / streams: %.2fx (this machine, never asserted)\n", serial_ms / streams_ms);

    float span = 0.0f;                               // one scale for both timelines
    for (const Row& r : serial_rows) span = r.down1 > span ? r.down1 : span;
    for (const Row& r : stream_rows) span = r.down1 > span ? r.down1 : span;
    if (!serial_rows.empty()) print_timeline("serial timeline", serial_rows, span);
    if (!stream_rows.empty()) print_timeline("streams timeline", stream_rows, span);

    check(cudaFreeHost(h_in));
    check(cudaFreeHost(h_out));
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
