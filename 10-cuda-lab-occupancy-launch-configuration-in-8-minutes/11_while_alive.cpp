// CUDA Occupancy and Launch Configuration in 8 Minutes - slide 11: thank you
// Build: make 11_while_alive
#include <iostream>
#include <string>
#include <vector>

// the imaginary tomjnet.h of the slide
struct Topic { std::string name; };
static int episodes = 0;
static bool alive() { return episodes < 3; }                 // three episodes, then the demo ends
static Topic next_cuda_topic() {
    static const char* topics[] = {"Profile CUDA with NVIDIA Nsight", "CPU vs GPU: An Honest Benchmark",
                                   "Back to the Low Latency C++ Lab"};
    return Topic{topics[episodes++]};
}
static Topic viewer_request() { return Topic{"viewer request #" + std::to_string(episodes)}; }
static void subscribe() { std::cout << "  subscribed to TomJNet\n"; }

int main() {
    std::vector<Topic> blocks;
    while (alive()) {
        blocks.push_back(next_cuda_topic()); // next: Nsight profiling
        blocks.push_back(viewer_request());  // one more resident block
        subscribe();                         // lifetime benefit
    }

    std::cout << blocks.size() << " topics queued:\n";
    for (const Topic& t : blocks) std::cout << "  " << t.name << '\n';
    return 0;
}
