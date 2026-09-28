// Memory Coalescing in CUDA in 8 Minutes - slide 11: thank you
// Build: make 11_while_alive
#include <iostream>
#include <string>
#include <vector>

// the imaginary tomjnet.h of the slide
struct Topic { std::string name; };
static int episodes = 0;
static bool alive() { return episodes < 3; }                 // three episodes, then the demo ends
static Topic next_cuda_topic() {
    static const char* topics[] = {"CUDA Occupancy and Launch Configuration", "Profile CUDA with NVIDIA Nsight",
                                   "CPU vs GPU: An Honest Benchmark"};
    return Topic{topics[episodes++]};
}
static Topic viewer_request() { return Topic{"viewer request #" + std::to_string(episodes)}; }
static void subscribe() { std::cout << "  subscribed to TomJNet\n"; }

int main() {
    std::vector<Topic> grid;
    while (alive()) {
        grid.push_back(next_cuda_topic()); // next: occupancy
        grid.push_back(viewer_request());  // neighbours, 1 transaction
        subscribe();                       // lifetime benefit
    }

    std::cout << grid.size() << " topics queued, side by side like a coalesced read:\n";
    for (const Topic& t : grid) std::cout << "  " << t.name << '\n';
    return 0;
}
