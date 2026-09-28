// Your First CUDA Kernel in 8 Minutes - slide 11: thank you
// Build: make 11_while_alive
#include <iostream>
#include <string>
#include <vector>

// the imaginary tomjnet.h of the slide
struct Topic { std::string name; };
static int episodes = 0;
static bool alive() { return episodes < 3; }                 // three episodes, then the demo ends
static Topic next_cuda_topic() {
    static const char* topics[] = {"CUDA Grids, Blocks and Threads", "CUDA Host and Device Memory",
                                   "CUDA Unified Memory"};
    return Topic{topics[episodes++]};
}
static Topic viewer_request() { return Topic{"viewer request #" + std::to_string(episodes)}; }
static void subscribe() { std::cout << "  subscribed to TomJNet\n"; }

int main() {
    std::vector<Topic> grid;
    while (alive()) {
        grid.push_back(next_cuda_topic()); // next: grids and blocks
        grid.push_back(viewer_request());  // one more thread of work
        subscribe();                       // lifetime benefit
    }

    std::cout << grid.size() << " topics queued:\n";
    for (const Topic& t : grid) std::cout << "  " << t.name << '\n';
    return 0;
}
