// CUDA Grids, Blocks and Threads in 8 Minutes - slide 11: thank you
// Build: make 11_while_alive
#include <iostream>
#include <string>
#include <vector>

// the imaginary tomjnet.h of the slide
struct Topic { std::string name; };
static int episodes = 0;
static bool alive() { return episodes < 3; }                 // three episodes, then the demo ends
static Topic next_cuda_topic() {
    static const char* topics[] = {"CUDA host and device memory", "CUDA unified memory", "CUDA shared memory and tiling"};
    return Topic{topics[episodes++]};
}
static Topic viewer_request() { return Topic{"viewer request #" + std::to_string(episodes)}; }
static void subscribe() { std::cout << "  subscribed to TomJNet\n"; }

int main() {
    std::vector<Topic> blocks;
    while (alive()) {
        blocks.push_back(next_cuda_topic()); // next: device memory
        blocks.push_back(viewer_request());  // one more thread of work
        subscribe();                         // lifetime benefit
    }

    std::cout << "topics queued: " << blocks.size() << "\n";
    for (const Topic& t : blocks) std::cout << "  " << t.name << "\n";
    return 0;
}
