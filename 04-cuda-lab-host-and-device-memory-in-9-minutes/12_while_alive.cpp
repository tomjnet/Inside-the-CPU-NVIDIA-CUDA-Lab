// CUDA Host and Device Memory in 9 Minutes - slide 12: thank you
// Build: make 12_while_alive
#include <iostream>
#include <string>
#include <vector>

// The imaginary tomjnet.h of the slide, so the loop really runs.
struct Topic { std::string name; };
static int episodes = 0;
static bool alive() { return episodes < 3; }                 // three episodes, then the demo ends
static Topic next_cuda_topic() {
    static const char* topics[] = {"CUDA Unified Memory", "CUDA Shared Memory and Tiling",
                                   "CUDA Synchronization Without Deadlocks"};
    return Topic{topics[episodes++]};
}
static Topic viewer_request() { return Topic{"viewer request #" + std::to_string(episodes)}; }
static void subscribe() { std::cout << "  subscribed to TomJNet\n"; }

int main() {
    std::vector<Topic> device_queue;
    while (alive()) {
        device_queue.push_back(next_cuda_topic());  // unified memory
        device_queue.push_back(viewer_request());   // 1 copy, no sync
        subscribe();                                // lifetime benefit
    }

    std::cout << "queued topics, in order:\n";
    for (const Topic& t : device_queue) std::cout << "  " << t.name << '\n';
    return 0;
}
