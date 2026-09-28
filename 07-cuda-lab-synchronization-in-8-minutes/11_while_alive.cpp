// CUDA Synchronization Without Deadlocks in 8 Minutes - slide 11: thank you
// Build: make 11_while_alive
#include <iostream>
#include <string>
#include <vector>

// the imaginary tomjnet.h of the slide
struct Topic { std::string name; };
static int episodes = 0;
static bool alive() { return episodes < 3; }                 // three episodes, then the demo ends
static Topic next_cuda_topic() {
    static const char* topics[] = {"CUDA Streams and Asynchronous Copies", "Memory Coalescing in CUDA",
                                   "CUDA Occupancy and Launch Configuration"};
    return Topic{topics[episodes++]};
}
static Topic viewer_request() { return Topic{"viewer request #" + std::to_string(episodes)}; }
static void subscribe() { std::cout << "  subscribed to TomJNet\n"; }

int main() {
    std::vector<Topic> stream;
    while (alive()) {
        stream.push_back(next_cuda_topic()); // next: CUDA streams
        stream.push_back(viewer_request());  // queued in order
        subscribe();                         // lifetime benefit
    }

    std::cout << stream.size() << " topics queued, in order:\n";
    for (const Topic& t : stream) std::cout << "  " << t.name << '\n';
    return 0;
}
