// CUDA Streams and Asynchronous Copies in 10 Minutes - slide 13: thank you
// Build: make 13_while_alive
#include <iostream>
#include <queue>
#include <string>

// the imaginary tomjnet.h of the slide
struct Topic { std::string name; };
static int episodes = 0;
static bool alive() { return episodes < 3; }                 // three episodes, then the demo ends
static Topic next_cuda_topic() {
    static const char* topics[] = {"memory coalescing", "occupancy and launch configuration",
                                   "profile with NVIDIA Nsight"};
    return Topic{topics[episodes++]};
}
static Topic viewer_request() { return Topic{"viewer request #" + std::to_string(episodes)}; }
static void subscribe() { std::cout << "  subscribed to TomJNet\n"; }

int main() {
    std::queue<Topic> stream;            // an ordered queue: a stream
    while (alive()) {
        stream.push(next_cuda_topic());  // next: memory coalescing
        stream.push(viewer_request());   // queued, never blocking
        subscribe();                     // lifetime benefit
    }

    std::cout << "queued " << stream.size() << " topics, in order:\n";
    while (!stream.empty()) {
        std::cout << "  " << stream.front().name << '\n';
        stream.pop();
    }
    return 0;
}
