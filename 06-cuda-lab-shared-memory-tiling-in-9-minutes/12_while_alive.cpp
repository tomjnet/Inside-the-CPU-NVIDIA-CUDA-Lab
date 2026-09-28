// CUDA Shared Memory and Tiling in 9 Minutes - slide 12: thank you
// Build: make 12_while_alive
#include <iostream>
#include <queue>
#include <string>

// The imaginary tomjnet.h of the slide, so the loop really runs.
struct Topic { std::string name; };
static int episodes = 0;
static bool alive() { return episodes < 3; }                 // three episodes, then the demo ends
static Topic next_cuda_topic() {
    static const char* topics[] = {"CUDA Synchronization Without Deadlocks", "CUDA Streams and Asynchronous Copies",
                                   "Memory Coalescing in CUDA"};
    return Topic{topics[episodes++]};
}
static Topic viewer_request() { return Topic{"viewer request #" + std::to_string(episodes)}; }
static void subscribe() { std::cout << "  subscribed to TomJNet\n"; }

int main() {
    std::queue<Topic> lab;
    while (alive()) {
        lab.push(next_cuda_topic());     // next: synchronization
        lab.push(viewer_request());      // barriers without deadlocks
        subscribe();                     // lifetime benefit
    }

    std::cout << "queued topics, in order:\n";
    while (!lab.empty()) {
        std::cout << "  " << lab.front().name << '\n';
        lab.pop();
    }
    return 0;
}
