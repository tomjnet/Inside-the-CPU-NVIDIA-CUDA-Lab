// Profile CUDA with NVIDIA Nsight in 10 Minutes - slide 13: thank you
// Build: make 13_while_alive
#include <iostream>
#include <queue>
#include <string>

// the imaginary tomjnet.h of the slide
struct Topic { std::string name; };
static int episodes = 0;
static bool alive() { return episodes < 3; }                 // three episodes, then the demo ends
static Topic next_cuda_topic() {
    static const char* topics[] = {"CPU vs GPU: an honest benchmark", "the break even size, measured",
                                   "back to the Low Latency C++ Lab"};
    return Topic{topics[episodes++]};
}
static Topic viewer_request() { return Topic{"viewer request #" + std::to_string(episodes)}; }
static void subscribe() { std::cout << "  subscribed to TomJNet\n"; }

int main() {
    std::queue<Topic> todo;
    while (alive()) {
        todo.push(next_cuda_topic());    // next: CPU vs GPU, honestly
        todo.push(viewer_request());     // profiled before optimized
        subscribe();                     // no gaps on this timeline
    }

    std::cout << "queued " << todo.size() << " topics, in order:\n";
    while (!todo.empty()) {
        std::cout << "  " << todo.front().name << '\n';
        todo.pop();
    }
    return 0;
}
