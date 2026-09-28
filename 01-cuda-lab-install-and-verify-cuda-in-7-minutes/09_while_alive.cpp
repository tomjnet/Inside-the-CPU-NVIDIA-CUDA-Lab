// Install and Verify CUDA in 7 Minutes: Driver, Toolkit and Runtime - slide 9: thank you
// Build: make 09_while_alive
#include <iostream>
#include <queue>
#include <string>

// the imaginary tomjnet.h of the slide
struct Topic { std::string name; };
static int episodes = 0;
static bool alive() { return episodes < 3; }                 // three episodes, then the demo ends
static Topic next_cuda_topic() {
    static const char* topics[] = {"your first CUDA kernel", "grids, blocks and threads", "host and device memory"};
    return Topic{topics[episodes++]};
}
static Topic viewer_request() { return Topic{"viewer request #" + std::to_string(episodes)}; }
static void subscribe() { std::cout << "  subscribed to TomJNet\n"; }

int main() {
    std::queue<Topic> todo;
    while (alive()) {
        todo.push(next_cuda_topic());    // next: your first CUDA kernel
        todo.push(viewer_request());     // checked like a runtime call
        subscribe();                     // compatible with every driver
    }

    std::cout << "queued " << todo.size() << " topics, in order:\n";
    while (!todo.empty()) {
        std::cout << "  " << todo.front().name << '\n';
        todo.pop();
    }
    return 0;
}
