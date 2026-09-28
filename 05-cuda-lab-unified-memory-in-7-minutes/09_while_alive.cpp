// CUDA Unified Memory in 7 Minutes - slide 9: thank you
// Build: make 09_while_alive
#include <iostream>
#include <string>
#include <vector>

// the imaginary tomjnet.h of the slide
struct Topic { std::string name; };
static int episodes = 0;
static bool alive() { return episodes < 3; }                 // three episodes, then the demo ends
static Topic next_cuda_topic() {
    static const char* topics[] = {"shared memory and tiling", "synchronization without deadlocks",
                                   "streams and asynchronous copies"};
    return Topic{topics[episodes++]};
}
static Topic viewer_request() { return Topic{"viewer request #" + std::to_string(episodes)}; }
static void subscribe() { std::cout << "  subscribed to TomJNet\n"; }

int main() {
    std::vector<Topic> pages;
    while (alive()) {
        pages.push_back(next_cuda_topic()); // next: shared memory
        pages.push_back(viewer_request());  // migrates on first touch
        subscribe();                        // lifetime benefit
    }

    std::cout << "queued topics:\n";
    for (const Topic& t : pages) std::cout << "  " << t.name << '\n';
    return 0;
}
