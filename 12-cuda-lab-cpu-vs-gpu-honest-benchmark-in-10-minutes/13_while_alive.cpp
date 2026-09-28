// CPU vs GPU: An Honest Benchmark in 10 Minutes - slide 13: thank you
// Build: make 13_while_alive
#include <iostream>
#include <string>
#include <vector>

// the imaginary tomjnet.h of the slide
struct Topic { std::string name; };
static int episodes = 0;
static bool alive() { return episodes < 3; }                 // three topics, then the demo ends
static Topic next_lab_topic() {
    static const char* topics[] = {"Low Latency C++ Lab: your first benchmark",
                                   "Low Latency C++ Lab: cache and memory latency",
                                   "Low Latency C++ Lab: a ring buffer"};
    return Topic{topics[episodes++]};
}
static Topic viewer_request() { return Topic{"viewer request #" + std::to_string(episodes)}; }
static void subscribe() { std::cout << "  subscribed to TomJNet\n"; }

int main() {
    std::vector<Topic> results;
    while (alive()) {
        results.push_back(next_lab_topic());  // Low Latency C++ Lab
        results.push_back(viewer_request());  // measured end to end
        subscribe();                          // lifetime benefit
    }

    std::cout << "topics queued: " << results.size() << "\n";
    for (const Topic& t : results) std::cout << "  " << t.name << "\n";
    return 0;
}
