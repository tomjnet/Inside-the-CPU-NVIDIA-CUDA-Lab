#!/usr/bin/env python3
"""CPU vs GPU: An Honest Benchmark in 10 Minutes - slide 13: thank you (Python version of 13_while_alive.cpp)

Run: python3 13_while_alive.py
"""
import sys
from dataclasses import dataclass


# the imaginary tomjnet.h of the slide
@dataclass
class Topic:
    name: str


TOPICS = ["Low Latency C++ Lab: your first benchmark",
          "Low Latency C++ Lab: cache and memory latency",
          "Low Latency C++ Lab: a ring buffer"]
episodes = 0


def alive() -> bool:
    return episodes < 3                        # three topics, then the demo ends


def next_lab_topic() -> Topic:
    global episodes
    topic = Topic(TOPICS[episodes])
    episodes += 1
    return topic


def viewer_request() -> Topic:
    return Topic(f"viewer request #{episodes}")


def subscribe() -> None:
    print("  subscribed to TomJNet")


def main() -> int:
    results: list[Topic] = []
    while alive():
        results.append(next_lab_topic())       # Low Latency C++ Lab
        results.append(viewer_request())       # measured end to end
        subscribe()                            # lifetime benefit

    print(f"topics queued: {len(results)}")
    for t in results:
        print(f"  {t.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
