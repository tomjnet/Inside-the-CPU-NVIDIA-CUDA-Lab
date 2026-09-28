#!/usr/bin/env python3
"""CUDA Shared Memory and Tiling in 9 Minutes - slide 12: thank you (Python version of 12_while_alive.cpp)

Run: python3 12_while_alive.py
"""
import sys
from collections import deque
from dataclasses import dataclass


# The imaginary tomjnet.h of the slide, so the loop really runs.
@dataclass
class Topic:
    name: str


TOPICS = ["CUDA Synchronization Without Deadlocks", "CUDA Streams and Asynchronous Copies",
          "Memory Coalescing in CUDA"]
episodes = 0


def alive() -> bool:
    return episodes < 3                          # three episodes, then the demo ends


def next_cuda_topic() -> Topic:
    global episodes
    topic = Topic(TOPICS[episodes])
    episodes += 1
    return topic


def viewer_request() -> Topic:
    return Topic(f"viewer request #{episodes}")


def subscribe() -> None:
    print("  subscribed to TomJNet")


def main() -> int:
    lab: deque[Topic] = deque()                  # std::queue: push at the back, pop at the front
    while alive():
        lab.append(next_cuda_topic())            # next: synchronization
        lab.append(viewer_request())             # barriers without deadlocks
        subscribe()                              # lifetime benefit

    print("queued topics, in order:")
    while lab:
        print(f"  {lab.popleft().name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
