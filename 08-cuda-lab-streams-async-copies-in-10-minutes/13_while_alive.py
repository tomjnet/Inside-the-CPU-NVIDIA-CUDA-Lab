#!/usr/bin/env python3
"""CUDA Streams and Asynchronous Copies in 10 Minutes - slide 13: thank you (Python version of 13_while_alive.cpp)

Run: python3 13_while_alive.py
"""
import sys
from collections import deque
from dataclasses import dataclass


# The imaginary tomjnet.h of the slide, so the loop really runs.
@dataclass
class Topic:
    name: str


TOPICS = ["memory coalescing", "occupancy and launch configuration", "profile with NVIDIA Nsight"]
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
    stream: deque[Topic] = deque()               # an ordered queue: a stream
    while alive():
        stream.append(next_cuda_topic())         # next: memory coalescing
        stream.append(viewer_request())          # queued, never blocking
        subscribe()                              # lifetime benefit

    print(f"queued {len(stream)} topics, in order:")
    while stream:
        print(f"  {stream.popleft().name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
