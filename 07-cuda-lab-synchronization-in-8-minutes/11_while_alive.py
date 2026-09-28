#!/usr/bin/env python3
"""CUDA Synchronization Without Deadlocks in 8 Minutes - slide 11: thank you (Python version of 11_while_alive.cpp)

Run: python3 11_while_alive.py
"""
import sys
from dataclasses import dataclass


# The imaginary tomjnet.h of the slide, so the loop really runs.
@dataclass
class Topic:
    name: str


TOPICS = ["CUDA Streams and Asynchronous Copies", "Memory Coalescing in CUDA",
          "CUDA Occupancy and Launch Configuration"]
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
    stream: list[Topic] = []
    while alive():
        stream.append(next_cuda_topic())         # next: CUDA streams
        stream.append(viewer_request())          # queued in order
        subscribe()                              # lifetime benefit

    print(f"{len(stream)} topics queued, in order:")
    for t in stream:
        print(f"  {t.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
