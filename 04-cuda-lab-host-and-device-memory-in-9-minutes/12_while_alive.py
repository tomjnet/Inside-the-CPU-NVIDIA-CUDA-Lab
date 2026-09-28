#!/usr/bin/env python3
"""CUDA Host and Device Memory in 9 Minutes - slide 12: thank you (Python version of 12_while_alive.cpp)

Run: python3 12_while_alive.py
"""
import sys
from dataclasses import dataclass


# The imaginary tomjnet.h of the slide, so the loop really runs.
@dataclass
class Topic:
    name: str


TOPICS = ["CUDA Unified Memory", "CUDA Shared Memory and Tiling", "CUDA Synchronization Without Deadlocks"]
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
    device_queue: list[Topic] = []
    while alive():
        device_queue.append(next_cuda_topic())   # unified memory
        device_queue.append(viewer_request())    # 1 copy, no sync
        subscribe()                              # lifetime benefit

    print("queued topics, in order:")
    for t in device_queue:
        print(f"  {t.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
