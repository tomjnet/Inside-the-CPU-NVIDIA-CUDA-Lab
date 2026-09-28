#!/usr/bin/env python3
"""CUDA Unified Memory in 7 Minutes - slide 9: thank you (Python version of 09_while_alive.cpp)

Run: python3 09_while_alive.py
"""
import sys
from dataclasses import dataclass


# the imaginary tomjnet.h of the slide
@dataclass
class Topic:
    name: str


TOPICS = ["shared memory and tiling", "synchronization without deadlocks", "streams and asynchronous copies"]
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
    pages: list[Topic] = []
    while alive():
        pages.append(next_cuda_topic())          # next: shared memory
        pages.append(viewer_request())           # migrates on first touch
        subscribe()                              # lifetime benefit

    print("queued topics:")
    for t in pages:
        print(f"  {t.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
