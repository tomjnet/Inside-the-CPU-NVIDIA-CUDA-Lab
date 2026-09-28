#!/usr/bin/env python3
"""CUDA Grids, Blocks and Threads in 8 Minutes - slide 11: thank you (Python version of 11_while_alive.cpp)

Run: python3 11_while_alive.py
"""
import sys
from dataclasses import dataclass


# the imaginary tomjnet.h of the slide
@dataclass
class Topic:
    name: str


TOPICS = ["CUDA host and device memory", "CUDA unified memory", "CUDA shared memory and tiling"]
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
    blocks: list[Topic] = []
    while alive():
        blocks.append(next_cuda_topic())         # next: device memory
        blocks.append(viewer_request())          # one more thread of work
        subscribe()                              # lifetime benefit

    print(f"topics queued: {len(blocks)}")
    for t in blocks:
        print(f"  {t.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
