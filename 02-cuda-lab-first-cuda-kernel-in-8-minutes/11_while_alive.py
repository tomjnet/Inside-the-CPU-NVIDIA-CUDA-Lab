#!/usr/bin/env python3
"""Your First CUDA Kernel in 8 Minutes - slide 11: thank you (Python version of 11_while_alive.cpp)

Run: python3 11_while_alive.py
"""
import sys
from dataclasses import dataclass


# the imaginary tomjnet.h of the slide
@dataclass
class Topic:
    name: str


TOPICS = ["CUDA Grids, Blocks and Threads", "CUDA Host and Device Memory", "CUDA Unified Memory"]
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
    grid: list[Topic] = []
    while alive():
        grid.append(next_cuda_topic())           # next: grids and blocks
        grid.append(viewer_request())            # one more thread of work
        subscribe()                              # lifetime benefit

    print(f"{len(grid)} topics queued:")
    for t in grid:
        print(f"  {t.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
