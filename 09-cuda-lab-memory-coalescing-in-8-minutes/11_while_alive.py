#!/usr/bin/env python3
"""Memory Coalescing in CUDA in 8 Minutes - slide 11: thank you (Python version of 11_while_alive.cpp)

Run: python3 11_while_alive.py
"""
import sys
from dataclasses import dataclass


# The imaginary tomjnet.h of the slide, so the loop really runs.
@dataclass
class Topic:
    name: str


TOPICS = ["CUDA Occupancy and Launch Configuration", "Profile CUDA with NVIDIA Nsight",
          "CPU vs GPU: An Honest Benchmark"]
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
    grid: list[Topic] = []                       # std::vector: contiguous, side by side
    while alive():
        grid.append(next_cuda_topic())           # next: occupancy
        grid.append(viewer_request())            # neighbours, 1 transaction
        subscribe()                              # lifetime benefit

    print(f"{len(grid)} topics queued, side by side like a coalesced read:")
    for t in grid:
        print(f"  {t.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
