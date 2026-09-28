#!/usr/bin/env python3
"""Profile CUDA with NVIDIA Nsight in 10 Minutes - slide 13: thank you (Python version of 13_while_alive.cpp)

Run: python3 13_while_alive.py
"""
import sys
from collections import deque
from dataclasses import dataclass


# The imaginary tomjnet.h of the slide, so the loop really runs.
@dataclass
class Topic:
    name: str


TOPICS = ["CPU vs GPU: an honest benchmark", "the break even size, measured", "back to the Low Latency C++ Lab"]
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
    todo: deque[Topic] = deque()
    while alive():
        todo.append(next_cuda_topic())           # next: CPU vs GPU, honestly
        todo.append(viewer_request())            # profiled before optimized
        subscribe()                              # no gaps on this timeline

    print(f"queued {len(todo)} topics, in order:")
    while todo:
        print(f"  {todo.popleft().name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
