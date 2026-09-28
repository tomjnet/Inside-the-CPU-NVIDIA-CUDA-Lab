#!/usr/bin/env python3
"""Install and Verify CUDA in 7 Minutes: Driver, Toolkit and Runtime - slide 9: thank you (Python version of 09_while_alive.cpp)

Run: python3 09_while_alive.py
"""
import sys
from collections import deque
from dataclasses import dataclass


# the imaginary tomjnet.h of the slide
@dataclass
class Topic:
    name: str


TOPICS = ["your first CUDA kernel", "grids, blocks and threads", "host and device memory"]
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
    todo: deque[Topic] = deque()                 # the std::queue of the slide
    while alive():
        todo.append(next_cuda_topic())           # next: your first CUDA kernel
        todo.append(viewer_request())            # checked like a runtime call
        subscribe()                              # compatible with every driver

    print(f"queued {len(todo)} topics, in order:")
    while todo:
        print(f"  {todo.popleft().name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
