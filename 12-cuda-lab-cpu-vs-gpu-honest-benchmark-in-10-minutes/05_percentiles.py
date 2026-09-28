#!/usr/bin/env python3
"""CPU vs GPU: An Honest Benchmark in 10 Minutes - slide 5: warm up, repeat, median and p99 (Python version of 05_percentiles.cpp)

Run: python3 05_percentiles.py
"""
import sys
import time

_sink = 0


def sink(x: int) -> None:
    """Keeps a result alive, the twin of sink() in the Low Latency C++ Lab harness."""
    global _sink
    _sink = (_sink + x) & 0xFFFFFFFFFFFFFFFF


# samples in ns: median and p99 by nearest rank
def pct(s: list[float], p: float) -> float:
    s.sort()                                   # O(n log n)
    k = int(p / 100.0 * len(s))
    return s[min(k, len(s) - 1)]


def main() -> int:
    # 1. the rank rule on known data: 1, 2, ..., 201 in reverse order
    known = [float(i) for i in range(201, 0, -1)]   # any order: pct sorts
    m, q, top = pct(known, 50.0), pct(known, 99.0), pct(known, 100.0)
    print(f"known data 1 to 201: median {m:.0f}, p99 {q:.0f}, max {top:.0f}")
    print("p99 is the 3rd slowest of 201 runs (199), not the max (201)\n")
    ok = m == 101.0 and q == 199.0 and top == 201.0

    # 2. real samples: sum 64 K integers, 5 warm ups thrown away, then 201 timed runs
    data = [(i * 2654435761) & 0xFFFFFFFF for i in range(1 << 16)]   # std::uint32_t wraps: mask to 32 bits

    def work() -> None:
        s = 0
        for v in data:
            s += v
        sink(s)

    for _ in range(5):                         # warm up: caches, page faults, clocks
        work()
    samples: list[float] = []
    for _ in range(201):
        t0 = time.perf_counter_ns()
        work()
        samples.append(float(time.perf_counter_ns() - t0))
    # 201 runs: p99 is the 3rd slowest, not the max
    median = pct(samples, 50.0)
    p99 = pct(samples, 99.0)                   # the tail you ship
    # Python runs this loop far slower than the C++ -O2 build: compare the shape (median vs p99), not the ns
    print("sum of 64 K integers, 201 runs on this machine, perf_counter_ns")
    print(f"  min    {samples[0]:10.1f} ns")
    print(f"  median {median:10.1f} ns")
    print(f"  p99    {p99:10.1f} ns   ({p99 / median:.2f}x the median)")
    print(f"  max    {samples[-1]:10.1f} ns   (one run: a page fault, an interrupt, another process)")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
