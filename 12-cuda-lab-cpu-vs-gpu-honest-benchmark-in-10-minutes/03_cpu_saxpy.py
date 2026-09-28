#!/usr/bin/env python3
"""CPU vs GPU: An Honest Benchmark in 10 Minutes - slide 3: the cpu baseline, no strawman (Python version of 03_cpu_saxpy.cpp)

Run: python3 03_cpu_saxpy.py   (needs pip install numpy; without it the program says why and exits 0)
"""
import statistics
import sys
import threading
import time

try:
    import numpy as np
except ImportError as e:                       # samples exit 0 without the libraries
    print(f"needs numpy: {e.name} is missing (pip install numpy)")
    sys.exit(0)


# the harness of the Low Latency C++ Lab
_sink = 0


def sink(x: int) -> None:
    """Keeps a result alive, the twin of sink() in the Low Latency C++ Lab harness."""
    global _sink
    _sink = (_sink + x) & 0xFFFFFFFFFFFFFFFF


def bench_ns(f, warmup: int = 3, repeats: int = 21) -> tuple[float, float]:
    """Warm up, repeat, keep the minimum and the median in nanoseconds."""
    for _ in range(warmup):
        f()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        f()
        samples.append(time.perf_counter_ns() - t0)
    return float(min(samples)), float(statistics.median_low(samples))


def uniform_inputs(n: int, seed: int = 42) -> tuple["np.ndarray", "np.ndarray"]:
    """std::mt19937(seed) with std::uniform_real_distribution<float>(-1, 1), drawn x[i] then y0[i] as the C++ loop
    does. numpy's legacy RandomState(seed) is the same MT19937 stream; libstdc++ turns one 32 bit draw u into
    u / 2^32 in float (kept below 1), then r * 2 - 1, all in float32 as here."""
    v = np.random.RandomState(seed).randint(0, 1 << 32, size=2 * n, dtype=np.uint32).astype(np.float32)
    v /= np.float32(1 << 32)
    np.minimum(v, np.nextafter(np.float32(1), np.float32(0)), out=v)
    v *= np.float32(2)
    v += np.float32(-1)
    return v[0::2].copy(), v[1::2].copy()


# y = a * x + y: 2 flops and 12 bytes of memory per element
def saxpy_cpu(a: "np.float32", x: "np.ndarray", y: "np.ndarray") -> None:
    y += a * x                                 # same math as the kernel: float multiply, float add
# The C++ baseline is a plain loop built with -O2. A numpy CPU loop is not the C++ -O2 loop: numpy runs its own
# compiled (and vectorized) loop and writes the product to a temporary array first, so it moves a little more
# memory. Same inputs, same result check, same kind of clock as the GPU side; one core here, threads below.


def saxpy_threads(a: "np.float32", x: "np.ndarray", y: "np.ndarray", threads: int) -> None:
    n = len(x)
    chunk = (n + threads - 1) // threads       # one contiguous slice per thread
    pool = []
    for t in range(threads):
        begin = min(n, t * chunk)
        end = min(n, begin + chunk)
        # numpy releases the GIL inside the array operation, so the four slices really run in parallel
        th = threading.Thread(target=saxpy_cpu, args=(a, x[begin:end], y[begin:end]))
        th.start()
        pool.append(th)
    for th in pool:                            # thread start and join are on the clock too
        th.join()


def main() -> int:
    n = 1 << 22                                # 4 M floats: 16 MiB per vector
    x, y0 = uniform_inputs(n)                  # fixed seed: same inputs on every toolchain
    y = y0.copy()
    a = np.float32(2.0)

    def run_one() -> None:
        saxpy_cpu(a, x, y)
        sink(int(y[n // 2] > 0.0))

    def run_four() -> None:
        saxpy_threads(a, x, y, 4)
        sink(int(y[n // 3] > 0.0))

    one = bench_ns(run_one)
    four = bench_ns(run_four)

    # the check: fresh inputs, one run each way, compared with a double precision reference
    y1 = y0.copy()
    y4 = y0.copy()
    saxpy_cpu(a, x, y1)
    saxpy_threads(a, x, y4, 4)
    ref = 2.0 * x.astype(np.float64) + y0.astype(np.float64)
    tol = 1e-6 * np.abs(ref) + 1e-6
    bad = (np.abs(y1.astype(np.float64) - ref) > tol) | (y4 != y1)
    wrong = int(np.count_nonzero(bad))

    nbytes = 12.0 * n                          # read x, read y, write y
    print(f"saxpy on the CPU, {n} floats, perf_counter_ns, minimum and median of 21 runs after 3 warm ups")
    print(f"{'threads':<10} {'min us':>12} {'median us':>12} {'GB/s':>10}")
    print(f"{1:<10d} {one[0] / 1e3:12.1f} {one[1] / 1e3:12.1f} {nbytes / one[1]:10.1f}")
    print(f"{4:<10d} {four[0] / 1e3:12.1f} {four[1] / 1e3:12.1f} {nbytes / four[1]:10.1f}")
    print(f"4 threads vs 1: {one[1] / four[1]:.2f}x on this machine; memory bandwidth, not cores, sets the ceiling")
    print(f"elements outside the tolerance: {wrong} of {n}")
    print("PASS" if wrong == 0 else "FAIL")    # the verdict is the check, never a timing
    return 0 if wrong == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
