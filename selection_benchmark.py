"""Empirical comparison of deterministic vs randomized selection.

AI has been used to write parts of this code. I have reviewed and modified it as needed.
"""
import csv
import random
import statistics
import sys
import time

from selection_algorithms import deterministic_select, randomized_select, _partition3

sys.setrecursionlimit(10000)

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
REPS = 7


def gen(dist, n, rng):
    if dist == "random":
        return [rng.random() for _ in range(n)]
    if dist == "sorted":
        return list(range(n))
    if dist == "reverse":
        return list(range(n, 0, -1))
    if dist == "few_unique":            # heavy duplicates: 10 distinct values
        return [rng.randint(0, 9) for _ in range(n)]
    raise ValueError(dist)


def naive_quickselect(arr, k):
    """Quickselect with the FIRST element as pivot (for contrast only)."""
    a, lo, hi, t = list(arr), 0, len(arr) - 1, k - 1
    while lo < hi:
        lt, gt = _partition3(a, lo, hi, a[lo])
        if t < lt:
            hi = lt - 1
        elif t > gt:
            lo = gt + 1
        else:
            return a[t]
    return a[t]


def time_it(f, arr, k):
    t0 = time.perf_counter()
    f(arr, k)
    return time.perf_counter() - t0


def main():
    rng = random.Random(42)
    rows = []
    algos = [("deterministic", deterministic_select),
             ("randomized", randomized_select)]
    for dist in ["random", "sorted", "reverse", "few_unique"]:
        for n in SIZES:
            k = (n + 1) // 2
            times = {name: [] for name, _ in algos}
            for _ in range(REPS):
                arr = gen(dist, n, rng)
                for name, f in algos:
                    times[name].append(time_it(f, arr, k))
            for name, _ in algos:
                ts = times[name]
                rows.append(dict(algorithm=name, distribution=dist, n=n,
                                 median_ms=statistics.median(ts) * 1e3,
                                 min_ms=min(ts) * 1e3, max_ms=max(ts) * 1e3,
                                 ns_per_elem=statistics.median(ts) / n * 1e9))
                print(f"{dist:10s} n={n:>9,} {name:13s} "
                      f"{statistics.median(ts)*1e3:9.2f} ms", flush=True)

    # Contrast: naive first-element-pivot quickselect on sorted input.
    for n in [500, 1_000, 2_000, 4_000, 8_000]:
        arr = list(range(n))
        ts = [time_it(naive_quickselect, arr, (n + 1) // 2) for _ in range(3)]
        rows.append(dict(algorithm="naive_first_pivot", distribution="sorted",
                         n=n, median_ms=statistics.median(ts) * 1e3,
                         min_ms=min(ts) * 1e3, max_ms=max(ts) * 1e3,
                         ns_per_elem=statistics.median(ts) / n * 1e9))
        print(f"naive sorted n={n:>6} {statistics.median(ts)*1e3:9.2f} ms")

    # Variance of randomized quickselect on one fixed random input.
    arr = gen("random", 100_000, rng)
    rt = [time_it(randomized_select, arr, 50_000) * 1e3 for _ in range(200)]
    dt = [time_it(deterministic_select, arr, 50_000) * 1e3 for _ in range(30)]
    with open("variance.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["algorithm", "ms"])
        w.writerows([("randomized", x) for x in rt])
        w.writerows([("deterministic", x) for x in dt])
    q = statistics.quantiles(rt, n=100)
    print(f"randomized n=1e5 x200: mean {statistics.mean(rt):.2f} "
          f"sd {statistics.stdev(rt):.2f} p1 {q[0]:.2f} p99 {q[98]:.2f} max {max(rt):.2f}")
    print(f"deterministic n=1e5 x30: mean {statistics.mean(dt):.2f} "
          f"sd {statistics.stdev(dt):.2f}")

    with open("results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()