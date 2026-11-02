"""Timing experiments for the data structures in data_structures.py.
Each scenario runs n operations and reports the median of 5 runs in
nanoseconds per operation. Results go to results_ds.csv.

Some parts of this code have been written using AI. I have reviewed and modified them as needed."""
import csv
import random
import statistics
import time
import tracemalloc

from data_structures import (DynamicArray, ArrayStack, ArrayQueue, ShiftingArrayQueue,
                             SinglyLinkedList, LinkedStack, LinkedQueue)

SIZES = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000]
REPS = 5


def timed(fn):
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def stack_run(S, n):
    def run():
        s = S()
        for i in range(n): s.push(i)
        for _ in range(n): s.pop()
    return run


def queue_run(Q, n):
    def run():
        q = Q()
        for i in range(n): q.enqueue(i)
        for _ in range(n): q.dequeue()
    return run


def front_insert(kind, n):
    def run():
        if kind == "array":
            a = DynamicArray()
            for i in range(n): a.insert(0, i)
        else:
            l = SinglyLinkedList()
            for i in range(n): l.insert_front(i)
    return run


def random_access(kind, n):
    idx = [random.randrange(n) for _ in range(1000)]
    if kind == "array":
        a = DynamicArray()
        for i in range(n): a.append(i)
        return lambda: [a.get(i) for i in idx]
    l = SinglyLinkedList(range(n))
    return lambda: [l.get(i) for i in idx]


def memory_bytes(kind, n):
    tracemalloc.start()
    if kind == "array":
        a = DynamicArray()
        for i in range(n): a.append(i)
    else:
        a = SinglyLinkedList(range(n))
    cur, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return cur


def main():
    rows = []
    def add(scenario, name, n, secs, ops):
        ns = secs / ops * 1e9
        rows.append(dict(scenario=scenario, structure=name, n=n, ns_per_op=round(ns, 1)))
        print(f"{scenario:14s} {name:22s} n={n:>6} {ns:10.1f} ns/op", flush=True)

    for n in SIZES:
        for name, S in [("ArrayStack", ArrayStack), ("LinkedStack", LinkedStack)]:
            add("stack", name, n, timed(stack_run(S, n)), 2 * n)
        for name, Q in [("ArrayQueue (circular)", ArrayQueue), ("LinkedQueue", LinkedQueue),
                        ("ShiftingArrayQueue", ShiftingArrayQueue)]:
            add("queue", name, n, timed(queue_run(Q, n)), 2 * n)
        for kind, name in [("array", "DynamicArray"), ("list", "SinglyLinkedList")]:
            add("front_insert", name, n, timed(front_insert(kind, n)), n)
            add("random_access", name, n, timed(random_access(kind, n)), 1000)

    for kind, name in [("array", "DynamicArray"), ("list", "SinglyLinkedList")]:
        n = 100_000
        b = memory_bytes(kind, n)
        rows.append(dict(scenario="memory_bytes_per_elem", structure=name, n=n,
                         ns_per_op=round(b / n, 1)))
        print(f"memory {name}: {b / n:.1f} bytes per element (n={n})")

    with open("results_ds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    random.seed(0)
    main()