"""
Order statistics: select the k-th smallest element (1-indexed) of an array.

  - deterministic_select : Median of Medians (BFPRT), worst-case O(n)
  - randomized_select    : Randomized Quickselect,    expected O(n)

"""

import random
INSERTION_CUTOFF = 5   # groups of 5 (classic BFPRT)
SMALL_N = 32           # below this, sorting a tiny sub-array is cheaper


def _check(a, k):
    n = len(a)
    if n == 0:
        raise ValueError("cannot select from an empty array")
    if not isinstance(k, int) or not 1 <= k <= n:
        raise ValueError(f"k must be an integer in [1, {n}], got {k!r}")

def _partition3(a, lo, hi, pivot):
    """Partition of a[lo..hi] (inclusive) around pivot.

    Afterwards: a[lo..lt-1] < pivot, a[lt..gt] == pivot, a[gt+1..hi] > pivot.
    Returns (lt, gt). One pass, O(hi - lo + 1) time, O(1) extra space.
    """
    # Three pointers sweep the range:
    #   a[lo..lt-1]  already known to be < pivot
    #   a[lt..i-1]   already known to be == pivot
    #   a[i..gt]     not examined yet
    #   a[gt+1..hi]  already known to be > pivot
    lt, i, gt = lo, lo, hi
    while i <= gt:
        x = a[i]
        if x < pivot:
            # Swap x into the "<" region; the element swapped back is an
            # "==" element (or x itself), so i can safely move on.
            a[lt], a[i] = a[i], a[lt]
            lt += 1
            i += 1
        elif x > pivot:
            # Swap x to the ">" region. The element swapped in from gt has
            # not been examined yet, so i stays put and checks it next.
            a[i], a[gt] = a[gt], a[i]
            gt -= 1
        else:
            # Equal to the pivot: leave it in the middle region.
            i += 1
    return lt, gt


def _insertion_sort(a, lo, hi):
    """Sort a[lo..hi] in place. Quadratic in general, but very fast for the
    tiny ranges it is used on (groups of 5, ranges under SMALL_N)."""
    for i in range(lo + 1, hi + 1):
        # Take a[i] and slide larger elements one slot right until the
        # correct position for it opens up.
        x = a[i]
        j = i - 1
        while j >= lo and a[j] > x:
            a[j + 1] = a[j]         # shift the element one position to the right
            j -= 1
        a[j + 1] = x


# --------------------------------------------------------------------------
# Randomized Quickselect
# --------------------------------------------------------------------------
def randomized_select(arr, k, rng=random):
    """Return the k-th smallest (1-indexed) element. Expected O(n) time."""
    _check(arr, k)
    a = list(arr)                 # work on a copy
    # `target` is an absolute 0-based position: once the whole array were
    # sorted, the answer would sit at a[target]. The search window a[lo..hi]
    # always contains that position and shrinks every round.
    lo, hi, target = 0, len(a) - 1, k - 1
    while True:
        # Small window: sorting it directly is cheaper than more partitions.
        if hi - lo < SMALL_N:
            _insertion_sort(a, lo, hi)
            return a[target]
        # A uniformly random pivot is what makes the expected time O(n):
        # no fixed input ordering can force bad pivots.
        pivot = a[rng.randint(lo, hi)]
        lt, gt = _partition3(a, lo, hi, pivot)
        # After partitioning, everything left of lt is smaller and everything
        # right of gt is larger, so keep only the side that holds `target`.
        if target < lt:
            hi = lt - 1
        elif target > gt:
            lo = gt + 1
        else:                                    # target falls in the == block
            return pivot


# --------------------------------------------------------------------------
# Deterministic Median of Medians (BFPRT)
# --------------------------------------------------------------------------
def _median_of_medians(a, lo, hi):
    """Return a pivot guaranteed to lie between the 30th and 70th percentile
    of a[lo..hi]. Rearranges a[lo..hi] in place (allowed: we own the copy)."""
    n = hi - lo + 1
    # Five or fewer elements: just sort them and take the middle one.
    if n <= INSERTION_CUTOFF:
        _insertion_sort(a, lo, hi)
        return a[lo + (n - 1) // 2]

    # 1. Sort each group of 5 and move its median to the front of the range.
    #    Collecting the medians in a[lo..m-1] (instead of a new list) keeps
    #    the extra memory at O(1) per call. The swap is safe because slot m
    #    is never ahead of the group currently being processed.
    m = lo
    for g in range(lo, hi + 1, 5):
        g_hi = min(g + 4, hi)
        _insertion_sort(a, g, g_hi)
        mid = g + (g_hi - g) // 2
        a[m], a[mid] = a[mid], a[m]
        m += 1
    # 2. Recursively select the median of the ceil(n/5) medians a[lo..m-1].
    #    This is the only recursion in the algorithm. Because it runs on a
    #    fifth of the range, the call stack stays O(log n) deep.
    #    The resulting pivot is guaranteed to have at least 3n/10 - 6
    #    elements on each side, which is what makes the worst case O(n).
    num = m - lo
    return _select_in_place(a, lo, m - 1, lo + (num - 1) // 2)


def _select_in_place(a, lo, hi, target):
    """Place the element of rank `target` (0-indexed, absolute) and return it,
    using median-of-medians pivots. Worst-case O(hi - lo + 1)."""
    # Same loop shape as randomized_select; only the pivot rule differs.
    while True:
        if hi - lo < SMALL_N:
            _insertion_sort(a, lo, hi)
            return a[target]
        # Deterministic "good" pivot: each round discards at least ~30% of
        # the window, whatever the input looks like.
        pivot = _median_of_medians(a, lo, hi)
        lt, gt = _partition3(a, lo, hi, pivot)
        if target < lt:
            hi = lt - 1
        elif target > gt:
            lo = gt + 1
        else:
            return pivot


def deterministic_select(arr, k):
    """Return the k-th smallest (1-indexed) element. Worst-case O(n) time."""
    _check(arr, k)
    a = list(arr)
    return _select_in_place(a, 0, len(a) - 1, k - 1)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    rnd = random.Random(0)
    cases = [
        [7], [2, 1], [5, 5, 5, 5, 5], [3, 1, 2, 3, 1, 2, 3],
        list(range(100)), list(range(100, 0, -1)),
        [rnd.randint(0, 3) for _ in range(500)],          # heavy duplicates
        [rnd.random() for _ in range(1000)],
        [-5, 0, -5, 10, 3.5, -1e9, 1e9],
    ]
    for _ in range(200):                                    # random fuzzing
        n = rnd.randint(1, 400)
        cases.append([rnd.randint(-50, 50) for _ in range(n)])

    # Reference answer: the k-th smallest is simply sorted(arr)[k - 1].
    # Every k is checked, so the extremes (min, max) and the median are covered.
    for arr in cases:
        s = sorted(arr)
        for k in range(1, len(arr) + 1):
            assert deterministic_select(arr, k) == s[k - 1], (arr, k)
            assert randomized_select(arr, k) == s[k - 1], (arr, k)
    # Invalid requests must raise ValueError rather than return garbage.
    for bad in [([], 1), ([1, 2], 0), ([1, 2], 3)]:
        for f in (deterministic_select, randomized_select):
            try:
                f(*bad)
                raise AssertionError("expected ValueError")
            except ValueError:
                pass
    print(f"All tests passed ({len(cases)} arrays, every k).")