import random


def probably_sorted(a, epsilon=0.01):
    if not a:
        return True

    for i in range(int(epsilon ** -1) + 1):
        index = random.randrange(len(a))
        lo, hi = 0, len(a)
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] + mid < a[index] + index:
                lo = mid + 1
            else:
                hi = mid
        bisect_index = lo
        if bisect_index >= len(a) or bisect_index != index:
            return False

    return True
