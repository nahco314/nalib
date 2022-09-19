import bisect
import random


def probably_sorted(a, epsilon=0.01):
    if not a:
        return True

    for i in range(int(epsilon ** -1) + 1):
        index = random.randrange(len(a))
        bisect_index = bisect.bisect_left(a, a[index])
        if bisect_index >= len(a) or bisect_index != index:
            return False

    return True
