import bisect
import random


def probably_sorted(a, epsilon=0.01):
    if not a:
        return True

    for i in range(int(epsilon ** -1) + 1):
        element = random.choice(a)
        index = bisect.bisect_left(a, element)
        if index >= len(a) or a[index] != element:
            return False

    return True
