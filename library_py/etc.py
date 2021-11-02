# base_10_to_n(123456, 25) -> [7, 22, 13, 6]
def base_10_to_n(val, base):
    res = []
    if val == 0:
        return [0]
    while val:
        res += [val % base]
        val //= base
    return res[::-1]


# base_n_to_10([3, 1, 4], 5) -> 84
def base_n_to_10(val, base):
    res = 0
    for i in val:
        res *= base
        res += i
    return res
