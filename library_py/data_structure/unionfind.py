class UnionFind:
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

    def unite(self, x, y):
        x, y = self.find(x), self.find(y)
        if x != y:
            if self.parents[x] > self.parents[y]:
                x, y = y, x
            self.parents[x] += self.parents[y]
            self.parents[y] = x
            return True

        return False

    def find(self, x):
        y = x
        while self.parents[x] >= 0:
            x = self.parents[x]
        while y != x:
            t = y
            y = self.parents[y]
            self.parents[t] = x
        return x

    def size(self, x):
        return -self.parents[self.find(x)]

    def same(self, x, y):
        return self.find(x) == self.find(y)
