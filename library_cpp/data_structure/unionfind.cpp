#include<bits/stdc++.h>


class UnionFind {
public:
    int n;
    std::vector<int> parents;

    UnionFind(int n) {
        this->n = n;
        parents = std::vector<int>(n, -1);
    }

    bool unite(int x, int y) {
        x = find(x);
        y = find(y);
        if (x != y) {
            if (parents[x] > parents[y]) {
                std::swap(x, y);
            }
            parents[x] += parents[y];
            parents[y] = x;
            return true;
        }
        return false;
    }

    int find(int x) {
        int y = x;
        while (parents[x] >= 0) {
            x = parents[x];
        }
        while (y != x) {
            int t = y;
            y = parents[y];
            parents[t] = x;
        }
        return x;
    }

    int size(int x) {
        return -parents[find(x)];
    }

    bool same(int x, int y) {
        return find(x) == find(y);
    }
};
