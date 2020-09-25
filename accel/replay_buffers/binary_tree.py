import numpy as np


def plus(x, y):
    return x + y


class BinaryTree:
    def __init__(self, capacity, operator, initial_value):
        # Type of initial value decides dtype of self.tree
        self.capacity = capacity
        self.tree = np.full(2 * capacity - 1, initial_value)
        self.operator = operator

    def _propagate(self, idx):
        parent = (idx - 1) // 2
        left = 2 * parent + 1
        right = left + 1
        self.tree[parent] = self.operator(self.tree[left], self.tree[right])

        if parent != 0:
            self._propagate(parent)

    def top(self):
        return self.tree[0]

    def update(self, data_idx, p):
        idx = data_idx + self.capacity - 1

        self.tree[idx] = p
        self._propagate(idx)


class SumTree(BinaryTree):
    def __init__(self, capacity):
        super().__init__(capacity, plus, 0.)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return data_idx, self.tree[idx]


class MinTree(BinaryTree):
    def __init__(self, capacity):
        super().__init__(capacity, min, float('inf'))

