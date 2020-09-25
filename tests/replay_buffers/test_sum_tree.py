import pytest
from accel.replay_buffers.sum_tree import SumTree
from collections import defaultdict
import random

P = [6, 48, 31, 26, 49, 43, 93, 74, 79, 13]

def test_overwrite():
    tree = SumTree(4)
    for i, p in enumerate(P):
        tree.add(p, i % 4)
    assert list(tree.tree[-4:]) == [79, 13, 93, 74]


def test_total():
    tree = SumTree(10)
    for i, p in enumerate(P):
        tree.add(p, i)
    assert tree.total() == pytest.approx(sum(P))

