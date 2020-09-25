import pytest
from accel.replay_buffers.sum_tree import SumTree, MinTree
from collections import defaultdict
import random

P = [6, 48, 31, 26, 49, 43, 93, 74, 79, 13]


def test_sum_tree_overwrite():
    tree = SumTree(4)
    for i, p in enumerate(P):
        tree.update(i % 4, p)
    assert list(tree.tree[-4:]) == [79, 13, 93, 74]


def test_sum_tree_top():
    tree = SumTree(10)
    for i, p in enumerate(P):
        tree.update(i, p)
    assert tree.top() == pytest.approx(sum(P))


def test_min_tree_overwrite():
    tree = MinTree(4)
    for i, p in enumerate(P):
        tree.update(i % 4, p)
    assert list(tree.tree[-4:]) == [79, 13, 93, 74]


def test_min_tree_top():
    tree = MinTree(10)
    for i, p in enumerate(P):
        tree.update(i, p)
    assert tree.top() == pytest.approx(min(P))

