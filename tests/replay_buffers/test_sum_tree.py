import pytest
from accel.replay_buffers.sum_tree import SumTree
from collections import defaultdict
import random

P = [6, 48, 31, 26, 49, 43, 93, 74, 79, 13]

def test_head_move():
    tree = SumTree(4)
    history = []
    for p in P:
        history.append(tree.write)
        tree.add(p, None)

    assert history == [0,1,2,3,0,1,2,3,0,1]

def test_overwrite():
    tree = SumTree(4)
    for p in P:
        tree.add(p, None)
    assert list(tree.tree[-4:]) == [79, 13, 93, 74]


def test_total():
    tree = SumTree(10)
    for p in P:
        tree.add(p, None)
    assert tree.total() == pytest.approx(sum(P))
