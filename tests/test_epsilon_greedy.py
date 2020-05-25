from accel.explorers import epsilon_greedy
import pytest
import math


def test_const_eps_calc():
    explorer = epsilon_greedy.ConstantEpsilonGreedy(eps=0.1)

    assert explorer.calc_eps(0) == explorer.calc_eps(10000) == 0.1


def test_linear_decay_eps_calc():
    decay_steps = 10000

    explorer = epsilon_greedy.LinearDecayEpsilonGreedy(
        start_eps=1.0, end_eps=0.0, decay_steps=decay_steps)

    assert explorer.calc_eps(0) == pytest.approx(1.0)
    assert explorer.calc_eps(decay_steps/2) == pytest.approx(0.5)
    assert explorer.calc_eps(decay_steps * 2) == pytest.approx(0)


def test_exp_decay_eps_calc():
    decay_steps = 10000

    explorer = epsilon_greedy.ExpDecayEpsilonGreedy(
        start_eps=1.0, end_eps=0.5, decay=decay_steps)

    assert explorer.calc_eps(0) == pytest.approx(1.0)
    assert explorer.calc_eps(decay_steps) == pytest.approx(0.5 + 0.5 / math.e)
    assert explorer.calc_eps(decay_steps ** 2) >= 0.5
