"""Reference models recover what they are built to recover, on synthetic grids."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from djembe.baselines import MarkovModel, PositionModel, histogram_intersection  # noqa: E402
from djembe.grid import TakeGrid  # noqa: E402


def synthetic(rng, n_cycles=400, rho=0.0, take="T_1"):
    """One instrument (J1) striking pulses 1, 4, 7, 10 with offset mean -0.1 * (k % 3) and sd 0.05."""
    x = np.zeros((n_cycles, 12, 4, 3), np.float32)
    present = np.array([True, False, False, False])
    r_prev = 0.0
    for c in range(n_cycles):
        for k in range(12):
            if k % 3 == 0 or (k % 3 == 1 and rng.random() < 0.5):
                r = rho * r_prev + rng.normal(0, 0.05)
                x[c, k, 0] = (1.0, rng.normal(0, 1), -0.1 * (k % 3) + r)
                r_prev = r
    return TakeGrid(take, "P", x, np.zeros(n_cycles, np.float32), np.zeros(n_cycles, int),
                    np.zeros(n_cycles, bool), present)


def test_position_model_recovers_rates_and_feel():
    rng = np.random.default_rng(0)
    m = PositionModel().fit([synthetic(rng)])
    k = np.arange(12)
    p = m.p_hit("P", 0, k)
    assert np.allclose(p[k % 3 == 0], 1.0, atol=0.01) and np.allclose(p[k % 3 == 1], 0.5, atol=0.05)
    assert np.allclose(p[k % 3 == 2], 0.0, atol=0.01)
    mu, sd = m.mu_sd("d", "P", 0, np.array([0, 1]))
    assert np.allclose(mu, [0.0, -0.1], atol=0.01) and np.allclose(sd, 0.05, atol=0.01)


def test_markov_model_recovers_the_timing_memory_and_beats_b1_on_it():
    rng = np.random.default_rng(1)
    train, test = [synthetic(rng, rho=0.7)], synthetic(rng, rho=0.7, take="T_2")
    b2 = MarkovModel().fit(train)
    assert abs(b2.rho[("d", "P", 0)] - 0.7) < 0.05
    b1 = PositionModel().fit(train)
    assert b2.nll(test)["delta"].mean() < b1.nll(test)["delta"].mean()


def test_markov_model_gains_nothing_without_memory():
    rng = np.random.default_rng(2)
    train, test = [synthetic(rng, rho=0.0)], synthetic(rng, rho=0.0, take="T_2")
    b2, b1 = MarkovModel().fit(train), PositionModel().fit(train)
    assert abs(b2.nll(test)["delta"].mean() - b1.nll(test)["delta"].mean()) < 0.02


def test_histogram_intersection_bounds():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.1, 5000)
    assert histogram_intersection(a, a) == 1.0
    assert histogram_intersection(a, a + 0.3) < 0.3
