"""Generation (J5): windows, strike masking, generators and measures, on synthetic grids."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip("torch")

from djembe import gen_eval as ge  # noqa: E402
from djembe import generate as gen  # noqa: E402
from djembe import gnn  # noqa: E402
from djembe import render  # noqa: E402
from djembe.baselines import MarkovModel, PositionModel  # noqa: E402
from djembe.grid import TakeGrid, position_features  # noqa: E402


def synthetic_grid(take="Maraka_1", n_cycles=10, seed=0, present=(True, True, True, False)):
    rng = np.random.default_rng(seed)
    x = np.zeros((n_cycles, 12, 4, 3), np.float32)
    pattern = rng.random((12, 4)) < 0.4
    for c in range(n_cycles):
        hit = pattern & (rng.random((12, 4)) < 0.9)
        hit[:, ~np.array(present)] = False
        x[c, :, :, 0] = hit
        x[c, :, :, 1] = hit * rng.normal(0, 1, (12, 4))
        x[c, :, :, 2] = hit * rng.normal(0, 0.1, (12, 4))
    return TakeGrid(take_id=take, piece=take.split("_")[0], x=x, log_T=np.full(n_cycles, np.log(1.5), np.float32),
                    section=np.zeros(n_cycles, int), in_signal=np.zeros(n_cycles, bool), present=np.array(present),
                    cycle_idx=np.arange(n_cycles))


def test_window_at_matches_make_windows():
    g = synthetic_grid()
    for t in range(3, g.x.shape[0]):
        a, b = gen.window_at(g, t, 4), gnn.make_windows(g, 4)[t - 3]
        assert np.array_equal(a.x, b.x) and np.array_equal(a.delta, b.delta) and a.cycle == b.cycle == t


def test_strike_window_hides_only_target_strikes():
    g = synthetic_grid()
    w = gnn.make_windows(g, 4)[0]
    sw = gen.strike_window(w)
    target = slice(-gen.N_PULSE_INST, None)
    assert not sw.x[target, 0].any()
    assert np.array_equal(sw.x[:-gen.N_PULSE_INST], w.x[:-gen.N_PULSE_INST])     # context untouched
    assert np.array_equal(sw.hit[target], w.x[target, 0])
    inst = np.arange(sw.x.shape[0]) % 4
    assert not sw.mask[:-gen.N_PULSE_INST].any() and not sw.mask[inst == 3].any()  # D2 absent


def tiny_models(K=4):
    torch.manual_seed(0)
    n_in = gnn.n_base_features() + position_features("full", 1).shape[1]
    return gen.StrikeGNN(n_in, dim=8, layers=1).eval(), gnn.CycleGNN(n_in, dim=8, layers=1).eval()


@pytest.mark.parametrize("which", ["gnn", "markov"])
def test_generators_keep_context_and_present_instruments(which):
    real = synthetic_grid()
    if which == "gnn":
        s, t = tiny_models()
        g = gen.generate_gnn(real, s, t, K=4, rng=np.random.default_rng(1))
    else:
        b2 = MarkovModel(memory_scope="cycle").fit([synthetic_grid("Maraka_2", seed=2), synthetic_grid("Maraka_3", seed=3)])
        g = gen.generate_markov(real, b2, K=4, rng=np.random.default_rng(1))
    assert np.array_equal(g.x[:3], real.x[:3])                       # real context kept
    assert not g.x[:, :, 3].any()                                    # nothing on the absent instrument
    hit = g.x[..., 0] > 0
    assert np.all(np.abs(g.x[..., 2]) <= gen.DELTA_CLIP)
    assert not g.x[..., 1][~hit].any() and not g.x[..., 2][~hit].any()   # no offset without a stroke
    assert hit[3:].any()                                             # it did generate strokes
    assert real.x[3:].any()                                          # and left the real grid alone


def test_compare_is_zero_against_itself():
    real = synthetic_grid(n_cycles=40)
    train = [synthetic_grid("Maraka_2", n_cycles=12, seed=2), synthetic_grid("Woloso_1", n_cycles=12, seed=4)]
    rec = ge.train_recogniser(train)
    m = ge.compare(real, real, train, rec, start_gen=3, start_real=3)
    assert m["profile_l1"] == 0 and m["density_ratio"] == pytest.approx(1)
    assert m["timing_w1"] == 0 and m["intensity_w1"] == 0


def test_novelty_of_a_training_take_is_zero():
    t = synthetic_grid("Maraka_2", seed=2)
    nov, copy = ge.novelty(t, [t], 0)
    assert nov == 0 and copy == 1


def test_render_is_finite_stereo():
    g = synthetic_grid(n_cycles=4)
    y = render.render(g.x, g.log_T, g.present)
    assert y.ndim == 2 and y.shape[1] == 2 and np.isfinite(y).all() and np.abs(y).max() <= 0.91
