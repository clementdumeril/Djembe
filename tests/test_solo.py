"""Soloist generation (J5b): the ensemble stays real, only jembe 1 is generated, strikes pulse by pulse."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip("torch")

from djembe import gnn, solo  # noqa: E402
from djembe.baselines import MarkovModel  # noqa: E402
from djembe.generate import StrikeGNN  # noqa: E402
from djembe.grid import position_features  # noqa: E402
from test_generate import synthetic_grid  # noqa: E402


def test_timing_window_knows_the_ensemble_not_the_soloist():
    g = synthetic_grid()
    w = solo.timing_window(g, 5, 4)
    x = w.x.reshape(4, 12, 4, -1)
    assert (x[-1, :, 1:, 3] == 1).all() and (x[-1, :, 0, 3] == 0).all()          # known flag
    assert np.allclose(x[-1, :, 1:3, 2], g.x[5, :, 1:3, 2])                       # ensemble offsets given
    assert not x[-1, :, 0, 1:3].any()                                             # soloist timing hidden
    m = w.mask_delta.reshape(4, 12, 4)
    assert not m[:, :, 1:].any() and not m[:-1].any()                             # targets: soloist only


def test_strike_window_hides_soloist_strikes_and_reveals_prefixes():
    g = synthetic_grid()
    sw = solo.strike_window(g, 5, 4)
    x = sw.x.reshape(4, 12, 4, -1)
    assert not x[-1, :, 0, 0].any() and (x[-1, :, 0, 3] == solo.HIDDEN).all()
    assert np.array_equal(sw.hit, g.x[5, :, 0, 0])
    xr, labels, mask = sw.reveal(5)
    xr = xr.reshape(4, 12, 4, -1)
    assert np.array_equal(xr[-1, :5, 0, 0], g.x[5, :5, 0, 0]) and (xr[-1, :5, 0, 3] == 0).all()
    assert (xr[-1, 5:, 0, 3] == solo.HIDDEN).all()
    m = mask.reshape(4, 12, 4)
    assert m[-1, 5:, 0].all() and m.sum() == 7


@pytest.mark.parametrize("which", ["gnn", "markov"])
def test_generation_changes_only_the_soloist(which):
    real = synthetic_grid(n_cycles=8)
    if which == "gnn":
        torch.manual_seed(0)
        n_in = gnn.n_base_features() + position_features("full", 1).shape[1]
        g = solo.generate(real, StrikeGNN(n_in, dim=8, layers=1).eval(), gnn.CycleGNN(n_in, dim=8, layers=1).eval(),
                          K=4, rng=np.random.default_rng(1))
    else:
        b2 = MarkovModel(memory_scope="cycle").fit([synthetic_grid("Maraka_2", seed=2)])
        g = solo.generate_markov(real, b2, K=4, rng=np.random.default_rng(1))
    assert np.array_equal(g.x[:, :, 1:], real.x[:, :, 1:])        # ensemble untouched everywhere
    assert np.array_equal(g.x[:3], real.x[:3])                    # real context
    hit = g.x[3:, :, 0, 0] > 0
    assert hit.any() and np.all(np.abs(g.x[3:, :, 0, 2]) <= solo.DELTA_CLIP)
    assert not g.x[3:, :, 0, 1][~hit].any()


def test_compare_against_itself():
    real = synthetic_grid(n_cycles=40)
    m = solo.compare(real, real, [synthetic_grid("Maraka_2", seed=2)], start_gen=3, start_real=3)
    assert m["profile_l1"] == 0 and m["timing_w1_ms"] == 0 and 0 < m["distinct_rate"] <= 1
