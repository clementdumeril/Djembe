"""J8: figure planner and executor."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip("torch")

from djembe import planner as pl, solo, solo_seq  # noqa: E402
from test_generate import synthetic_grid  # noqa: E402

A = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], np.float32)
B = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1], np.float32)


def line(figures):
    g = synthetic_grid(n_cycles=len(figures))
    g.x[:, :, solo.SOLO, 0] = np.array(figures)
    g.x[:, :, solo.SOLO, 1] *= g.x[:, :, solo.SOLO, 0]
    g.x[:, :, solo.SOLO, 2] *= g.x[:, :, solo.SOLO, 0]
    return g


def test_candidates_are_distinct_newest_first_with_counts():
    g = line([A, B, A, A, B, A])
    c = pl.candidates(g, 5)
    assert len(c.pattern) == 2
    assert np.array_equal(c.pattern[0], B) and np.array_equal(c.pattern[1], A)          # B played last (cycle 4)
    assert np.allclose(c.last_row[1], g.x[3, :, solo.SOLO])                              # A's latest occurrence
    assert c.feats[0, 14] == 1.0 and c.feats[1, 14] == 0.0                               # last-cycle flag
    assert np.isclose(np.expm1(c.feats[1, 13]), 3)                                       # A played 3 times


def test_target_is_the_nearest_figure_within_two_strokes():
    g = line([A, B, A])
    c = pl.candidates(g, 2)
    near_b = B.copy()
    near_b[1] = 0
    assert pl.target_base(c, near_b) == int(np.flatnonzero((c.pattern == B).all(1))[0])
    far = 1 - A
    assert pl.target_base(c, far) == len(c.pattern)                                      # no base


def test_batched_and_single_planner_scores_agree():
    torch.manual_seed(0)
    g = line([A, B, A, B, B, A, A, B])
    ex = pl.plan_examples([g])
    m = pl.Planner().eval()
    f, mask, ctx, _ = pl._pad(ex)
    with torch.no_grad():
        batch = m.logits_batch(f, mask, ctx)
        for i, (c, _) in enumerate(ex):
            single = m.logits(c)
            n = len(c.pattern)
            assert torch.allclose(single[:n], batch[i, :n], atol=1e-5) and torch.allclose(single[-1], batch[i, -1])


def test_base_columns_sit_on_the_target_soloist_nodes():
    g = line([A, B, A, B, A])
    w = pl.exec_window(g, 4, 4, g.x[2, :, solo.SOLO])
    x = w.x.reshape(4, 12, 4, -1)
    assert np.allclose(x[-1, :, solo.SOLO, -4], A) and (x[-1, :, solo.SOLO, -1] == 1).all()
    assert not x[:-1, :, :, -4:].any() and not x[-1, :, 1:, -4:].any()
    assert not pl.exec_window(g, 4, 4, None).x[:, -4:].any()


def test_two_stage_generation_changes_only_the_soloist():
    g = line([A, B, A, A, B, A, B, B, A, A])
    planner = pl.train_planner(pl.plan_examples([g]), epochs=1)
    executor = solo_seq.train(pl.exec_windows([g]), epochs=1, dim=16, layers=1)
    log = []
    out = pl.generate(g, planner, executor, rng=np.random.default_rng(0), log=log)
    assert np.array_equal(out.x[:, :, 1:], g.x[:, :, 1:]) and np.array_equal(out.x[:3], g.x[:3])
    assert len(log) == 7
