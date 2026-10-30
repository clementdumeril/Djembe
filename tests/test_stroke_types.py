"""Stroke-type model and its use in rendering, on synthetic grids (no corpus audio)."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("sklearn")

from djembe import render, stroke_types as stt  # noqa: E402
from test_generate import synthetic_grid  # noqa: E402


def rule_labels(g):
    """Loud -> slap, quiet -> bass, else tone: a rule the model must find."""
    inten, hit = g.x[:, :, 0, 1], g.x[:, :, 0, 0] > 0
    lab = np.where(inten > 0.5, 2, np.where(inten < -0.5, 0, 1))
    return np.where(hit, lab, -1)


def test_labels_for_joins_on_cycle_and_pulse():
    g = synthetic_grid(n_cycles=4)
    c, k = np.argwhere(g.x[:, :, 0, 0] > 0)[0]
    table = pd.DataFrame({"take_id": [g.take_id, g.take_id, "other"], "cycle_idx": [g.cycle_idx[c], g.cycle_idx[c], 0],
                          "sd": [k + 1, (k + 1) % 12 + 1, 1], "stroke": ["slap", "bass", "tone"]})
    lab = stt.labels_for(g, table)
    assert lab[c, k] == stt.TYPES.index("slap")
    assert (lab >= 0).sum() == 1 + int(g.x[c, (k + 1) % 12, 0, 0] > 0)
    assert (lab[g.x[:, :, 0, 0] == 0] == -1).all()


def test_model_learns_an_intensity_rule_and_samples_only_on_strokes():
    grids = [synthetic_grid(take=f"Maraka_{s + 1}", n_cycles=30, seed=s) for s in range(4)]
    m = stt.StrokeTypeModel().fit(grids[:3], [rule_labels(g) for g in grids[:3]])
    assert m.score(grids[3], rule_labels(grids[3]))["acc_type"] > 0.9
    drawn = m.sample(grids[3], np.random.default_rng(0), start_cycle=5, given=rule_labels(grids[3]))
    hit = grids[3].x[:, :, 0, 0] > 0
    assert ((drawn >= 0) == hit).all()
    assert np.array_equal(drawn[:5], rule_labels(grids[3])[:5])
    assert (drawn[5:][hit[5:]] == rule_labels(grids[3])[5:][hit[5:]]).mean() > 0.8


def test_render_plays_the_given_stroke_type():
    assert render.sample_name(0, 2.0) == "slap"
    assert render.sample_name(0, 2.0, stt.TYPES.index("bass")) == "bass"
    assert render.sample_name(2, 0.0, 1) == "dundunba"
    assert render.STROKE_TYPES == stt.TYPES
    g = synthetic_grid(n_cycles=2)
    strokes = {0: np.full((2, 12), stt.TYPES.index("tone"))}
    y = render.render(g.x, g.log_T, g.present, voice="synth", strokes=strokes)
    assert y.ndim == 2 and np.isfinite(y).all()
