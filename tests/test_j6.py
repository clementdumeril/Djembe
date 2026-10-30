"""J6: stroke-by-stroke soloist, graph model and transformer on the same information."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "j6"))
torch = pytest.importorskip("torch")

from djembe import solo, solo_seq, transformer as tf  # noqa: E402
from test_generate import synthetic_grid  # noqa: E402


def test_seq_window_reveals_prefix_in_full_and_hides_the_rest():
    g = synthetic_grid()
    w = solo_seq.seq_window(g, 5, 4)
    x = w.reveal(4).reshape(4, 12, 4, -1)
    tgt = g.x[5, :, 0]
    assert np.array_equal(x[-1, :4, 0, 0], tgt[:4, 0]) and (x[-1, :4, 0, 3] == 1).all()
    assert np.allclose(x[-1, :4, 0, 1], tgt[:4, 0] * tgt[:4, 1])
    assert not x[-1, 4:, 0, :3].any() and (x[-1, 4:, 0, 3] == solo.HIDDEN).all()
    assert np.allclose(x[-1, :, 1:3, 2], g.x[5, :, 1:3, 2])                     # ensemble timing known


def test_piece_input_off_removes_the_piece_everywhere():
    g = synthetic_grid()
    assert solo_seq.seq_window(g, 5, 4, piece_input=True).x[:, solo_seq.PIECE_COLS].any()
    assert not solo_seq.seq_window(g, 5, 4, piece_input=False).x[:, solo_seq.PIECE_COLS].any()
    on, off = tf.tok_window(g, 5, 4, True), tf.tok_window(g, 5, 4, False)
    assert on.enc.sum() > off.enc.sum() and np.array_equal(on.solo, off.solo)
    other = replace(g, piece="Woloso")
    assert np.array_equal(tf.tok_window(other, 5, 4, False).enc, off.enc)


def test_transformer_never_sees_the_stroke_it_predicts():
    torch.manual_seed(0)
    model = tf.SoloTransformer(K=4, dim=32, ff=64).eval()
    w = tf.tok_window(synthetic_grid(), 5, 4)
    enc, solo_t, _ = tf._tensors([w], "cpu")
    base = [o.detach() for o in model(enc, solo_t)]
    p = 40
    changed = solo_t.clone()
    changed[0, p] = torch.tensor([1.0, 2.5, 0.3])
    after = [o.detach() for o in model(enc, changed)]
    for a, b in zip(base, after):
        assert torch.allclose(a[0, :p + 1], b[0, :p + 1], atol=1e-6)               # up to p: unchanged
    assert not torch.allclose(base[0][0, p + 1:], after[0][0, p + 1:])            # later pulses see it
    enc2 = enc.clone()
    enc2[0, 45, 0] = 1 - enc2[0, 45, 0]                                           # a future ensemble stroke
    assert not torch.allclose(base[0][0, 36:], model(enc2, solo_t)[0][0, 36:].detach())


@pytest.mark.parametrize("which", ["gnn", "transformer"])
def test_generation_changes_only_the_soloist_after_context(which):
    g = synthetic_grid(n_cycles=8)
    if which == "gnn":
        sw = [solo_seq.seq_window(g, t, 4) for t in range(3, 8)]
        m = solo_seq.train(sw, epochs=1, dim=16, layers=1)
        out = solo_seq.generate(g, m, rng=np.random.default_rng(0))
        assert solo_seq.score(m, sw)["n_hit"] == 5 * 12
    else:
        tw = [tf.tok_window(g, t, 4) for t in range(3, 8)]
        m = tf.train(tw, epochs=1, dim=16, ff=32)
        out = tf.generate(g, m, rng=np.random.default_rng(0))
        assert tf.score(m, tw)["n_hit"] == 5 * 12
    assert np.array_equal(out.x[:, :, 1:], g.x[:, :, 1:]) and np.array_equal(out.x[:3], g.x[:3])
    s = out.x[3:, :, 0]
    assert set(np.unique(s[..., 0])) <= {0.0, 1.0}
    assert not s[s[..., 0] == 0][:, 1:].any() and (np.abs(s[..., 2]) <= 0.5).all()


def test_timing_lag1_sees_linked_deviations():
    rng = np.random.default_rng(0)
    g = synthetic_grid(n_cycles=60)
    g.x[:, :, 0, 0] = 1.0
    ar = np.zeros(60 * 12)
    for t in range(1, ar.size):
        ar[t] = 0.8 * ar[t - 1] + rng.normal(0, 0.05)
    linked = replace(g, x=g.x.copy())
    linked.x[:, :, 0, 2] = ar.reshape(60, 12)
    indep = replace(g, x=g.x.copy())
    indep.x[:, :, 0, 2] = rng.normal(0, 0.1, (60, 12))
    assert solo.timing_lag1(linked) > 0.6 and abs(solo.timing_lag1(indep)) < 0.15


def test_task_lists():
    import run

    assert len(run.tasks("take", [0, 1, 2])) == 45 and len(run.tasks("piece", [0, 1, 2])) == 9
    assert len(set(run.tasks("take_nopiece", [0, 1, 2]))) == 45
