"""Live listening: stroke detection from a stream, and the grid tracker, on synthetic audio and strokes."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from djembe.live import J2_PULSES, GridTracker, OnsetStream  # noqa: E402

SR = 22050
FEEL = {k: (0.15 if k % 3 == 0 else -0.05) for k in J2_PULSES}


def clicks(times, sr=SR, dur=None, level=0.5, seed=0):
    rng = np.random.default_rng(seed)
    n = int(sr * (dur or max(times) + 0.5))
    y = rng.normal(0, 1e-4, n).astype(np.float32)
    t = np.arange(int(0.08 * sr)) / sr
    hit = (level * np.sin(2 * np.pi * 400 * t) * np.exp(-t / 0.02)).astype(np.float32)
    for s in times:
        i = int(s * sr)
        y[i:i + hit.size] += hit[:max(0, min(hit.size, n - i))]
    return y


def j2_times(n_cycles, T0=1.5, drift=0.0, start=0.3):
    out, s, T = [], start, T0
    for _ in range(n_cycles):
        out += [s + (k + o) / 12 * T for k, o in FEEL.items()]
        s, T = s + T, T * (1 + drift)
    return np.array(out)


def test_stream_finds_strokes_shortly_after_they_happen():
    truth = j2_times(4)
    y = clicks(truth)
    st = OnsetStream(SR)
    found, confirmed = [], []
    for b in range(0, len(y), 256):
        for t, _ in st.push(y[b:b + 256]):
            found.append(t)
            confirmed.append((b + 256) / SR)
    found = np.array(found)
    assert len(found) == len(truth)
    assert np.abs(found - truth).max() < 0.012
    assert np.median(np.array(confirmed) - found) < 0.04


def test_tracker_starts_and_follows_an_accelerando():
    truth = j2_times(30, T0=1.6, drift=-0.004)                         # tempo rises by about 11 %
    tr = GridTracker(FEEL)
    heard = [(t, 1.0) for t in truth if t <= truth[0] + tr.listen]
    tr.start_from(heard)
    errs = []
    for i in range(len(heard), len(truth)):
        k = J2_PULSES[i % 8]
        c = i // 8
        errs.append(tr.pulse_time(k + FEEL[k], c) - truth[i])           # predicted before hearing it
        tr.observe(truth[i], 1.0)
    assert np.abs(errs[16:]).max() < 0.01


def test_tracker_ignores_faint_bleed_between_strokes():
    truth = j2_times(12)
    rng = np.random.default_rng(1)
    bleed = [(t + rng.uniform(0.03, 0.08), 0.15) for t in truth[::3]]
    strokes = sorted([(t, 1.0) for t in truth] + bleed)
    tr = GridTracker(FEEL)
    heard = [s for s in strokes if s[0] <= truth[0] + tr.listen]
    tr.start_from(heard)
    for s in strokes[len(heard):]:
        tr.observe(*s)
    assert abs(tr.x[1] - 1.5) < 0.01 and tr.cycle == 11


def test_live_session_plays_on_the_tracked_grid():
    import pytest

    torch = pytest.importorskip("torch")
    from djembe import gnn
    from djembe.grid import position_features
    from djembe.live import LiveSession
    from djembe.solo_seq import SoloGNN

    torch.manual_seed(0)
    model = SoloGNN(gnn.n_base_features() + position_features("full", 1).shape[1], dim=16, layers=1).eval()
    truth = j2_times(8)
    tr = GridTracker(FEEL)
    seed_rows = np.zeros((3, 12, 4, 3), np.float32)
    seed_rows[:, list(J2_PULSES), 1, 0] = 1.0
    s = LiveSession(model, tr, "Maraka", seed_rows, rng=np.random.default_rng(0))
    heard = [(t, 1.0) for t in truth if t <= truth[0] + tr.listen]
    s.start(heard)
    assert s.x[3, list(J2_PULSES), 1, 0].all()                          # the listened cycle is written as jembe 2
    plays = []
    for i in range(len(heard), len(truth)):
        c, k = divmod(i, 8)
        out = s.decide(c, J2_PULSES[k])
        if out is not None:
            plays.append((out[0], c, J2_PULSES[k]))
        s.hear(truth[i], 1.0)
    assert plays
    for t, c, k in plays:
        assert abs(t - (0.3 + c * 1.5 + k / 12 * 1.5)) <= 0.5 / 12 * 1.5 + 0.005   # half a pulse at most, plus tracking error
