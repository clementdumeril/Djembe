"""The real accompaniment's microphones with a generated soloist on top (local listening only).

The sample renderer (``djembe.render``) rebuilds every drum from one-shots, which
sounds mechanical. Here the accompaniment is the recording itself: the jembe-2 and
dundun microphones of the take. The soloist's own microphone is left out and the
generated soloist line is played with real soloist strokes cut from that microphone.

- Bleed: the soloist leaks into the other microphones (on Maraka_2, about -19 dB of
  a jembe-2 stroke in the jembe-2 microphone, -13 dB in the dundun one), so the real
  solo would ghost under the generated one. Each accompaniment track is gated on its
  own corpus onsets: open from 3 ms before a stroke until its next stroke (at most
  ``HOLD`` s, then a 60 ms fade), ``FLOOR_DB`` below in between.
- Soloist strokes: the take's jembe-1 strokes with a confident stroke type
  (``djembe.strokes``), cut from 4 ms before the onset until the next jembe-1 stroke
  (at most 0.35 s). Clean strokes (no other drum within 30 ms, so no bleed in the cut)
  are used first; a type with fewer than 12 clean strokes (often the bass, which lands
  with the dundun) uses all of its strokes. A generated stroke of type t and intensity
  z plays a randomly chosen one of the 6 cut strokes of type t closest in intensity,
  rescaled to z, and is damped (8 ms fade) when the next generated stroke comes, as a
  hand on the skin would.
- Placement: absolute cycle starts of the real take, ``(pulse + offset) / 12`` of the
  cycle, the take's median offset added back.

Everything here reads the corpus audio (CC BY-NC-ND): the output stays in ``output/``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from djembe import dataset as d
from djembe.grid import TakeGrid

TRACKS = {"J2": ("Jembe2", 0.35, 0.40), "D1": ("Dundun1", 0.0, 0.70), "D2": ("Dundun2", 0.10, 0.70)}  # pan, hold [s]
SOLO_PAN = -0.25
FLOOR_DB = -24.0
CUT_MAX = 0.35


def _read(path: Path):
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32", always_2d=True)
    return y.mean(axis=1), sr


def _raw_onsets(base: Path, take: str, name: str) -> np.ndarray:
    f = base / "annotations" / f"MJ_{take}_Onsets_Raw_{name}.csv"
    return np.sort(pd.read_csv(f)["Time"].to_numpy()) if f.exists() else np.zeros(0)


def gate(n: int, sr: int, onsets: np.ndarray, hold: float, floor_db: float = FLOOR_DB) -> np.ndarray:
    """Gain curve: 1 around each stroke (to the next one, at most ``hold``), ``floor_db`` elsewhere."""
    g = np.full(n, 10 ** (floor_db / 20), np.float32)
    fade = int(0.060 * sr)
    ramp = np.linspace(1, g[0], fade, dtype=np.float32)
    for j, t in enumerate(onsets):
        a = max(int((t - 0.003) * sr), 0)
        end = min(onsets[j + 1] if j + 1 < len(onsets) else t + hold, t + hold)
        b = min(int(end * sr), n)
        if b <= a:
            continue
        g[a:b] = 1.0
        c = min(b + fade, n)
        g[b:c] = np.maximum(g[b:c], ramp[:c - b])
    return g


def cycle_starts(events: pd.DataFrame, grid: TakeGrid) -> np.ndarray:
    ev = events[events["take_id"] == grid.take_id]
    s = ev.groupby("cycle_idx")["cycle_start_s"].first()
    return s.reindex(grid.cycle_idx).to_numpy()


def stroke_bank(take: str, table: pd.DataFrame, corpus=d.DEFAULT_CORPUS, min_conf: float = 0.8):
    """Cut jembe-1 strokes of a take: list of (type, intensity z, audio, clean), the sample rate, log-peak sd."""
    piece = take.split("_")[0]
    base = Path(corpus) / f"MJ_{piece}"
    y, sr = _read(base / f"MJ_{take}_J1.wav")
    own = _raw_onsets(base, take, "Jembe1")
    others = np.sort(np.concatenate([_raw_onsets(base, take, v[0]) for v in TRACKS.values()]))
    rows = table[(table["take_id"] == take) & (table["confidence"] >= min_conf)].copy()
    lp = np.log(table.loc[table["take_id"] == take, "peak"])
    rows["z"] = (np.log(rows["peak"]) - lp.mean()) / lp.std()
    bank = []
    for t, s, z in zip(rows["onset_s"], rows["stroke"], rows["z"]):
        clean = not (len(others) and np.abs(others - t).min() < 0.030)
        nxt = own[own > t + 0.02]
        end = min(nxt[0] - 0.002 if len(nxt) else t + CUT_MAX, t + CUT_MAX)
        a, b = int((t - 0.004) * sr), int(end * sr)
        if b - a < int(0.04 * sr):
            continue
        w = y[a:b].copy()
        f = min(int(0.008 * sr), len(w))
        w[-f:] *= np.linspace(1, 0, f)
        bank.append((s, float(z), w, clean))
    return bank, sr, float(lp.std())


def backing(take: str, t0: float, t1: float, corpus=d.DEFAULT_CORPUS, gated: bool = True) -> tuple[np.ndarray, int]:
    """Stereo mix of the accompaniment microphones over [t0, t1] s."""
    piece = take.split("_")[0]
    base = Path(corpus) / f"MJ_{piece}"
    out, sr = None, None
    for inst, (name, pan, hold) in TRACKS.items():
        f = base / f"MJ_{take}_{inst}.wav"
        if not f.exists():
            continue
        y, sr = _read(f)
        if gated:
            y = y * gate(len(y), sr, _raw_onsets(base, take, name), hold)
        seg = y[int(t0 * sr):int(t1 * sr)]
        if out is None:
            out = np.zeros((len(seg), 2), np.float32)
        n = min(len(seg), len(out))
        out[:n, 0] += seg[:n] * (1 - pan) / 2
        out[:n, 1] += seg[:n] * (1 + pan) / 2
    return out, sr


def soloist_track(grid: TakeGrid, types: np.ndarray, starts: np.ndarray, bank, sr: int, logpeak_sd: float,
                  t0: float, t1: float, start_cycle: int, rng=None, n_near: int = 6) -> np.ndarray:
    """Mono soloist over [t0, t1] s from the cut strokes, following the grid's soloist line."""
    rng = rng or np.random.default_rng(0)
    out = np.zeros(int((t1 - t0) * sr), np.float32)
    T = np.exp(grid.log_T)
    by_type = {}
    for s in ("bass", "tone", "slap"):
        clean = [(z, w) for s2, z, w, ok in bank if s2 == s and ok]
        by_type[s] = clean if len(clean) >= 2 * n_near else [(z, w) for s2, z, w, _ in bank if s2 == s]
    from djembe.stroke_types import TYPES

    cells = [(starts[c] + (k + grid.x[c, k, 0, 2] + grid.delta_offset) / 12 * T[c], c, k)
             for c in range(start_cycle, grid.x.shape[0]) for k in range(12) if grid.x[c, k, 0, 0] > 0]
    cells = sorted(x for x in cells if t0 <= x[0] < t1)
    last = None
    for j, (t, c, k) in enumerate(cells):
        z = float(grid.x[c, k, 0, 1])
        name = TYPES[types[c, k]] if types[c, k] >= 0 else "tone"
        pool = by_type[name] or [(zz, w) for _, zz, w, _ in bank]
        near = np.argsort([abs(zz - z) for zz, _ in pool])[:n_near]
        choice = [i for i in near if i != last] or list(near)
        i = int(rng.choice(choice))
        last = i
        zz, w = pool[i]
        w = w * float(np.exp(np.clip((z - zz) * logpeak_sd, -1.0, 1.0)))
        nxt = cells[j + 1][0] if j + 1 < len(cells) else t + CUT_MAX
        L = min(len(w), int((nxt - t + 0.004) * sr))
        w = w[:L].copy()
        f = min(int(0.008 * sr), L)
        w[L - f:] *= np.linspace(1, 0, f)
        a = int((t - t0 - 0.004) * sr)
        if a < 0 or a >= len(out):
            continue
        b = min(a + L, len(out))
        out[a:b] += w[:b - a]
    return out


def mix(back: np.ndarray, solo: np.ndarray, solo_gain: float = 1.0) -> np.ndarray:
    n = min(len(back), len(solo))
    y = back[:n].copy()
    y[:, 0] += solo[:n] * solo_gain * (1 - SOLO_PAN) / 2
    y[:, 1] += solo[:n] * solo_gain * (1 + SOLO_PAN) / 2
    return y / max(np.abs(y).max(), 1e-9) * 0.9
