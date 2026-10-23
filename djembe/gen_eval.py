"""How close is a generated take to the real one? (roadmap step J5)

Every measure compares the generated cycles of a held-out take (after its K-1 real
context cycles) with the real cycles of that take:

- ``profile_l1``: mean absolute difference of the strike rate per pulse and instrument;
- ``density_ratio``: strikes per cycle, generated over real;
- ``timing_w1`` / ``intensity_w1``: Wasserstein-1 distance between the offset (fraction
  of a pulse) or intensity distributions, averaged over the (pulse, instrument) cells
  with at least ``min_strokes`` strokes on both sides; ``timing_w1_ms`` converts the
  offset distance with the take's mean cycle duration;
- ``novelty``: per cycle, the smallest Hamming distance between its strike pattern
  and any training cycle of the same piece, as a fraction of the compared bits;
  ``copy_rate``: fraction of cycles identical to a training cycle. The same numbers for
  the real held-out cycles give the level of novelty of real playing;
- ``recognised``: fraction of non-overlapping 4-cycle windows that a classifier trained
  on the training takes assigns to the take's piece.

Scored on jembe 1, jembe 2 and dundun 1 (the instruments present in most takes).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from djembe.grid import TakeGrid

MAIN = (0, 1, 2)
DELTA_MAX = 0.5
WINDOW = 4


def _instruments(*grids: TakeGrid) -> list[int]:
    return [i for i in MAIN if all(g.present[i] for g in grids)]


def strike_profile(g: TakeGrid, start: int = 0) -> np.ndarray:
    return g.x[start:, :, :, 0].mean(axis=0)                      # (12, 4)


def _cell_values(g: TakeGrid, start: int, k: int, i: int, f: int) -> np.ndarray:
    x = g.x[start:, k, i]
    v = x[x[:, 0] > 0, f]
    return v[np.abs(v) <= DELTA_MAX] if f == 2 else v


def distribution_distance(gen: TakeGrid, real: TakeGrid, f: int, start_gen: int, start_real: int,
                          min_strokes: int = 10) -> float:
    d = []
    for i in _instruments(gen, real):
        for k in range(12):
            a, b = _cell_values(gen, start_gen, k, i, f), _cell_values(real, start_real, k, i, f)
            if len(a) >= min_strokes and len(b) >= min_strokes:
                d.append(wasserstein_distance(a, b))
    return float(np.mean(d)) if d else float("nan")


def _patterns(g: TakeGrid, insts: list[int], start: int = 0) -> np.ndarray:
    return g.x[start:, :, insts, 0].reshape(g.x.shape[0] - start, -1) > 0


def novelty(g: TakeGrid, train: list[TakeGrid], start: int) -> tuple[float, float]:
    """(mean nearest-neighbour Hamming fraction, exact-copy rate) of g's cycles from ``start``."""
    same = [t for t in train if t.piece == g.piece]
    insts = [i for i in MAIN if g.present[i] and all(t.present[i] for t in same)]
    bank = np.concatenate([_patterns(t, insts) for t in same])
    mine = _patterns(g, insts, start)
    dist = (mine[:, None, :] != bank[None, :, :]).mean(axis=2).min(axis=1)
    return float(dist.mean()), float((dist == 0).mean())


def window_features(g: TakeGrid, start: int = 0) -> np.ndarray:
    """Non-overlapping 4-cycle windows: strike rate per pulse (J1, J2, D1) and mean offset per beat position."""
    feats = []
    for s in range(start, g.x.shape[0] - WINDOW + 1, WINDOW):
        x = g.x[s:s + WINDOW][:, :, list(MAIN)]
        rate = x[..., 0].mean(axis=0).reshape(-1)                  # 36
        off = []
        for pos in range(3):
            for i in range(len(MAIN)):
                h = x[:, pos::3, i, 0] > 0
                d = x[:, pos::3, i, 2][h]
                off.append(float(d.mean()) if d.size else 0.0)
        feats.append(np.concatenate([rate, off]))
    return np.array(feats)


def train_recogniser(train: list[TakeGrid]):
    X = np.concatenate([window_features(t) for t in train])
    y = np.concatenate([[t.piece] * len(window_features(t)) for t in train])
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)


def compare(gen: TakeGrid, real: TakeGrid, train: list[TakeGrid], recogniser, *, start_gen: int,
            start_real: int) -> dict:
    insts = _instruments(gen, real)
    pg, pr = strike_profile(gen, start_gen)[:, insts], strike_profile(real, start_real)[:, insts]
    nov, copy = novelty(gen, train, start_gen)
    X = window_features(gen, start_gen)
    mean_T = float(np.exp(real.log_T).mean())
    w1 = distribution_distance(gen, real, 2, start_gen, start_real)
    return {
        "profile_l1": float(np.abs(pg - pr).mean()),
        "density_ratio": float(pg.sum() / max(pr.sum(), 1e-9)),
        "timing_w1": w1,
        "timing_w1_ms": w1 * mean_T / 12 * 1000,
        "intensity_w1": distribution_distance(gen, real, 1, start_gen, start_real),
        "novelty": nov,
        "copy_rate": copy,
        "recognised": float((recogniser.predict(X) == gen.piece).mean()) if len(X) else float("nan"),
        "n_cycles": int(gen.x.shape[0] - start_gen),
    }
