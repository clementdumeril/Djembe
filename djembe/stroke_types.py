"""Which stroke (bass, tone, slap) a jembe plays, given what the grid says about it.

The stroke labels come from ``djembe.strokes`` (the audio of the isolated track,
checked by ear at 75 %). This model only reads the cycle grid, so it applies as well
to a generated line as to a real one: for each stroke of one instrument, in time order,

- piece, and pulse k (one-hot);
- intensity (z-scored log peak), its square, and intensity x pulse;
- offset from the grid, and log time since the instrument's previous stroke;
- the two previous stroke types (or "none") and the previous stroke's intensity.

A multinomial logistic regression maps these to the three types. Leave-one-take-out
on the soloist (jembe 1, ``experiments/strokes/type_eval.py``): piece and pulse
frequencies reach 46 % (NLL 1.003, uniform 1.099); pulse and intensity 71 % (0.676);
everything above 72 % (0.656, better on 14 of 15 takes).
Loud strokes are slaps (91 % in the loudest third); the soloist's choice between bass
and tone is much less tied to position than the accompanist's.

``sample`` draws the types stroke after stroke, feeding its own draws back as the
previous types, as a renderer needs for a generated soloist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from djembe.dataset import PIECES
from djembe.grid import TakeGrid

TYPES = ("bass", "tone", "slap")
NONE = len(TYPES)
INSTRUMENT_NAMES = ("J1", "J2", "D1", "D2")


def labels_for(grid: TakeGrid, table: pd.DataFrame, instrument: int = 0) -> np.ndarray:
    """``(n_cycles, 12)`` type index per cell of ``instrument``, -1 where there is no labelled stroke."""
    out = np.full(grid.x.shape[:2], -1, int)
    rows = table[table["take_id"] == grid.take_id]
    pos = {c: n for n, c in enumerate(grid.cycle_idx)}
    for c, sd, s in zip(rows["cycle_idx"], rows["sd"], rows["stroke"]):
        if c in pos and grid.x[pos[c], sd - 1, instrument, 0] > 0:
            out[pos[c], sd - 1] = TYPES.index(s)
    return out


def _strokes(grid: TakeGrid, instrument: int):
    """Cells (c, k) of the instrument's strokes in time order, with their times [s]."""
    T = np.exp(grid.log_T)
    starts = np.concatenate([[0.0], np.cumsum(T)[:-1]])
    c, k = np.nonzero(grid.x[:, :, instrument, 0] > 0)
    t = starts[c] + (k + grid.x[c, k, instrument, 2]) / 12 * T[c]
    order = np.argsort(t, kind="stable")
    return c[order], k[order], t[order]


def _row(piece: str, k: int, inten: float, delta: float, log_ioi: float, p1: int, p2: int, prev_int: float) -> np.ndarray:
    pulse = np.eye(12)[k]
    return np.concatenate([np.eye(len(PIECES))[PIECES.index(piece)], pulse, [inten, inten ** 2], pulse * inten,
                           [delta, log_ioi, prev_int], np.eye(NONE + 1)[p1], np.eye(NONE + 1)[p2]])


def _log_ioi(t: np.ndarray, j: int) -> float:
    return float(np.log(np.clip(t[j] - t[j - 1], 0.02, 2.0))) if j > 0 else 0.0


def features(grid: TakeGrid, types: np.ndarray, instrument: int = 0):
    """Feature matrix of every stroke, its cell and its type (-1 if unlabelled)."""
    c, k, t = _strokes(grid, instrument)
    x = grid.x[:, :, instrument]
    y = types[c, k]
    prev = np.where(y >= 0, y, NONE)
    rows = [_row(grid.piece, k[j], x[c[j], k[j], 1], x[c[j], k[j], 2], _log_ioi(t, j),
                 prev[j - 1] if j > 0 else NONE, prev[j - 2] if j > 1 else NONE,
                 x[c[j - 1], k[j - 1], 1] if j > 0 else 0.0) for j in range(len(c))]
    return (np.array(rows) if rows else np.zeros((0, _row(PIECES[0], 0, 0, 0, 0, NONE, NONE, 0).size))), c, k, y


N_POSITION = len(PIECES) + 12 + 2 + 12          # piece, pulse, intensity terms: the columns before the context


class StrokeTypeModel:
    """``context=False`` keeps only piece, pulse and intensity (the ablation in the docstring)."""

    def __init__(self, instrument: int = 0, C: float = 1.0, context: bool = True):
        self.instrument, self.C, self.context = instrument, C, context

    def _cols(self, X: np.ndarray) -> np.ndarray:
        return X if self.context else X[:, :N_POSITION]

    def fit(self, grids: list[TakeGrid], labels: list[np.ndarray]) -> "StrokeTypeModel":
        from sklearn.linear_model import LogisticRegression

        X, Y = [], []
        for g, lab in zip(grids, labels):
            f, _, _, y = features(g, lab, self.instrument)
            X.append(f[y >= 0])
            Y.append(y[y >= 0])
        self.clf = LogisticRegression(max_iter=2000, C=self.C).fit(self._cols(np.concatenate(X)), np.concatenate(Y))
        return self

    def proba(self, X: np.ndarray) -> np.ndarray:
        """``(n, 3)`` probabilities in ``TYPES`` order (a type absent from training gets 0)."""
        out = np.zeros((len(X), len(TYPES)))
        out[:, self.clf.classes_] = self.clf.predict_proba(self._cols(X))
        return out

    def score(self, grid: TakeGrid, labels: np.ndarray) -> dict:
        """NLL and accuracy on the labelled strokes, previous types taken from the labels."""
        f, _, _, y = features(grid, labels, self.instrument)
        keep = y >= 0
        p = self.proba(f[keep])
        return {"nll_type": float(-np.log(np.clip(p[np.arange(keep.sum()), y[keep]], 1e-12, 1)).mean()),
                "acc_type": float((p.argmax(1) == y[keep]).mean()), "n_type": int(keep.sum())}

    def sample(self, grid: TakeGrid, rng=None, temperature: float = 1.0, start_cycle: int = 0,
               given: np.ndarray | None = None) -> np.ndarray:
        """Types for every stroke from ``start_cycle`` on, drawn in time order; ``given`` fixes earlier cells."""
        rng = rng or np.random.default_rng(0)
        types = np.full(grid.x.shape[:2], -1, int) if given is None else given.copy()
        types[start_cycle:] = -1
        c, k, t = _strokes(grid, self.instrument)
        x = grid.x[:, :, self.instrument]
        prev = lambda j: (types[c[j], k[j]] if types[c[j], k[j]] >= 0 else NONE) if j >= 0 else NONE
        for j in np.flatnonzero(c >= start_cycle):
            row = _row(grid.piece, k[j], x[c[j], k[j], 1], x[c[j], k[j], 2], _log_ioi(t, j), prev(j - 1), prev(j - 2),
                       x[c[j - 1], k[j - 1], 1] if j > 0 else 0.0)
            p = self.proba(row[None])[0] ** (1.0 / temperature)
            types[c[j], k[j]] = int(rng.choice(len(TYPES), p=p / p.sum()))
        return types
