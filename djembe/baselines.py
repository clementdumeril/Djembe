"""Reference models for the cycle grid (roadmap step J2).

Both models answer the same three questions for every node (cycle c, pulse k,
instrument i) of a held-out take, so that the graph model can be scored on exactly
the same footing later:

- does the instrument strike?          -> Bernoulli, scored by its negative log-likelihood;
- if it strikes, with which offset?    -> Gaussian density on delta (fraction of a pulse);
- and how hard?                        -> Gaussian density on the z-scored log peak.

``PositionModel`` (B1) is Polak's description of the feel turned into a model: per
piece, instrument and pulse, a strike probability and a Gaussian for the offset and
for the intensity. Nothing depends on what happened before.

``MarkovModel`` (B2) adds memory along each instrument's line:
- the strike probability depends on the two previous pulses (an order-2 chain),
  shrunk towards B1 when a context is rare (Dirichlet smoothing, strength ``alpha``);
- the offset and the intensity are B1's mean plus ``rho`` times the residual of the
  instrument's previous stroke (within the last 12 pulses), which captures drift and
  the persistence of a player's timing.

Offsets are scored only where |delta| <= 0.5 (3.5 % of strokes sit further from their
nominal pulse and are kept as annotated but not used as targets).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from djembe.grid import TakeGrid

LOG2PI = np.log(2 * np.pi)
EPS = 1e-3
DELTA_MAX = 0.5


def _gauss_nll(x, mu, sd):
    return 0.5 * (LOG2PI + 2 * np.log(sd) + ((x - mu) / sd) ** 2)


def _bern_nll(h, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(h * np.log(p) + (1 - h) * np.log(1 - p))


def _lines(grid: TakeGrid):
    """Per present instrument: flattened (cycle*12) arrays hit, delta, intensity, pulse."""
    n = grid.x.shape[0] * 12
    pulse = np.tile(np.arange(12), grid.x.shape[0])
    for i in np.flatnonzero(grid.present):
        yield i, grid.x[:, :, i, 0].reshape(n), grid.x[:, :, i, 2].reshape(n), grid.x[:, :, i, 1].reshape(n), pulse


class PositionModel:
    """B1: independent Gaussian feel per (piece, instrument, pulse)."""

    def __init__(self, min_strokes: int = 5, min_sd: float = 0.01):
        self.min_strokes, self.min_sd = min_strokes, min_sd

    def fit(self, grids: list[TakeGrid]) -> "PositionModel":
        hits, tot = defaultdict(float), defaultdict(float)
        vals = {"d": defaultdict(list), "I": defaultdict(list)}
        for g in grids:
            for i, h, d, I, k in _lines(g):
                for kk in range(12):
                    m = k == kk
                    key = (g.piece, i, kk)
                    hits[key] += h[m].sum()
                    tot[key] += m.sum()
                    ok = m & (h > 0)
                    vals["d"][key] += d[ok & (np.abs(d) <= DELTA_MAX)].tolist()
                    vals["I"][key] += I[ok].tolist()
        self.p = {key: (hits[key] + 0.5) / (tot[key] + 1.0) for key in tot}
        self.params = {}
        for name in ("d", "I"):
            pooled = defaultdict(list)
            for (piece, i, kk), v in vals[name].items():
                pooled[(piece, i)] += v
            for key in tot:
                v = vals[name][key] if len(vals[name][key]) >= self.min_strokes else pooled[key[:2]]
                v = np.asarray(v) if len(v) else np.zeros(1)
                self.params[(name,) + key] = (float(v.mean()), max(float(v.std()), self.min_sd))
        return self

    def mu_sd(self, name, piece, i, k):
        table = [self.params.get((name, piece, i, kk), (0.0, 1.0)) for kk in range(12)]
        mu = np.array([t[0] for t in table])[k]
        sd = np.array([t[1] for t in table])[k]
        return mu, sd

    def p_hit(self, piece, i, k):
        return np.array([self.p.get((piece, i, kk), 0.5) for kk in range(12)])[k]

    def nll(self, g: TakeGrid) -> dict:
        out = {key: [] for key in ("hit", "delta", "intensity", "hit_inst", "delta_inst", "intensity_inst")}
        for i, h, d, I, k in _lines(g):
            out["hit"].append(_bern_nll(h, self.p_hit(g.piece, i, k)))
            ok = h > 0
            okd = ok & (np.abs(d) <= DELTA_MAX)
            mu, sd = self.mu_sd("d", g.piece, i, k)
            out["delta"].append(_gauss_nll(d[okd], mu[okd], sd[okd]))
            mu, sd = self.mu_sd("I", g.piece, i, k)
            out["intensity"].append(_gauss_nll(I[ok], mu[ok], sd[ok]))
            for key, n in (("hit", h.size), ("delta", okd.sum()), ("intensity", ok.sum())):
                out[key + "_inst"].append(np.full(n, i))
        return {key: np.concatenate(v) for key, v in out.items()}

    def sample_delta(self, g: TakeGrid, rng) -> dict:
        """Offsets drawn at the take's real strokes (the 'humanise a score' setting)."""
        out = {}
        for i, h, d, I, k in _lines(g):
            mu, sd = self.mu_sd("d", g.piece, i, k)
            out[i] = np.where(h > 0, rng.normal(mu, sd), np.nan)
        return out


class MarkovModel:
    """B2: order-2 strike chain shrunk towards B1, AR(1) on offset and intensity residuals."""

    def __init__(self, alpha: float = 5.0, memory: int = 12):
        self.alpha, self.memory = alpha, memory
        self.base = PositionModel()

    @staticmethod
    def _prev_residual(h, r, memory):
        """Residual of the instrument's previous stroke within ``memory`` pulses (0 if none)."""
        prev = np.zeros_like(r)
        last_t, last_r = -10 ** 9, 0.0
        for t in range(len(h)):
            prev[t] = last_r if t - last_t <= memory else 0.0
            if h[t] > 0:
                last_t, last_r = t, r[t]
        return prev

    def _residuals(self, g, i, h, x, k, name):
        mu, _ = self.base.mu_sd(name, g.piece, i, k)
        r = np.where(h > 0, x - mu, 0.0)
        if name == "d":
            r = np.where(np.abs(x) <= DELTA_MAX, r, 0.0)
        return mu, r

    def fit(self, grids: list[TakeGrid]) -> "MarkovModel":
        self.base.fit(grids)
        hits, tot = defaultdict(float), defaultdict(float)
        xy = {"d": defaultdict(lambda: [[], [], []]), "I": defaultdict(lambda: [[], [], []])}
        for g in grids:
            for i, h, d, I, k in _lines(g):
                h1 = np.concatenate([[0], h[:-1]])
                h2 = np.concatenate([[0, 0], h[:-2]])
                for t in range(len(h)):
                    key = (g.piece, i, k[t], h1[t], h2[t])
                    hits[key] += h[t]
                    tot[key] += 1
                for name, x in (("d", d), ("I", I)):
                    _, r = self._residuals(g, i, h, x, k, name)
                    prev = self._prev_residual(h, r, self.memory)
                    ok = h > 0
                    if name == "d":
                        ok &= np.abs(d) <= DELTA_MAX
                    xy[name][(g.piece, i)][0] += prev[ok].tolist()
                    xy[name][(g.piece, i)][1] += r[ok].tolist()
                    xy[name][(g.piece, i)][2] += k[ok].tolist()
        self.hits, self.tot = dict(hits), dict(tot)
        self.rho, self.sd = {}, {}
        for name in ("d", "I"):
            for key, (px, ry, kk) in xy[name].items():
                px, ry, kk = map(np.asarray, (px, ry, kk))
                rho = float((px @ ry) / (px @ px)) if (px @ px) > 0 else 0.0
                self.rho[(name,) + key] = rho
                e = ry - rho * px
                for pulse in range(12):
                    m = kk == pulse
                    self.sd[(name,) + key + (pulse,)] = max(float(e[m].std()) if m.sum() >= 5 else float(e.std()), 0.01)
        return self

    def p_hit(self, piece, i, k, h1, h2):
        base = self.base.p_hit(piece, i, k)
        n = np.array([self.tot.get((piece, i, a, b, c), 0.0) for a, b, c in zip(k, h1, h2)])
        s = np.array([self.hits.get((piece, i, a, b, c), 0.0) for a, b, c in zip(k, h1, h2)])
        return (s + self.alpha * base) / (n + self.alpha)

    def _ar(self, name, piece, i, k):
        rho = self.rho.get((name, piece, i), 0.0)
        sd = np.array([self.sd.get((name, piece, i, kk), 1.0) for kk in range(12)])[k]
        return rho, sd

    def nll(self, g: TakeGrid) -> dict:
        out = {key: [] for key in ("hit", "delta", "intensity", "hit_inst", "delta_inst", "intensity_inst")}
        for i, h, d, I, k in _lines(g):
            h1 = np.concatenate([[0], h[:-1]])
            h2 = np.concatenate([[0, 0], h[:-2]])
            out["hit"].append(_bern_nll(h, self.p_hit(g.piece, i, k, h1, h2)))
            out["hit_inst"].append(np.full(h.size, i))
            for name, x, key in (("d", d, "delta"), ("I", I, "intensity")):
                mu, r = self._residuals(g, i, h, x, k, name)
                prev = self._prev_residual(h, r, self.memory)
                rho, sd = self._ar(name, g.piece, i, k)
                ok = h > 0
                if name == "d":
                    ok &= np.abs(d) <= DELTA_MAX
                out[key].append(_gauss_nll(x[ok], mu[ok] + rho * prev[ok], sd[ok]))
                out[key + "_inst"].append(np.full(ok.sum(), i))
        return {key: np.concatenate(v) for key, v in out.items()}

    def sample_delta(self, g: TakeGrid, rng) -> dict:
        """Offsets drawn sequentially at the take's real strokes, each conditioned on the
        model's own previous draw (free running, no access to the real offsets)."""
        out = {}
        for i, h, d, I, k in _lines(g):
            mu, _ = self.base.mu_sd("d", g.piece, i, k)
            rho, sd = self._ar("d", g.piece, i, k)
            sample = np.full(len(h), np.nan)
            last_t, last_r = -10 ** 9, 0.0
            for t in range(len(h)):
                if h[t] > 0:
                    prev = last_r if t - last_t <= self.memory else 0.0
                    r = rho * prev + rng.normal(0.0, sd[t])
                    sample[t] = mu[t] + r
                    last_t, last_r = t, r
            out[i] = sample
        return out


def histogram_intersection(a, b, bins=np.linspace(-DELTA_MAX, DELTA_MAX, 21)) -> float:
    """Overlap of two normalised histograms (1 = identical), as in Mishra et al. (2025)."""
    ha, _ = np.histogram(a, bins)
    hb, _ = np.histogram(b, bins)
    if ha.sum() == 0 or hb.sum() == 0:
        return np.nan
    return float(np.minimum(ha / ha.sum(), hb / hb.sum()).sum())
