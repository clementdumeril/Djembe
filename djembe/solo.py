"""The soloist (jembe 1) over the real ensemble (roadmap step J5b).

Jembe 1 is the soloist in this corpus: its most frequent cycle pattern covers 17 % of
its cycles on average (36 % of its cycles are distinct patterns), against 95 % for
jembe 2 and 68 % for dundun 1. Here the graph model generates only that line. The
accompaniment (jembe 2, dunduns) stays real everywhere, including in the cycle being
generated, because a soloist answers the ensemble as it plays.

Node features follow ``djembe.gnn.make_windows`` with one change for the target cycle:
accompaniment nodes are fully known (strike, offset, intensity, ``known = 1``), soloist
nodes are not. In the strike model a soloist node whose strike is still hidden has
``hit = 0`` and ``known = -1``; once its strike is revealed, ``known = 0`` (strike known,
timing not).

Strikes are generated one pulse after the other: the strike model sees the soloist's
strikes already drawn in this cycle (a prefix of pulses), and it is trained the same
way, on random prefixes, with the loss on every pulse still hidden. Strike scores are
therefore sequential, like B2's order-2 chain: the NLL of pulse k given pulses 0..k-1.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from djembe import gnn
from djembe.generate import StrikeGNN, slice_grid, window_at
from djembe.grid import TakeGrid, position_features

SOLO = 0                                   # jembe 1
N_NODES_CYCLE = 48
DELTA_CLIP = 0.5
HIDDEN = -1.0


def _target_nodes(n_nodes: int) -> np.ndarray:
    idx = np.zeros(n_nodes, bool)
    idx[-N_NODES_CYCLE:] = True
    return idx


def _solo_nodes(n_nodes: int) -> np.ndarray:
    """Target-cycle nodes of the soloist, in pulse order 0..11."""
    return np.flatnonzero(_target_nodes(n_nodes) & (np.arange(n_nodes) % 4 == SOLO))


def timing_window(grid: TakeGrid, t: int, K: int) -> gnn.Window:
    """Soloist strikes of cycle t known, its offsets and intensities to predict; ensemble fully known."""
    w = window_at(grid, t, K)
    x = w.x.copy()
    target = _target_nodes(x.shape[0])
    acc = target & (np.arange(x.shape[0]) % 4 != SOLO)
    h = x[acc, 0]
    tgt = grid.x[t].reshape(-1, 3)                       # (48, 3): hit, intensity, delta, row-major (k, i)
    x[acc, 1] = h * tgt[acc[-N_NODES_CYCLE:]][:, 1]
    x[acc, 2] = h * np.clip(tgt[acc[-N_NODES_CYCLE:]][:, 2], -1, 1)
    x[acc, 3] = 1.0
    solo = np.zeros(x.shape[0], bool)
    solo[_solo_nodes(x.shape[0])] = True
    return replace(w, x=x, mask_delta=w.mask_delta & solo, mask_intensity=w.mask_intensity & solo)


@dataclass
class SoloStrikeWindow:
    x: np.ndarray            # node features, every soloist strike of the target cycle hidden
    hit: np.ndarray          # (12,) soloist strikes of the target cycle
    present: tuple
    piece: str

    def reveal(self, k: int, hits: np.ndarray | None = None):
        """Features with pulses 0..k-1 of the soloist revealed (``hits`` or the true ones); loss mask on the rest."""
        hits = self.hit if hits is None else hits
        x = self.x.copy()
        nodes = _solo_nodes(x.shape[0])
        x[nodes[:k], 0] = hits[:k]
        x[nodes[:k], 3] = 0.0
        mask = np.zeros(x.shape[0], bool)
        mask[nodes[k:]] = True
        labels = np.zeros(x.shape[0], np.float32)
        labels[nodes] = self.hit
        return x, labels, mask


def strike_window(grid: TakeGrid, t: int, K: int) -> SoloStrikeWindow:
    w = timing_window(grid, t, K)
    x = w.x.copy()
    nodes = _solo_nodes(x.shape[0])
    hit = x[nodes, 0].copy()
    x[nodes, 0] = 0.0
    x[nodes, 3] = HIDDEN
    return SoloStrikeWindow(x, hit.astype(np.float32), w.present, w.piece)


def _collate(xs: list[np.ndarray], presents: list[tuple], K: int, position: str, helix: bool, device) -> dict:
    pos = position_features(position, K)
    edges, n0 = [[] for _ in gnn.EDGE_TYPES], 0
    for x, present in zip(xs, presents):
        for r, e in enumerate(gnn._edges(K, present, helix)):
            edges[r].append(e + n0)
        n0 += x.shape[0]
    return {"x": torch.as_tensor(np.concatenate([np.concatenate([x, pos], 1) for x in xs]), dtype=torch.float32,
                                 device=device),
            "edges": [torch.as_tensor(np.concatenate(e, axis=1), dtype=torch.long, device=device) for e in edges]}


def train_strikes(windows: list[SoloStrikeWindow], *, K=4, position="full", helix=True, dim=64, layers=3, dropout=0.1,
                  lr=2e-3, weight_decay=1e-4, epochs=40, batch_size=32, seed=0, device="cpu") -> StrikeGNN:
    """Random prefix k in 0..11 per example and epoch; BCE on the soloist pulses k..11."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = StrikeGNN(gnn.n_base_features() + position_features(position, 1).shape[1], dim, layers, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(windows))
        for b in range(0, len(order), batch_size):
            parts = [windows[j].reveal(int(rng.integers(0, 12))) for j in order[b:b + batch_size]]
            batch = _collate([p[0] for p in parts], [windows[j].present for j in order[b:b + batch_size]],
                             K, position, helix, device)
            labels = torch.as_tensor(np.concatenate([p[1] for p in parts]), device=device)
            mask = torch.as_tensor(np.concatenate([p[2] for p in parts]), device=device)
            loss = F.binary_cross_entropy_with_logits(model(batch["x"], batch["edges"])[mask], labels[mask])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def strike_nll(model, windows: list[SoloStrikeWindow], *, K=4, position="full", helix=True, device="cpu",
               batch_size=256) -> dict:
    """Sequential NLL: pulse k given the true pulses 0..k-1, summed into a per-stroke-slot mean."""
    out = []
    for k in range(12):
        for b in range(0, len(windows), batch_size):
            chunk = windows[b:b + batch_size]
            parts = [w.reveal(k) for w in chunk]
            batch = _collate([p[0] for p in parts], [w.present for w in chunk], K, position, helix, device)
            logit = model(batch["x"], batch["edges"]).cpu().numpy()
            n = chunk[0].x.shape[0]
            node = _solo_nodes(n)[k]
            lk = logit[node::n][:len(chunk)]
            y = np.array([w.hit[k] for w in chunk])
            out.append(np.logaddexp(0, -lk) * y + np.logaddexp(0, lk) * (1 - y))
    v = np.concatenate(out)
    return {"nll_hit": float(v.mean()), "n_hit": int(v.size)}


@torch.no_grad()
def generate(grid: TakeGrid, strike_model, timing_model, *, K=4, position="full", helix=True, rng=None,
             temperature: float = 1.0, device="cpu", n_context: int | None = None) -> TakeGrid:
    """Soloist generated after ``n_context`` (default K-1) real cycles; the ensemble is copied from the real take."""
    rng = rng or np.random.default_rng(0)
    n_context = K - 1 if n_context is None else n_context
    g = replace(grid, x=grid.x.copy())
    g.x[n_context:, :, SOLO] = 0.0
    for t in range(n_context, g.x.shape[0]):
        sw = strike_window(g, t, K)
        n = sw.x.shape[0]
        node_k = _solo_nodes(n)
        hits = np.zeros(12, np.float32)
        for k in range(12):
            x, _, _ = sw.reveal(k, hits)
            b = _collate([x], [sw.present], K, position, helix, device)
            logit = float(strike_model(b["x"], b["edges"])[node_k[k]])
            hits[k] = float(rng.random() < 1.0 / (1.0 + np.exp(-logit / temperature)))
        g.x[t, :, SOLO, 0] = hits
        w = timing_window(g, t, K)
        bt = gnn.collate([w], K, position, device, helix=helix)
        dm, dls, im, ils = (v.cpu().numpy()[node_k] for v in timing_model(bt["x"], bt["edges"], bt["prior"]))
        g.x[t, :, SOLO, 2] = hits * np.clip(rng.normal(dm, np.exp(dls)), -DELTA_CLIP, DELTA_CLIP)
        g.x[t, :, SOLO, 1] = hits * rng.normal(im, np.exp(ils))
    return g


def generate_markov(grid: TakeGrid, b2, *, K=4, rng=None, n_context: int | None = None) -> TakeGrid:
    """Reference: B2's order-2 strike chain and offsets for the soloist line only, B1 intensities."""
    rng = rng or np.random.default_rng(0)
    n_context = K - 1 if n_context is None else n_context
    g = replace(grid, x=grid.x.copy())
    g.x[n_context:, :, SOLO] = 0.0
    n_c = g.x.shape[0]
    h = g.x[:, :, SOLO, 0].reshape(-1)
    k = np.tile(np.arange(12), n_c)
    for t in range(n_context * 12, h.size):
        p = b2.p_hit(g.piece, SOLO, k[t:t + 1], h[t - 1:t], h[t - 2:t - 1])
        h[t] = float(rng.random() < p[0])
    g.x[:, :, SOLO, 0] = h.reshape(n_c, 12)
    d = np.nan_to_num(np.clip(b2.sample_delta(g, rng)[SOLO], -DELTA_CLIP, DELTA_CLIP)).reshape(n_c, 12)
    mu, sd = b2.base.mu_sd("I", g.piece, SOLO, np.arange(12))
    inten = rng.normal(mu[None, :], sd[None, :], size=(n_c, 12))
    hit = g.x[:, :, SOLO, 0]
    g.x[n_context:, :, SOLO, 2] = (hit * d)[n_context:]
    g.x[n_context:, :, SOLO, 1] = (hit * inten)[n_context:]
    return g


# ───────────────────────────── measures ─────────────────────────────
def _solo_cells(g: TakeGrid, start: int, k: int, f: int) -> np.ndarray:
    x = g.x[start:, k, SOLO]
    v = x[x[:, 0] > 0, f]
    return v[np.abs(v) <= DELTA_CLIP] if f == 2 else v


def compare(gen: TakeGrid, real: TakeGrid, train: list[TakeGrid], *, start_gen: int, start_real: int,
            min_strokes: int = 10) -> dict:
    """Soloist line of ``gen`` against the real soloist line (see module docstring for each measure)."""
    from scipy.stats import wasserstein_distance

    pg = gen.x[start_gen:, :, SOLO, 0]
    pr = real.x[start_real:, :, SOLO, 0]
    dist = {}
    for f, name in ((2, "timing_w1"), (1, "intensity_w1")):
        vals = []
        for k in range(12):
            a, b = _solo_cells(gen, start_gen, k, f), _solo_cells(real, start_real, k, f)
            if len(a) >= min_strokes and len(b) >= min_strokes:
                vals.append(wasserstein_distance(a, b))
        dist[name] = float(np.mean(vals)) if vals else float("nan")
    bank = np.concatenate([t.x[:, :, SOLO, 0] for t in train if t.piece == gen.piece]) > 0
    mine = pg > 0
    nn_dist = (mine[:, None, :] != bank[None, :, :]).mean(axis=2).min(axis=1)
    distinct = len({tuple(r) for r in mine.astype(int)}) / len(mine)
    return {
        "profile_l1": float(np.abs(pg.mean(0) - pr.mean(0)).mean()),
        "density_ratio": float(pg.sum(1).mean() / max(pr.sum(1).mean(), 1e-9)),
        "timing_w1_ms": dist["timing_w1"] * float(np.exp(real.log_T).mean()) / 12 * 1000,
        "intensity_w1": dist["intensity_w1"],
        "novelty": float(nn_dist.mean()),
        "copy_rate": float((nn_dist == 0).mean()),
        "distinct_rate": float(distinct),
        "n_cycles": int(len(pg)),
    }
