"""The soloist generated stroke after stroke, timing included (roadmap step J6, graph model).

In J5b the soloist's strikes were drawn pulse after pulse, but the offsets and
intensities of a whole cycle were then drawn at once, each independently of the
others. The real soloist's deviations hang together (two successive strokes late
together: lag-1 correlation of the residuals 0.35 to 0.76 depending on the piece);
the J5b soloist's barely do (0.07 to 0.52).

Here every soloist pulse of the target cycle is handled in order, k = 0..11, with the
same factorisation as the transformer (``djembe.transformer``) so that the two can be
compared on the same numbers:

    p(hit_k | past) and, if it strikes, p(offset_k, intensity_k | hit_k = 1, past),

where "past" is everything in the K-1 context cycles, the whole accompaniment of the
target cycle (as in J5b, it plays as the soloist plays), and the soloist's pulses
0..k-1 of the target cycle with their offsets and intensities. The network has one
head per quantity on each soloist node; the offset and intensity heads are trained
on strokes only, which is what conditioning on ``hit_k = 1`` means.

Training: a random prefix k per example and epoch; pulses 0..k-1 revealed in full
(``known = 1``), pulses k..11 hidden (``known = -1``), losses on the hidden pulses.
Scoring: pulse k from the state with exactly pulses 0..k-1 revealed.

``piece_input=False`` zeroes the piece columns of the node features: the network then
cannot tell which rhythm it plays except by listening to it (the leave-one-piece-out
test).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from djembe import gnn, solo
from djembe.grid import TakeGrid, position_features

PIECE_COLS = slice(9, 12)                 # piece one-hot in gnn.make_windows' node features
DELTA_CLIP = 0.5


@dataclass
class SeqWindow:
    x: np.ndarray            # node features, every soloist value of the target cycle hidden
    hit: np.ndarray          # (12,) soloist target cycle
    delta: np.ndarray
    inten: np.ndarray
    present: tuple
    piece: str

    def reveal(self, k: int, hit=None, delta=None, inten=None) -> np.ndarray:
        """Features with the soloist's pulses 0..k-1 of the target cycle fully known."""
        hit = self.hit if hit is None else hit
        delta = self.delta if delta is None else delta
        inten = self.inten if inten is None else inten
        x = self.x.copy()
        nodes = solo._solo_nodes(x.shape[0])[:k]
        x[nodes, 0] = hit[:k]
        x[nodes, 1] = hit[:k] * inten[:k]
        x[nodes, 2] = hit[:k] * np.clip(delta[:k], -1, 1)
        x[nodes, 3] = 1.0
        return x


def seq_window(grid: TakeGrid, t: int, K: int, piece_input: bool = True) -> SeqWindow:
    sw = solo.strike_window(grid, t, K)              # ensemble known, soloist target hidden (known = -1)
    x = sw.x.copy()
    if not piece_input:
        x[:, PIECE_COLS] = 0.0
    tgt = grid.x[t, :, solo.SOLO]
    return SeqWindow(x, tgt[:, 0].astype(np.float32), tgt[:, 2].astype(np.float32), tgt[:, 1].astype(np.float32),
                     sw.present, sw.piece)


class SoloGNN(gnn.CycleGNN):
    """Message passing of ``djembe.gnn``; per node a strike logit and two Gaussians."""

    def __init__(self, n_in: int, dim: int = 64, layers: int = 3, dropout: float = 0.1):
        super().__init__(n_in, dim, layers, dropout)
        self.head = nn.Linear(dim, 5)

    def forward(self, x, edges, prior=None):
        h = self.inp(x)
        for layer in self.layers:
            h = self.drop(layer(h, edges))
        o = self.head(h)
        return o[:, 0], o[:, 1], o[:, 2].clamp(-5, 2), o[:, 3], o[:, 4].clamp(-5, 2)


def _batch(windows: list[SeqWindow], ks: list[int], K: int, position: str, helix: bool, device):
    xs = [w.reveal(k) for w, k in zip(windows, ks)]
    b = solo._collate(xs, [w.present for w in windows], K, position, helix, device)
    n = windows[0].x.shape[0]
    nodes = np.concatenate([solo._solo_nodes(n) + j * n for j in range(len(windows))])
    t = lambda a: torch.as_tensor(np.concatenate(a), dtype=torch.float32, device=device)
    hidden = np.concatenate([np.arange(12) >= k for k in ks])
    return b, torch.as_tensor(nodes, device=device), t([w.hit for w in windows]), t([w.delta for w in windows]), \
        t([w.inten for w in windows]), torch.as_tensor(hidden, device=device)


def nll_terms(out, nodes, hit, delta, inten):
    logit, dm, dls, im, ils = (o[nodes] for o in out)
    l_hit = F.binary_cross_entropy_with_logits(logit, hit, reduction="none")
    l_d = gnn.gaussian_nll(delta, dm, dls)
    l_i = gnn.gaussian_nll(inten, im, ils)
    return l_hit, l_d, l_i


def train(windows: list[SeqWindow], *, K=4, position="full", helix=True, dim=64, layers=3, dropout=0.1, lr=2e-3,
          weight_decay=1e-4, epochs=40, batch_size=32, seed=0, device="cpu") -> SoloGNN:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SoloGNN(gnn.n_base_features() + position_features(position, 1).shape[1], dim, layers, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(windows))
        for b in range(0, len(order), batch_size):
            chunk = [windows[j] for j in order[b:b + batch_size]]
            ks = [int(rng.integers(0, 12)) for _ in chunk]
            batch, nodes, hit, delta, inten, hidden = _batch(chunk, ks, K, position, helix, device)
            l_hit, l_d, l_i = nll_terms(model(batch["x"], batch["edges"]), nodes, hit, delta, inten)
            stroke = hidden & (hit > 0)
            md = stroke & (delta.abs() <= DELTA_CLIP)
            loss = l_hit[hidden].mean() + (l_d[md].mean() if md.any() else 0) + (l_i[stroke].mean() if stroke.any() else 0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def score(model, windows: list[SeqWindow], *, K=4, position="full", helix=True, device="cpu", batch_size=256) -> dict:
    """Pulse k scored from the state with the true pulses 0..k-1 revealed; means over slots / strokes."""
    hs, ds, is_ = [], [], []
    for k in range(12):
        for b in range(0, len(windows), batch_size):
            chunk = windows[b:b + batch_size]
            batch, nodes, hit, delta, inten, _ = _batch(chunk, [k] * len(chunk), K, position, helix, device)
            l_hit, l_d, l_i = (v.cpu().numpy().reshape(len(chunk), 12)[:, k]
                               for v in nll_terms(model(batch["x"], batch["edges"]), nodes, hit, delta, inten))
            h = np.array([w.hit[k] for w in chunk]) > 0
            d = np.array([w.delta[k] for w in chunk])
            hs.append(l_hit)
            ds.append(l_d[h & (np.abs(d) <= DELTA_CLIP)])
            is_.append(l_i[h])
    h, d, i = (np.concatenate(v) for v in (hs, ds, is_))
    return {"nll_hit": float(h.mean()), "n_hit": int(h.size), "nll_delta": float(d.mean()), "n_delta": int(d.size),
            "nll_intensity": float(i.mean()), "n_intensity": int(i.size)}


@torch.no_grad()
def generate(grid: TakeGrid, model, *, K=4, position="full", helix=True, piece_input=True, rng=None,
             temperature: float = 1.0, device="cpu", n_context: int | None = None) -> TakeGrid:
    """Soloist drawn stroke by stroke after ``n_context`` real cycles; the ensemble is the real one."""
    rng = rng or np.random.default_rng(0)
    n_context = K - 1 if n_context is None else n_context
    g = replace(grid, x=grid.x.copy())
    g.x[n_context:, :, solo.SOLO] = 0.0
    for t in range(n_context, g.x.shape[0]):
        w = seq_window(g, t, K, piece_input)
        node = solo._solo_nodes(w.x.shape[0])
        hit, delta, inten = (np.zeros(12, np.float32) for _ in range(3))
        for k in range(12):
            b = solo._collate([w.reveal(k, hit, delta, inten)], [w.present], K, position, helix, device)
            logit, dm, dls, im, ils = (float(o[node[k]]) for o in model(b["x"], b["edges"]))
            if rng.random() < 1.0 / (1.0 + np.exp(-logit / temperature)):
                hit[k] = 1.0
                delta[k] = np.clip(rng.normal(dm, np.exp(dls)), -DELTA_CLIP, DELTA_CLIP)
                inten[k] = rng.normal(im, np.exp(ils))
        g.x[t, :, solo.SOLO, 0], g.x[t, :, solo.SOLO, 1], g.x[t, :, solo.SOLO, 2] = hit, hit * inten, hit * delta
    return g
