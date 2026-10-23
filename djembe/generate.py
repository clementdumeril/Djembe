"""Generation of whole cycles with the graph model (roadmap step J5).

Two networks share the architecture of ``djembe.gnn``:

- the timing model is the "humanise a cycle" network (V5): strikes of the target
  cycle known, Gaussian offset and intensity for each stroke;
- the strike model sees the same window with the target cycle's strikes hidden too,
  and predicts, for every pulse and instrument of that cycle, the probability that it
  strikes (Bernoulli, scored like B2's strike model).

Generation runs cycle by cycle from K-1 real cycles of context: draw the strikes of
the next cycle, then their offsets and intensities, write them into the grid and move
on, so every later cycle is conditioned on the model's own output. Tempo (cycle
durations) and form labels are copied from the real take: the model generates the
feel, not the tempo curve.

The Markov reference generates the same way from B2: order-2 strike chain along each
instrument's line, offsets from B2 conditioned on its own previous draws, intensities
from B1.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from djembe import gnn
from djembe.grid import TakeGrid, position_features

N_PULSE_INST = 48                       # nodes of one cycle: 12 pulses x 4 instruments
DELTA_CLIP = 0.5


# ───────────────────────────── windows ─────────────────────────────
def slice_grid(grid: TakeGrid, start: int, stop: int) -> TakeGrid:
    """Cycles [start, stop) of a take grid, as a grid of its own."""
    return replace(grid, x=grid.x[start:stop].copy(), log_T=grid.log_T[start:stop], section=grid.section[start:stop],
                   in_signal=grid.in_signal[start:stop], cycle_idx=grid.cycle_idx[start:stop])


def window_at(grid: TakeGrid, t: int, K: int) -> gnn.Window:
    """The single window whose target is cycle t (identical to ``gnn.make_windows(grid)[t - K + 1]``)."""
    (w,) = gnn.make_windows(slice_grid(grid, t - K + 1, t + 1), K)
    return replace(w, cycle=t)


@dataclass
class StrikeWindow:
    x: np.ndarray            # node features, target strikes hidden
    hit: np.ndarray          # (N,) 1 where the instrument strikes, target cycle only
    mask: np.ndarray         # (N,) target-cycle nodes of present instruments
    present: tuple
    piece: str


def strike_window(w: gnn.Window) -> StrikeWindow:
    x = w.x.copy()
    target = np.zeros(x.shape[0], bool)
    target[-N_PULSE_INST:] = True
    hit = np.where(target, w.x[:, 0], 0.0).astype(np.float32)
    x[target, 0] = 0.0                                  # the strikes to predict
    inst = np.arange(x.shape[0]) % 4
    mask = target & np.isin(inst, w.present)
    return StrikeWindow(x, hit, mask, w.present, w.piece)


def collate_strikes(windows: list[StrikeWindow], K: int, position: str, device="cpu", helix: bool = False) -> dict:
    pos = position_features(position, K)
    xs, edges, n0 = [], [[] for _ in gnn.EDGE_TYPES], 0
    for w in windows:
        xs.append(np.concatenate([w.x, pos], axis=1))
        for r, e in enumerate(gnn._edges(K, w.present, helix)):
            edges[r].append(e + n0)
        n0 += w.x.shape[0]
    t = lambda a, dt=torch.float32: torch.as_tensor(np.concatenate(a), dtype=dt, device=device)
    return {"x": t(xs), "edges": [torch.as_tensor(np.concatenate(e, axis=1), dtype=torch.long, device=device)
                                  for e in edges],
            "hit": t([w.hit for w in windows]), "mask": t([w.mask for w in windows], torch.bool)}


# ───────────────────────────── strike model ─────────────────────────────
class StrikeGNN(gnn.CycleGNN):
    """Same message passing as the timing model, one logit per node."""

    def __init__(self, n_in: int, dim: int = 64, layers: int = 3, dropout: float = 0.1):
        super().__init__(n_in, dim, layers, dropout)
        self.head = nn.Linear(dim, 1)

    def forward(self, x, edges, prior=None):
        h = self.inp(x)
        for layer in self.layers:
            h = self.drop(layer(h, edges))
        return self.head(h)[:, 0]


def train_strikes(train_w: list[StrikeWindow], *, K=4, position="full", helix=True, dim=64, layers=3, dropout=0.1,
                  lr=2e-3, weight_decay=1e-4, epochs=40, batch_size=32, seed=0, device="cpu") -> StrikeGNN:
    """Binary cross-entropy on the target cycle's strikes; all epochs run, last weights kept."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_in = gnn.n_base_features() + position_features(position, 1).shape[1]
    model = StrikeGNN(n_in, dim, layers, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(train_w))
        for b in range(0, len(order), batch_size):
            batch = collate_strikes([train_w[j] for j in order[b:b + batch_size]], K, position, device, helix)
            logit = model(batch["x"], batch["edges"])
            loss = F.binary_cross_entropy_with_logits(logit[batch["mask"]], batch["hit"][batch["mask"]])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def strike_nll(model, windows: list[StrikeWindow], *, K=4, position="full", helix=True, device="cpu",
               instruments=(0, 1, 2, 3), batch_size=256) -> dict:
    """Mean Bernoulli NLL of the target strikes, in nats, comparable to B2's ``nll()['hit']``."""
    out = []
    for b in range(0, len(windows), batch_size):
        batch = collate_strikes(windows[b:b + batch_size], K, position, device, helix)
        keep = batch["mask"] & torch.isin(torch.arange(batch["x"].shape[0], device=device) % 4,
                                          torch.as_tensor(instruments, device=device))
        logit = model(batch["x"], batch["edges"])
        out.append(F.binary_cross_entropy_with_logits(logit[keep], batch["hit"][keep], reduction="none").cpu().numpy())
    v = np.concatenate(out)
    return {"nll_hit": float(v.mean()), "n_hit": int(v.size)}


# ───────────────────────────── generators ─────────────────────────────
def _blank_from(grid: TakeGrid, n_context: int) -> TakeGrid:
    g = replace(grid, x=grid.x.copy())
    g.x[n_context:] = 0.0
    return g


@torch.no_grad()
def generate_gnn(grid: TakeGrid, strike_model, timing_model, *, K=4, position="full", helix=True, prior=None,
                 rng=None, device="cpu", n_context: int | None = None) -> TakeGrid:
    """Free-running generation of every cycle after the first ``n_context`` (default K-1) real ones."""
    rng = rng or np.random.default_rng(0)
    n_context = K - 1 if n_context is None else n_context
    g = _blank_from(grid, n_context)
    present = np.flatnonzero(grid.present)
    for t in range(n_context, g.x.shape[0]):
        sw = strike_window(window_at(g, t, K))
        b = collate_strikes([sw], K, position, device, helix)
        p = torch.sigmoid(strike_model(b["x"], b["edges"])[-N_PULSE_INST:]).cpu().numpy().reshape(12, 4)
        hits = np.zeros((12, 4), np.float32)
        hits[:, present] = rng.random((12, len(present))) < p[:, present]
        g.x[t, :, :, 0] = hits
        w = window_at(g, t, K)                              # strikes now known, offsets still hidden
        bt = gnn.collate([w], K, position, device, helix=helix, prior=prior)
        dm, dls, im, ils = (v[-N_PULSE_INST:].cpu().numpy().reshape(12, 4)
                            for v in timing_model(bt["x"], bt["edges"], bt["prior"]))
        delta = np.clip(rng.normal(dm, np.exp(dls)), -DELTA_CLIP, DELTA_CLIP)
        inten = rng.normal(im, np.exp(ils))
        g.x[t, :, :, 1] = hits * inten
        g.x[t, :, :, 2] = hits * delta
    return g


def generate_markov(grid: TakeGrid, b2, *, K=4, rng=None, n_context: int | None = None) -> TakeGrid:
    """The same free-running generation with B2 (strikes, offsets) and B1 (intensities)."""
    rng = rng or np.random.default_rng(0)
    n_context = K - 1 if n_context is None else n_context
    g = _blank_from(grid, n_context)
    n_c, start = g.x.shape[0], n_context * 12
    for i in np.flatnonzero(grid.present):
        h = g.x[:, :, i, 0].reshape(-1)
        k = np.tile(np.arange(12), n_c)
        for t in range(start, h.size):
            p = b2.p_hit(g.piece, i, k[t:t + 1], h[t - 1:t] if t else np.zeros(1), h[t - 2:t - 1] if t > 1 else np.zeros(1))
            h[t] = float(rng.random() < p[0])
        g.x[:, :, i, 0] = h.reshape(n_c, 12)
    deltas = b2.sample_delta(g, rng)
    for i, d in deltas.items():
        d = np.nan_to_num(np.clip(d, -DELTA_CLIP, DELTA_CLIP)).reshape(n_c, 12)
        mu, sd = b2.base.mu_sd("I", g.piece, i, np.arange(12))
        inten = rng.normal(mu[None, :], sd[None, :], size=(n_c, 12))
        hit = g.x[:, :, i, 0]
        g.x[n_context:, :, i, 2] = (hit * d)[n_context:]
        g.x[n_context:, :, i, 1] = (hit * inten)[n_context:]
    return g
