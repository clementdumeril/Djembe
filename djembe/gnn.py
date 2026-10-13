"""Graph model for the cycle grid (roadmap step J3: the "humanise a cycle" task).

A training example is a window of ``K`` consecutive cycles of one take. The first
K-1 cycles are fully known (strikes, offsets, intensities); in the last cycle, the
target, only the strikes are known. For every stroke of the target cycle the
network predicts a Gaussian for its offset and one for its intensity. Scores are
negative log-likelihoods in the same units as the reference models
(``djembe.baselines``), on the same strokes.

The network is plain relational message passing on the graph of ``djembe.grid``:
for each node, a linear map of itself plus, for each edge type, the mean of a
linear map of its in-neighbours; then GELU, a residual connection and LayerNorm.
Every operation is node-wise or a mean over an edge set that rotation of the cycle
maps onto itself, so the network is exactly as symmetric as its positional input:
``position="none"`` (rotation by any number of pulses, C12), ``"subpulse"``
(rotation by whole beats, C4), ``"full"`` (no symmetry). ``tests/test_gnn.py``
checks this numerically.

Plain PyTorch (``index_add_``), no graph library, so it runs unchanged on a laptop
CPU or a Colab GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from djembe.dataset import PIECES
from djembe.grid import EDGE_TYPES, SECTIONS, TakeGrid, graph_edges, position_features

DELTA_MAX = 0.5
LOG_T_REF, LOG_T_SCALE = np.log(1.5), 0.2      # cycle durations are 1.2-2.6 s


@dataclass
class Window:
    """One example: node features x (N, F), targets and masks for the target cycle."""
    x: np.ndarray
    delta: np.ndarray
    intensity: np.ndarray
    mask_delta: np.ndarray
    mask_intensity: np.ndarray
    present: tuple
    take_id: str
    cycle: int                      # index of the target cycle in the take grid


def n_base_features() -> int:
    return 5 + 4 + len(PIECES) + len(SECTIONS) + 1


def make_windows(grid: TakeGrid, K: int = 4) -> list[Window]:
    """All windows whose target cycle has K-1 cycles of context (cycles K-1 ... end)."""
    n_c = grid.x.shape[0]
    piece = np.eye(len(PIECES), dtype=np.float32)[PIECES.index(grid.piece)]
    inst = np.eye(4, dtype=np.float32)
    out = []
    for t in range(K - 1, n_c):
        cyc = range(t - K + 1, t + 1)
        x = np.zeros((K, 12, 4, n_base_features()), np.float32)
        for w, c in enumerate(cyc):
            known = 1.0 if c < t else 0.0
            h = grid.x[c, :, :, 0]
            x[w, :, :, 0] = h
            x[w, :, :, 1] = known * h * grid.x[c, :, :, 1]
            x[w, :, :, 2] = known * h * np.clip(grid.x[c, :, :, 2], -1, 1)
            x[w, :, :, 3] = known
            x[w, :, :, 4] = grid.present[None, :]
            x[w, :, :, 5:9] = inst[None, :, :]
            x[w, :, :, 9:12] = piece
            x[w, :, :, 12:16] = np.eye(len(SECTIONS), dtype=np.float32)[grid.section[c]]
            x[w, :, :, 16] = (grid.log_T[c] - LOG_T_REF) / LOG_T_SCALE
        hit = grid.x[t, :, :, 0] > 0
        d, I = grid.x[t, :, :, 2], grid.x[t, :, :, 1]
        md = np.zeros((K, 12, 4), bool)
        mi = np.zeros((K, 12, 4), bool)
        md[-1] = hit & (np.abs(d) <= DELTA_MAX) & grid.present[None, :]
        mi[-1] = hit & grid.present[None, :]
        dd = np.zeros((K, 12, 4), np.float32)
        ii = np.zeros((K, 12, 4), np.float32)
        dd[-1], ii[-1] = d, I
        out.append(Window(x.reshape(-1, x.shape[-1]), dd.reshape(-1), ii.reshape(-1), md.reshape(-1),
                          mi.reshape(-1), tuple(np.flatnonzero(grid.present)), grid.take_id, t))
    return out


@lru_cache(maxsize=None)
def _edges(K: int, present: tuple) -> tuple:
    e = graph_edges(K, instruments=present)
    return tuple(e[name] for name in EDGE_TYPES)


def collate(windows: list[Window], K: int, position: str, device="cpu"):
    """Stack windows into one disconnected graph."""
    pos = position_features(position, K)
    xs, edges, n0 = [], [[] for _ in EDGE_TYPES], 0
    for w in windows:
        xs.append(np.concatenate([w.x, pos], axis=1))
        for r, e in enumerate(_edges(K, w.present)):
            edges[r].append(e + n0)
        n0 += w.x.shape[0]
    t = lambda a, dt=torch.float32: torch.as_tensor(np.concatenate(a), dtype=dt, device=device)
    return {
        "x": t(xs),
        "edges": [torch.as_tensor(np.concatenate(e, axis=1), dtype=torch.long, device=device) for e in edges],
        "delta": t([w.delta for w in windows]),
        "intensity": t([w.intensity for w in windows]),
        "mask_delta": t([w.mask_delta for w in windows], torch.bool),
        "mask_intensity": t([w.mask_intensity for w in windows], torch.bool),
        "inst": t([np.tile(np.arange(4), w.x.shape[0] // 4) for w in windows], torch.long),
    }


class RelationalLayer(nn.Module):
    def __init__(self, dim: int, n_rel: int = len(EDGE_TYPES)):
        super().__init__()
        self.self_lin = nn.Linear(dim, dim)
        self.rel = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(n_rel))
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edges):
        out = self.self_lin(h)
        ones = torch.ones(h.size(0), 1, device=h.device)
        for lin, e in zip(self.rel, edges):
            src, dst = e[0], e[1]
            agg = torch.zeros_like(h).index_add_(0, dst, lin(h)[src])
            deg = torch.zeros_like(ones).index_add_(0, dst, ones[src]).clamp(min=1.0)
            out = out + agg / deg
        return self.norm(h + F.gelu(out))


class CycleGNN(nn.Module):
    """Message passing on the performance graph; Gaussian heads for offset and intensity."""

    def __init__(self, n_in: int, dim: int = 64, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, dim), nn.GELU(), nn.Linear(dim, dim))
        self.layers = nn.ModuleList(RelationalLayer(dim) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(dim, 4)     # delta mean, delta log-sd, intensity mean, intensity log-sd

    def forward(self, x, edges):
        h = self.inp(x)
        for layer in self.layers:
            h = self.drop(layer(h, edges))
        out = self.head(h)
        return out[:, 0], out[:, 1].clamp(-5, 2), out[:, 2], out[:, 3].clamp(-5, 2)


def gaussian_nll(y, mu, log_sd):
    return 0.5 * np.log(2 * np.pi) + log_sd + 0.5 * ((y - mu) / log_sd.exp()) ** 2


def losses(model, batch):
    dm, dls, im, ils = model(batch["x"], batch["edges"])
    md, mi = batch["mask_delta"], batch["mask_intensity"]
    ld = gaussian_nll(batch["delta"][md], dm[md], dls[md])
    li = gaussian_nll(batch["intensity"][mi], im[mi], ils[mi])
    return ld, li


def train(train_w, val_w, *, K=4, position="subpulse", dim=64, layers=3, dropout=0.1, lr=2e-3,
          weight_decay=1e-4, epochs=40, batch_size=32, patience=8, seed=0, device="cpu", log=None):
    """Train with early stopping on the validation windows; returns the best model and its history."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_in = n_base_features() + position_features(position, 1).shape[1]
    model = CycleGNN(n_in, dim, layers, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    val_batch = collate(val_w, K, position, device)
    best, best_state, wait, history = np.inf, None, 0, []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_w))
        tot, n = 0.0, 0
        for b in range(0, len(order), batch_size):
            batch = collate([train_w[j] for j in order[b:b + batch_size]], K, position, device)
            ld, li = losses(model, batch)
            loss = ld.mean() + li.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, n = tot + loss.item(), n + 1
        sched.step()
        model.eval()
        with torch.no_grad():
            ld, li = losses(model, val_batch)
            val = (ld.mean() + li.mean()).item()
        history.append({"epoch": epoch, "train_loss": tot / n, "val_loss": val,
                        "val_nll_delta": ld.mean().item(), "val_nll_intensity": li.mean().item()})
        if log:
            log(history[-1])
        if val < best - 1e-4:
            best, wait = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate(model, windows, *, K=4, position="subpulse", device="cpu", batch_size=256,
             instruments=(0, 1, 2, 3)) -> dict:
    """Mean NLL over the target strokes of the listed instruments (0-3 = J1, J2, D1, D2)."""
    model.eval()
    ld_all, li_all = [], []
    for b in range(0, len(windows), batch_size):
        batch = collate(windows[b:b + batch_size], K, position, device)
        keep = torch.isin(batch["inst"], torch.as_tensor(instruments, device=device))
        batch["mask_delta"] &= keep
        batch["mask_intensity"] &= keep
        ld, li = losses(model, batch)
        ld_all.append(ld.cpu().numpy())
        li_all.append(li.cpu().numpy())
    ld, li = np.concatenate(ld_all), np.concatenate(li_all)
    return {"nll_delta": float(ld.mean()), "n_delta": int(ld.size),
            "nll_intensity": float(li.mean()), "n_intensity": int(li.size)}
