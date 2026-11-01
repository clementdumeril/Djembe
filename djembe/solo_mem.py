"""Soloist with a memory of its figures and a groove encoder (roadmap step J7).

The J6 graph model sees K = 4 cycles, about 6 s. Within that span its timing hangs
together like a real soloist's, but everything played earlier is forgotten: real
soloists return to a figure they already played in 64 % of their cycles, the model in
32 %, and it changes figure almost every cycle. A soloist improvises on a base: a
vocabulary of figures, taken up again and varied, in a feel that stays the same.

Two additions, each optional, on top of ``djembe.solo_seq`` (same windows, same
factorisation, same training on random prefixes):

- groove encoder: the H cycles before the window (every drum, the soloist included,
  plus the cycle duration and the age of the cycle) are embedded one by one and averaged
  into one vector, added to every node's input. It summarises the performance so far:
  its density, accents and feel. In generation the history contains what the model
  itself played, so its own style feeds back.
- figure memory: the soloist's figures of those H cycles are memory slots. After the
  second-to-last message-passing layer, each soloist node of the target cycle (pulse k)
  attends over the slots, with a learned "no figure" slot, and receives what the chosen
  figures play at pulse k (the value of a slot is pulse-aligned). Because the pulses
  already played in the target cycle are revealed, the query knows how the cycle has
  begun and can pick a figure that began the same way.

With both off the network is the J6 model (``SoloGNN``), so the four variants are
compared on exactly the same code and data.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from djembe import gnn, solo, solo_seq
from djembe.grid import TakeGrid, position_features

H_DEFAULT = 32
N_CYCLE_FEAT = 12 * 4 * 3 + 1          # every drum's (hit, intensity, offset) per pulse, and the cycle duration
DELTA_CLIP = 0.5


@dataclass
class MemWindow:
    seq: solo_seq.SeqWindow
    hist: np.ndarray        # (H, N_CYCLE_FEAT) cycles before the window, most recent last
    age: np.ndarray         # (H,) cycles between that cycle and the target
    mask: np.ndarray        # (H,) True where a cycle exists


def mem_window(grid: TakeGrid, t: int, K: int = 4, piece_input: bool = True, causal: bool = False,
               H: int = H_DEFAULT) -> MemWindow:
    seq = solo_seq.seq_window(grid, t, K, piece_input, causal)
    hist = np.zeros((H, N_CYCLE_FEAT), np.float32)
    age = np.zeros(H, np.float32)
    mask = np.zeros(H, bool)
    last = t - K                                      # newest cycle before the window
    for j in range(H):
        c = last - (H - 1 - j)
        if c < 0:
            continue
        x = grid.x[c].copy()
        x[..., 2] = np.clip(x[..., 2], -1, 1)
        hist[j, :-1] = x.reshape(-1)
        hist[j, -1] = (grid.log_T[c] - gnn.LOG_T_REF) / gnn.LOG_T_SCALE
        age[j] = t - c
        mask[j] = True
    return MemWindow(seq, hist, age, mask)


def _solo_figure(hist: torch.Tensor) -> torch.Tensor:
    """(..., N_CYCLE_FEAT) -> (..., 12, 3): the soloist's (hit, intensity, offset) per pulse."""
    return hist[..., :-1].reshape(*hist.shape[:-1], 12, 4, 3)[..., solo.SOLO, :]


class SoloMemGNN(nn.Module):
    def __init__(self, n_in: int, dim: int = 64, layers: int = 3, dropout: float = 0.1, use_memory: bool = True,
                 use_groove: bool = True):
        super().__init__()
        self.use_memory, self.use_groove = use_memory, use_groove
        self.inp = nn.Sequential(nn.Linear(n_in, dim), nn.GELU(), nn.Linear(dim, dim))
        self.layers = nn.ModuleList(gnn.RelationalLayer(dim) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(dim, 5)
        if use_groove:
            self.cyc_enc = nn.Sequential(nn.Linear(N_CYCLE_FEAT + 1, dim), nn.GELU(), nn.Linear(dim, dim))
            self.groove_out = nn.Linear(dim, dim)
        if use_memory:
            self.fig_enc = nn.Sequential(nn.Linear(12 * 3 + 1, dim), nn.GELU(), nn.Linear(dim, dim))
            self.val = nn.Sequential(nn.Linear(3 + dim, dim), nn.GELU(), nn.Linear(dim, dim))
            self.q, self.k = nn.Linear(dim, dim), nn.Linear(dim, dim)
            self.null_key = nn.Parameter(torch.zeros(dim))
            self.mem_out = nn.Linear(dim, dim)
            nn.init.zeros_(self.mem_out.weight)
            nn.init.zeros_(self.mem_out.bias)
            self.mem_norm = nn.LayerNorm(dim)

    def forward(self, x, edges, hist, age, mask, n_nodes: int):
        B = hist.shape[0]
        h = self.inp(x)
        log_age = torch.log1p(age)[..., None] / 4.0
        if self.use_groove:
            e = self.cyc_enc(torch.cat([hist, log_age], -1))                    # (B, H, d)
            w = mask.float()[..., None]
            g = (e * w).sum(1) / w.sum(1).clamp(min=1.0)                         # (B, d)
            h = h + self.groove_out(g).repeat_interleave(n_nodes, dim=0)
        for i, layer in enumerate(self.layers):
            h = self.drop(layer(h, edges))
            if self.use_memory and i == len(self.layers) - 2:
                h = self._recall(h, hist, log_age, mask, n_nodes, B)
        o = self.head(h)
        return o[:, 0], o[:, 1], o[:, 2].clamp(-5, 2), o[:, 3], o[:, 4].clamp(-5, 2)

    def _recall(self, h, hist, log_age, mask, n_nodes, B):
        fig = _solo_figure(hist)                                                 # (B, H, 12, 3)
        e = self.fig_enc(torch.cat([fig.reshape(B, fig.shape[1], 36), log_age], -1))   # (B, H, d)
        v = self.val(torch.cat([fig, e[:, :, None, :].expand(-1, -1, 12, -1)], -1))     # (B, H, 12, d)
        idx = torch.as_tensor(solo._solo_nodes(n_nodes), device=h.device)
        nodes = (torch.arange(B, device=h.device)[:, None] * n_nodes + idx[None, :]).reshape(-1)
        q = self.q(h[nodes]).view(B, 12, -1)                                     # (B, 12, d)
        keys = torch.cat([self.null_key.expand(B, 1, -1), self.k(e)], 1)        # (B, 1+H, d)
        score = q @ keys.transpose(1, 2) / np.sqrt(q.shape[-1])                  # (B, 12, 1+H)
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=h.device), mask], 1)
        score = score.masked_fill(~valid[:, None, :], float("-inf"))
        a = score.softmax(-1)[:, :, 1:]                                          # weight of each figure (null: nothing)
        r = torch.einsum("bkm,bmkd->bkd", a, v)                                  # pulse-aligned recall
        h = h.clone()
        h[nodes] = self.mem_norm(h[nodes] + self.mem_out(r.reshape(-1, r.shape[-1])))
        return h


def n_in(position: str = "full") -> int:
    return gnn.n_base_features() + position_features(position, 1).shape[1]


def _batch(windows: list[MemWindow], ks, K, position, helix, device):
    b, nodes, hit, delta, inten, hidden = solo_seq._batch([w.seq for w in windows], ks, K, position, helix, device)
    t = lambda a, dt=torch.float32: torch.as_tensor(np.stack(a), dtype=dt, device=device)
    return b, nodes, hit, delta, inten, hidden, t([w.hist for w in windows]), t([w.age for w in windows]), \
        t([w.mask for w in windows], torch.bool)


def _forward(model, b, hist, age, mask, windows):
    return model(b["x"], b["edges"], hist, age, mask, windows[0].seq.x.shape[0])


def train(windows: list[MemWindow], *, K=4, position="full", helix=True, dim=64, layers=3, dropout=0.1, lr=2e-3,
          weight_decay=1e-4, epochs=40, batch_size=32, seed=0, device="cpu", use_memory=True,
          use_groove=True) -> SoloMemGNN:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SoloMemGNN(n_in(position), dim, layers, dropout, use_memory, use_groove).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(windows))
        for s in range(0, len(order), batch_size):
            chunk = [windows[j] for j in order[s:s + batch_size]]
            ks = [int(rng.integers(0, 12)) for _ in chunk]
            b, nodes, hit, delta, inten, hidden, hist, age, mask = _batch(chunk, ks, K, position, helix, device)
            l_hit, l_d, l_i = solo_seq.nll_terms(_forward(model, b, hist, age, mask, chunk), nodes, hit, delta, inten)
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
def score(model, windows: list[MemWindow], *, K=4, position="full", helix=True, device="cpu", batch_size=256) -> dict:
    hs, ds, is_ = [], [], []
    for k in range(12):
        for s in range(0, len(windows), batch_size):
            chunk = windows[s:s + batch_size]
            b, nodes, hit, delta, inten, _, hist, age, mask = _batch(chunk, [k] * len(chunk), K, position, helix, device)
            out = _forward(model, b, hist, age, mask, chunk)
            l_hit, l_d, l_i = (v.cpu().numpy().reshape(len(chunk), 12)[:, k]
                               for v in solo_seq.nll_terms(out, nodes, hit, delta, inten))
            h = np.array([w.seq.hit[k] for w in chunk]) > 0
            d = np.array([w.seq.delta[k] for w in chunk])
            hs.append(l_hit)
            ds.append(l_d[h & (np.abs(d) <= DELTA_CLIP)])
            is_.append(l_i[h])
    h, d, i = (np.concatenate(v) for v in (hs, ds, is_))
    return {"nll_hit": float(h.mean()), "n_hit": int(h.size), "nll_delta": float(d.mean()), "n_delta": int(d.size),
            "nll_intensity": float(i.mean()), "n_intensity": int(i.size)}


@torch.no_grad()
def generate(grid: TakeGrid, model, *, K=4, position="full", helix=True, piece_input=True, causal=False, rng=None,
             temperature: float = 1.0, device="cpu", n_context: int | None = None, H: int = H_DEFAULT) -> TakeGrid:
    """Soloist drawn stroke by stroke; memory and groove are read from the line generated so far."""
    rng = rng or np.random.default_rng(0)
    n_context = K - 1 if n_context is None else n_context
    g = replace(grid, x=grid.x.copy())
    g.x[n_context:, :, solo.SOLO] = 0.0
    for t in range(n_context, g.x.shape[0]):
        w = mem_window(g, t, K, piece_input, causal, H)
        node = solo._solo_nodes(w.seq.x.shape[0])
        hist, age, mask = (torch.as_tensor(a[None], device=device) for a in (w.hist, w.age, w.mask))
        hit, delta, inten = (np.zeros(12, np.float32) for _ in range(3))
        for k in range(12):
            b = solo._collate([w.seq.reveal(k, hit, delta, inten)], [w.seq.present], K, position, helix, device)
            logit, dm, dls, im, ils = (float(o[node[k]]) for o in model(b["x"], b["edges"], hist, age, mask,
                                                                          w.seq.x.shape[0]))
            if rng.random() < 1.0 / (1.0 + np.exp(-logit / temperature)):
                hit[k] = 1.0
                delta[k] = np.clip(rng.normal(dm, np.exp(dls)), -DELTA_CLIP, DELTA_CLIP)
                inten[k] = rng.normal(im, np.exp(ils))
        g.x[t, :, solo.SOLO, 0], g.x[t, :, solo.SOLO, 1], g.x[t, :, solo.SOLO, 2] = hit, hit * inten, hit * delta
    return g
