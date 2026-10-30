"""A small Music-Transformer-style model for the soloist (roadmap step J6).

The comparison point for the graph model of ``djembe.solo_seq``: same windows (K
cycles, the last one the target), same information, same factorisation, scored on the
same strokes.

- Encoder (bidirectional) over the accompaniment: one token per pulse of the window
  (K x 12), holding jembe 2 and the dunduns at that pulse (strike, intensity, offset,
  present), the section, the cycle duration, the piece (zeroed with
  ``piece_input=False``) and a target-cycle flag. As in J5b the accompaniment of the
  target cycle is known: it plays while the soloist plays.
- Decoder (causal) over the soloist line: the token at pulse p carries the soloist's
  strike, intensity and offset at pulse p - 1, so the output at p sees the soloist up
  to p - 1 and, through cross-attention, the whole accompaniment.
- Heads at p: a strike logit, and Gaussians for offset and intensity trained on strokes
  only, i.e. p(hit_p | past) and p(offset_p, intensity_p | hit_p = 1, past).

Positions: a learned embedding of the pulse in the cycle (0-11), like the graph model's
``position="full"``, plus a learned relative-position bias per attention head over the
distance between tokens (self- and cross-attention). This is the relative attention of
the Music Transformer (Huang et al., 2018) in the simpler additive-bias form of
Raffel et al. (2020); it is our own implementation, not the Magenta code.

The decoder is trained on every pulse of the window (teacher forcing); scores are taken
on the target cycle only, where its information matches the graph model's.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from djembe import gnn
from djembe.dataset import PIECES
from djembe.grid import SECTIONS, TakeGrid

SOLO = 0
ENS = (1, 2, 3)
DELTA_CLIP = 0.5
N_ENC = 4 * len(ENS) + len(SECTIONS) + 1 + len(PIECES) + 2
N_DEC = 4


@dataclass
class TokWindow:
    enc: np.ndarray          # (K*12, N_ENC)
    solo: np.ndarray         # (K*12, 3) soloist hit, intensity, offset (targets and teacher-forcing inputs)
    solo_present: bool
    piece: str


def tok_window(grid: TakeGrid, t: int, K: int, piece_input: bool = True) -> TokWindow:
    cyc = np.arange(t - K + 1, t + 1)
    x = grid.x[cyc]                                       # (K, 12, 4, 3)
    enc = np.zeros((K, 12, N_ENC), np.float32)
    for j, i in enumerate(ENS):
        h = x[:, :, i, 0]
        enc[:, :, 4 * j] = h
        enc[:, :, 4 * j + 1] = h * x[:, :, i, 1]
        enc[:, :, 4 * j + 2] = h * np.clip(x[:, :, i, 2], -1, 1)
        enc[:, :, 4 * j + 3] = float(grid.present[i])
    o = 4 * len(ENS)
    enc[:, :, o:o + len(SECTIONS)] = np.eye(len(SECTIONS), dtype=np.float32)[grid.section[cyc]][:, None, :]
    o += len(SECTIONS)
    enc[:, :, o] = ((grid.log_T[cyc] - gnn.LOG_T_REF) / gnn.LOG_T_SCALE)[:, None]
    o += 1
    if piece_input:
        enc[:, :, o + PIECES.index(grid.piece)] = 1.0
    o += len(PIECES)
    enc[:, :, o] = float(grid.present[SOLO])
    enc[-1, :, o + 1] = 1.0                              # target cycle
    return TokWindow(enc.reshape(K * 12, N_ENC), x[:, :, SOLO].reshape(K * 12, 3).astype(np.float32),
                     bool(grid.present[SOLO]), grid.piece)


def _dec_inputs(solo: torch.Tensor) -> torch.Tensor:
    """(B, L, 3) soloist values -> (B, L, N_DEC): values at p - 1, plus a start flag at p = 0."""
    prev = torch.zeros(solo.shape[0], solo.shape[1], N_DEC, device=solo.device)
    h = solo[:, :-1, 0]
    prev[:, 1:, 0] = h
    prev[:, 1:, 1] = h * solo[:, :-1, 1]
    prev[:, 1:, 2] = h * solo[:, :-1, 2].clamp(-1, 1)
    prev[:, 0, 3] = 1.0
    return prev


class RelAttention(nn.Module):
    def __init__(self, dim: int, heads: int, max_len: int):
        super().__init__()
        self.h, self.dk, self.max_len = heads, dim // heads, max_len
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.rel = nn.Parameter(torch.zeros(heads, 2 * max_len - 1))

    def forward(self, x, mem, causal: bool = False):
        B, L, D = x.shape
        M = mem.shape[1]
        q = self.q(x).view(B, L, self.h, self.dk).transpose(1, 2)
        k = self.k(mem).view(B, M, self.h, self.dk).transpose(1, 2)
        v = self.v(mem).view(B, M, self.h, self.dk).transpose(1, 2)
        att = q @ k.transpose(-1, -2) / np.sqrt(self.dk)
        rel = torch.arange(L, device=x.device)[:, None] - torch.arange(M, device=x.device)[None, :]
        att = att + self.rel[:, rel + self.max_len - 1][None]
        if causal:
            att = att.masked_fill(rel[None, None] < 0, float("-inf"))
        out = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, L, D)
        return self.o(out)


class Block(nn.Module):
    """Pre-norm block: self-attention, optional cross-attention, feed-forward."""

    def __init__(self, dim, heads, ff, dropout, max_len, cross: bool):
        super().__init__()
        self.n1, self.att = nn.LayerNorm(dim), RelAttention(dim, heads, max_len)
        self.cross = cross
        if cross:
            self.n2, self.catt = nn.LayerNorm(dim), RelAttention(dim, heads, max_len)
        self.n3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem=None, causal=False):
        y = self.n1(x)
        x = x + self.drop(self.att(y, y, causal))
        if self.cross:
            x = x + self.drop(self.catt(self.n2(x), mem))
        return x + self.drop(self.ff(self.n3(x)))


class SoloTransformer(nn.Module):
    def __init__(self, K: int = 4, dim: int = 64, heads: int = 4, enc_layers: int = 2, dec_layers: int = 2,
                 ff: int = 128, dropout: float = 0.1):
        super().__init__()
        L = K * 12
        self.enc_in = nn.Sequential(nn.Linear(N_ENC, dim), nn.GELU(), nn.Linear(dim, dim))
        self.dec_in = nn.Sequential(nn.Linear(N_DEC, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pulse = nn.Embedding(12, dim)
        self.enc = nn.ModuleList(Block(dim, heads, ff, dropout, L, cross=False) for _ in range(enc_layers))
        self.dec = nn.ModuleList(Block(dim, heads, ff, dropout, L, cross=True) for _ in range(dec_layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 5)
        self.register_buffer("pulse_idx", torch.arange(L) % 12, persistent=False)

    def forward(self, enc, solo):
        pos = self.pulse(self.pulse_idx)[None]
        m = self.enc_in(enc) + pos
        for blk in self.enc:
            m = blk(m)
        h = self.dec_in(_dec_inputs(solo)) + pos
        for blk in self.dec:
            h = blk(h, m, causal=True)
        o = self.head(self.norm(h))
        return o[..., 0], o[..., 1], o[..., 2].clamp(-5, 2), o[..., 3], o[..., 4].clamp(-5, 2)


def _tensors(windows, device):
    t = lambda a: torch.as_tensor(np.stack(a), dtype=torch.float32, device=device)
    return t([w.enc for w in windows]), t([w.solo for w in windows]), \
        torch.as_tensor([w.solo_present for w in windows], device=device)


def nll_terms(out, solo):
    logit, dm, dls, im, ils = out
    hit, inten, delta = solo[..., 0], solo[..., 1], solo[..., 2]
    return (F.binary_cross_entropy_with_logits(logit, hit, reduction="none"), gnn.gaussian_nll(delta, dm, dls),
            gnn.gaussian_nll(inten, im, ils))


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def train(windows: list[TokWindow], *, K=4, dim=64, heads=4, enc_layers=2, dec_layers=2, ff=128, dropout=0.1,
          lr=1e-3, weight_decay=1e-4, epochs=40, batch_size=32, seed=0, device="cpu") -> SoloTransformer:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SoloTransformer(K, dim, heads, enc_layers, dec_layers, ff, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    windows = [w for w in windows if w.solo_present]
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(windows))
        for b in range(0, len(order), batch_size):
            enc, solo, _ = _tensors([windows[j] for j in order[b:b + batch_size]], device)
            l_hit, l_d, l_i = nll_terms(model(enc, solo), solo)
            stroke = solo[..., 0] > 0
            md = stroke & (solo[..., 2].abs() <= DELTA_CLIP)
            loss = l_hit.mean() + l_d[md].mean() + l_i[stroke].mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def score(model, windows: list[TokWindow], device="cpu", batch_size=256) -> dict:
    """NLL on the target cycle's soloist pulses (the last 12 tokens)."""
    hs, ds, is_ = [], [], []
    for b in range(0, len(windows), batch_size):
        enc, solo, _ = _tensors(windows[b:b + batch_size], device)
        l_hit, l_d, l_i = (v[:, -12:].cpu().numpy() for v in nll_terms(model(enc, solo), solo))
        s = solo[:, -12:].cpu().numpy()
        h = s[..., 0] > 0
        hs.append(l_hit.reshape(-1))
        ds.append(l_d[h & (np.abs(s[..., 2]) <= DELTA_CLIP)])
        is_.append(l_i[h])
    h, d, i = (np.concatenate(v) for v in (hs, ds, is_))
    return {"nll_hit": float(h.mean()), "n_hit": int(h.size), "nll_delta": float(d.mean()), "n_delta": int(d.size),
            "nll_intensity": float(i.mean()), "n_intensity": int(i.size)}


@torch.no_grad()
def generate(grid: TakeGrid, model, *, K=4, piece_input=True, rng=None, temperature: float = 1.0, device="cpu",
             n_context: int | None = None) -> TakeGrid:
    """Soloist drawn stroke by stroke after ``n_context`` real cycles; the ensemble is the real one."""
    rng = rng or np.random.default_rng(0)
    n_context = K - 1 if n_context is None else n_context
    g = replace(grid, x=grid.x.copy())
    g.x[n_context:, :, SOLO] = 0.0
    for t in range(n_context, g.x.shape[0]):
        w = tok_window(g, t, K, piece_input)
        enc, solo, _ = _tensors([w], device)
        for k in range(12):
            p = (K - 1) * 12 + k
            logit, dm, dls, im, ils = (float(o[0, p]) for o in model(enc, solo))
            if rng.random() < 1.0 / (1.0 + np.exp(-logit / temperature)):
                solo[0, p, 0] = 1.0
                solo[0, p, 2] = float(np.clip(rng.normal(dm, np.exp(dls)), -DELTA_CLIP, DELTA_CLIP))
                solo[0, p, 1] = float(rng.normal(im, np.exp(ils)))
        s = solo[0, -12:].cpu().numpy()
        g.x[t, :, SOLO, 0], g.x[t, :, SOLO, 1], g.x[t, :, SOLO, 2] = s[:, 0], s[:, 0] * s[:, 1], s[:, 0] * s[:, 2]
    return g
