"""Two-stage soloist: a figure planner, then the graph model playing the cycle (roadmap step J8).

Real soloists build their cycles from figures they have just played: 58 % of their
cycles repeat exactly a figure of the last 32 cycles, 20 % change it by one stroke, 11 %
by two; 11 % are new. Of the exact repeats, 43 % repeat the previous cycle and 73 % a
figure of the last 3. The J6 graph model sees those 3 cycles, yet repeats a figure in
only 31 % of its cycles: it draws the 12 pulses one by one, and even at 90 % per pulse
twelve right draws in a row happen in 0.9^12 = 28 % of cycles. The structure of a solo
is a decision about the whole cycle.

Stage 1, the planner, decides it. At the start of cycle t the candidates are the
distinct figures (strike patterns) of the soloist in the last ``H`` cycles, plus "no
base". Each candidate is scored from its pattern, how recently and how often it was
played, whether it was the last cycle's figure, its distance to the last figure, and
the context (the accompaniment of cycle t, the last figure, the section, the piece); a
softmax over the candidates gives the plan. Training target: the candidate nearest the
real figure if it is at most ``MAX_EDIT`` strokes away (most recent on ties), else no
base.

Stage 2, the executor, is the stroke-by-stroke graph model of ``djembe.solo_seq`` with
four more input columns on the target cycle's soloist nodes: the base figure's strike,
intensity and offset at that pulse (from the last time it was played) and a flag that
there is a base. It is trained with the planner's target as base, so it learns how
closely soloists follow, or vary, the figure they take up.

In generation the planner draws a base for each cycle from the line generated so far,
then the executor plays the cycle pulse by pulse.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from djembe import solo, solo_seq
from djembe.dataset import PIECES
from djembe.grid import SECTIONS, TakeGrid

H_PLAN = 32
MAX_EDIT = 2
N_BASE_COLS = 4
DELTA_CLIP = 0.5


@dataclass
class Candidates:
    pattern: np.ndarray      # (M, 12) strike pattern of each distinct earlier figure
    last_row: np.ndarray     # (M, 12, 3) the soloist's (hit, intensity, offset) the last time it was played
    feats: np.ndarray        # (M, N_CAND) planner features
    context: np.ndarray      # (N_CTX,) features shared by every candidate


N_CAND = 12 + 5
N_CTX = 36 + 12 + len(SECTIONS) + len(PIECES) + 2


def candidates(grid: TakeGrid, t: int, H: int = H_PLAN) -> Candidates:
    """Distinct soloist figures of cycles [t-H, t-1], most recent occurrence kept, and their features."""
    lo = max(0, t - H)
    rows = grid.x[lo:t, :, solo.SOLO]
    pats, last, count, age = {}, {}, {}, {}
    for j in range(len(rows) - 1, -1, -1):                        # newest first
        key = tuple((rows[j, :, 0] > 0).astype(int))
        if key not in pats:
            pats[key], last[key], age[key] = np.array(key, np.float32), rows[j], t - (lo + j)
        count[key] = count.get(key, 0) + 1
    keys = list(pats)
    prev = (grid.x[t - 1, :, solo.SOLO, 0] > 0).astype(np.float32) if t > 0 else np.zeros(12, np.float32)
    feats = np.array([np.concatenate([pats[k], [np.log1p(age[k]), np.log1p(count[k]), float(age[k] == 1),
                                                np.abs(pats[k] - prev).sum() / 12, pats[k].sum() / 12]])
                      for k in keys], np.float32).reshape(len(keys), N_CAND)
    ens = (grid.x[t, :, 1:, 0] > 0).astype(np.float32).reshape(-1)
    ctx = np.concatenate([ens, prev, np.eye(len(SECTIONS), dtype=np.float32)[grid.section[t]],
                          np.eye(len(PIECES), dtype=np.float32)[PIECES.index(grid.piece)]
                          if grid.piece in PIECES else np.zeros(len(PIECES), np.float32),
                          [prev.sum() / 12, np.log1p(len(keys))]]).astype(np.float32)
    return Candidates(np.array([pats[k] for k in keys], np.float32).reshape(len(keys), 12),
                      np.array([last[k] for k in keys], np.float32).reshape(len(keys), 12, 3), feats, ctx)


def target_base(cands: Candidates, figure: np.ndarray, max_edit: int = MAX_EDIT) -> int:
    """Index of the candidate nearest the real figure (most recent on ties) if within max_edit, else M (no base)."""
    if len(cands.pattern) == 0:
        return 0
    d = np.abs(cands.pattern - (figure > 0)).sum(1)
    i = int(np.argmin(d))                                          # candidates are newest first
    return i if d[i] <= max_edit else len(cands.pattern)


class Planner(nn.Module):
    def __init__(self, dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.ctx = nn.Sequential(nn.Linear(N_CTX, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.cand = nn.Sequential(nn.Linear(N_CAND + dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1))
        self.none = nn.Sequential(nn.GELU(), nn.Linear(dim, 1))

    def logits_batch(self, feats: torch.Tensor, mask: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """(B, M+1) scores for padded candidates (B, M, N_CAND) with mask (B, M); the last column is "no base"."""
        h = self.ctx(ctx)                                                                 # (B, d)
        s = self.cand(torch.cat([feats, h[:, None, :].expand(-1, feats.shape[1], -1)], -1))[..., 0]
        s = s.masked_fill(~mask, float("-inf"))
        return torch.cat([s, self.none(h)], 1)

    def logits(self, c: Candidates) -> torch.Tensor:
        """(M+1,) scores: the candidates, then "no base"."""
        f, m, x, _ = _pad([(c, 0)])
        lg = self.logits_batch(f, m, x)[0]
        return torch.cat([lg[:len(c.pattern)], lg[-1:]])


def _pad(examples):
    M = max(1, max(len(c.pattern) for c, _ in examples))
    B = len(examples)
    feats = torch.zeros(B, M, N_CAND)
    mask = torch.zeros(B, M, dtype=torch.bool)
    ctx = torch.zeros(B, N_CTX)
    y = torch.zeros(B, dtype=torch.long)
    for i, (c, t) in enumerate(examples):
        n = len(c.pattern)
        feats[i, :n] = torch.as_tensor(c.feats)
        mask[i, :n] = True
        ctx[i] = torch.as_tensor(c.context)
        y[i] = t if t < n else M                                   # "no base" is the last column
    return feats, mask, ctx, y


def plan_examples(grids: list[TakeGrid], start: int = 1, H: int = H_PLAN):
    out = []
    for g in grids:
        for t in range(start, g.x.shape[0]):
            c = candidates(g, t, H)
            out.append((c, target_base(c, g.x[t, :, solo.SOLO, 0])))
    return out


def train_planner(examples, *, epochs: int = 30, lr: float = 2e-3, seed: int = 0, batch_size: int = 64) -> Planner:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Planner()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    f_all, m_all, x_all, y_all = _pad(examples)                     # padded once
    for _ in range(epochs):
        model.train()
        order = torch.as_tensor(rng.permutation(len(examples)))
        for s in range(0, len(order), batch_size):
            i = order[s:s + batch_size]
            loss = F.cross_entropy(model.logits_batch(f_all[i], m_all[i], x_all[i]), y_all[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def planner_scores(model: Planner, examples) -> dict:
    """NLL and accuracy of the plan, against two rules: always take the last cycle's figure, and the empirical
    frequency of "no base"."""
    nll, acc, last_acc = [], [], []
    for c, y in examples:
        lg = model.logits(c)
        nll.append(float(F.cross_entropy(lg[None], torch.tensor([y]))))
        acc.append(int(lg.argmax()) == y)
        is_last = np.flatnonzero(c.feats[:, 14] > 0) if len(c.feats) else []
        last_acc.append((int(is_last[0]) if len(is_last) else len(c.pattern)) == y)
    ys = np.array([y == len(c.pattern) for c, y in examples])
    return {"plan_nll": float(np.mean(nll)), "plan_acc": float(np.mean(acc)),
            "rule_last_acc": float(np.mean(last_acc)), "no_base_share": float(ys.mean())}


def base_columns(n_nodes: int, base_row: np.ndarray | None) -> np.ndarray:
    """(n_nodes, 4) extra input: the base figure on the target cycle's soloist nodes."""
    out = np.zeros((n_nodes, N_BASE_COLS), np.float32)
    if base_row is not None:
        nodes = solo._solo_nodes(n_nodes)
        h = base_row[:, 0]
        out[nodes, 0] = h
        out[nodes, 1] = h * base_row[:, 1]
        out[nodes, 2] = h * np.clip(base_row[:, 2], -1, 1)
        out[nodes, 3] = 1.0
    return out


def exec_window(grid: TakeGrid, t: int, K: int, base_row: np.ndarray | None, piece_input: bool = True):
    w = solo_seq.seq_window(grid, t, K, piece_input)
    return replace(w, x=np.concatenate([w.x, base_columns(w.x.shape[0], base_row)], 1))


def exec_windows(grids: list[TakeGrid], K: int = 4, H: int = H_PLAN):
    """Executor training windows, each with the planner's target as base."""
    out = []
    for g in grids:
        for t in range(K - 1, g.x.shape[0]):
            c = candidates(g, t, H)
            y = target_base(c, g.x[t, :, solo.SOLO, 0])
            out.append(exec_window(g, t, K, c.last_row[y] if y < len(c.pattern) else None))
    return out


@torch.no_grad()
def generate(grid: TakeGrid, planner: Planner, executor, *, K: int = 4, rng=None, temperature: float = 1.0,
             plan_temperature: float = 1.0, n_context: int | None = None, H: int = H_PLAN, log: list | None = None):
    """Each cycle: the planner draws a base from the line so far, the executor plays the cycle stroke by stroke."""
    rng = rng or np.random.default_rng(0)
    n_context = K - 1 if n_context is None else n_context
    g = replace(grid, x=grid.x.copy())
    g.x[n_context:, :, solo.SOLO] = 0.0
    for t in range(n_context, g.x.shape[0]):
        c = candidates(g, t, H)
        p = torch.softmax(planner.logits(c) / plan_temperature, 0).numpy()
        y = int(rng.choice(len(p), p=p / p.sum()))
        base = c.last_row[y] if y < len(c.pattern) else None
        if log is not None:
            log.append(y < len(c.pattern))
        w = exec_window(g, t, K, base)
        node = solo._solo_nodes(w.x.shape[0])
        hit, delta, inten = (np.zeros(12, np.float32) for _ in range(3))
        for k in range(12):
            b = solo._collate([w.reveal(k, hit, delta, inten)], [w.present], K, "full", True, "cpu")
            logit, dm, dls, im, ils = (float(o[node[k]]) for o in executor(b["x"], b["edges"]))
            if rng.random() < 1.0 / (1.0 + np.exp(-logit / temperature)):
                hit[k] = 1.0
                delta[k] = np.clip(rng.normal(dm, np.exp(dls)), -DELTA_CLIP, DELTA_CLIP)
                inten[k] = rng.normal(im, np.exp(ils))
        g.x[t, :, solo.SOLO, 0], g.x[t, :, solo.SOLO, 1], g.x[t, :, solo.SOLO, 2] = hit, hit * inten, hit * delta
    return g
