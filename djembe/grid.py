"""Cycle grid and performance graph built from the event table.

The grid of a take is an array ``x[c, k, i, f]``: cycle ``c``, pulse ``k`` (0-11),
instrument ``i`` (J1, J2, D1, D2) and three features ``f``:

- ``hit``: 1 if the instrument plays on that pulse, else 0;
- ``intensity``: log peak amplitude, z-scored per take and instrument (0 when no hit);
- ``delta``: offset from the even 12-pulse grid in fractions of a pulse
  (``dev_sd`` in the event table; 0 when no hit).

Times are never fed as seconds. An onset is described by its cycle, its pulse and
its offset; the cycle duration ``T_c`` is kept apart as ``log_T`` so that tempo
(and the accelerando) can be modelled separately from the feel.

The graph over a window of cycles has one node per (cycle, pulse, instrument)
and three directed edge types:

- ``next_pulse``: (c, k, i) -> (c, k+1 mod 12, i), a ring of 12 per cycle and instrument;
- ``same_pulse``: (c, k, i) -> (c, k, j) for every other instrument j;
- ``prev_cycle``: (c-1, k, i) -> (c, k, i).

Rotating every cycle by r pulses maps each edge set onto itself (tests check it
for all r), so a message-passing network without positional input is equivariant
to the cyclic group C12. ``position_features`` decides how much of that symmetry
the model keeps: ``"none"`` (C12), ``"subpulse"`` (k mod 3, so only rotations by
whole beats, C4) or ``"full"`` (no symmetry).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from djembe.dataset import INSTRUMENTS, N_SUBDIVISIONS

FEATURES = ("hit", "intensity", "delta")
SECTIONS = ("basic", "heating", "impro", "other")
EDGE_TYPES = ("next_pulse", "same_pulse", "prev_cycle")


def section_class(label) -> str:
    """Map the corpus' free-text form labels to four classes."""
    if not isinstance(label, str):
        return "other"
    s = label.lower()
    if re.search(r"heat|finale", s):
        return "heating"
    if re.search(r"impro|diverse", s):
        return "impro"
    if re.search(r"basic|theme|pattern|groove", s):
        return "basic"
    return "other"


@dataclass
class TakeGrid:
    take_id: str
    piece: str
    x: np.ndarray                 # (n_cycles, 12, 4, 3) float32
    log_T: np.ndarray             # (n_cycles,) log cycle duration [log s]
    section: np.ndarray           # (n_cycles,) index into SECTIONS
    in_signal: np.ndarray         # (n_cycles,) True if a signal phrase (break) overlaps the cycle
    present: np.ndarray           # (4,) True if the instrument is recorded in this take
    cycle_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, int))


def take_grid(events: pd.DataFrame, take_id: str) -> TakeGrid:
    """Grid of one take. Cycles without an end time (the last one) are dropped."""
    ev = events[(events["take_id"] == take_id) & events["cycle_dur_s"].notna()].copy()
    cycles = np.sort(ev["cycle_idx"].unique())
    pos = {c: n for n, c in enumerate(cycles)}
    x = np.zeros((len(cycles), N_SUBDIVISIONS, len(INSTRUMENTS), len(FEATURES)), np.float32)

    # Intensity: log peak, z-scored per instrument within the take.
    ev["log_peak"] = np.log(ev["peak"].clip(lower=1e-4))
    grp = ev.groupby("instrument")["log_peak"]
    ev["intensity"] = (ev["log_peak"] - grp.transform("mean")) / grp.transform("std").replace(0, 1)

    c = ev["cycle_idx"].map(pos).to_numpy()
    k = ev["sd"].to_numpy() - 1
    i = ev["instrument"].map({name: n for n, name in enumerate(INSTRUMENTS)}).to_numpy()
    x[c, k, i, 0] = 1.0
    x[c, k, i, 1] = ev["intensity"].fillna(0.0).to_numpy()
    x[c, k, i, 2] = ev["dev_sd"].fillna(0.0).to_numpy()

    per_cycle = ev.groupby("cycle_idx").agg(T=("cycle_dur_s", "first"),
                                            section=("section", lambda s: s.dropna().mode().iloc[0]
                                                     if s.notna().any() else None),
                                            in_signal=("in_signal", "any"))
    per_cycle = per_cycle.reindex(cycles)
    return TakeGrid(
        take_id=take_id,
        piece=ev["piece"].iloc[0],
        x=x,
        log_T=np.log(per_cycle["T"].to_numpy()).astype(np.float32),
        section=np.array([SECTIONS.index(section_class(s)) for s in per_cycle["section"]]),
        in_signal=per_cycle["in_signal"].to_numpy(bool),
        present=np.array([name in set(ev["instrument"]) for name in INSTRUMENTS]),
        cycle_idx=cycles,
    )


def node_index(n_cycles: int, n_inst: int = len(INSTRUMENTS)) -> np.ndarray:
    """``idx[c, k, i]`` = node number, in row-major order."""
    return np.arange(n_cycles * N_SUBDIVISIONS * n_inst).reshape(n_cycles, N_SUBDIVISIONS, n_inst)


def graph_edges(n_cycles: int, instruments=range(len(INSTRUMENTS))) -> dict[str, np.ndarray]:
    """Edge lists (2, E) of the three relations over a window of ``n_cycles`` cycles.

    Nodes of instruments not listed in ``instruments`` get no edges (a take without
    D2 simply has isolated, masked D2 nodes).
    """
    idx = node_index(n_cycles)
    inst = list(instruments)
    src, dst = {t: [] for t in EDGE_TYPES}, {t: [] for t in EDGE_TYPES}
    for c in range(n_cycles):
        for k in range(N_SUBDIVISIONS):
            for i in inst:
                src["next_pulse"].append(idx[c, k, i])
                dst["next_pulse"].append(idx[c, (k + 1) % N_SUBDIVISIONS, i])
                for j in inst:
                    if j != i:
                        src["same_pulse"].append(idx[c, k, j])
                        dst["same_pulse"].append(idx[c, k, i])
                if c > 0:
                    src["prev_cycle"].append(idx[c - 1, k, i])
                    dst["prev_cycle"].append(idx[c, k, i])
    return {t: np.array([src[t], dst[t]], dtype=np.int64).reshape(2, -1) for t in EDGE_TYPES}


def rotate_nodes(n_cycles: int, r: int) -> np.ndarray:
    """Permutation ``perm[n]`` = node that n becomes when every cycle is rotated by r pulses."""
    # Node numbers are row-major, so position n of this array is node (c, k, i) itself.
    return node_index(n_cycles)[:, (np.arange(N_SUBDIVISIONS) + r) % N_SUBDIVISIONS, :].reshape(-1)


def position_features(mode: str, n_cycles: int, n_inst: int = len(INSTRUMENTS)) -> np.ndarray:
    """Per-node positional input that sets which rotations the model can tell apart.

    ``"none"``: no column (rotations by any number of pulses are symmetries, C12);
    ``"subpulse"``: one-hot of k mod 3 (only rotations by whole beats, C4);
    ``"full"``: one-hot of k (no rotation is a symmetry).
    """
    k = np.broadcast_to(np.arange(N_SUBDIVISIONS)[None, :, None], (n_cycles, N_SUBDIVISIONS, n_inst)).reshape(-1)
    if mode == "none":
        return np.zeros((k.size, 0), np.float32)
    if mode == "subpulse":
        return np.eye(3, dtype=np.float32)[k % 3]
    if mode == "full":
        return np.eye(N_SUBDIVISIONS, dtype=np.float32)[k]
    raise ValueError(f"unknown mode {mode!r}")
