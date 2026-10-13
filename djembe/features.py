"""Window-level features for rhythm recognition (roadmap steps a1 and a2).

A window is k consecutive annotated cycles of one take, non-overlapping. Each
window gets one label (the piece) and one group (the take), so folds from
``splits.json`` keep every window of a take on the same side.

Feature sets:
- ``tempo``          mean cycle duration and its trend over the window (a
                     shortcut check: rhythms are played at different tempi).
- ``d1_pattern``     fraction of cycles where D1 plays each of the 12 pulses.
- ``jembe_pattern``  same for the jembe accompaniment (J1 or J2).
- ``jembe_timing12`` mean offset from the even grid at each of the 12 pulses,
                     jembe strokes only (in fractions of a pulse).
- ``jembe_feel3``    mean and spread of the jembe offset at each position
                     inside the beat (1st, 2nd, 3rd pulse of 3), pooled over
                     the four beats. This is the "feel without the pattern"
                     test: which pulses are played does not enter.

Missing values (no stroke at a position in that window) stay NaN; models
impute them inside each training fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_SD = 12
FEATURE_SETS = ("tempo", "d1_pattern", "jembe_pattern", "jembe_timing12", "jembe_feel3")


def assign_windows(events: pd.DataFrame, k: int) -> pd.DataFrame:
    """Add a ``window`` id: k consecutive cycles per take; incomplete tails dropped."""
    usable = events[events["cycle_dur_s"].notna()].copy()
    # Renumber cycles densely per take, in time order, after dropping the last one.
    rank = usable.groupby("take_id")["cycle_idx"].rank(method="dense").astype(int) - 1
    usable["window_in_take"] = rank // k
    full = usable.groupby(["take_id", "window_in_take"])["cycle_label"].transform("nunique") == k
    usable = usable[full]
    usable["window"] = usable["take_id"] + "#" + usable["window_in_take"].astype(str)
    return usable


def _presence(win: pd.DataFrame, mask: pd.Series, n_cycles: pd.Series, prefix: str) -> pd.DataFrame:
    hits = win[mask].drop_duplicates(["window", "cycle_label", "sd"])
    counts = hits.groupby(["window", "sd"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=n_cycles.index, columns=range(1, N_SD + 1), fill_value=0)
    out = counts.div(n_cycles, axis=0)
    out.columns = [f"{prefix}_sd{c}" for c in out.columns]
    return out


def window_features(events: pd.DataFrame, k: int) -> pd.DataFrame:
    """One row per window: take_id, piece, n_cycles, then all feature columns."""
    win = assign_windows(events, k)
    meta = win.groupby("window").agg(take_id=("take_id", "first"), piece=("piece", "first"),
                                     n_cycles=("cycle_label", "nunique"))
    jembe = win["instrument"].isin(["J1", "J2"])

    cycles = win.drop_duplicates(["window", "cycle_label"]).sort_values("cycle_start_s")
    tempo = cycles.groupby("window")["cycle_dur_s"].agg(
        tempo_mean_dur="mean",
        tempo_trend=lambda s: float(np.polyfit(np.arange(len(s)), s.to_numpy(), 1)[0]) if len(s) > 1 else 0.0,
    )

    timing12 = win[jembe].groupby(["window", "sd"])["dev_sd"].mean().unstack()
    timing12 = timing12.reindex(index=meta.index, columns=range(1, N_SD + 1))
    timing12.columns = [f"jembe_timing12_sd{c}" for c in timing12.columns]

    j = win[jembe].assign(pos=(win.loc[jembe, "sd"] - 1) % 3)
    feel = j.groupby(["window", "pos"])["dev_sd"].agg(["mean", "std"]).unstack("pos")
    feel.columns = [f"jembe_feel3_{stat}_p{pos + 1}" for stat, pos in feel.columns]
    feel = feel.reindex(meta.index)

    parts = [
        meta,
        tempo,
        _presence(win, win["instrument"] == "D1", meta["n_cycles"], "d1_pattern"),
        _presence(win, jembe, meta["n_cycles"], "jembe_pattern"),
        timing12,
        feel,
    ]
    return pd.concat(parts, axis=1).reset_index()


def columns(frame: pd.DataFrame, feature_set: str) -> list[str]:
    return [c for c in frame.columns if c.startswith(feature_set + "_")]
