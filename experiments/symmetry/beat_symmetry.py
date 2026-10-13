"""Roadmap step J1: does the feel respect the beat symmetry of the cycle?

The 12 pulses of a cycle are 4 beats of 3 pulses. If the micro-timing were the same
on every beat (symmetry under rotation by one beat, the group C4), the mean offset
at pulse k = 3b + s would depend on s only. Two numbers per piece and instrument,
both in fractions of a pulse, computed from the mean offset mu[b, s]:

- feel   A = spread over s of the beat-averaged profile: the size of the
             short-medium-long pattern, i.e. how far the feel is from an even grid
             (and so from rotation by one pulse, C12);
- breach B = root-mean-square over s of the spread over b: how much the four beats
             differ at the same position in the beat (zero under exact C4).

Noise floor for B: every cycle is rotated by a random whole number of beats
(which makes C4 exact while keeping the data), 200 times. Intervals: bootstrap over
takes within each piece, 1,000 resamples. Strokes in signal phrases (breaks), the
43 without an end of cycle and the 3.5 % more than half a pulse off their position
are left out. A position enters only if every beat has at least 30 strokes there.

    python experiments/symmetry/beat_symmetry.py     # results/j1_beat_symmetry.json, docs/figures/j1_beat_symmetry.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from djembe import dataset as d

ROOT = Path(__file__).resolve().parents[2]
INSTRUMENTS = ("J1", "J2", "D1")
MIN_STROKES = 30


def clean(events: pd.DataFrame) -> pd.DataFrame:
    ev = events[~events["in_signal"] & events["dev_sd"].notna() & (events["dev_sd"].abs() <= 0.5)].copy()
    ev["beat"] = (ev["sd"] - 1) // 3
    ev["sub"] = (ev["sd"] - 1) % 3
    return ev


def profile(ev: pd.DataFrame, value: str = "dev_sd") -> pd.DataFrame:
    """mean[b, s] and count[b, s] as two 4x3 frames."""
    g = ev.groupby(["beat", "sub"])[value]
    mean = g.mean().unstack().reindex(index=range(4), columns=range(3))
    count = g.size().unstack().reindex(index=range(4), columns=range(3)).fillna(0)
    return mean, count


def stats(ev: pd.DataFrame, subs=None) -> dict:
    mean, count = profile(ev)
    if subs is None:
        subs = [s for s in range(3) if (count[s] >= MIN_STROKES).all()]
    if not subs:
        return {"subs": [], "feel_A": np.nan, "breach_B": np.nan}
    m = mean[subs]
    feel = float(m.mean(axis=0).std(ddof=0)) if len(subs) > 1 else np.nan
    breach = float(np.sqrt((m.std(axis=0, ddof=0) ** 2).mean()))
    return {"subs": subs, "feel_A": feel, "breach_B": breach}


def null_breach(ev: pd.DataFrame, subs, n: int, rng) -> np.ndarray:
    """B after rotating each cycle by a random whole number of beats (C4 made exact)."""
    key = ev["take_id"] + "#" + ev["cycle_idx"].astype(str)
    codes, uniq = pd.factorize(key)
    out = np.empty(n)
    for t in range(n):
        shift = rng.integers(0, 4, size=len(uniq))[codes]
        rot = ev.assign(beat=(ev["beat"].to_numpy() + shift) % 4)
        out[t] = stats(rot, subs)["breach_B"]
    return out


def bootstrap(ev: pd.DataFrame, subs, n: int, rng) -> dict:
    takes = ev["take_id"].unique()
    by_take = {t: ev[ev["take_id"] == t] for t in takes}
    feel, breach = [], []
    for _ in range(n):
        sample = pd.concat([by_take[t] for t in rng.choice(takes, size=len(takes), replace=True)])
        s = stats(sample, subs)
        feel.append(s["feel_A"])
        breach.append(s["breach_B"])
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    return {"feel_A_ci95": ci(feel), "breach_B_ci95": ci(breach)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", default=str(d.DEFAULT_EVENTS))
    ap.add_argument("--out", default=str(ROOT / "results" / "j1_beat_symmetry.json"))
    ap.add_argument("--figure", default=str(ROOT / "docs" / "figures" / "j1_beat_symmetry.png"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--null", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ev = clean(d.load_events(args.events))
    results, profiles = [], {}
    for piece in d.PIECES:
        for inst in INSTRUMENTS:
            sub = ev[(ev["piece"] == piece) & (ev["instrument"] == inst)]
            s = stats(sub)
            row = {"piece": piece, "instrument": inst, "n_strokes": int(len(sub)),
                   "n_takes": int(sub["take_id"].nunique()), **s}
            if s["subs"]:
                nb = null_breach(sub, s["subs"], args.null, rng)
                row["breach_B_null_mean"] = float(nb.mean())
                row["breach_B_null_p95"] = float(np.percentile(nb, 95))
                row.update(bootstrap(sub, s["subs"], args.boot, rng))
                pulse_ms = float((sub["cycle_dur_s"] / 12).median() * 1000)
                row["pulse_ms_median"] = pulse_ms
                row["feel_A_ms"] = s["feel_A"] * pulse_ms if s["feel_A"] == s["feel_A"] else None
                row["breach_B_ms"] = s["breach_B"] * pulse_ms
            results.append(row)
            profiles[(piece, inst)] = profile(sub, "dev_ms")
            print(f"{piece:9s} {inst}  subs {s['subs']}  feel {s['feel_A']:.3f}  breach {s['breach_B']:.3f}"
                  + (f"  null p95 {row['breach_B_null_p95']:.3f}  CI {row['breach_B_ci95'][0]:.3f}-{row['breach_B_ci95'][1]:.3f}"
                     if s["subs"] else ""))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"config": {"boot": args.boot, "null": args.null, "seed": args.seed,
                                                     "min_strokes": MIN_STROKES, "unit": "fraction of a pulse"},
                                          "results": results}, indent=2))

    fig, axes = plt.subplots(len(INSTRUMENTS), len(d.PIECES), figsize=(11, 8), sharex=True)
    for col, piece in enumerate(d.PIECES):
        for row_i, inst in enumerate(INSTRUMENTS):
            ax = axes[row_i, col]
            mean, count = profiles[(piece, inst)]
            for b in range(4):
                ok = count.loc[b] >= MIN_STROKES
                ax.plot(np.arange(1, 4)[ok.to_numpy()], mean.loc[b][ok].to_numpy(), "o-", label=f"beat {b + 1}")
            ax.axhline(0, color="0.7", lw=0.8)
            ax.set_title(f"{piece}, {inst}", fontsize=10)
            if col == 0:
                ax.set_ylabel("mean offset [ms]")
            if row_i == len(INSTRUMENTS) - 1:
                ax.set_xlabel("pulse within the beat")
            ax.set_xticks([1, 2, 3])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Mean offset from the even grid at each pulse of each beat (equal beats = C4 symmetry)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=130)
    print("->", args.out, args.figure)


if __name__ == "__main__":
    main()
