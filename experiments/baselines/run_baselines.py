"""Roadmap step J2: score the reference models on held-out takes.

Leave-one-take-out over the 15 takes (folds from splits.json). For every test take:
- negative log-likelihood, in nats, of the strikes (per node of a recorded
  instrument), of the offsets (per stroke with |delta| <= 0.5) and of the intensities
  (per stroke);
- for the offsets, how well the model's *distribution* matches the take, per
  (instrument, pulse) with at least 20 strokes: histogram intersection and error of
  the mean in ms, after drawing offsets at the take's real strokes (the setting of
  Mishra et al. 2025, which lets the scores be compared with theirs).

A third, cruder reference ("global") ignores the pulse: one strike rate and one
Gaussian per piece and instrument. It shows how much the pulse position alone buys.

    python experiments/baselines/run_baselines.py    # results/j2_baselines.json, docs/figures/j2_baselines.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from djembe import dataset as d
from djembe import grid as gr
from djembe.baselines import MarkovModel, PositionModel, histogram_intersection, _lines

ROOT = Path(__file__).resolve().parents[2]


class GlobalModel(PositionModel):
    """Pulse-blind reference: B1 with every pulse pooled."""

    def fit(self, grids):
        pooled = []
        for g in grids:
            x = g.x.copy()
            pooled.append(gr.TakeGrid(g.take_id, g.piece, x, g.log_T, g.section, g.in_signal, g.present, g.cycle_idx))
        super().fit(pooled)
        # replace every per-pulse entry by the instrument-level pool
        for key in list(self.p):
            piece, i, _ = key
            same = [k for k in self.p if k[:2] == (piece, i)]
            self.p[key] = float(np.mean([self.p[k] for k in same]))
        for name in ("d", "I"):
            for key in [k for k in self.params if k[0] == name]:
                _, piece, i, _ = key
                same = [self.params[k] for k in self.params if k[:3] == (name, piece, i)]
                self.params[key] = (float(np.mean([s[0] for s in same])),
                                    float(np.sqrt(np.mean([s[1] ** 2 for s in same]) + np.var([s[0] for s in same]))))
        return self


def distribution_scores(model, g, rng, instruments, min_strokes=20):
    sampled = model.sample_delta(g, rng)
    hi, err = [], []
    pulse_ms = float(np.exp(g.log_T).mean() / 12 * 1000)
    for i, h, dlt, _, k in _lines(g):
        if i not in instruments:
            continue
        for kk in range(12):
            sel = (k == kk) & (h > 0) & (np.abs(dlt) <= 0.5)
            if sel.sum() < min_strokes:
                continue
            hi.append(histogram_intersection(dlt[sel], sampled[i][sel]))
            err.append(abs(dlt[sel].mean() - np.nanmean(sampled[i][sel])) * pulse_ms)
    return (float(np.nanmean(hi)), float(np.mean(err))) if hi else (np.nan, np.nan)


MAIN = (0, 1, 2)   # J1, J2, D1: D2 is recorded on 4 takes only and is scored apart
D2 = (3,)


def score(model, g, rng, instruments):
    nll = model.nll(g)
    row = {}
    for key in ("hit", "delta", "intensity"):
        m = np.isin(nll[key + "_inst"], instruments)
        row["nll_" + key] = float(nll[key][m].mean()) if m.any() else np.nan
        row["n_" + key] = int(m.sum())
    row["hist_intersection"], row["mean_error_ms"] = distribution_scores(model, g, rng, instruments)
    return row


def summarise(rows, models):
    out = {}
    for name in models:
        sel = [r for r in rows if r["model"] == name and r["n_hit"] > 0]
        s = {}
        for key in ("hit", "delta", "intensity"):
            w = [r["n_" + key] for r in sel]
            s["nll_" + key] = float(np.average([r["nll_" + key] for r in sel], weights=w)) if sum(w) else None
        for key in ("hist_intersection", "mean_error_ms"):
            v = [r[key] for r in sel if r[key] == r[key]]
            s[key] = float(np.mean(v)) if v else None
        s["n_takes"] = len(sel)
        out[name] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", default=str(d.DEFAULT_EVENTS))
    ap.add_argument("--out", default=str(ROOT / "results" / "j2_baselines.json"))
    ap.add_argument("--figure", default=str(ROOT / "docs" / "figures" / "j2_baselines.png"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    events = d.load_events(args.events)
    splits = d.load_splits()
    models = {"global": GlobalModel, "B1_position": PositionModel, "B2_markov": MarkovModel}
    report = {"config": {"seed": args.seed, "delta_max": 0.5, "unit_nll": "nats",
                         "folds": "leave-one-take-out (splits.json)", "main_instruments": "J1, J2, D1"}}
    for align in (True, False):
        grids = {t: gr.take_grid(events, t, align=align) for t in d.TAKES}
        rng = np.random.default_rng(args.seed)
        rows = {"main": [], "D2": []}
        for fold in splits["folds"]:
            test = fold["test"][0]
            g = grids[test]
            for name, cls in models.items():
                m = cls().fit([grids[t] for t in fold["train"]])
                for part, inst in (("main", MAIN), ("D2", D2)):
                    rows[part].append({"take": test, "piece": g.piece, "model": name, **score(m, g, rng, inst)})
        tag = "aligned" if align else "raw"
        report[tag] = {"take_offsets_pulse": {t: grids[t].delta_offset for t in d.TAKES} if align else None,
                       "summary": {part: summarise(rows[part], models) for part in rows},
                       "per_take": rows}
        print(f"--- {tag} (J1, J2, D1)")
        for name, s in report[tag]["summary"]["main"].items():
            print(f"{name:12s} hit {s['nll_hit']:.3f}  delta {s['nll_delta']:+.3f}  intensity {s['nll_intensity']:.3f}"
                  f"  HI {s['hist_intersection']:.3f}  mean err {s['mean_error_ms']:.1f} ms")
        print(f"--- {tag} (D2, {report[tag]['summary']['D2']['B1_position']['n_takes']} takes)")
        for name, s in report[tag]["summary"]["D2"].items():
            print(f"{name:12s} hit {s['nll_hit']:.3f}  delta {s['nll_delta']:+.3f}")
    Path(args.out).write_text(json.dumps(report, indent=2))

    rows = report["aligned"]["per_take"]["main"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    takes = list(d.TAKES)
    for ax, key, title in zip(axes, ("nll_hit", "nll_delta", "nll_intensity"),
                              ("Strikes (nats per node)", "Offsets (nats per stroke)", "Intensity (nats per stroke)")):
        for name in models:
            v = [next(r[key] for r in rows if r["take"] == t and r["model"] == name) for t in takes]
            ax.plot(range(len(takes)), v, "o-", ms=4, label=name)
        ax.set_xticks(range(len(takes)))
        ax.set_xticklabels([t[:3] + t[-2:] for t in takes],
                           rotation=90, fontsize=8)
        ax.set_title(title + ", lower is better", fontsize=10)
    axes[0].legend(fontsize=8)
    fig.suptitle("Reference models on each held-out take (J1, J2, D1; takes aligned)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=130)
    print("->", args.out, args.figure)


if __name__ == "__main__":
    main()
