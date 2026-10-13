"""Does the jembe "feel" recognise the piece only through tempo?

The pieces are played at different tempi (median cycle 1.37 s for Manjanin,
1.53 s for Maraka, 1.68 s for Woloso), and timing offsets measured in
fractions of a pulse could follow tempo. Two controls, same folds as a2:

1. tempo band: keep only windows whose cycles all lie in the tempo range
   shared by the three pieces (between the highest 5th percentile and the
   lowest 95th percentile of cycle duration). Inside it, tempo alone should
   fall towards chance.
2. tempo regressed out: inside each training fold, regress every feature on
   the window's mean cycle duration and keep the residuals; the test fold is
   transformed with the training fit.

    python tempo_control.py      # -> results/a2_tempo_control.json
"""
from __future__ import annotations

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "2")

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from djembe import dataset as d
from djembe import features as f
from recognition_baselines import cross_validate, summarise

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = (1, 2, 4)
SETS = ("tempo", "d1_pattern", "jembe_timing12", "jembe_feel3")
TEMPO_COL = "tempo_mean_dur"


class RegressOutTempo(BaseEstimator, TransformerMixin):
    """Replace each feature by its residual after a linear fit on tempo (last column)."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.reg_ = LinearRegression().fit(X[:, -1:], X[:, :-1])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, :-1] - self.reg_.predict(X[:, -1:])


def models(regress_out: bool):
    head = [SimpleImputer(strategy="mean")] + ([RegressOutTempo()] if regress_out else [])
    return {
        "nearest_centroid": make_pipeline(*head, StandardScaler(), NearestCentroid()),
        "logistic_regression": make_pipeline(*head, StandardScaler(), LogisticRegression(max_iter=2000)),
    }


def tempo_band(events) -> tuple[float, float]:
    cycles = events.drop_duplicates(["take_id", "cycle_label"]).dropna(subset=["cycle_dur_s"])
    q = cycles.groupby("piece")["cycle_dur_s"]
    return float(q.quantile(0.05).max()), float(q.quantile(0.95).min())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default=str(d.DEFAULT_EVENTS))
    parser.add_argument("--out", default=str(ROOT / "results" / "a2_tempo_control.json"))
    args = parser.parse_args()

    events = d.load_events(args.events)
    splits = d.load_splits()
    rng = np.random.default_rng(0)
    lo, hi = tempo_band(events)
    in_band_cycles = events["cycle_dur_s"].between(lo, hi)
    report = {"tempo_band_s": [lo, hi], "results": {}}
    print(f"tempo band {lo:.3f}-{hi:.3f} s")

    for k in WINDOWS:
        frame = f.window_features(events, k)
        # windows whose every cycle is inside the band
        win = f.assign_windows(events, k)
        ok = win.assign(inb=in_band_cycles.reindex(win.index)).groupby("window")["inb"].all()
        banded = frame[frame["window"].isin(ok[ok].index)].reset_index(drop=True)
        res_k = report["results"].setdefault(str(k), {})
        for control, data, regress in (("all_windows", frame, False), ("tempo_band", banded, False),
                                       ("tempo_regressed_out", frame, True)):
            res_k[control] = {"n_windows": int(len(data)),
                              "takes_per_piece": data.groupby("piece")["take_id"].nunique().to_dict()}
            for fs in SETS:
                if regress and fs == "tempo":
                    continue
                cols = f.columns(data, fs) + ([TEMPO_COL] if regress else [])
                for name, model in models(regress).items():
                    r = summarise(cross_validate(data, cols, model, splits), rng)
                    res_k[control].setdefault(fs, {})[name] = r
                    print(f"k={k} {control:20s} {fs:15s} {name:20s} n={len(data):4d} "
                          f"bal {r['balanced_accuracy']:.3f} [{r['ci95_bootstrap_takes'][0]:.2f}, "
                          f"{r['ci95_bootstrap_takes'][1]:.2f}]", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
