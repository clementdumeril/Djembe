"""Steps a1/a2: recognise the piece from short windows, with and without the pattern.

For each window length k (cycles) and each feature set in features.py, four
models are scored with the frozen leave-one-take-out folds:
majority class, nearest centroid, logistic regression, random forest.
Imputation and scaling are fitted inside each training fold.

Metrics: window-level balanced accuracy pooled over the 15 test folds, a 95 %
interval from a bootstrap over takes stratified by piece (windows of one
take are not independent), and take-level accuracy by majority vote over the take's windows.

    python djembe_dataset.py build
    python recognition_baselines.py      # -> results/a2_recognition.json, docs/figures/a2_window_accuracy.png
"""
from __future__ import annotations

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "2")

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import djembe_dataset as d
import features as f

ROOT = Path(__file__).resolve().parent
WINDOWS = (1, 2, 4, 8)
N_BOOT = 1000
LABELS = list(d.PIECES)
SET_COLORS = {  # validated categorical order (dataviz palette, light mode)
    "tempo": "#2a78d6", "d1_pattern": "#eb6834", "jembe_pattern": "#1baf7a",
    "jembe_timing12": "#eda100", "jembe_feel3": "#e87ba4",
}


def models():
    def prep():
        return [SimpleImputer(strategy="mean"), StandardScaler()]
    return {
        "majority": make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0), DummyClassifier(strategy="most_frequent")),
        "nearest_centroid": make_pipeline(*prep(), NearestCentroid()),
        "logistic_regression": make_pipeline(*prep(), LogisticRegression(C=1.0, max_iter=2000)),
        "random_forest": make_pipeline(SimpleImputer(strategy="mean"),
                                       RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=0, n_jobs=2)),
    }


def cross_validate(frame: pd.DataFrame, cols: list[str], model, splits: dict) -> pd.DataFrame:
    out = []
    for fold, train, test in d.iter_folds(frame, splits):
        if test.empty:  # possible on filtered subsets (e.g. a tempo band)
            continue
        model.fit(train[cols], train["piece"])
        out.append(test[["window", "take_id", "piece"]].assign(pred=model.predict(test[cols])))
    return pd.concat(out, ignore_index=True)


def _balanced_from_counts(counts: np.ndarray) -> float:
    """Balanced accuracy from a (true x pred) count matrix."""
    seen = counts.sum(axis=1) > 0
    return float(np.mean(np.diag(counts)[seen] / counts.sum(axis=1)[seen]))


def summarise(pred: pd.DataFrame, rng: np.random.Generator) -> dict:
    # One confusion matrix per take; the bootstrap resamples takes within each
    # piece (stratified, so every resample keeps all three classes) and sums them.
    index = {p: i for i, p in enumerate(LABELS)}
    per_take = {}
    for take, g in pred.groupby("take_id"):
        m = np.zeros((len(LABELS), len(LABELS)))
        np.add.at(m, (g["piece"].map(index).to_numpy(), g["pred"].map(index).to_numpy()), 1)
        per_take[take] = m
    strata = [s for s in ([t for t in per_take if t.startswith(p)] for p in LABELS) if s]
    boot = []
    for _ in range(N_BOOT):
        total = sum(per_take[t] for takes in strata for t in rng.choice(takes, size=len(takes), replace=True))
        boot.append(_balanced_from_counts(total))
    vote = pred.groupby("take_id").agg(piece=("piece", "first"), pred=("pred", lambda s: s.value_counts().index[0]))
    take_acc = pred.assign(ok=pred["piece"] == pred["pred"]).groupby("take_id")["ok"].mean()
    return {
        "n_windows": int(len(pred)),
        "balanced_accuracy": float(balanced_accuracy_score(pred["piece"], pred["pred"])),
        "ci95_bootstrap_takes": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "take_vote_accuracy": float((vote["piece"] == vote["pred"]).mean()),
        "worst_take": {"take_id": take_acc.idxmin(), "window_accuracy": float(take_acc.min())},
    }


def plot(results: dict, path: Path, model: str = "logistic_regression") -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for fs, color in SET_COLORS.items():
        ys = [results[str(k)][fs][model]["balanced_accuracy"] for k in WINDOWS]
        lo = [results[str(k)][fs][model]["ci95_bootstrap_takes"][0] for k in WINDOWS]
        hi = [results[str(k)][fs][model]["ci95_bootstrap_takes"][1] for k in WINDOWS]
        ax.fill_between(WINDOWS, lo, hi, color=color, alpha=0.12, lw=0)
        ax.plot(WINDOWS, ys, "-o", color=color, lw=2, ms=6, mec="white", mew=1.2)
        ax.annotate(fs.replace("_", " "), (WINDOWS[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#1f1f1e")
    ax.axhline(1 / 3, color="#6b6a63", lw=1, ls="--")
    ax.text(WINDOWS[0], 1 / 3 + 0.015, "chance (balanced)", fontsize=8.5, color="#6b6a63")
    ax.set_xscale("log", base=2)
    ax.set_xticks(WINDOWS, [str(k) for k in WINDOWS])
    ax.set_xlim(0.85, 20)
    ax.set_ylim(0.2, 1.02)
    ax.set_xlabel("window length (annotated cycles)")
    ax.set_ylabel("balanced accuracy, leave-one-take-out")
    ax.set_title("Recognising the piece from short windows (logistic regression; band = 95 % CI over takes)",
                 loc="left", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e4e3dc", lw=0.8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default=str(d.DEFAULT_EVENTS))
    parser.add_argument("--out", default=str(ROOT / "results" / "a2_recognition.json"))
    parser.add_argument("--figure", default=str(ROOT / "docs" / "figures" / "a2_window_accuracy.png"))
    args = parser.parse_args()

    events = d.load_events(args.events)
    splits = d.load_splits()
    rng = np.random.default_rng(0)
    results = {}
    for k in WINDOWS:
        frame = f.window_features(events, k)
        results[str(k)] = {}
        for fs in f.FEATURE_SETS:
            cols = f.columns(frame, fs)
            results[str(k)][fs] = {}
            for name, model in models().items():
                res = summarise(cross_validate(frame, cols, model, splits), rng)
                results[str(k)][fs][name] = res
                print(f"k={k} {fs:15s} {name:20s} bal {res['balanced_accuracy']:.3f} "
                      f"[{res['ci95_bootstrap_takes'][0]:.2f}, {res['ci95_bootstrap_takes'][1]:.2f}] "
                      f"vote {res['take_vote_accuracy']:.2f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"windows_in_cycles": list(WINDOWS), "feature_sets": list(f.FEATURE_SETS),
              "splits": "splits.json (leave-one-take-out)", "results": results}
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot(results, Path(args.figure))
    print("->", out, "and", args.figure)


if __name__ == "__main__":
    main()
