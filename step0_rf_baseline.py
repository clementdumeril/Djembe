"""Step 0 baseline: which rhythm is this take, from the D1 subdivision profile?

Recomputes the random forest of notebook 03 from the event table and the frozen
leave-one-take-out folds, next to two reference points (majority class and
nearest centroid). Writes results/step0_rf.json (metrics and per-take
predictions only, no corpus data).

    python djembe_dataset.py build
    python step0_rf_baseline.py [--events output/events.csv] [--out results/step0_rf.json]
"""
from __future__ import annotations

import os

# The machine is shared: keep BLAS and joblib to two threads.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "2")

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.neighbors import NearestCentroid

import djembe_dataset as d

OUT = Path(__file__).resolve().parent / "results" / "step0_rf.json"


def models():
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "nearest_centroid": NearestCentroid(),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=2),
    }


def exact_ci(n_correct: int, n: int) -> list[float]:
    """Clopper-Pearson 95 % interval on accuracy over takes. With 15 takes a
    bootstrap collapses to [1, 1] at perfect accuracy; this one does not."""
    ci = binomtest(n_correct, n).proportion_ci(confidence_level=0.95, method="exact")
    return [float(ci.low), float(ci.high)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default=str(d.DEFAULT_EVENTS))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    out = Path(args.out)

    events = d.load_events(args.events)
    profile = d.subdivision_profile(events, "D1")
    features = [c for c in profile.columns if c.startswith("sd")]
    labels = list(d.PIECES)

    report = {
        "task": "piece from per-take D1 subdivision profile (12 features)",
        "n_takes": int(len(profile)),
        "class_counts": profile["piece"].value_counts().reindex(labels).astype(int).to_dict(),
        "splits": "splits.json (leave-one-take-out)",
        "models": {},
    }
    for name, model in models().items():
        rows = []
        for fold, train, test in d.iter_folds(profile):
            model.fit(train[features].to_numpy(), train["piece"].to_numpy())
            pred = model.predict(test[features].to_numpy())
            rows += [{"take_id": t, "true": str(y), "pred": str(p)} for t, y, p in zip(test["take_id"], test["piece"], pred)]
        res = pd.DataFrame(rows)
        n_correct = int((res["true"] == res["pred"]).sum())
        report["models"][name] = {
            "accuracy": float(accuracy_score(res["true"], res["pred"])),
            "balanced_accuracy": float(balanced_accuracy_score(res["true"], res["pred"])),
            "accuracy_ci95_exact": exact_ci(n_correct, len(res)),
            "confusion": {"labels": labels, "matrix": confusion_matrix(res["true"], res["pred"], labels=labels).tolist()},
            "errors": res[res["true"] != res["pred"]].to_dict("records"),
        }
        m = report["models"][name]
        print(f"{name:17s} acc {m['accuracy']:.3f}  bal {m['balanced_accuracy']:.3f}  "
              f"CI95 [{m['accuracy_ci95_exact'][0]:.2f}, {m['accuracy_ci95_exact'][1]:.2f}]  "
              f"errors {[(e['take_id'], e['pred']) for e in m['errors']]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
