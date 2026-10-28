"""Agreement between listening labels and the automatic stroke types.

    python experiments/strokes/agreement.py stroke_labels.json        # key: output/stroke_check_key.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("labels")
    ap.add_argument("--key", default=str(ROOT / "output" / "stroke_check_key.json"))
    args = ap.parse_args()
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    df = pd.DataFrame([{**key[i], "heard": l} for i, l in labels.items() if i in key])
    sure = df[df["heard"] != "unsure"]
    print(f"{len(df)} labelled, {len(df) - len(sure)} unsure")
    print(pd.crosstab(sure["heard"], sure["predicted"], margins=True).to_string())
    acc = (sure["heard"] == sure["predicted"]).mean()
    print(f"agreement {acc:.1%}, Cohen's kappa {cohen_kappa_score(sure['heard'], sure['predicted']):.2f}")
    for name, g in sure.groupby("predicted"):
        print(f"  predicted {name:5s}: heard as such {(g['heard'] == name).mean():.0%} of {len(g)}")
    for lo, hi in ((0, 0.9), (0.9, 1.01)):
        g = sure[(sure["confidence"] >= lo) & (sure["confidence"] < hi)]
        if len(g):
            print(f"  confidence {lo:.1f}-{min(hi, 1):.1f}: agreement {(g['heard'] == g['predicted']).mean():.0%} ({len(g)})")
    for flag, g in sure.groupby("bleed"):
        print(f"  dundun at the same time = {flag}: agreement {(g['heard'] == g['predicted']).mean():.0%} ({len(g)})")


if __name__ == "__main__":
    main()
