"""Merge the parts of a J5b (soloist) run and summarise them.

Each take's value is averaged over seeds, then over the 15 takes. Prediction: soloist
strikes (sequential NLL), offsets and intensities, network against B2, take by take.
Generation: the network's soloist against the Markov one, "closer" meaning closer to the
real soloist (for the distinct-pattern and copy rates, closer to the real take's own
rate). Two-sided Wilcoxon on the per-take seed means.

    python experiments/generation/summarise_solo.py <run dir>/part_*.json --out results/j5b_solo_savio.json --expect-takes 15 --expect-seeds 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

GENERATORS = ("GNN_solo", "Markov_B2", "real_other_take")
MEASURES = ["profile_l1", "density_ratio", "timing_w1_ms", "intensity_w1", "novelty", "copy_rate", "distinct_rate"]


def _test(d: pd.Series) -> dict:
    return {"gnn_better_on": int((d < 0).sum()), "takes": int(d.notna().sum()),
            "wilcoxon_p": float(wilcoxon(d.dropna()).pvalue) if (d.dropna() != 0).any() else 1.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-takes", type=int, default=None)
    ap.add_argument("--expect-seeds", type=int, default=None)
    args = ap.parse_args()

    rows = [r for p in args.parts for r in json.loads(Path(p).read_text(encoding="utf-8"))["rows"]]
    df = pd.DataFrame(rows).rename(columns={"take": "tk"})
    cov = df.groupby(["kind", "model"]).agg(takes=("tk", "nunique"), seeds=("seed", "nunique"))
    bad = cov[((cov.takes != args.expect_takes) if args.expect_takes else False)
              | ((cov.seeds != args.expect_seeds) if args.expect_seeds else False)]
    if len(bad) or df.duplicated(["kind", "model", "tk", "seed"]).any():
        raise SystemExit(f"incomplete or duplicated parts:\n{bad}")

    pred = df[df["kind"] == "prediction"]
    summary = {"prediction": {}, "generation_mean": {}, "gnn_vs_markov": {}}
    for key in ("nll_hit", "nll_delta", "nll_intensity"):
        t = pred.groupby(["model", "tk"])[key].mean().unstack(0)
        summary["prediction"][key] = {"mean": t.mean().round(4).to_dict(), **_test(t["GNN_solo"] - t["B2_markov_cycle"])}

    gen = df[df["kind"] == "generation"]
    per = {m: gen[gen["model"] == m].groupby("tk")[MEASURES].mean() for m in GENERATORS + ("real_heldout",)}
    summary["generation_mean"] = {m: per[m].mean().round(4).to_dict() for m in GENERATORS + ("real_heldout",)}
    temps = gen[gen["model"] == "GNN_solo"]["temperature"].value_counts().to_dict()
    summary["temperature_chosen"] = {str(k): int(v) for k, v in temps.items()}
    g, mk, real = per["GNN_solo"], per["Markov_B2"], per["real_heldout"]
    for m in ("profile_l1", "timing_w1_ms", "intensity_w1"):
        summary["gnn_vs_markov"][m] = _test(g[m] - mk[m])
    for m, target in (("density_ratio", 1.0), ("distinct_rate", real["distinct_rate"]), ("copy_rate", real["copy_rate"])):
        summary["gnn_vs_markov"][m] = _test((g[m] - target).abs() - (mk[m] - target).abs())

    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    for key, s in summary["prediction"].items():
        print(f"prediction {key:13s} {s['mean']}  GNN better on {s['gnn_better_on']}/{s['takes']}, p = {s['wilcoxon_p']:.4f}")
    print(pd.DataFrame(summary["generation_mean"]).round(3).to_string())
    print("temperature chosen on the validation take:", summary["temperature_chosen"])
    for m, s in summary["gnn_vs_markov"].items():
        print(f"  {m:14s} GNN closer to the real soloist than Markov on {s['gnn_better_on']}/{s['takes']}, p = {s['wilcoxon_p']:.4f}")
    print("->", args.out)


if __name__ == "__main__":
    main()
