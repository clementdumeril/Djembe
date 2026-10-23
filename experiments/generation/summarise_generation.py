"""Merge the parts of a J5 run and summarise them.

Per measure, each take's value is averaged over seeds, then over the 15 takes. The GNN
generator is compared with the Markov one take by take (two-sided Wilcoxon on the seed
means); for the distances "better" means closer to the real take. Strike prediction:
GNN strike model against B2, same test.

    python experiments/generation/summarise_generation.py <run dir>/part_*.json --out results/j5_generation_savio.json --expect-takes 15 --expect-seeds 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

LOWER_IS_CLOSER = ("profile_l1", "timing_w1_ms", "intensity_w1")
GENERATORS = ("GNN", "Markov_B2", "real_other_take")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-takes", type=int, default=None)
    ap.add_argument("--expect-seeds", type=int, default=None)
    args = ap.parse_args()

    rows = [r for p in args.parts for r in json.loads(Path(p).read_text(encoding="utf-8"))["rows"]]
    df = pd.DataFrame(rows).rename(columns={"take": "tk"})
    keys = df.groupby(["kind", "model"]).apply(lambda s: (s["tk"].nunique(), s["seed"].nunique()), include_groups=False)
    problems = [f"{k}: {v[0]} takes, {v[1]} seeds" for k, v in keys.items()
                if (args.expect_takes and v[0] != args.expect_takes) or (args.expect_seeds and v[1] != args.expect_seeds)]
    if df.duplicated(["kind", "model", "tk", "seed"]).any():
        problems.append("duplicated (kind, model, take, seed) rows")
    if problems:
        raise SystemExit("incomplete or duplicated: " + "; ".join(problems))

    summary = {}
    sp = df[df["kind"] == "strike_prediction"].groupby(["model", "tk"])["nll_hit"].mean().unstack(0)
    diff = sp["GNN_strikes"] - sp["B2_markov_cycle"]
    summary["strike_prediction"] = {
        "nll_hit": sp.mean().round(4).to_dict(),
        "gnn_better_on": int((diff < 0).sum()), "takes": int(len(diff)), "wilcoxon_p": float(wilcoxon(diff).pvalue)}

    gen = df[df["kind"] == "generation"]
    real_nov = gen[gen["model"] == "real_heldout_novelty"].groupby("tk")[["novelty", "copy_rate"]].mean()
    measures = ["profile_l1", "density_ratio", "timing_w1_ms", "intensity_w1", "novelty", "copy_rate", "recognised"]
    per_take = {m: gen[gen["model"] == m].groupby("tk")[measures].mean() for m in GENERATORS}
    summary["generation_mean"] = {m: per_take[m].mean().round(4).to_dict() for m in GENERATORS}
    summary["real_heldout_novelty"] = real_nov.mean().round(4).to_dict()
    comp = {}
    g, mk = per_take["GNN"], per_take["Markov_B2"]
    for m in LOWER_IS_CLOSER:
        d = g[m] - mk[m]
        comp[m] = {"gnn_closer_on": int((d < 0).sum()), "wilcoxon_p": float(wilcoxon(d).pvalue)}
    for m, target in (("density_ratio", 1.0), ("novelty", real_nov["novelty"]), ("copy_rate", real_nov["copy_rate"])):
        d = (g[m] - target).abs() - (mk[m] - target).abs()
        comp[m] = {"gnn_closer_on": int((d < 0).sum()), "wilcoxon_p": float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0}
    summary["gnn_vs_markov_per_take"] = comp

    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    s = summary
    print(f"strike prediction (nats, lower better): {s['strike_prediction']['nll_hit']}  "
          f"GNN better on {s['strike_prediction']['gnn_better_on']}/{s['strike_prediction']['takes']}, "
          f"p = {s['strike_prediction']['wilcoxon_p']:.3f}")
    print(pd.DataFrame(s["generation_mean"]).round(3).to_string())
    print("real held-out takes:", s["real_heldout_novelty"])
    for m, c in comp.items():
        print(f"  {m:14s} GNN closer to the real take than Markov on {c['gnn_closer_on']}/{len(g)}, p = {c['wilcoxon_p']:.3f}")
    print("->", args.out)


if __name__ == "__main__":
    main()
