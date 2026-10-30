"""Merge the parts of a J6 run (one protocol) and summarise them.

Each take's value is averaged over seeds, then over the 15 takes. Prediction: soloist
NLL of strikes, offsets and intensities for every model, and pairwise comparisons take
by take (graph model against each transformer, each network against B2). Generation:
the measures of ``djembe.solo.compare`` per model; "closer" compares the distance to the
real soloist's own value (``timing_lag1``, ``distinct_rate``) or the distance itself
(``profile_l1``, ``timing_w1_ms``, ``intensity_w1``). Two-sided Wilcoxon on the per-take
seed means.

    python experiments/j6/summarise.py <run dir>/part_*.json --out results/j6_piece_savio.json --expect-seeds 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

NETS = ("GNN_seq", "Transformer", "Transformer_matched")
PRED = ("nll_hit", "nll_delta", "nll_intensity")
GEN_DIST = ("profile_l1", "timing_w1_ms", "intensity_w1")
GEN_TO_REAL = ("timing_lag1", "distinct_rate", "density_ratio")


def _test(d: pd.Series) -> dict:
    d = d.dropna()
    return {"first_better_on": int((d < 0).sum()), "takes": int(d.size),
            "wilcoxon_p": float(wilcoxon(d).pvalue) if (d != 0).any() and d.size > 1 else 1.0}


def summarise(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    out = {"protocol": sorted(df["protocol"].unique()), "n_takes": int(df["take"].nunique()),
           "seeds": sorted(int(s) for s in df["seed"].unique())}
    pred = df[df["kind"] == "prediction"].groupby(["take", "model"])[list(PRED)].mean()
    out["prediction_mean"] = pred.groupby("model").mean().round(4).to_dict(orient="index")
    pairs = [(a, b) for a in NETS for b in NETS if a < b] + [(a, "B2_markov") for a in NETS]
    out["prediction_pairs"] = {}
    for a, b in pairs:
        if a not in pred.index.get_level_values(1) or b not in pred.index.get_level_values(1):
            continue
        out["prediction_pairs"][f"{a} vs {b}"] = {m: _test(pred.xs(a, level=1)[m] - pred.xs(b, level=1)[m])
                                                  for m in PRED}
    gen = df[df["kind"] == "generation"].groupby(["take", "model"])[list(GEN_DIST + GEN_TO_REAL)].mean()
    out["generation_mean"] = gen.groupby("model").mean().round(4).to_dict(orient="index")
    real = gen.xs("real_heldout", level=1)
    dist = {}
    for m in gen.index.get_level_values(1).unique():
        if m == "real_heldout":
            continue
        g = gen.xs(m, level=1)
        dist[m] = pd.concat([g[list(GEN_DIST)], (g[list(GEN_TO_REAL)] - real[list(GEN_TO_REAL)]).abs()], axis=1)
    out["generation_pairs"] = {}
    for a in NETS:
        for b in [*NETS, "Markov_B2"]:
            if a < b or b == "Markov_B2":
                if a in dist and b in dist and a != b:
                    out["generation_pairs"][f"{a} vs {b}"] = {m: _test(dist[a][m] - dist[b][m])
                                                              for m in GEN_DIST + GEN_TO_REAL}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-seeds", type=int, default=None)
    args = ap.parse_args()
    rows = [r for p in args.parts for r in json.loads(Path(p).read_text())["rows"]]
    seen = {(r["take"], r["model"], r["seed"], r["kind"]) for r in rows}
    if len(seen) != len(rows):
        raise SystemExit(f"duplicate rows: {len(rows) - len(seen)}")
    takes = {r["take"] for r in rows}
    if args.expect_seeds is not None:
        for t in takes:
            n = len({r["seed"] for r in rows if r["take"] == t})
            if n != args.expect_seeds:
                raise SystemExit(f"{t}: {n} seeds, expected {args.expect_seeds}")
    summary = summarise(rows)
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps({k: summary[k] for k in ("n_takes", "seeds", "prediction_mean", "generation_mean")}, indent=1))
    print("->", args.out)


if __name__ == "__main__":
    main()
