"""Merge the partial result files of a split run (job array, --takes / --seeds parts).

Each part is a train_humanize.py output. Network rows are unique per (take, model, seed);
reference rows (B1, B2) are recomputed by every part that touches a take, so identical
copies are kept once. Two parts disagreeing on the same row is an error, not a silent pick.

    python experiments/gnn/merge_results.py parts/*.json --out results/j3b_humanize_3seeds_savio.json
    python experiments/gnn/merge_results.py parts/*.json --out merged.json --expect-takes 15 --expect-seeds 3
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from djembe.results import summarise, summarise_curve

KEY = ("take", "model", "seed", "n_train", "rep")  # the last two exist only in data-efficiency runs
TOL = 1e-9


def _same(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float):
            if not math.isclose(x, y, rel_tol=0, abs_tol=TOL):
                return False
        elif k != "seconds" and x != y:
            return False
    return True


def merge(parts: list[dict]) -> list[dict]:
    rows: dict[tuple, dict] = {}
    for part in parts:
        for row in part["rows"]:
            key = tuple(row.get(k) for k in KEY)
            if key in rows and not _same(rows[key], row):
                raise ValueError(f"conflicting rows for {key}")
            rows.setdefault(key, row)
    return sorted(rows.values(), key=lambda r: (r["take"], r["model"], -1 if r["seed"] is None else r["seed"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-takes", type=int, default=None, help="fail unless every model covers this many takes")
    ap.add_argument("--expect-seeds", type=int, default=None, help="fail unless every network has this many seeds")
    args = ap.parse_args()

    parts = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.parts]
    rows = merge(parts)
    # Coverage is checked per model and, in data-efficiency runs, per training size too:
    # every (take, repetition) must be present at every size, references included.
    coverage = {}
    for r in rows:
        key = r["model"] if "n_train" not in r else f"{r['model']} n={r['n_train']}"
        c = coverage.setdefault(key, {"takes": set(), "seeds": set(), "units": set()})
        c["takes"].add(r["take"])
        c["seeds"].add(r["seed"] if "rep" not in r else r["rep"])
        c["units"].add((r["take"], r.get("rep", r["seed"])))
    problems = []
    for key, c in coverage.items():
        if args.expect_takes and len(c["takes"]) != args.expect_takes:
            problems.append(f"{key}: {len(c['takes'])} takes")
        repeated = key.startswith("GNN") or " n=" in key
        if args.expect_seeds and repeated and len(c["seeds"]) != args.expect_seeds:
            problems.append(f"{key}: {len(c['seeds'])} seeds")
        if args.expect_takes and args.expect_seeds and repeated and len(c["units"]) != args.expect_takes * args.expect_seeds:
            problems.append(f"{key}: {len(c['units'])} of {args.expect_takes * args.expect_seeds} (take, seed) units")
    if problems:
        raise SystemExit("incomplete merge: " + "; ".join(problems))

    configs = [p["config"] for p in parts]
    curve = any("n_train" in r for r in rows)
    out = {"config": {"merged_from": [str(Path(p).name) for p in args.parts], "parts": configs},
           "summary": summarise_curve(rows) if curve else summarise(rows), "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    blocks = out["summary"].items() if curve else [("all", out["summary"])]
    for n, summary in blocks:
        if curve:
            print(f"n_train = {n}")
        for name, s in summary.items():
            print(f"{name:16s} delta {s['nll_delta']:+.3f}  intensity {s['nll_intensity']:.3f}  ({s['runs']} runs)")
    print("->", args.out)


if __name__ == "__main__":
    main()
