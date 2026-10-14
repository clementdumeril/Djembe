"""Rebuild a J3 results file from the runs logged in Weights & Biases.

Each training run logged its test take, variant and seed (config) and its final scores
(summary). B1 and B2 are deterministic and are not logged; their rows are copied from a
local results file of the same task.

    pip install wandb && wandb login          # once; the key is typed in your own terminal
    python experiments/gnn/fetch_wandb.py --project dumerilclement-clementdumeril/djembe \
        --refs results/j3_humanize_seed0_cpu.json --out results/j3_humanize_3seeds_colab.json

Alternatively, without installing anything: in the W&B project, open the runs table,
"Export as CSV", and pass that file with --csv instead of --project.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

KEYS = ("nll_delta", "n_delta", "nll_intensity", "n_intensity", "epochs_run", "best_val", "seconds")


def from_api(project: str, group_prefix: str) -> list[dict]:
    import wandb

    rows = []
    for run in wandb.Api().runs(project):
        if not (run.group or "").startswith(group_prefix) or run.state != "finished":
            continue
        c, s = run.config, run.summary
        if "nll_delta" not in s:
            continue
        rows.append({"take": c["test_take"], "model": f"GNN_{c['variant']}", "seed": c["seed"],
                     **{k: s.get(k) for k in KEYS}, "wandb_id": run.id})
    return rows


def from_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("nll_delta"):
                continue
            rows.append({"take": r["test_take"], "model": f"GNN_{r['variant']}", "seed": int(float(r["seed"])),
                         **{k: float(r[k]) for k in KEYS if r.get(k)}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--project", help="entity/project, e.g. dumerilclement-clementdumeril/djembe")
    src.add_argument("--csv", help="runs table exported from the W&B web page")
    ap.add_argument("--group-prefix", default="j3-")
    ap.add_argument("--refs", required=True, help="local results file holding the B1/B2 rows")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = from_api(args.project, args.group_prefix) if args.project else from_csv(args.csv)
    # keep the last run per (take, model, seed) if a fold was rerun
    unique = {(r["take"], r["model"], r["seed"]): r for r in rows}
    rows = list(unique.values())
    refs = [r for r in json.load(open(args.refs))["rows"] if r["seed"] is None]
    pieces = {r["take"]: r["piece"] for r in refs}
    for r in rows:
        r["piece"] = pieces.get(r["take"])
    rows = refs + rows
    summary = {}
    for name in sorted({r["model"] for r in rows}):
        sel = [r for r in rows if r["model"] == name]
        summary[name] = {
            "nll_delta": float(np.average([r["nll_delta"] for r in sel], weights=[r["n_delta"] for r in sel])),
            "nll_intensity": float(np.average([r["nll_intensity"] for r in sel], weights=[r["n_intensity"] for r in sel])),
            "runs": len(sel)}
        print(f"{name:16s} delta {summary[name]['nll_delta']:+.3f}  intensity {summary[name]['nll_intensity']:.3f}"
              f"  ({len(sel)} runs)")
    Path(args.out).write_text(json.dumps({"config": {"source": args.project or args.csv}, "summary": summary,
                                          "rows": rows}, indent=2))
    print("->", args.out)


if __name__ == "__main__":
    main()
