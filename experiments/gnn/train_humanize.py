"""Roadmap step J3: graph model on the "humanise a cycle" task, against B1 and B2.

Leave-one-take-out (splits.json). In each fold one training take of the test take's
piece is held out for early stopping (the first in alphabetical order), so the
network trains on 13 takes while B1 and B2 are fitted on all 14. Scores are on the
target strokes of jembe 1, jembe 2 and dundun 1 in every cycle with K-1 cycles of
context, for the network and for the references alike.

Variants (positional input, see djembe/grid.py):
  V0 = "full"     (no symmetry)
  V1 = "subpulse" (rotation by whole beats, C4)
  V3 = "none"     (rotation by any pulse, C12: expected to fail, see J1)

    python experiments/gnn/train_humanize.py --variants V0 V1 --seeds 0 1 2
    python experiments/gnn/train_humanize.py --takes Woloso_2 --epochs 5        # quick check
    python experiments/gnn/train_humanize.py --wandb-project djembe              # log to Weights & Biases

Weights & Biases is optional: it is used only with --wandb-project and an API key
(WANDB_API_KEY in the environment, e.g. from Colab's Secrets panel). Only metrics and
settings are logged, never corpus data.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from djembe import dataset as d
from djembe import gnn
from djembe import grid as gr
from djembe.baselines import MarkovModel, PositionModel

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {"V0": "full", "V1": "subpulse", "V3": "none"}
MAIN = (0, 1, 2)


def reference_scores(model, g, K):
    nll = model.nll(g)
    out = {}
    for key in ("delta", "intensity"):
        m = np.isin(nll[key + "_inst"], MAIN) & (nll[key + "_cyc"] >= K - 1)
        out["nll_" + key] = float(nll[key][m].mean())
        out["n_" + key] = int(m.sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", default=str(d.DEFAULT_EVENTS))
    ap.add_argument("--out", default=str(ROOT / "results" / "j3_humanize.json"))
    ap.add_argument("--variants", nargs="+", default=["V0", "V1"], choices=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--takes", nargs="+", default=None, help="test takes (default: all 15 folds)")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wandb-project", default=None)
    args = ap.parse_args()

    wandb = None
    if args.wandb_project:
        import wandb  # noqa: F811

    events = d.load_events(args.events)
    grids = {t: gr.take_grid(events, t) for t in d.TAKES}
    windows = {t: gnn.make_windows(grids[t], args.K) for t in d.TAKES}
    folds = [f for f in d.load_splits()["folds"] if args.takes is None or f["test"][0] in args.takes]
    rows = []
    for fold in folds:
        test = fold["test"][0]
        piece = grids[test].piece
        val = sorted(t for t in fold["train"] if grids[t].piece == piece)[0]
        train_takes = [t for t in fold["train"] if t != val]
        fitted = [grids[t] for t in fold["train"]]
        refs = {"B1_position": PositionModel().fit(fitted),
                "B2_markov_cycle": MarkovModel(memory_scope="cycle").fit(fitted)}
        for name, m in refs.items():
            rows.append({"take": test, "piece": piece, "model": name, "seed": None,
                         **reference_scores(m, grids[test], args.K)})
        train_w = [w for t in train_takes for w in windows[t]]
        for variant in args.variants:
            for seed in args.seeds:
                position = VARIANTS[variant]
                run = None
                if wandb:
                    run = wandb.init(project=args.wandb_project, group=f"j3-{variant}", reinit=True,
                                     name=f"{variant}-{test}-s{seed}",
                                     config={"variant": variant, "position": position, "test_take": test,
                                             "val_take": val, "seed": seed, "K": args.K, "dim": args.dim,
                                             "layers": args.layers, "epochs": args.epochs})
                t0 = time.time()
                model, hist = gnn.train(train_w, windows[val], K=args.K, position=position, dim=args.dim,
                                        layers=args.layers, epochs=args.epochs, seed=seed, device=args.device,
                                        log=(lambda h: run.log(h)) if run else None)
                res = gnn.evaluate(model, windows[test], K=args.K, position=position, device=args.device,
                                   instruments=MAIN)
                row = {"take": test, "piece": piece, "model": f"GNN_{variant}", "seed": seed, **res,
                       "epochs_run": len(hist), "best_val": min(h["val_loss"] for h in hist),
                       "seconds": round(time.time() - t0, 1)}
                rows.append(row)
                if run:
                    run.summary.update({k: v for k, v in row.items() if isinstance(v, (int, float))})
                    run.finish()
                print(f"{test:11s} {variant} seed {seed}: delta {res['nll_delta']:+.3f}  intensity "
                      f"{res['nll_intensity']:.3f}  ({len(hist)} epochs, {row['seconds']} s)", flush=True)
        refs_rows = [r for r in rows if r["take"] == test and r["seed"] is None]
        print(f"{test:11s} refs: " + "  ".join(
            f"{r['model']} delta {r['nll_delta']:+.3f} int {r['nll_intensity']:.3f}" for r in refs_rows), flush=True)

    summary = {}
    for name in sorted({r["model"] for r in rows}):
        sel = [r for r in rows if r["model"] == name]
        summary[name] = {
            "nll_delta": float(np.average([r["nll_delta"] for r in sel], weights=[r["n_delta"] for r in sel])),
            "nll_intensity": float(np.average([r["nll_intensity"] for r in sel],
                                              weights=[r["n_intensity"] for r in sel])),
            "runs": len(sel)}
    for name, s in summary.items():
        print(f"{name:16s} delta {s['nll_delta']:+.3f}  intensity {s['nll_intensity']:.3f}  ({s['runs']} runs)")
    Path(args.out).write_text(json.dumps({"config": vars(args), "summary": summary, "rows": rows}, indent=2))
    print("->", args.out)


if __name__ == "__main__":
    main()
