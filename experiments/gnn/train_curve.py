"""Roadmap step J4: data-efficiency curve of the graph model, with and without symmetry.

Same task, folds and scores as train_humanize.py, but the network AND the references
are fitted on only n of the training takes, for n in --n-train. The question: does
building a symmetry into the network (V1 beat rotation C4, V3 pulse rotation C12)
pay off when data is scarce, even if it does not with all the data?

Training subsets, per fold and repetition r (the value given in --seeds):
- the pool is the fold's 13 network-training takes (the validation take stays out,
  as in train_humanize.py; with --select last it is only logged, never used to pick);
- takes are ordered by drawing round-robin across pieces, the test take's piece
  first, each piece's takes shuffled with a generator seeded by (r, fold); the
  subset of size n is the first n of that order. Subsets are therefore nested
  (the 2 takes are among the 4, the 4 among the 8) and, from n = 1, always contain
  a take of the test piece; from n = 3 every piece is present;
- r also seeds the network, so repetitions vary both the subset and the initialisation.
At n = 13 the subset is the whole pool: B1 and B2 are then fitted on 13 takes, not 14
as in J3/J3b, so their scores differ slightly from those tables.

    python experiments/gnn/train_curve.py --n-train 2 4 8 13 --seeds 0 1 2 --variants V0 V1 V3 V5
    python experiments/gnn/train_curve.py --takes Woloso_5 --n-train 2 --seeds 0 --epochs 2   # quick check
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
from djembe.results import summarise_curve
from train_humanize import MAIN, VARIANTS, reference_scores

ROOT = Path(__file__).resolve().parents[2]


def subset_order(pool: list[str], test_piece: str, rep: int, fold_index: int) -> list[str]:
    """Order of the pool for nested subsets: round-robin across pieces, test piece first."""
    rng = np.random.default_rng([rep, fold_index])
    by_piece: dict[str, list[str]] = {}
    for t in sorted(pool):
        by_piece.setdefault(t.split("_")[0], []).append(t)
    for takes in by_piece.values():
        rng.shuffle(takes)
    others = sorted(p for p in by_piece if p != test_piece)
    rng.shuffle(others)
    pieces = ([test_piece] if test_piece in by_piece else []) + others
    order = []
    while any(by_piece[p] for p in pieces):
        for p in pieces:
            if by_piece[p]:
                order.append(by_piece[p].pop(0))
    return order


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", default=str(d.DEFAULT_EVENTS))
    ap.add_argument("--out", default=str(ROOT / "results" / "j4_curve.json"))
    ap.add_argument("--variants", nargs="+", default=["V0", "V1", "V3", "V5"], choices=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0], help="repetitions: subset draw and network seed")
    ap.add_argument("--n-train", nargs="+", type=int, default=[2, 4, 8, 13])
    ap.add_argument("--takes", nargs="+", default=None, help="test takes (default: all 15 folds)")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--select", choices=("best", "last"), default="last")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    events = d.load_events(args.events)
    grids = {t: gr.take_grid(events, t) for t in d.TAKES}
    windows = {t: gnn.make_windows(grids[t], args.K) for t in d.TAKES}
    all_folds = d.load_splits()["folds"]
    rows = []
    for fold_index, fold in enumerate(all_folds):
        test = fold["test"][0]
        if args.takes is not None and test not in args.takes:
            continue
        piece = grids[test].piece
        val = sorted(t for t in fold["train"] if grids[t].piece == piece)[0]
        pool = [t for t in fold["train"] if t != val]
        for rep in args.seeds:
            order = subset_order(pool, piece, rep, fold_index)
            for n in args.n_train:
                subset = order[:n]
                common = {"take": test, "piece": piece, "rep": rep, "n_train": len(subset), "subset": subset}
                fitted = [grids[t] for t in subset]
                refs = {"B1_position": PositionModel().fit(fitted),
                        "B2_markov_cycle": MarkovModel(memory_scope="cycle").fit(fitted)}
                for name, m in refs.items():
                    rows.append({**common, "model": name, "seed": None, **reference_scores(m, grids[test], args.K)})
                train_w = [w for t in subset for w in windows[t]]
                b1_prior = gnn.prior_from_b1(PositionModel().fit(fitted))
                for variant in args.variants:
                    cfg = VARIANTS[variant]
                    graph = {"helix": cfg.get("helix", False), "prior": b1_prior if cfg.get("prior") else None}
                    t0 = time.time()
                    model, hist = gnn.train(train_w, windows[val], K=args.K, position=cfg["position"], dim=args.dim,
                                            layers=args.layers, epochs=args.epochs, patience=args.patience,
                                            select=args.select, seed=rep, device=args.device, **graph)
                    res = gnn.evaluate(model, windows[test], K=args.K, position=cfg["position"], device=args.device,
                                       instruments=MAIN, **graph)
                    rows.append({**common, "model": f"GNN_{variant}", "seed": rep, **res,
                                 "epochs_run": len(hist), "seconds": round(time.time() - t0, 1)})
                    print(f"{test:11s} n={len(subset):2d} rep {rep} {variant}: delta {res['nll_delta']:+.3f}  "
                          f"intensity {res['nll_intensity']:.3f}  ({rows[-1]['seconds']} s)", flush=True)
                b = {r["model"]: r["nll_delta"] for r in rows[-(2 + len(args.variants)):] if r["seed"] is None}
                print(f"{test:11s} n={len(subset):2d} rep {rep} refs: B1 {b['B1_position']:+.3f}  "
                      f"B2 {b['B2_markov_cycle']:+.3f}  subset {subset}", flush=True)

    summary = summarise_curve(rows)
    for n, models in summary.items():
        print(f"n = {n}: " + "  ".join(f"{m} {s['nll_delta']:+.3f}" for m, s in models.items()))
    Path(args.out).write_text(json.dumps({"config": vars(args), "summary": summary, "rows": rows}, indent=2))
    print("->", args.out)


if __name__ == "__main__":
    main()
