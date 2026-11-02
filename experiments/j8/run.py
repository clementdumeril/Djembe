"""Roadmap step J8: two-stage soloist (figure planner, then the graph model) against the graph model alone.

Leave-one-take-out, piece given, whole accompaniment known (the J6/J7 ``take`` setting).
Per task: the J6 graph model (``GNN_base``), and the planner plus executor of
``djembe.planner`` (``TwoStage``), both 40 epochs; the planner is scored against the rule
"take the last cycle's figure". Generation at temperature 1 and 0.8 (suffix ``_T08``,
applied to the strokes; the planner draws at temperature 1), compared with the real
soloist by ``djembe.solo.compare`` (groove measures included). Markov as reference.

    python experiments/j8/run.py --index 0 --epochs 2      # quick check
    python experiments/j8/run.py --count                   # -> 45
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from djembe import dataset as d
from djembe import grid as gr
from djembe import planner as pl
from djembe import solo, solo_seq
from djembe.baselines import MarkovModel
from djembe.transformer import n_params

ROOT = Path(__file__).resolve().parents[2]
TEMPERATURES = {1.0: "", 0.8: "_T08"}
GRAPH = {"position": "full", "helix": True}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int)
    group.add_argument("--count", action="store_true")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--plan-epochs", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()
    todo = [(t, s) for s in args.seeds for t in d.TAKES]
    if args.count:
        print(len(todo))
        return
    test, seed = todo[args.index]
    K = args.K
    events = d.load_events()
    grids = {t: gr.take_grid(events, t) for t in d.TAKES}
    real = grids[test]
    train = [grids[t] for t in d.TAKES if t != test]
    common = {"protocol": "take", "unit": test, "take": test, "piece": real.piece, "seed": seed}
    rows = []

    def save(g, name):
        if args.save_dir:
            out = Path(args.save_dir)
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / f"j8_{test}_s{seed}_{name}.npz", x=g.x, log_T=g.log_T, present=g.present,
                                piece=real.piece, n_context=K - 1)

    def gen_rows(name, make):
        for T, suffix in TEMPERATURES.items():
            g, extra = make(T)
            rows.append({**common, "kind": "generation", "model": name + suffix, "temperature": T, **extra,
                         **solo.compare(g, real, train, start_gen=K - 1, start_real=K - 1)})
            save(g, name + suffix)
            r = rows[-1]
            print(test, name + suffix, " ".join(f"{k}={r[k]:.2f}" for k in ("self_return", "distinct_rate",
                                                                           "timing_lag1", "accent_stability",
                                                                           "profile_stability", "density_ratio")),
                  flush=True)

    # graph model alone (J6)
    t0 = time.time()
    base_w = [solo_seq.seq_window(g, c, K) for g in train for c in range(K - 1, g.x.shape[0])]
    base = solo_seq.train(base_w, K=K, seed=seed, epochs=args.epochs, **GRAPH)
    rows.append({**common, "kind": "prediction", "model": "GNN_base", "params": n_params(base),
                 "train_s": time.time() - t0,
                 **solo_seq.score(base, [solo_seq.seq_window(real, c, K) for c in range(K - 1, real.x.shape[0])],
                                  K=K, **GRAPH)})
    gen_rows("GNN_base", lambda T: (solo_seq.generate(real, base, K=K, rng=np.random.default_rng(1000 + seed),
                                                      temperature=T, **GRAPH), {}))

    # two stages
    t0 = time.time()
    planner = pl.train_planner(pl.plan_examples(train), epochs=args.plan_epochs, seed=seed)
    executor = solo_seq.train(pl.exec_windows(train, K), K=K, seed=seed, epochs=args.epochs, **GRAPH)
    rows.append({**common, "kind": "prediction", "model": "TwoStage_plan", "train_s": time.time() - t0,
                 "params": n_params(planner) + n_params(executor),
                 **pl.planner_scores(planner, pl.plan_examples([real], start=K - 1))})

    def two_stage(T):
        log = []
        g = pl.generate(real, planner, executor, K=K, rng=np.random.default_rng(1000 + seed), temperature=T, log=log)
        return g, {"base_used": float(np.mean(log))}

    gen_rows("TwoStage", two_stage)

    b2 = MarkovModel(memory_scope="stroke").fit(train)
    rows.append({**common, "kind": "generation", "model": "Markov_B2",
                 **solo.compare(solo.generate_markov(real, b2, K=K, rng=np.random.default_rng(1000 + seed)), real,
                                train, start_gen=K - 1, start_real=K - 1)})
    rows.append({**common, "kind": "generation", "model": "real_heldout",
                 **solo.compare(real, real, train, start_gen=K - 1, start_real=K - 1)})
    out = Path(args.out or ROOT / "results" / f"j8_{test}_s{seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, indent=1))
    print("->", out)


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
