"""Roadmap step J5: generate whole held-out takes with the graph model and judge them.

For each fold (test take) and seed:
1. train the timing model (V5: full position, helix, last epoch) and the strike model
   on the fold's 13 network-training takes, as in J3b; fit B1/B2 on all 14;
2. score next-cycle strike prediction on the held-out take: strike model against B2's
   strike chain (Bernoulli NLL, jembe 1, jembe 2, dundun 1);
3. generate the held-out take from its first K-1 real cycles, with the GNN and with
   the Markov reference (B2), and compare each with the real take (djembe.gen_eval);
4. as a yardstick, compare another real take of the same piece with the held-out one.

    python experiments/generation/generate_eval.py --takes Woloso_5 --seeds 0 --epochs 2   # quick check
    python experiments/generation/generate_eval.py --seeds 0 --save-dir output/generated
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from djembe import dataset as d
from djembe import gen_eval as ge
from djembe import generate as gen
from djembe import gnn
from djembe import grid as gr
from djembe.baselines import MarkovModel

ROOT = Path(__file__).resolve().parents[2]
MAIN = (0, 1, 2)
TIMING = {"position": "full", "helix": True}


def markov_strike_nll(b2, g: gr.TakeGrid, K: int) -> dict:
    out = b2.nll(g)
    keep = (out["hit_cyc"] >= K - 1) & np.isin(out["hit_inst"], MAIN)
    return {"nll_hit": float(out["hit"][keep].mean()), "n_hit": int(keep.sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", default=str(d.DEFAULT_EVENTS))
    ap.add_argument("--out", default=str(ROOT / "results" / "j5_generation.json"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--takes", nargs="+", default=None)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--save-dir", default=None, help="write generated grids (.npz) here, for rendering")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    K = args.K

    events = d.load_events(args.events)
    grids = {t: gr.take_grid(events, t) for t in d.TAKES}
    windows = {t: gnn.make_windows(grids[t], K) for t in d.TAKES}
    rows = []
    for fold in d.load_splits()["folds"]:
        test = fold["test"][0]
        if args.takes is not None and test not in args.takes:
            continue
        real = grids[test]
        piece = real.piece
        val = sorted(t for t in fold["train"] if grids[t].piece == piece)[0]
        train_takes = [t for t in fold["train"] if t != val]
        train_grids = [grids[t] for t in fold["train"]]
        b2 = MarkovModel(memory_scope="cycle").fit(train_grids)
        recogniser = ge.train_recogniser(train_grids)
        train_w = [w for t in train_takes for w in windows[t]]
        for seed in args.seeds:
            t0 = time.time()
            timing, _ = gnn.train(train_w, windows[val], K=K, select="last", seed=seed, device=args.device,
                                  epochs=args.epochs, **TIMING)
            strikes = gen.train_strikes([gen.strike_window(w) for w in train_w], K=K, seed=seed, device=args.device,
                                        epochs=args.epochs, **TIMING)
            test_sw = [gen.strike_window(w) for w in windows[test]]
            common = {"take": test, "piece": piece, "seed": seed}
            rows.append({**common, "kind": "strike_prediction", "model": "GNN_strikes",
                         **gen.strike_nll(strikes, test_sw, K=K, device=args.device, instruments=MAIN, **TIMING)})
            rows.append({**common, "kind": "strike_prediction", "model": "B2_markov_cycle",
                         **markov_strike_nll(b2, real, K)})

            rng = np.random.default_rng(seed)
            generated = {
                "GNN": gen.generate_gnn(real, strikes, timing, K=K, rng=rng, device=args.device, **TIMING),
                "Markov_B2": gen.generate_markov(real, b2, K=K, rng=rng),
            }
            for name, g in generated.items():
                rows.append({**common, "kind": "generation", "model": name,
                             **ge.compare(g, real, train_grids, recogniser, start_gen=K - 1, start_real=K - 1)})
            other = grids[val]                                   # another real take of the same piece
            rows.append({**common, "kind": "generation", "model": "real_other_take", "other": val,
                         **ge.compare(other, real, [t for t in train_grids if t.take_id != val], recogniser,
                                      start_gen=0, start_real=K - 1)})
            nov, copy = ge.novelty(real, train_grids, K - 1)
            rows.append({**common, "kind": "generation", "model": "real_heldout_novelty", "novelty": nov,
                         "copy_rate": copy})
            if args.save_dir:
                out = Path(args.save_dir)
                out.mkdir(parents=True, exist_ok=True)
                for name, g in generated.items():
                    np.savez_compressed(out / f"{test}_s{seed}_{name}.npz", x=g.x, log_T=g.log_T,
                                        present=g.present, piece=g.piece, n_context=K - 1)
            gnn_row = next(r for r in rows[::-1] if r["model"] == "GNN")
            print(f"{test:11s} seed {seed}: strikes GNN {rows[-6]['nll_hit']:.3f} vs B2 {rows[-5]['nll_hit']:.3f} | "
                  f"GNN profile {gnn_row['profile_l1']:.3f} timing {gnn_row['timing_w1_ms']:.1f} ms "
                  f"recognised {gnn_row['recognised']:.2f} novelty {gnn_row['novelty']:.3f} "
                  f"({time.time() - t0:.0f} s)", flush=True)

    Path(args.out).write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2))
    print("->", args.out)


if __name__ == "__main__":
    main()
