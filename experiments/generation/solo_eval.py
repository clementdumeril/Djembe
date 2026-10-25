"""Roadmap step J5b: generate the soloist (jembe 1) over the real ensemble, and judge it.

For each fold (test take) and seed:
1. train the soloist timing model (V5 settings, ensemble fully known) and the soloist
   strike model (pulse-by-pulse, random prefixes) on the fold's 13 network-training
   takes; fit B1/B2 on all 14;
2. score prediction on the held-out take's soloist line, network against B2:
   strikes (sequential NLL, pulse k given pulses 0..k-1) and offsets/intensities;
3. pick the sampling temperature on the validation take (never on the test take):
   the one whose generated soloist is closest to the real one there (strike profile
   plus the gap in distinct-pattern rate);
4. generate the held-out take's soloist with the network and with B2 (Markov), over the
   real ensemble, from 3 real cycles, and compare both with the real soloist; as a
   yardstick, compare another real take of the same piece with it.

    python experiments/generation/solo_eval.py --takes Woloso_5 --seeds 0 --epochs 2   # quick check
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
from djembe import solo
from djembe.baselines import MarkovModel

ROOT = Path(__file__).resolve().parents[2]
TIMING = {"position": "full", "helix": True}
TEMPERATURES = (0.8, 1.0)


def windows_of(g: gr.TakeGrid, K: int, make) -> list:
    return [make(g, t, K) for t in range(K - 1, g.x.shape[0])]


def b2_solo_scores(b2, g: gr.TakeGrid, K: int) -> dict:
    out = b2.nll(g)
    res = {}
    for key in ("hit", "delta", "intensity"):
        keep = (out[key + "_cyc"] >= K - 1) & (out[key + "_inst"] == solo.SOLO)
        res[f"nll_{key}"] = float(out[key][keep].mean())
        res[f"n_{key}"] = int(keep.sum())
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", default=str(d.DEFAULT_EVENTS))
    ap.add_argument("--out", default=str(ROOT / "results" / "j5b_solo.json"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--takes", nargs="+", default=None)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    K = args.K

    events = d.load_events(args.events)
    grids = {t: gr.take_grid(events, t) for t in d.TAKES}
    rows = []
    for fold in d.load_splits()["folds"]:
        test = fold["test"][0]
        if args.takes is not None and test not in args.takes:
            continue
        real = grids[test]
        val = sorted(t for t in fold["train"] if grids[t].piece == real.piece)[0]
        train_takes = [t for t in fold["train"] if t != val]
        train_grids = [grids[t] for t in fold["train"]]
        no_val = [g for g in train_grids if g.take_id != val]
        b2 = MarkovModel(memory_scope="cycle").fit(train_grids)
        tw = [w for t in train_takes for w in windows_of(grids[t], K, solo.timing_window)]
        sw = [w for t in train_takes for w in windows_of(grids[t], K, solo.strike_window)]
        for seed in args.seeds:
            t0 = time.time()
            common = {"take": test, "piece": real.piece, "seed": seed}
            timing, _ = gnn.train(tw, windows_of(grids[val], K, solo.timing_window), K=K, select="last", seed=seed,
                                  device=args.device, epochs=args.epochs, **TIMING)
            strikes = solo.train_strikes(sw, K=K, seed=seed, device=args.device, epochs=args.epochs, **TIMING)

            net = solo.strike_nll(strikes, windows_of(real, K, solo.strike_window), K=K, device=args.device, **TIMING)
            net.update(gnn.evaluate(timing, windows_of(real, K, solo.timing_window), K=K, device=args.device,
                                    instruments=(solo.SOLO,), **TIMING))
            rows.append({**common, "kind": "prediction", "model": "GNN_solo", **net})
            rows.append({**common, "kind": "prediction", "model": "B2_markov_cycle", **b2_solo_scores(b2, real, K)})

            # temperature chosen on the validation take
            choice = {}
            for T in TEMPERATURES:
                gv = solo.generate(grids[val], strikes, timing, K=K, rng=np.random.default_rng(seed), temperature=T,
                                   device=args.device, **TIMING)
                m = solo.compare(gv, grids[val], no_val, start_gen=K - 1, start_real=K - 1)
                rv = solo.compare(grids[val], grids[val], no_val, start_gen=K - 1, start_real=K - 1)
                choice[T] = m["profile_l1"] + abs(m["distinct_rate"] - rv["distinct_rate"])
            T_best = min(choice, key=choice.get)

            rng = np.random.default_rng(1000 + seed)
            generated = {"GNN_solo": solo.generate(real, strikes, timing, K=K, rng=rng, temperature=T_best,
                                                   device=args.device, **TIMING),
                         "Markov_B2": solo.generate_markov(real, b2, K=K, rng=rng)}
            for name, g in generated.items():
                rows.append({**common, "kind": "generation", "model": name,
                             "temperature": T_best if name == "GNN_solo" else None,
                             **solo.compare(g, real, train_grids, start_gen=K - 1, start_real=K - 1)})
            rows.append({**common, "kind": "generation", "model": "real_other_take", "other": val,
                         **solo.compare(grids[val], real, no_val, start_gen=0, start_real=K - 1)})
            rows.append({**common, "kind": "generation", "model": "real_heldout",
                         **solo.compare(real, real, train_grids, start_gen=K - 1, start_real=K - 1)})
            if args.save_dir:
                out = Path(args.save_dir)
                out.mkdir(parents=True, exist_ok=True)
                for name, g in generated.items():
                    np.savez_compressed(out / f"{test}_s{seed}_{name}.npz", x=g.x, log_T=g.log_T, present=g.present,
                                        piece=g.piece, n_context=K - 1)
            gr_ = rows[-4]
            print(f"{test:11s} seed {seed} T={T_best}: strikes GNN {net['nll_hit']:.3f} vs B2 {rows[-5]['nll_hit']:.3f} | "
                  f"offsets GNN {net['nll_delta']:+.3f} vs B2 {rows[-5]['nll_delta']:+.3f} | gen profile "
                  f"{gr_['profile_l1']:.3f} distinct {gr_['distinct_rate']:.2f} (real {rows[-1]['distinct_rate']:.2f}) "
                  f"({time.time() - t0:.0f} s)", flush=True)

    Path(args.out).write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2))
    print("->", args.out)


if __name__ == "__main__":
    main()
