"""Roadmap step L2: the soloist model in live conditions, against the offline one.

Live, one player accompanies on jembe 2 and the model solos. When it decides pulse k
it has heard the accompaniment only up to pulse k-1, does not know how long the current
cycle will last, has no form labels, and hears one drum instead of three
(``djembe.solo_seq``, ``causal=True`` and ``only_drums``).

Per task (test take or test piece, and seed), on the takes that have a jembe 2:
- ``GNN_offline``: the J6 graph model (whole accompaniment of the cycle, every drum);
- ``GNN_live``: one causal model trained on every drum and on jembe 2 alone (half of its
  windows each, so it works in both), scored causally with every drum
  (``GNN_live_alldrums``) and with jembe 2 alone (``GNN_live``, the live setting);
- ``B2_markov``: the soloist's own line only, so unaffected by the accompaniment.

Generation: the live model solos over jembe 2 alone, causally, compared with the real
soloist (``djembe.solo.compare``); Markov as reference.

    python experiments/live/causal_eval.py --protocol take --index 0 --epochs 2      # quick check
    python experiments/live/causal_eval.py --protocol take --count                   # -> 36
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from djembe import dataset as d
from djembe import grid as gr
from djembe import solo, solo_seq
from djembe.baselines import MarkovModel
from djembe.transformer import n_params

ROOT = Path(__file__).resolve().parents[2]
GRAPH = {"position": "full", "helix": True}
J2 = 1


def with_j2(grids: dict) -> list[str]:
    return [t for t in d.TAKES if grids[t].present[J2]]


def tasks(protocol: str, seeds: list[int], takes_with_j2: list[str]) -> list[tuple[str, int]]:
    units = list(d.PIECES) if protocol == "piece" else takes_with_j2
    return [(u, s) for s in seeds for u in units]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocol", choices=("take", "piece"), required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int)
    group.add_argument("--count", action="store_true")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    events = d.load_events()
    grids = {t: gr.take_grid(events, t) for t in d.TAKES}
    todo = tasks(args.protocol, args.seeds, with_j2(grids))
    if args.count:
        print(len(todo))
        return
    unit, seed = todo[args.index]
    K, piece_input = args.K, args.protocol == "take"
    if args.protocol == "piece":
        test_takes = [t for t in with_j2(grids) if grids[t].piece == unit]
        train_takes = [t for t in d.TAKES if grids[t].piece != unit]
        pooled = lambda g: replace(g, piece="any")
    else:
        test_takes = [unit]
        train_takes = [t for t in d.TAKES if t != unit]
        pooled = lambda g: g
    train_grids = [grids[t] for t in train_takes]
    b2 = MarkovModel(memory_scope="stroke").fit([pooled(g) for g in train_grids])

    def windows(gs, causal):
        return [solo_seq.seq_window(g, c, K, piece_input, causal) for g in gs for c in range(K - 1, g.x.shape[0])]

    solo_j2 = [solo_seq.only_drums(g) for g in train_grids if g.present[J2]]
    t0 = time.time()
    offline = solo_seq.train(windows(train_grids, False), K=K, seed=seed, epochs=args.epochs, device=args.device,
                             **GRAPH)
    t_off = time.time() - t0
    t0 = time.time()
    live = solo_seq.train(windows(train_grids, True) + windows(solo_j2, True), K=K, seed=seed, epochs=args.epochs,
                          device=args.device, **GRAPH)
    t_live = time.time() - t0

    rows = []
    common_run = {"protocol": args.protocol, "unit": unit, "seed": seed, "train_takes": train_takes}
    for test in test_takes:
        real = grids[test]
        real_j2 = solo_seq.only_drums(real)
        common = {**common_run, "take": test, "piece": real.piece}
        score = lambda m, g, causal: solo_seq.score(m, windows([g], causal), K=K, device=args.device, **GRAPH)
        mine = [
            {**common, "kind": "prediction", "model": "GNN_offline", "params": n_params(offline), "train_s": t_off,
             **score(offline, real, False)},
            {**common, "kind": "prediction", "model": "GNN_live_alldrums", **score(live, real, True)},
            {**common, "kind": "prediction", "model": "GNN_live", "params": n_params(live), "train_s": t_live,
             **score(live, real_j2, True)},
        ]
        out = b2.nll(pooled(real))
        keep = {k: (out[k + "_cyc"] >= K - 1) & (out[k + "_inst"] == solo.SOLO) for k in ("hit", "delta", "intensity")}
        mine.append({**common, "kind": "prediction", "model": "B2_markov",
                     **{f"nll_{k}": float(out[k][m].mean()) for k, m in keep.items()},
                     **{f"n_{k}": int(m.sum()) for k, m in keep.items()}})
        rng = np.random.default_rng(1000 + seed)
        gen_live = solo_seq.generate(real_j2, live, K=K, piece_input=piece_input, causal=True, rng=rng,
                                     device=args.device, **GRAPH)
        generated = {"GNN_live": replace(gen_live, x=np.concatenate([gen_live.x[:, :, :1], real.x[:, :, 1:]], 2),
                                         present=real.present),
                     "Markov_B2": replace(solo.generate_markov(pooled(real), b2, K=K, rng=rng), piece=real.piece)}
        for name, g in generated.items():
            mine.append({**common, "kind": "generation", "model": name,
                         **solo.compare(g, real, train_grids, start_gen=K - 1, start_real=K - 1)})
        mine.append({**common, "kind": "generation", "model": "real_heldout",
                     **solo.compare(real, real, train_grids, start_gen=K - 1, start_real=K - 1)})
        if args.save_dir:
            sd = Path(args.save_dir)
            sd.mkdir(parents=True, exist_ok=True)
            for name, g in generated.items():
                np.savez_compressed(sd / f"live_{args.protocol}_{test}_s{seed}_{name}.npz", x=g.x, log_T=g.log_T,
                                    present=g.present, piece=real.piece, n_context=K - 1)
        for r in mine:
            keys = [k for k in ("nll_hit", "nll_delta", "nll_intensity", "timing_lag1", "distinct_rate") if k in r]
            print(test, r["kind"][:4], f"{r['model']:18s}", " ".join(f"{k}={r[k]:.3f}" for k in keys), flush=True)
        rows += mine
    out = Path(args.out or ROOT / "results" / f"live_{args.protocol}_{unit}_s{seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
