"""Roadmap step J6: graph model against transformer, stroke by stroke, on seen and unseen rhythms.

The soloist (jembe 1) over the real accompaniment, as in J5b, with the timing now drawn
stroke by stroke too. Three protocols:

- ``take``: leave-one-take-out, the piece given as input (the J5b setting);
- ``take_nopiece``: leave-one-take-out, no piece input (what removing the label costs);
- ``piece``: leave-one-piece-out, no piece input: trained on two rhythms, the soloist of
  the third is predicted and generated without the model ever having heard it.

Models, all scored on the same soloist pulses of the target cycles (K-1 context cycles):
- ``GNN_seq`` (``djembe.solo_seq``, 56 k parameters) and ``Transformer``
  (``djembe.transformer``, 181 k) with ``Transformer_matched`` (width 36, 60 k, the
  graph model's size), same factorisation: strike of pulse k, then offset and intensity
  given it strikes, from the context, the whole accompaniment and the soloist's pulses
  before k;
- ``B2_markov`` (memory along the line, previous stroke included) and ``B1_position``.
  Under ``piece`` they are fitted with the piece label removed (one pooled table).

Generation: temperature 1 for every model (no tuning on any take), from 3 real cycles;
``djembe.solo.compare`` against the real soloist, including ``timing_lag1``.

One task = one training set and seed: (test take, seed) for the take protocols,
(test piece, seed) for ``piece``, which then scores every take of that piece.

    python experiments/j6/run.py --protocol take --index 0 --epochs 2      # quick check
    python experiments/j6/run.py --protocol piece --count --seeds 0 1 2   # -> 9
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
from djembe import transformer as tf
from djembe.baselines import MarkovModel, PositionModel

ROOT = Path(__file__).resolve().parents[2]
GRAPH = {"position": "full", "helix": True}
TRANSFORMERS = {"Transformer": {}, "Transformer_matched": {"dim": 36, "ff": 72}}


def tasks(protocol: str, seeds: list[int]) -> list[tuple[str, int]]:
    units = list(d.PIECES) if protocol == "piece" else list(d.TAKES)
    return [(u, s) for s in seeds for u in units]


def solo_scores(model, g: gr.TakeGrid, K: int) -> dict:
    out = model.nll(g)
    res = {}
    for key in ("hit", "delta", "intensity"):
        keep = (out[key + "_cyc"] >= K - 1) & (out[key + "_inst"] == solo.SOLO)
        res[f"nll_{key}"] = float(out[key][keep].mean())
        res[f"n_{key}"] = int(keep.sum())
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocol", choices=("take", "take_nopiece", "piece"), required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int)
    group.add_argument("--count", action="store_true")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--models", nargs="+", default=["GNN_seq", *TRANSFORMERS])
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    todo = tasks(args.protocol, args.seeds)
    if args.count:
        print(len(todo))
        return
    unit, seed = todo[args.index]
    K, piece_input = args.K, args.protocol == "take"

    events = d.load_events()
    grids = {t: gr.take_grid(events, t) for t in d.TAKES}
    if args.protocol == "piece":
        test_takes = [t for t in d.TAKES if grids[t].piece == unit]
        train_takes = [t for t in d.TAKES if grids[t].piece != unit]
        pooled = lambda g: replace(g, piece="any")
    else:
        test_takes = [unit]
        train_takes = [t for t in d.TAKES if t != unit]
        pooled = lambda g: g
    train_grids = [grids[t] for t in train_takes]
    b1 = PositionModel().fit([pooled(g) for g in train_grids])
    b2 = MarkovModel(memory_scope="stroke").fit([pooled(g) for g in train_grids])

    common_run = {"protocol": args.protocol, "unit": unit, "seed": seed, "train_takes": train_takes}
    models, info = {}, {}
    windows_of = lambda make, takes: [make(grids[t], c, K, piece_input) for t in takes
                                      for c in range(K - 1, grids[t].x.shape[0])]
    if "GNN_seq" in args.models:
        t0 = time.time()
        models["GNN_seq"] = solo_seq.train(windows_of(solo_seq.seq_window, train_takes), K=K, seed=seed,
                                           epochs=args.epochs, device=args.device, **GRAPH)
        info["GNN_seq"] = {"params": tf.n_params(models["GNN_seq"]), "train_s": time.time() - t0}
    for name, size in TRANSFORMERS.items():
        if name not in args.models:
            continue
        t0 = time.time()
        models[name] = tf.train(windows_of(tf.tok_window, train_takes), K=K, seed=seed, epochs=args.epochs,
                                device=args.device, **size)
        info[name] = {"params": tf.n_params(models[name]), "train_s": time.time() - t0}

    rows = []
    for test in test_takes:
        real = grids[test]
        common = {**common_run, "take": test, "piece": real.piece}
        mine = []
        for name, m in models.items():
            if name == "GNN_seq":
                s = solo_seq.score(m, windows_of(solo_seq.seq_window, [test]), K=K, device=args.device, **GRAPH)
            else:
                s = tf.score(m, windows_of(tf.tok_window, [test]), device=args.device)
            mine.append({**common, "kind": "prediction", "model": name, **info[name], **s})
        mine.append({**common, "kind": "prediction", "model": "B2_markov", **solo_scores(b2, pooled(real), K)})
        mine.append({**common, "kind": "prediction", "model": "B1_position", **solo_scores(b1, pooled(real), K)})

        rng = np.random.default_rng(1000 + seed)
        generated = {}
        for name, m in models.items():
            if name == "GNN_seq":
                generated[name] = solo_seq.generate(real, m, K=K, piece_input=piece_input, rng=rng,
                                                    device=args.device, **GRAPH)
            else:
                generated[name] = tf.generate(real, m, K=K, piece_input=piece_input, rng=rng, device=args.device)
        generated["Markov_B2"] = replace(solo.generate_markov(pooled(real), b2, K=K, rng=rng), piece=real.piece)
        for name, g in generated.items():
            mine.append({**common, "kind": "generation", "model": name,
                         **solo.compare(g, real, train_grids, start_gen=K - 1, start_real=K - 1)})
        mine.append({**common, "kind": "generation", "model": "real_heldout",
                     **solo.compare(real, real, train_grids, start_gen=K - 1, start_real=K - 1)})
        if args.save_dir:
            out = Path(args.save_dir)
            out.mkdir(parents=True, exist_ok=True)
            for name, g in generated.items():
                np.savez_compressed(out / f"{args.protocol}_{test}_s{seed}_{name}.npz", x=g.x, log_T=g.log_T,
                                    present=g.present, piece=real.piece, n_context=K - 1)
        for r in mine:
            keys = [k for k in ("nll_hit", "nll_delta", "nll_intensity", "timing_lag1", "distinct_rate") if k in r]
            print(test, r["kind"][:4], f"{r['model']:12s}", " ".join(f"{k}={r[k]:.3f}" for k in keys), flush=True)
        rows += mine

    out = Path(args.out or ROOT / "results" / f"j6_{args.protocol}_{unit}_s{seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
