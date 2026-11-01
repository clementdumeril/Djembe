"""Roadmap step J7: does a memory of figures and a groove encoder give the soloist a groove?

Leave-one-take-out, piece given, whole accompaniment known (the J6 ``take`` setting, where
the graph model's timing already hangs together like a real soloist's). Four graph
models on the same code (``djembe.solo_mem``): ``GNN_base`` (the J6 model), ``GNN_memory``,
``GNN_groove`` and ``GNN_memory_groove``; B2 as reference.

Scores: soloist NLL (strikes, offsets, intensities). Generation at temperature 1 and
0.8 (suffix ``_T08``), compared with the real soloist by ``djembe.solo.compare``, which
includes the groove measures (``self_return``, ``timing_lag12``, ``timing_drift``,
``profile_stability``, ``accent_stability``).

    python experiments/j7/run.py --index 0 --epochs 2      # quick check
    python experiments/j7/run.py --count                   # -> 45
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
from djembe import solo, solo_mem
from djembe.baselines import MarkovModel
from djembe.transformer import n_params

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {"GNN_base": (False, False), "GNN_memory": (True, False), "GNN_groove": (False, True),
            "GNN_memory_groove": (True, True)}
TEMPERATURES = {1.0: "", 0.8: "_T08"}


def tasks(seeds: list[int]) -> list[tuple[str, int]]:
    return [(t, s) for s in seeds for t in d.TAKES]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int)
    group.add_argument("--count", action="store_true")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    todo = tasks(args.seeds)
    if args.count:
        print(len(todo))
        return
    test, seed = todo[args.index]
    K = args.K
    events = d.load_events()
    grids = {t: gr.take_grid(events, t) for t in d.TAKES}
    real = grids[test]
    train_takes = [t for t in d.TAKES if t != test]
    train_grids = [grids[t] for t in train_takes]
    windows = [solo_mem.mem_window(grids[t], c, K) for t in train_takes for c in range(K - 1, grids[t].x.shape[0])]
    test_windows = [solo_mem.mem_window(real, c, K) for c in range(K - 1, real.x.shape[0])]
    b2 = MarkovModel(memory_scope="stroke").fit(train_grids)
    common = {"protocol": "take", "unit": test, "take": test, "piece": real.piece, "seed": seed}
    rows = []
    for name in args.variants:
        memory, groove = VARIANTS[name]
        t0 = time.time()
        m = solo_mem.train(windows, K=K, seed=seed, epochs=args.epochs, device=args.device, use_memory=memory,
                           use_groove=groove)
        info = {"params": n_params(m), "train_s": time.time() - t0}
        rows.append({**common, "kind": "prediction", "model": name, **info,
                     **solo_mem.score(m, test_windows, K=K, device=args.device)})
        for T, suffix in TEMPERATURES.items():
            g = solo_mem.generate(real, m, K=K, rng=np.random.default_rng(1000 + seed), temperature=T,
                                  device=args.device)
            rows.append({**common, "kind": "generation", "model": name + suffix, "temperature": T,
                         **solo.compare(g, real, train_grids, start_gen=K - 1, start_real=K - 1)})
            if args.save_dir:
                out = Path(args.save_dir)
                out.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out / f"j7_{test}_s{seed}_{name}{suffix}.npz", x=g.x, log_T=g.log_T,
                                    present=g.present, piece=real.piece, n_context=K - 1)
        print(test, name, " ".join(f"{k}={rows[-3][k]:.3f}" for k in ("nll_hit", "nll_delta", "nll_intensity")),
              "| T1:", " ".join(f"{k}={rows[-2][k]:.2f}" for k in ("self_return", "distinct_rate", "timing_lag1")),
              "| T0.8:", " ".join(f"{k}={rows[-1][k]:.2f}" for k in ("self_return", "distinct_rate", "timing_lag1")),
              flush=True)
    out = b2.nll(real)
    keep = {k: (out[k + "_cyc"] >= K - 1) & (out[k + "_inst"] == solo.SOLO) for k in ("hit", "delta", "intensity")}
    rows.append({**common, "kind": "prediction", "model": "B2_markov",
                 **{f"nll_{k}": float(out[k][v].mean()) for k, v in keep.items()},
                 **{f"n_{k}": int(v.sum()) for k, v in keep.items()}})
    rows.append({**common, "kind": "generation", "model": "Markov_B2",
                 **solo.compare(solo.generate_markov(real, b2, K=K, rng=np.random.default_rng(1000 + seed)), real,
                                train_grids, start_gen=K - 1, start_real=K - 1)})
    rows.append({**common, "kind": "generation", "model": "real_heldout",
                 **solo.compare(real, real, train_grids, start_gen=K - 1, start_real=K - 1)})
    out_path = Path(args.out or ROOT / "results" / f"j7_{test}_s{seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows}, indent=1))
    print("->", out_path)


if __name__ == "__main__":
    main()
