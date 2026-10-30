"""Listen to the generated soloist (J5b) with stroke types, against the real one.

For one held-out take: the stroke-type model (``djembe.stroke_types``) is fitted on the
soloist labels of the other takes, then draws a type for every generated soloist stroke
(network and Markov), in time order, after the real context cycles. The real soloist and
the real accompanying jembe keep the types read from their own audio (strokes without a
label fall back to the intensity rule). Rendered with the CC0 samples; the audio stays
in ``output/`` (it is derived from the corpus' timings).

    python experiments/strokes/render_solo_types.py --take Maraka_2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from djembe import dataset as d
from djembe import grid as gr
from djembe import render
from djembe import stroke_types as stt

ROOT = Path(__file__).resolve().parents[2]


def from_npz(path: Path, real: gr.TakeGrid) -> gr.TakeGrid:
    z = np.load(path, allow_pickle=False)
    return gr.TakeGrid(take_id=real.take_id, piece=str(z["piece"]), x=z["x"], log_T=z["log_T"], section=real.section,
                       in_signal=real.in_signal, present=z["present"], cycle_idx=real.cycle_idx), int(z["n_context"])


def shares(types: np.ndarray, start: int) -> str:
    t = types[start:][types[start:] >= 0]
    return "  ".join(f"{n} {np.mean(t == i):.0%}" for i, n in enumerate(stt.TYPES))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--take", default="Maraka_2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=24)
    ap.add_argument("--grids", default=str(ROOT / "output" / "j5b_grids"))
    ap.add_argument("--solo-emphasis", type=float, default=1.8)
    args = ap.parse_args()
    events = d.load_events()
    grids = {t: gr.take_grid(events, t) for t in d.TAKES}
    tables = {i: pd.read_csv(ROOT / "output" / f"strokes_J{i + 1}.csv") for i in (0, 1)}
    labels = {i: {t: stt.labels_for(g, tables[i], i) for t, g in grids.items()} for i in (0, 1)}
    others = [t for t in d.TAKES if t != args.take and (labels[0][t] >= 0).any()]
    model = stt.StrokeTypeModel().fit([grids[t] for t in others], [labels[0][t] for t in others])
    real = grids[args.take]
    rng = np.random.default_rng(args.seed)
    out_dir = ROOT / "output"
    pieces = []
    for name in ("real", "GNN_solo", "Markov_B2"):
        if name == "real":
            g, start, solo_types = real, 3, labels[0][args.take]
        else:
            g, start = from_npz(Path(args.grids) / f"{args.take}_s{args.seed}_{name}.npz", real)
            solo_types = model.sample(g, rng, start_cycle=start, given=labels[0][args.take])
        print(f"{name:10s} soloist types: {shares(solo_types, start)}")
        y = render.render(g.x, g.log_T, g.present, start_cycle=start, n_cycles=args.cycles, seed=args.seed,
                          solo_emphasis=args.solo_emphasis, strokes={0: solo_types, 1: labels[1][args.take]})
        dst = out_dir / f"{args.take}_solo_types_{name}.wav"
        render.write_wav(dst, y)
        pieces += [y, np.zeros((render.SR * 2, 2))]
        print("->", dst)
    render.write_wav(out_dir / f"{args.take}_solo_types_compare_real_GNN_Markov.wav", np.concatenate(pieces))


if __name__ == "__main__":
    main()
