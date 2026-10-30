"""Generated soloist over the real accompaniment microphones (``djembe.backing``).

Writes, for one take, 24 cycles after the 3 real context cycles:
- ``original``: the recording as is (all microphones, soloist included);
- ``real``: gated accompaniment + the real soloist microphone;
- ``real_cut``: gated accompaniment + the real soloist line replayed from cut strokes
  (isolates what the cutting itself does to the sound);
- ``GNN_solo`` and ``Markov_B2``: gated accompaniment + the generated line, stroke types
  drawn by ``djembe.stroke_types`` (fitted without this take), played from cut strokes;
- ``compare``: the four of real, real_cut, GNN_solo, Markov_B2 in a row, 2 s apart.

The audio comes from the corpus: it stays in ``output/``.

    python experiments/strokes/render_over_backing.py --take Maraka_2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from djembe import backing as bk
from djembe import dataset as d
from djembe import grid as gr
from djembe import stroke_types as stt

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--take", default="Maraka_2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=24)
    ap.add_argument("--grids", default=str(ROOT / "output" / "j5b_grids"))
    args = ap.parse_args()
    events = d.load_events()
    grids = {t: gr.take_grid(events, t) for t in d.TAKES}
    table = pd.read_csv(ROOT / "output" / "strokes_J1.csv")
    labels = {t: stt.labels_for(g, table, 0) for t, g in grids.items()}
    others = [t for t in d.TAKES if t != args.take and (labels[t] >= 0).any()]
    model = stt.StrokeTypeModel().fit([grids[t] for t in others], [labels[t] for t in others])

    real = grids[args.take]
    start = 3
    starts = bk.cycle_starts(events, real)
    stop = min(start + args.cycles, len(starts) - 1)
    t0, t1 = float(starts[start]) - 0.05, float(starts[stop])
    back, sr = bk.backing(args.take, t0, t1)
    bank, sr_b, lp_sd = bk.stroke_bank(args.take, table)
    assert sr_b == sr
    kinds = pd.DataFrame([(b[0], b[3]) for b in bank], columns=["type", "clean"])
    print(f"{len(bank)} cut soloist strokes (clean / all):",
          {t: f"{int(g.clean.sum())}/{len(g)}" for t, g in kinds.groupby("type")})
    y_all, _ = bk._read(Path(d.DEFAULT_CORPUS) / f"MJ_{real.piece}" / f"MJ_{args.take}_J1.wav")
    solo_real = y_all[int(t0 * sr):int(t0 * sr) + len(back)]
    raw, _ = bk.backing(args.take, t0, t1, gated=False)

    rng = np.random.default_rng(args.seed)
    outs = {"original": bk.mix(raw, solo_real), "real": bk.mix(back, solo_real)}
    outs["real_cut"] = bk.mix(back, bk.soloist_track(real, labels[args.take], starts, bank, sr, lp_sd, t0, t1, start, rng))
    for name in ("GNN_solo", "Markov_B2"):
        z = np.load(Path(args.grids) / f"{args.take}_s{args.seed}_{name}.npz", allow_pickle=False)
        g = gr.TakeGrid(take_id=real.take_id, piece=real.piece, x=z["x"], log_T=z["log_T"], section=real.section,
                        in_signal=real.in_signal, present=z["present"], cycle_idx=real.cycle_idx,
                        delta_offset=real.delta_offset)
        types = model.sample(g, rng, start_cycle=start, given=labels[args.take])
        outs[name] = bk.mix(back, bk.soloist_track(g, types, starts, bank, sr, lp_sd, t0, t1, start, rng))
    gap = np.zeros((2 * sr, 2), np.float32)
    outs["compare"] = np.concatenate([x for n in ("real", "real_cut", "GNN_solo", "Markov_B2") for x in (outs[n], gap)])
    for name, y in outs.items():
        dst = ROOT / "output" / f"{args.take}_backing_{name}.wav"
        sf.write(dst, y, sr, subtype="PCM_16")
        print("->", dst)


if __name__ == "__main__":
    main()
