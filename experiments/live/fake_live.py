"""Roadmap L1, end to end: the soloist plays along with a recorded accompanist as if live.

The isolated jembe-2 track of a take plays the role of the person accompanying. It is
streamed through ``djembe.live.OnsetStream``; a ``LiveSession`` follows the cycle and
asks the causal soloist (trained on the other takes, every drum and jembe 2 alone) for
each pulse 100 ms before it falls, using only what has been heard so far. Stroke types
are then drawn by ``djembe.stroke_types``. Nothing of the take's own soloist is used,
except its microphone as a source of stroke sounds for listening.

Measured: the asynchrony between the soloist and the accompanist on pulses where both
strike (real soloist of the take against the generated one), strokes per cycle, the
link between successive timing deviations, and the share of distinct figures.

Writes ``output/<take>_fakelive_{mix,alone}.wav`` (corpus audio: local only) and
``results/fake_live_<take>.json`` (aggregates).

    python experiments/live/fake_live.py --take Maraka_2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from djembe import backing as bk
from djembe import dataset as d
from djembe import grid as gr
from djembe import solo, solo_seq
from djembe import stroke_types as stt
from djembe.live import GridTracker, LiveSession, j2_offsets

ROOT = Path(__file__).resolve().parents[2]
LEAD = 0.10


def live_model(take: str, grids: dict, epochs: int, seed: int, cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"live_{take}_s{seed}_e{epochs}.pt"
    train = [g for t, g in grids.items() if t != take]
    if f.exists():
        from djembe import gnn
        from djembe.grid import position_features

        m = solo_seq.SoloGNN(gnn.n_base_features() + position_features("full", 1).shape[1])
        m.load_state_dict(torch.load(f))
        return m.eval()
    win = lambda gs: [solo_seq.seq_window(g, c, 4, True, True) for g in gs for c in range(3, g.x.shape[0])]
    m = solo_seq.train(win(train) + win([solo_seq.only_drums(g) for g in train if g.present[1]]), K=4, seed=seed,
                       epochs=epochs, position="full", helix=True)
    torch.save(m.state_dict(), f)
    return m


def main() -> None:
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--take", default="Maraka_2")
    ap.add_argument("--cycles", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()
    torch.set_num_threads(2)
    events = d.load_events()
    grids = {t: gr.take_grid(events, t) for t in d.TAKES}
    real = grids[args.take]
    piece = real.piece
    model = live_model(args.take, grids, args.epochs, args.seed, ROOT / "output" / "live_models")

    on = [tuple(r) for r in np.load(ROOT / "output" / "live_onsets" / f"{args.take}_J2.npy")]
    cyc = events[events["take_id"] == args.take].groupby("cycle_idx")[["cycle_start_s", "cycle_dur_s"]].first().dropna()
    starts, durs = cyc["cycle_start_s"].to_numpy(), cyc["cycle_dur_s"].to_numpy()
    others = [t for t in d.TAKES if t != args.take and t.startswith(piece)]
    feel = j2_offsets(events, others, piece)
    seed_take = next(t for t in others if grids[t].present[1])
    seed_rows = grids[seed_take].x[3:6]

    t_c = starts[1] + feel[0] / 12 * durs[1]
    i0 = next(i for i, (t, _, _) in enumerate(on) if abs(t - t_c) < 0.06)
    tracker = GridTracker(feel)
    heard = [o for o in on[i0:] if o[0] <= on[i0][0] + tracker.listen]
    offset = float(np.median([grids[t].delta_offset for t in others]))
    session = LiveSession(model, tracker, piece, seed_rows, temperature=args.temperature,
                          rng=np.random.default_rng(args.seed), grid_offset=offset)
    session.start([(t, lev) for t, lev, _ in heard])
    now = heard[-1][2]
    rest = [o for o in on[i0 + len(heard):]]
    c, k = tracker.cycle, 0
    while tracker.pulse_time(k, c) - LEAD < now:          # first pulse still ahead of us
        k += 1
        if k == 12:
            c, k = c + 1, 0
    first_cycle = c
    played, j = [], 0
    while c < first_cycle + args.cycles:
        when = tracker.pulse_time(k, c) - LEAD
        while j < len(rest) and rest[j][2] <= when:
            session.hear(rest[j][0], rest[j][1])
            j += 1
        out = session.decide(c, k)
        if out is not None:
            played.append((out[0], c, k, out[1]))
        k += 1
        if k == 12:
            c, k = c + 1, 0
    while j < len(rest) and rest[j][0] < played[-1][0] + 1.0:
        session.hear(rest[j][0], rest[j][1])
        j += 1

    # stroke types on the generated line
    table = pd.read_csv(ROOT / "output" / "strokes_J1.csv")
    labels = {t: stt.labels_for(g, table, 0) for t, g in grids.items()}
    train_takes = [t for t in d.TAKES if t != args.take]
    tm = stt.StrokeTypeModel().fit([grids[t] for t in train_takes], [labels[t] for t in train_takes])
    g_live = session.grid(c - 1)
    types = tm.sample(g_live, np.random.default_rng(args.seed), start_cycle=session.n_seed + first_cycle)
    strokes = [(t, stt.TYPES[types[session.n_seed + cc, kk]] if types[session.n_seed + cc, kk] >= 0 else "tone", z)
               for t, cc, kk, z in played]

    # measures
    j2 = {(cc, kk): t for t, cc, kk, _ in tracker.log}
    asyn = [t - j2[(cc, kk)] for t, cc, kk, _ in played if (cc, kk) in j2]
    ev = events[events["take_id"] == args.take]
    both = ev[ev["instrument"].isin(["J1", "J2"])].pivot_table(index=["cycle_idx", "sd"], columns="instrument",
                                                               values="onset_s").dropna()
    real_asyn = (both["J1"] - both["J2"]).to_numpy()
    real_asyn = real_asyn[np.abs(real_asyn) < 0.1]
    gen_rows = slice(session.n_seed + first_cycle, session.n_seed + c)
    gen_grid = session.grid(c - 1)
    gen_hits = gen_grid.x[gen_rows, :, 0, 0]
    res = {
        "take": args.take, "cycles": int(c - first_cycle), "strokes": len(played),
        "asynchrony_ms": {"generated_median": float(np.median(asyn) * 1000),
                          "generated_iqr": float(np.subtract(*np.percentile(asyn, [75, 25])) * 1000),
                          "real_median": float(np.median(real_asyn) * 1000),
                          "real_iqr": float(np.subtract(*np.percentile(real_asyn, [75, 25])) * 1000),
                          "n_generated": len(asyn)},
        "strokes_per_cycle": {"generated": float(gen_hits.sum(1).mean()), "real": float(real.x[:, :, 0, 0].sum(1).mean())},
        "timing_lag1": {"generated": solo.timing_lag1(replace_x(gen_grid, gen_rows)), "real": solo.timing_lag1(real, 3)},
        "distinct_rate": {"generated": float(len({tuple(r) for r in gen_hits.astype(int)}) / len(gen_hits)),
                          "real": float(len({tuple(r) for r in real.x[3:, :, 0, 0].astype(int)}) / (len(real.x) - 3))},
        "tracker_reacquired": tracker.n_reacquired, "temperature": args.temperature, "grid_offset": offset,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"fake_live_{args.take}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

    # listening: the accompanist's microphone (gated on its own strokes) and the soloist from cut strokes
    t0, t1 = played[0][0] - 1.0, played[-1][0] + 1.0
    base = Path(d.DEFAULT_CORPUS) / f"MJ_{piece}"
    y, sr = bk._read(base / f"MJ_{args.take}_J2.wav")
    y = y * bk.gate(len(y), sr, bk._raw_onsets(base, args.take, "Jembe2"), 0.40)
    acc = y[int(t0 * sr):int(t1 * sr)]
    back = np.stack([acc * 0.65, acc * 0.35], axis=1)
    bank, _, lp_sd = bk.stroke_bank(args.take, table)
    solo_track = bk.strokes_track(strokes, bank, sr, lp_sd, t0, t1, np.random.default_rng(args.seed))
    sf.write(ROOT / "output" / f"{args.take}_fakelive_mix.wav", bk.mix(back, solo_track), sr, subtype="PCM_16")
    sf.write(ROOT / "output" / f"{args.take}_fakelive_alone.wav", bk.mix(np.zeros_like(back), solo_track), sr,
             subtype="PCM_16")
    print("->", ROOT / "output" / f"{args.take}_fakelive_mix.wav")


def replace_x(g, rows):
    from dataclasses import replace

    return replace(g, x=g.x[rows], log_T=g.log_T[rows], section=g.section[rows], in_signal=g.in_signal[rows],
                   cycle_idx=g.cycle_idx[rows])


if __name__ == "__main__":
    main()
