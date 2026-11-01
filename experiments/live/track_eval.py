"""Roadmap L1: follow the cycle from the jembe-2 track alone, as if it were played live.

For every take with a jembe 2, its isolated track is fed to ``djembe.live.OnsetStream``
in 256-sample blocks; each stroke reaches the tracker only when the stream confirms it.
The tracker starts on the first stroke of the take's second annotated cycle (the player
starts on the first pulse), listens to the strokes of the next ``GridTracker.listen``
seconds, then follows.

Two measures, both of predictions made ``lead`` seconds ahead (the time the soloist
has to decide and schedule a stroke):
- ``player``: the time of the accompanist's next stroke (the corpus' jembe-2 onsets).
  Live, the player is the reference the soloist must stay with. ``floor`` is the same
  error for a predictor that knows the annotated grid exactly and adds the piece's
  mean jembe-2 offsets: what the player's own variability leaves unpredictable.
- ``metre``: the annotated pulses of the corpus, which follow the whole ensemble; an
  accompanist alone can drift from them for a few cycles (on Manjanin_3 jembe 2 plays up
  to half a pulse ahead of the annotated grid for four cycles).

``lost`` is the share of predictions off by more than half a pulse, ``off_beat`` the
same ignoring whole-beat shifts (jembe-2 strokes alone repeat every beat).

The soloist and the dundun leak into the corpus' jembe-2 microphone; the tracker's level
floor is the only defence used here (live there is no dundun, and the system knows when
it played, ``GridTracker.ignore_own``), so this is harder than live.

Trackers: ``feel`` (offsets of jembe 2 per pulse from the other takes of the piece),
``isochronous`` (the same filter assuming jembe 2 plays on the pulses), ``frozen`` (the
start-up tempo kept for the whole take). Also the stream's strokes against the corpus'
raw jembe-2 onsets. Writes aggregates only.

    python experiments/live/track_eval.py      # -> results/live_tracking.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from djembe import dataset as d
from djembe import grid as gr
from djembe.live import J2_PULSES, GridTracker, OnsetStream, j2_offsets
from djembe.onsets import compare_with

ROOT = Path(__file__).resolve().parents[2]
BLOCK = 256
TRACKERS = ("feel", "isochronous", "frozen")


def stream(y: np.ndarray, sr: int) -> list[tuple[float, float, float]]:
    """(onset time, level, time it was confirmed)."""
    s = OnsetStream(sr)
    out = []
    for b in range(0, len(y), BLOCK):
        for t, lev in s.push(y[b:b + BLOCK]):
            out.append((t, lev, (b + BLOCK) / sr))
    return out


def run_tracker(onsets, offsets, starts, durs, queries, s0: int, lead: float, frozen: bool = False, own=(), **kw):
    """Errors [s] of predictions made ``lead`` s ahead; ``queries``: (true time, annotated cycle, pulse, tag)."""
    t_c = starts[s0] + offsets[0] / 12 * durs[s0]
    first = [i for i, (t, _, _) in enumerate(onsets) if abs(t - t_c) < 0.06]
    if not first:
        return None
    i0 = first[0]
    tr = GridTracker(offsets, **kw)
    tr.ignore_own(own)
    heard = [o for o in onsets[i0:] if o[0] <= onsets[i0][0] + tr.listen]
    tr.start_from([(t, lev) for t, lev, _ in heard])
    ready = heard[-1][2]
    events = [(conf, 0, (t, lev)) for t, lev, conf in onsets[i0 + len(heard):]]
    events += [(true - lead, 1, (true, c, k, tag)) for true, c, k, tag in queries if c > s0]
    events.sort(key=lambda e: (e[0], e[1]))
    errs = {}
    for when, kind, what in events:
        if when < ready:
            continue
        if kind == 0:
            if not frozen:
                tr.observe(*what)
        else:
            true, c, k, tag = what
            tr.advance_to(when)
            errs.setdefault(tag, []).append(tr.pulse_time(k, c - s0) - true)
    return {tag: np.array(v) for tag, v in errs.items()}, tr.n_reacquired


def stats(e: np.ndarray, pulse: float) -> dict:
    beat = 3 * pulse
    in_beat = np.abs(np.mod(e + beat / 2, beat) - beat / 2)
    return {"median_abs_ms": float(np.median(np.abs(e)) * 1000), "p90_abs_ms": float(np.percentile(np.abs(e), 90) * 1000),
            "lost_share": float(np.mean(np.abs(e) > 0.5 * pulse)), "off_beat_share": float(np.mean(in_beat > 0.5 * pulse))}


def main() -> None:
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lead", type=float, default=0.10)
    ap.add_argument("--out", default=str(ROOT / "results" / "live_tracking.json"))
    ap.add_argument("--cache", default=str(ROOT / "output" / "live_onsets"), help="stream onsets per take (local)")
    ap.add_argument("--tempo-drift", type=float, default=0.04)
    ap.add_argument("--obs-sd", type=float, default=0.06)
    args = ap.parse_args()
    kw = {"tempo_drift": args.tempo_drift, "obs_sd": args.obs_sd}
    events = d.load_events()
    rows = []
    for take in d.TAKES:
        g = gr.take_grid(events, take)
        if not g.present[1]:
            continue
        piece = g.piece
        base = Path(d.DEFAULT_CORPUS) / f"MJ_{piece}"
        cache = Path(args.cache) / f"{take}_J2.npy"
        if cache.exists():
            on = [tuple(r) for r in np.load(cache)]
        else:
            y, sr = sf.read(base / f"MJ_{take}_J2.wav", dtype="float32", always_2d=True)
            on = stream(y.mean(axis=1), sr)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, np.array(on))
        raw = pd.read_csv(base / "annotations" / f"MJ_{take}_Onsets_Raw_Jembe2.csv")["Time"].to_numpy()
        solo_raw = pd.read_csv(base / "annotations" / f"MJ_{take}_Onsets_Raw_Jembe1.csv")["Time"].to_numpy()
        det = compare_with(np.array([t for t, _, _ in on]), raw, tol=0.02)
        near = [t - raw[np.argmin(np.abs(raw - t))] for t, _, _ in on]
        ev = events[events["take_id"] == take]
        cyc = ev.groupby("cycle_idx")[["cycle_start_s", "cycle_dur_s"]].first().dropna()
        starts, durs = cyc["cycle_start_s"].to_numpy(), cyc["cycle_dur_s"].to_numpy()
        pos = {c: n for n, c in enumerate(cyc.index)}
        others = [t for t in d.TAKES if t != take and t.startswith(piece)]
        feel = j2_offsets(events, others, piece)
        j2 = ev[(ev["instrument"] == "J2") & ev["cycle_idx"].isin(pos) & ev["sd"].sub(1).isin(J2_PULSES)]
        metre_q = [(starts[c] + k / 12 * durs[c], c, k, "metre") for c in range(len(starts)) for k in range(12)]
        pulse = float(np.median(durs) / 12)
        row = {"take": take, "piece": piece, "recall": det["recall"], "precision": det["precision"],
               "bias_ms": float(np.median([n for n in near if abs(n) < 0.02]) * 1000),
               "confirm_latency_ms": float(np.median([conf - t for t, _, conf in on]) * 1000),
               "cycles": int(len(starts) - 2)}
        floor = [starts[pos[c]] + (k - 1 + feel[k - 1]) / 12 * durs[pos[c]] - t
                 for c, k, t in zip(j2["cycle_idx"], j2["sd"], j2["onset_s"]) if pos[c] > 1]
        row["floor"] = stats(np.array(floor), pulse)
        for name in TRACKERS:
            offs = {k: 0.0 for k in J2_PULSES} if name == "isochronous" else feel
            player_q = [(t, pos[c], k - 1 + offs[k - 1], "player") for c, k, t in zip(j2["cycle_idx"], j2["sd"], j2["onset_s"])]
            errs, n_re = run_tracker(on, offs, starts, durs, metre_q + player_q, 1, args.lead, name == "frozen",
                                     **kw)
            row[name] = {tag: stats(e, pulse) for tag, e in errs.items()}
            row[name]["reacquired"] = n_re
        rows.append(row)
        f = row["feel"]
        print(f"{take:10s} recall {row['recall']:.2f} prec {row['precision']:.2f} | player: feel "
              f"{f['player']['median_abs_ms']:.0f}/{f['player']['p90_abs_ms']:.0f} ms lost {f['player']['lost_share']:.0%} "
              f"(floor {row['floor']['median_abs_ms']:.0f}/{row['floor']['p90_abs_ms']:.0f}), isochronous "
              f"{row['isochronous']['player']['median_abs_ms']:.0f} ms | metre: feel {f['metre']['median_abs_ms']:.0f} ms "
              f"lost {f['metre']['lost_share']:.0%} off-beat {f['metre']['off_beat_share']:.0%} | re {f['reacquired']}",
              flush=True)
    # medians over takes: one lost take must not hide how the others behave (its share of lost
    # predictions still counts in lost_share)
    summary = {"floor": {m: float(np.median([r["floor"][m] for r in rows])) for m in rows[0]["floor"]}}
    for name in TRACKERS:
        summary[name] = {tag: {m: float(np.median([r[name][tag][m] for r in rows])) for m in rows[0][name][tag]}
                         for tag in ("player", "metre")}
        for tag in ("player", "metre"):
            summary[name][tag]["mean_lost_share"] = float(np.mean([r[name][tag]["lost_share"] for r in rows]))
            summary[name][tag]["takes_tracked"] = int(sum(r[name][tag]["lost_share"] < 0.05 for r in rows))
    summary.update({m: float(np.mean([r[m] for r in rows])) for m in ("recall", "precision", "bias_ms",
                                                                      "confirm_latency_ms")})
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows, "lead_s": args.lead, **kw}, indent=1))
    print(json.dumps({k: summary[k] for k in ("floor", "feel", "isochronous")}, indent=1))


if __name__ == "__main__":
    main()
