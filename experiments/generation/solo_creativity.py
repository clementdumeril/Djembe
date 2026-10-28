"""Does the generated soloist copy, vary or invent? (J5b analysis)

For every soloist cycle after the 3 context cycles (strike pattern of jembe 1, 12 pulses),
compared with the soloist lines of the 14 other takes of the same piece:

- ``exact_copy``: the figure exists as such in another take;
- ``ham1`` / ``ham2plus``: it is 1 pulse, or 2 or more pulses, away from the nearest known figure;
- ``in_run4``: it belongs to a run of at least 4 consecutive cycles identical to 4 consecutive
  cycles of one other take; ``run4_varied``: share of those runs that contain at least two
  different figures (a copied phrase, not one figure repeated);
- ``self_repeat``: the figure already appeared earlier in the same take (returning to one's
  own motifs).

- ``sources``: for the cycles that copy a known figure, which soloist (jembe 1 player) of the
  other takes plays it, each figure's weight split equally among the players who play it.

The real held-out soloist gives the reference level; the Markov soloist the contrast.
Results are aggregates only (no patterns).

    python experiments/generation/solo_creativity.py output/j5b_grids --out results/j5b_solo_creativity.json --figure docs/figures/j5b_solo.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from djembe import dataset as d
from djembe import grid as gr

N_CONTEXT = 3
SOURCES = {"real soloist": None, "GNN": "GNN_solo", "Markov": "Markov_B2"}


def patterns(x: np.ndarray) -> np.ndarray:
    return (x[:, :, 0, 0] > 0).astype(np.uint8)


def sources(p: np.ndarray, others: dict, player: dict) -> dict:
    share = {}
    for r in p:
        owners = {player[t] for t, q in others.items() if (q == r).all(1).any()}
        for w in owners:
            share[w] = share.get(w, 0.0) + 1 / len(owners)
    return share


def analyse(p: np.ndarray, others: list[np.ndarray]) -> dict:
    bank = np.concatenate(others)
    ham = (p[:, None, :] != bank[None, :, :]).sum(2).min(1)
    best, runs, varied = np.zeros(len(p), int), 0, 0
    for q in others:
        eq = (p[:, None, :] == q[None, :, :]).all(2)
        run = np.zeros_like(eq, int)
        for i in range(len(p)):
            for j in range(len(q)):
                if eq[i, j]:
                    run[i, j] = 1 + (run[i - 1, j - 1] if i and j else 0)
        best = np.maximum(best, run.max(1))
        for i, j in zip(*np.nonzero(run >= 4)):
            runs += 1
            varied += len({tuple(r) for r in p[i - run[i, j] + 1:i + 1]}) >= 2
    self_rep = [bool(i) and bool((p[i] == p[:i]).all(1).any()) for i in range(len(p))]
    return {"exact_copy": float((ham == 0).mean()), "ham1": float((ham == 1).mean()),
            "ham2plus": float((ham >= 2).mean()), "in_run4": float((best >= 4).mean()),
            "run4_varied": varied / runs if runs else float("nan"),
            "self_repeat": float(np.mean(self_rep)),
            "distinct": len({tuple(r) for r in p}) / len(p)}


def figure(summary: pd.DataFrame, path: Path) -> None:
    order = ["real soloist", "GNN", "Markov"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    parts = [("exact_copy", "known figure", "#184f95"), ("ham1", "1 pulse away", "#5598e7"),
             ("ham2plus", "2+ pulses away", "#b7d3f6")]
    left = np.zeros(len(order))
    for col, label, color in parts:
        v = summary.loc[order, col].to_numpy() * 100
        a.barh(order, v, left=left, color=color, edgecolor="white", linewidth=2, label=label)
        for y, (l, w) in enumerate(zip(left, v)):
            if w > 7:
                a.text(l + w / 2, y, f"{w:.0f}%", ha="center", va="center", fontsize=9,
                       color="white" if color != "#b7d3f6" else "#1f1f1e")
        left += v
    a.invert_yaxis()
    a.set_xlim(0, 100)
    a.set_xlabel("% of soloist cycles")
    a.set_title("Each cycle against the other takes' soloists", loc="left", fontsize=10)
    a.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3, frameon=False)
    x = np.arange(2)
    colors = {"real soloist": "#3d3d3a", "GNN": "#2a78d6", "Markov": "#eb6834"}
    for k, src in enumerate(order):
        v = summary.loc[src, ["self_repeat", "distinct"]].to_numpy() * 100
        bars = b.bar(x + (k - 1) * 0.26, v, 0.24, color=colors[src], label=src)
        for bar, val in zip(bars, v):
            b.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.0f}", ha="center", fontsize=8.5)
    b.set_xticks(x, ["returns to its own\nearlier figures", "distinct figures\nper cycle"])
    b.set_ylabel("%")
    b.set_ylim(0, 100)
    b.set_title("Phrasing within the take", loc="left", fontsize=10)
    b.legend(fontsize=8, frameon=False)
    for ax in (a, b):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Generated soloists over the real ensemble: copying, varying, inventing (15 takes, 3 seeds)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grids", help="directory of generated grids (*_s<seed>_GNN_solo.npz, *_Markov_B2.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--figure", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    events = d.load_events()
    grids = {t: gr.take_grid(events, t) for t in d.TAKES}
    player = {t: events[(events.take_id == t) & (events.instrument == "J1")].player.iloc[0] for t in d.TAKES}
    shares = {src: {} for src in SOURCES}
    rows = []
    for t in d.TAKES:
        other_d = {o: patterns(grids[o].x) for o in d.TAKES if o != t and grids[o].piece == grids[t].piece}
        others = list(other_d.values())
        mine = patterns(grids[t].x)[N_CONTEXT:]
        rows.append({"take": t, "source": "real soloist", **analyse(mine, others)})
        for w, v in sources(mine, other_d, player).items():
            shares["real soloist"][w] = shares["real soloist"].get(w, 0.0) + v
        for seed in args.seeds:
            for src, suffix in SOURCES.items():
                if suffix:
                    x = np.load(Path(args.grids) / f"{t}_s{seed}_{suffix}.npz")["x"]
                    rows.append({"take": t, "source": src, "seed": seed, **analyse(patterns(x)[N_CONTEXT:], others)})
                    for w, v in sources(patterns(x)[N_CONTEXT:], other_d, player).items():
                        shares[src][w] = shares[src].get(w, 0.0) + v
    df = pd.DataFrame(rows)
    per_take = df.groupby(["source", "take"]).mean(numeric_only=True).drop(columns="seed", errors="ignore")
    summary = per_take.groupby("source").mean()
    sources_share = {src: {w: round(v / sum(s.values()), 4) for w, v in sorted(s.items(), key=lambda kv: -kv[1])}
                     for src, s in shares.items()}
    takes_per_player = pd.Series(player).value_counts().to_dict()
    Path(args.out).write_text(json.dumps({"summary": summary.round(4).to_dict(orient="index"),
                                          "copied_figures_by_soloist": sources_share,
                                          "takes_per_soloist": takes_per_player,
                                          "per_take": per_take.reset_index().round(4).to_dict("records")}, indent=2))
    print(summary.round(3).to_string())
    print("copied figures by soloist:", sources_share, "| takes per soloist:", takes_per_player)
    if args.figure:
        figure(summary, Path(args.figure))
        print("->", args.figure)
    print("->", args.out)


if __name__ == "__main__":
    main()
