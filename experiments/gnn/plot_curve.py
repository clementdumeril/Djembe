"""Figure of the data-efficiency curve (J4): offset NLL against the number of training takes.

Left: each model's score (stroke-weighted mean over the 15 held-out takes), mean over
repetitions with the min-max range as a band. Right: the same minus B2 fitted on the same
takes, so a symmetry that pays off with little data shows up as a line that dips below 0
on the left of the plot.

    python experiments/gnn/plot_curve.py results/j4_curve_savio.json docs/figures/j4_curve.png
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"B1_position": "B1 position", "B2_markov_cycle": "B2 Markov",
          "GNN_V0": "GNN V0 (no symmetry)", "GNN_V1": "GNN V1 (beat symmetry C4)",
          "GNN_V3": "GNN V3 (pulse rotation C12)", "GNN_V5": "GNN V5 (V0 + helix edge)"}
STYLES = {"B1_position": ("0.6", "s--"), "B2_markov_cycle": ("0.3", "s--"), "GNN_V0": ("C3", "o-"),
          "GNN_V1": ("C0", "o-"), "GNN_V3": ("C2", "o-"), "GNN_V5": ("C4", "o-")}


def score(rows, model, n, rep):
    sel = [r for r in rows if r["model"] == model and r["n_train"] == n and r["rep"] == rep]
    return float(np.average([r["nll_delta"] for r in sel], weights=[r["n_delta"] for r in sel]))


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    rows = json.load(open(src))["rows"]
    sizes = sorted({r["n_train"] for r in rows})
    reps = sorted({r["rep"] for r in rows})
    models = [m for m in LABELS if any(r["model"] == m for r in rows)]
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in models:
        s = np.array([[score(rows, m, n, r) for r in reps] for n in sizes])
        b2 = np.array([[score(rows, "B2_markov_cycle", n, r) for r in reps] for n in sizes])
        c, fmt = STYLES[m]
        for a, v in ((ax, s), (axd, s - b2)):
            if a is axd and m == "B2_markov_cycle":
                continue
            a.plot(sizes, v.mean(1), fmt, color=c, ms=5, label=LABELS[m])
            if len(reps) > 1:
                a.fill_between(sizes, v.min(1), v.max(1), color=c, alpha=0.12, lw=0)
    axd.axhline(0, color="0.3", lw=1, ls="--")
    for a, title in ((ax, "Offsets: nats per stroke on the held-out take"),
                     (axd, "Difference to B2 fitted on the same takes")):
        a.set_xscale("log", base=2)
        a.set_xticks(sizes, [str(n) for n in sizes])
        a.set_xlabel("training takes")
        a.set_title(title + ", lower is better", fontsize=10)
        a.grid(axis="y", color="0.9")
    ax.legend(fontsize=8)
    fig.suptitle(f"Data efficiency: mean over {len(reps)} repetitions (subset and seed), band = min to max", fontsize=11)
    fig.tight_layout()
    fig.savefig(dst, dpi=130)
    print("saved", dst)


if __name__ == "__main__":
    main()
