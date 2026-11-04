"""Figure of J6: soloist prediction on seen and unseen rhythms, graph model against transformers.

One panel per quantity (strikes, offsets, intensities); for each protocol, the mean NLL
over the 15 held-out takes (seed means) as bars and every take as a dot.

    python experiments/j6/plot.py --out docs/figures/j6_gnn_transformer.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = [("GNN_seq", "GNN (56 k)", "#2a6fbb"), ("Transformer_matched", "Transformer (60 k)", "#e07b39"),
          ("Transformer", "Transformer (181 k)", "#c94c4c"), ("B2_markov", "Markov (B2)", "#8a8a8a"),
          ("B1_position", "Per position (B1)", "#c4c4c4")]
PROTOCOLS = [("take", "rhythm seen in training"), ("piece", "rhythm never heard")]
PANELS = [("nll_hit", "Strikes"), ("nll_delta", "Offsets"), ("nll_intensity", "Intensities")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures" / "j6_gnn_transformer.png"))
    args = ap.parse_args()
    data = {}
    for p, _ in PROTOCOLS:
        rows = pd.DataFrame(json.loads((ROOT / "results" / f"j6_{p}.json").read_text())["rows"])
        pred = rows[rows["kind"] == "prediction"]
        data[p] = pred.groupby(["take", "model"])[[m for m, _ in PANELS]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    width = 0.8 / len(MODELS)
    rng = np.random.default_rng(0)
    for ax, (metric, title) in zip(axes, PANELS):
        for j, (model, label, color) in enumerate(MODELS):
            for i, (p, _) in enumerate(PROTOCOLS):
                v = data[p].xs(model, level=1)[metric]
                x = i + (j - (len(MODELS) - 1) / 2) * width
                ax.bar(x, v.mean(), width * 0.92, color=color, label=label if i == 0 else None)
                ax.scatter(x + rng.uniform(-width / 4, width / 4, len(v)), v, s=6, color="black", alpha=0.5, zorder=3,
                           linewidths=0)
        ax.set_xticks(range(len(PROTOCOLS)), [lab for _, lab in PROTOCOLS])
        ax.set_title(title)
        ax.axhline(0, color="black", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("NLL per stroke slot / stroke (nats, lower is better)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9, loc="lower center", ncol=len(MODELS))
    fig.suptitle("Soloist over the real ensemble, held-out takes (15 takes x 3 seeds)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(args.out, dpi=150)
    print("->", args.out)


if __name__ == "__main__":
    main()
