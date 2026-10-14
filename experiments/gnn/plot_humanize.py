"""Figure of the J3 results: negative log-likelihood per held-out take, network against references.

    python experiments/gnn/plot_humanize.py results/j3_humanize_seed0_cpu.json docs/figures/j3_humanize.png
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from djembe import dataset as d

src, dst = sys.argv[1], sys.argv[2]
rows = json.load(open(src))["rows"]
models = ["B1_position", "B2_markov_cycle", "GNN_V0", "GNN_V1", "GNN_V3"]
labels = {"B1_position": "B1 position", "B2_markov_cycle": "B2 Markov (earlier cycles)",
          "GNN_V0": "GNN V0 (no symmetry)", "GNN_V1": "GNN V1 (beat symmetry C4)", "GNN_V3": "GNN V3 (pulse rotation C12)"}
styles = {"B1_position": ("0.6", "s--"), "B2_markov_cycle": ("0.3", "s--"), "GNN_V0": ("C3", "o-"),
          "GNN_V1": ("C0", "o-"), "GNN_V3": ("C2", "o-")}
takes = list(d.TAKES)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, key, title in zip(axes, ("nll_delta", "nll_intensity"), ("Offsets", "Intensity")):
    for m in models:
        v = [np.mean([r[key] for r in rows if r["take"] == t and r["model"] == m]) for t in takes]
        c, s = styles[m]
        ax.plot(range(len(takes)), v, s, color=c, ms=4, label=labels[m])
    ax.set_xticks(range(len(takes)))
    ax.set_xticklabels([t[:3] + t[-2:] for t in takes], rotation=90, fontsize=8)
    ax.set_title(f"{title}: nats per stroke on the held-out take, lower is better", fontsize=10)
axes[0].legend(fontsize=8)
fig.suptitle("Humanising a cycle (strikes known, offsets and intensities predicted); jembe 1, jembe 2, dundun 1", fontsize=11)
fig.tight_layout()
fig.savefig(dst, dpi=130)
print("saved", dst)
