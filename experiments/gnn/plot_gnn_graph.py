"""Figure of the graph the network runs on, drawn from ``djembe.grid.graph_edges``.

    python experiments/gnn/plot_gnn_graph.py      # -> docs/figures/gnn_graph.png and .svg

Left: one cycle with the default edges (pulse 12 wraps to pulse 1 of the same cycle), which a
rotation of the cycle maps onto itself. Right: the window of 4 cycles with the helix edge (V5),
the last cycle being the target. No corpus data is used: the hit pattern is made up.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, ConnectionPatch, FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge

from djembe.dataset import INSTRUMENTS, N_SUBDIVISIONS as P
from djembe.grid import graph_edges, node_index

OUT = ROOT / "docs" / "figures" / "gnn_graph"
JEMBE, DUNDUN, TARGET = "#2a78d6", "#d9622b", "#1a9e6e"
INK, MUTED, EDGE, FAINT, PREV, SHADE = "#1f1f1e", "#6b6a63", "#8d8c85", "#cdccc5", "#9a9992", "#f1f0eb"
COLOR = [JEMBE, JEMBE, DUNDUN, DUNDUN]
DASH = (0, (3, 2))
NI = len(INSTRUMENTS)
K = 4                                           # cycles in a window, as in the J3 runs

# Illustrative hit pattern (pulses 1-12), not taken from the corpus.
HITS = {"J1": (1, 3, 4, 6, 7, 9, 10, 12), "J2": (1, 2, 4, 5, 7, 8, 10, 11),
        "D1": (1, 4, 6, 7, 10), "D2": (1, 3, 7, 9)}
HIT = np.array([[k + 1 in HITS[name] for name in INSTRUMENTS] for k in range(P)])   # (12, 4)

plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 10, "text.color": INK,
                     "svg.fonttype": "path"})


def decode(n):
    """Node number -> (cycle, pulse, instrument), matching ``node_index``."""
    return n // (P * NI), (n // NI) % P, n % NI


assert all(decode(n) == tuple(int(v) for v in np.argwhere(node_index(K) == n)[0]) for n in range(K * P * NI))


def node(ax, x, y, i, hit, size=7.0, alpha=1.0, z=5):
    ax.plot(x, y, "o", ms=size, mfc=COLOR[i] if hit else "white", mec=COLOR[i], mew=1.3, alpha=alpha, zorder=z)


def arrow(ax, a, b, color, lw=0.9, rad=0.0, head=6, shrink=4.5, alpha=1.0, z=2, style="-|>", ls="-"):
    if style != "-":
        style = f"{style},head_length={head * 0.09:.2f},head_width={head * 0.05:.2f}"
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, connectionstyle=f"arc3,rad={rad}", color=color, lw=lw,
                                 shrinkA=shrink, shrinkB=shrink, alpha=alpha, zorder=z, mutation_scale=10,
                                 linestyle=ls))


# ---------------------------------------------------------------- left: one cycle
RADII = (1.0, 0.83, 0.66, 0.49)                # J1 outer ... D2 inner


def angle(k):
    """Pulse 1 at the top, time running clockwise."""
    return np.pi / 2 - 2 * np.pi * k / P


def one_cycle(ax):
    e = graph_edges(1)                          # default edges: pulse 12 wraps to pulse 1
    pos = {}
    for n in range(P * NI):
        _, k, i = decode(n)
        pos[n] = (RADII[i] * np.cos(angle(k)), RADII[i] * np.sin(angle(k)))

    for b in range(4):                          # beats of 3 pulses, shaded alternately
        start = np.degrees(angle(3 * b)) + 15
        if b % 2 == 0:
            ax.add_patch(Wedge((0, 0), 1.09, start - 90, start, width=0.72, fc=SHADE, ec="none", zorder=0))
        mid = np.radians(start - 45)
        ax.text(1.47 * np.cos(mid), 1.47 * np.sin(mid), f"beat {b + 1}", ha="center", va="center",
                fontsize=8.5, color=MUTED)

    for s, d in e["same_pulse"].T:              # complete graph between the 4 drums at a pulse
        i, j = decode(s)[2], decode(d)[2]
        if i < j:                               # both directions exist; draw each pair once
            arrow(ax, pos[s], pos[d], FAINT, lw=0.7, rad={1: 0.0, 2: 0.45, 3: 0.32}[j - i], style="-",
                  shrink=3.5, z=1)
    for s, d in e["next_pulse"].T:
        arrow(ax, pos[s], pos[d], EDGE, lw=1.0, rad=-0.14, head=5, shrink=4.2, z=2)
    for n, (x, y) in pos.items():
        _, k, i = decode(n)
        node(ax, x, y, i, HIT[k, i])

    for k in range(P):
        a = angle(k)
        ax.text(1.17 * np.cos(a), 1.17 * np.sin(a), str(k + 1), ha="center", va="center", fontsize=9.5)
    for i, name in enumerate(INSTRUMENTS):      # ring labels in the gap between pulse 12 and pulse 1
        a = angle(-0.5)
        ax.text(RADII[i] * np.cos(a), RADII[i] * np.sin(a), name, ha="center", va="center", fontsize=8,
                color=COLOR[i], fontweight="bold", zorder=4.5,
                bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none"))

    # Rotation, in the middle of the rings.
    ax.add_patch(Arc((0, 0), 0.62, 0.62, theta1=-60, theta2=200, color=TARGET, lw=1.6, zorder=3))
    tip = np.radians(-60)
    ax.annotate("", xy=(0.31 * np.cos(tip - 0.05), 0.31 * np.sin(tip - 0.05)),
                xytext=(0.31 * np.cos(tip + 0.25), 0.31 * np.sin(tip + 0.25)),
                arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.28", color=TARGET, lw=1.6))
    ax.text(0, 0.0, "rotate\nby r", ha="center", va="center", fontsize=8.5, color=TARGET, linespacing=1.05)

    ax.text(0, -1.5, "Rotating by r pulses maps every edge set\nonto itself. With no positional input, message\n"
            "passing is equivariant to C12.", ha="center", va="top", fontsize=9.5, linespacing=1.35)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-2.0, 1.6)


# ---------------------------------------------------------------- right: window of 4 cycles
ROW = 2.6                                       # vertical gap between cycles, in pulse widths


def window(ax):
    """The window unrolled: one row per cycle, read like lines of text, for one drum (J1)."""
    e = graph_edges(K, instruments=[0], helix=True)   # V5: pulse 12 feeds pulse 1 of the next cycle

    def xy(n):
        c, k, _ = decode(n)
        return k, -ROW * c

    for b in (0, 2):                            # same beat shading as the cycle on the left
        ax.add_patch(Rectangle((3 * b - 0.5, -ROW * (K - 1) - 0.7), 3, ROW * (K - 1) + 1.4, fc=SHADE, ec="none",
                               zorder=0))
    ax.add_patch(FancyBboxPatch((-0.5, -ROW * (K - 1) - 0.55), P, 1.1, boxstyle="round,pad=0,rounding_size=0.4",
                                fc="none", ec=TARGET, lw=1.0, zorder=1))
    for k in range(P):
        ax.text(k, 0.95, str(k + 1), ha="center", va="bottom", fontsize=9, color=MUTED)

    for s, d in e["prev_cycle"].T:
        arrow(ax, xy(s), xy(d), PREV, lw=0.9, style="-", shrink=5.5, z=1, ls=DASH)
    for s, d in e["next_pulse"].T:
        if decode(s)[0] != decode(d)[0]:        # the helix edge
            arrow(ax, xy(s), xy(d), TARGET, lw=1.5, head=6, shrink=5.5, z=2)
        else:
            arrow(ax, xy(s), xy(d), EDGE, lw=1.0, head=4.5, shrink=4.5, z=2)
    for n in range(K * P):
        c, k, i = decode(n * NI)
        node(ax, k, -ROW * c, i, HIT[k, i], z=5)

    for c in range(K):
        last = c == K - 1
        ax.text(-1.1, -ROW * c + 0.25, f"cycle {c + 1}", ha="right", va="center", fontsize=9.5,
                color=TARGET if last else INK, fontweight="bold" if last else "normal")
        ax.text(-1.1, -ROW * c - 0.35, "hits known,\npredict offset\nand intensity" if last else "all known",
                ha="right", va="top", fontsize=8.5, color=TARGET if last else MUTED, linespacing=1.2)

    (x0, y0), (x1, y1) = [(xy(s), xy(d)) for s, d in e["next_pulse"].T if decode(s)[0] != decode(d)[0]][0]
    t = 0.12
    ax.annotate("pulse 12 feeds\npulse 1 of the\nnext cycle (V5)", xy=(x0 + t * (x1 - x0), y0 + t * (y1 - y0)),
                xytext=(P + 0.2, -ROW * 0.55), ha="left", va="center", fontsize=9, color=TARGET,
                linespacing=1.2, arrowprops=dict(arrowstyle="-", color=TARGET, lw=0.8, shrinkA=3, shrinkB=0))
    ax.set_xlim(-5.2, P + 5.0)
    ax.set_ylim(-ROW * (K - 1) - 0.8, 1.7)
    return P - 1, -ROW * (K - 1)                 # a target stroke, for the output callout


def gaussians(fig, rect):
    """Per target stroke the head outputs two Gaussians (mean, sigma)."""
    ax = fig.add_axes(rect)
    x = np.linspace(-3, 3, 200)
    for off, lab in ((0.0, "offset"), (7.0, "intensity")):
        y = np.exp(-0.5 * x ** 2)
        ax.fill_between(x + off, y, color=TARGET, alpha=0.13, lw=0)
        ax.plot(x + off, y, color=TARGET, lw=1.4)
        ax.plot([off - 3.2, off + 3.2], [0, 0], color=FAINT, lw=0.8)
        ax.plot([off, off], [0, 1], color=TARGET, lw=0.9, ls=(0, (1.5, 1.5)))
        ax.plot([off, off + 1], [0.607, 0.607], color=TARGET, lw=1.0)
        ax.text(off + 0.5, 0.56, "σ", ha="center", va="top", fontsize=9, color=TARGET)
        ax.text(off, 1.05, "μ", ha="center", va="bottom", fontsize=9, color=TARGET)
        ax.text(off, -0.1, lab, ha="center", va="top", fontsize=9)
    ax.set_xlim(-3.3, 10.3)
    ax.set_ylim(-0.55, 1.35)
    ax.axis("off")
    return ax


def legend(fig):
    h = [Line2D([], [], color=EDGE, lw=1.2, marker=">", ms=4, label="next pulse"),
         Line2D([], [], color=FAINT, lw=2.0, label="same pulse"),
         Line2D([], [], color=PREV, lw=1.2, ls=DASH, label="previous cycle"),
         Line2D([], [], marker="o", ls="", ms=6.5, mfc=JEMBE, mec=JEMBE, label="jembe hit"),
         Line2D([], [], marker="o", ls="", ms=6.5, mfc=DUNDUN, mec=DUNDUN, label="dundun hit"),
         Line2D([], [], marker="o", ls="", ms=6.5, mfc="white", mec=MUTED, label="no hit (pattern illustrative)")]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, fontsize=9, handlelength=1.6,
               columnspacing=1.4, handletextpad=0.5, bbox_to_anchor=(0.5, 0.0))


def main():
    fig = plt.figure(figsize=(8, 4), dpi=200, facecolor="white")
    left = fig.add_axes([0.0, 0.07, 0.45, 0.93])
    right = fig.add_axes([0.44, 0.30, 0.56, 0.62])
    for ax in (left, right):
        ax.set_aspect("equal")
        ax.axis("off")
    one_cycle(left)
    tx, ty = window(right)
    g = gaussians(fig, [0.715, 0.085, 0.25, 0.165])
    fig.add_artist(ConnectionPatch((tx, ty), (3.5, 1.0), "data", "data", axesA=right, axesB=g,
                                   color=TARGET, lw=0.8, shrinkA=6, zorder=1))
    fig.text(0.832, 0.30, "per stroke: two\nGaussians (μ, σ)", ha="right", va="center", fontsize=8.5,
             color=TARGET, linespacing=1.2)
    fig.text(0.015, 0.975, "One cycle", ha="left", va="top", fontsize=10.5, fontweight="bold")
    fig.text(0.47, 0.975, "Window of 4 cycles", ha="left", va="top", fontsize=10.5, fontweight="bold")
    fig.text(0.47, 0.93, "J1 shown; the other three drums repeat it", ha="left", va="top", fontsize=8.5,
             color=MUTED)
    legend(fig)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
    print("saved", OUT.with_suffix(".png"), OUT.with_suffix(".svg"))


if __name__ == "__main__":
    main()
