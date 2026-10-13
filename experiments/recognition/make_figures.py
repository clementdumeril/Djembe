"""README figures, computed from the event table (aggregates only, no audio).

    python -m djembe.dataset build
    python make_figures.py          # -> docs/figures/*.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from djembe import dataset as d

FIG_DIR = Path(__file__).resolve().parents[2] / "docs" / "figures"
COLORS = {"Manjanin": "#2a78d6", "Maraka": "#eb6834", "Woloso": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
MIN_PROB = 0.10  # subdivisions D1 plays in at least 10 % of cycles

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def hero(events) -> Path:
    """Per rhythm: where D1 plays in the 12-pulse cycle, and how far from the even grid."""
    d1 = events[(events["instrument"] == "D1") & events["dev_ms"].notna()]
    profile = d.subdivision_profile(events, "D1")
    fig, axes = plt.subplots(2, 3, figsize=(11, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    sds = np.arange(1, 13)
    for col, piece in enumerate(d.PIECES):
        color = COLORS[piece]
        prob = profile[profile["piece"] == piece][[f"sd{s}" for s in sds]].mean().to_numpy()
        top, bottom = axes[0, col], axes[1, col]
        top.bar(sds, prob, width=0.6, color=color)
        top.set_ylim(0, 1)
        top.set_title(f"{piece}  ({d1[d1.piece == piece].take_id.nunique()} takes)", color=INK, fontsize=11, loc="left")
        played = sds[prob >= MIN_PROB]
        stats = d1[(d1.piece == piece) & d1.sd.isin(played)].groupby("sd")["dev_ms"]
        mean, q1, q3 = stats.mean(), stats.quantile(0.25), stats.quantile(0.75)
        bottom.axhline(0, color=MUTED, lw=1)
        bottom.vlines(mean.index, q1, q3, color=color, lw=2)
        bottom.plot(mean.index, mean, "o", color=color, ms=7, mec="white", mew=1.5)
        bottom.set_ylim(-75, 75)
        bottom.set_xticks(sds)
        bottom.set_xlabel("subdivision of the cycle")
        for ax in (top, bottom):
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
        if col == 0:
            top.set_ylabel("P(D1 plays)")
            bottom.set_ylabel("offset from even grid (ms)\nmean and IQR")
    fig.suptitle("Dundun 1 in three Malian jembe pieces: pattern (top) and micro-timing (bottom)",
                 x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.005, f"IEMP Malian Jembe corpus, 15 takes, {len(d1):,} D1 strokes. "
             "Negative = played early relative to an isochronous 12-pulse grid.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = FIG_DIR / "hero_d1_pattern_timing.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def step0_profiles(events) -> Path:
    """The 15 per-take D1 profiles the step 0 classifier sees."""
    profile = d.subdivision_profile(events, "D1")
    profile = profile.set_index("take_id").loc[list(d.TAKES)]
    matrix = profile[[f"sd{s}" for s in range(1, 13)]].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(12), range(1, 13))
    ax.set_yticks(range(len(profile)), profile.index)
    for label in ax.get_yticklabels():
        label.set_color(COLORS[label.get_text().split("_")[0]])
    for edge in np.cumsum([4, 6])[:2] - 0.5:
        ax.axhline(edge, color="white", lw=3)
    ax.set_xlabel("subdivision of the cycle")
    ax.set_title("Per-take D1 profile (fraction of cycles with a stroke)", loc="left", fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    path = FIG_DIR / "step0_d1_profiles.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    events = d.load_events()
    for path in (hero(events), step0_profiles(events)):
        print("->", path.relative_to(FIG_DIR.parents[1]))


if __name__ == "__main__":
    main()
