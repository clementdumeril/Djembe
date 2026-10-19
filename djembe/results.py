"""Result files of the generative experiments: summaries shared by training and merging."""
from __future__ import annotations

import numpy as np


def summarise(rows: list[dict]) -> dict:
    """Per model: NLL averaged over all target strokes (weighted by stroke count), and run count."""
    summary = {}
    for name in sorted({r["model"] for r in rows}):
        sel = [r for r in rows if r["model"] == name]
        summary[name] = {
            "nll_delta": float(np.average([r["nll_delta"] for r in sel], weights=[r["n_delta"] for r in sel])),
            "nll_intensity": float(np.average([r["nll_intensity"] for r in sel],
                                              weights=[r["n_intensity"] for r in sel])),
            "runs": len(sel)}
    return summary
