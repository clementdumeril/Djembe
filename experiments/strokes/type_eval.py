"""Leave-one-take-out scores of the stroke-type model (``djembe.stroke_types``).

Per held-out take: NLL and accuracy of the soloist's stroke types under
- ``piece``: the piece's type frequencies;
- ``piece_pulse``: frequencies per piece and pulse;
- ``logistic_pulse_intensity``: the model without context (piece, pulse, intensity);
- ``logistic_full``: the model with the previous strokes, offset and gap.

Needs ``output/strokes_J1.csv`` (``python -m djembe.strokes --instrument J1``); writes
scores only.

    python experiments/strokes/type_eval.py      # -> results/strokes_type_J1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from djembe import dataset as d
from djembe import grid as gr
from djembe import stroke_types as stt

ROOT = Path(__file__).resolve().parents[2]


def count_scores(train_labels, test_labels, pieces_train, piece, by_pulse: bool, alpha: float = 1.0) -> dict:
    counts = np.full((12 if by_pulse else 1, 3), alpha)
    for lab, p in zip(train_labels, pieces_train):
        if p != piece:
            continue
        for k in range(12):
            for y in range(3):
                counts[k if by_pulse else 0, y] += (lab[:, k] == y).sum()
    P = counts / counts.sum(1, keepdims=True)
    c, k = np.nonzero(test_labels >= 0)
    p = P[k if by_pulse else 0, test_labels[c, k]]
    best = P.argmax(1)[k if by_pulse else 0]
    return {"nll_type": float(-np.log(p).mean()), "acc_type": float((best == test_labels[c, k]).mean()),
            "n_type": int(len(c))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strokes", default=str(ROOT / "output" / "strokes_J1.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "strokes_type_J1.json"))
    args = ap.parse_args()
    events = d.load_events()
    table = pd.read_csv(args.strokes)
    grids = {t: gr.take_grid(events, t) for t in d.TAKES}
    labels = {t: stt.labels_for(g, table, 0) for t, g in grids.items()}
    rows = []
    for fold in d.load_splits()["folds"]:
        test = fold["test"][0]
        if (labels[test] >= 0).sum() == 0:
            continue
        tr = [t for t in fold["train"] if (labels[t] >= 0).any()]
        common = {"take": test, "piece": grids[test].piece}
        for name, by_pulse in (("piece", False), ("piece_pulse", True)):
            rows.append({**common, "model": name, **count_scores([labels[t] for t in tr], labels[test],
                                                                   [grids[t].piece for t in tr], grids[test].piece, by_pulse)})
        for name, ctx in (("logistic_pulse_intensity", False), ("logistic_full", True)):
            m = stt.StrokeTypeModel(context=ctx).fit([grids[t] for t in tr], [labels[t] for t in tr])
            rows.append({**common, "model": name, **m.score(grids[test], labels[test])})
    df = pd.DataFrame(rows)
    table_ = df.pivot(index="take", columns="model", values="nll_type")
    summary = {m: {"nll_type": float(df[df.model == m].nll_type.mean()), "acc_type": float(df[df.model == m].acc_type.mean())}
               for m in df.model.unique()}
    summary["full_beats_pulse_intensity"] = int((table_["logistic_full"] < table_["logistic_pulse_intensity"]).sum())
    summary["n_takes"] = int(len(table_))
    summary["uniform_nll"] = float(np.log(3))
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print("->", args.out)


if __name__ == "__main__":
    main()
