"""Windowing guards: a window never spans two takes and always holds k cycles."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from djembe import features as f  # noqa: E402


def synthetic_events(n_cycles=10, takes=("Maraka_1", "Woloso_1")):
    rows = []
    for take in takes:
        for c in range(n_cycles):
            for sd in (1, 4, 7, 10):
                for inst in ("D1", "J1"):
                    rows.append({
                        "take_id": take, "piece": take.split("_")[0], "cycle_label": str(c + 1), "cycle_idx": c,
                        "sd": sd, "instrument": inst, "dev_sd": 0.1 * (sd % 3), "cycle_start_s": 2.0 * c,
                        # last cycle has no duration, as in the real table
                        "cycle_dur_s": np.nan if c == n_cycles - 1 else 2.0,
                    })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_windows_hold_k_cycles_of_one_take(k):
    win = f.assign_windows(synthetic_events(), k)
    per_window = win.groupby("window").agg(takes=("take_id", "nunique"), cycles=("cycle_label", "nunique"))
    assert (per_window["takes"] == 1).all()
    assert (per_window["cycles"] == k).all()
    # 9 usable cycles per take (the last one has no duration)
    assert win["window"].nunique() == 2 * (9 // k)


def test_window_features_shape_and_labels():
    frame = f.window_features(synthetic_events(), 2)
    assert len(frame) == 2 * 4
    assert set(frame["piece"]) == {"Maraka", "Woloso"}
    assert (frame["d1_pattern_sd1"] == 1.0).all()
    assert (frame["d1_pattern_sd2"] == 0.0).all()
    for fs in f.FEATURE_SETS:
        assert f.columns(frame, fs), fs
