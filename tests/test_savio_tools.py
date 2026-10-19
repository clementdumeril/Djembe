"""Job-array helpers: every (take, seed) runs once, and merging parts is exact."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "gnn"))
import array_task  # noqa: E402
import merge_results  # noqa: E402
from djembe.dataset import TAKES  # noqa: E402
from djembe.results import summarise  # noqa: E402


@pytest.mark.parametrize("seeds", [[0], [0, 1, 2], [3, 7]])
def test_array_covers_every_take_and_seed_once(seeds):
    units = [array_task.task(i, seeds) for i in range(len(TAKES) * len(seeds))]
    assert sorted(units) == sorted((t, s) for t in TAKES for s in seeds)
    with pytest.raises(IndexError):
        array_task.task(len(units), seeds)


def row(take, model, seed, delta, n=100):
    return {"take": take, "piece": take.split("_")[0], "model": model, "seed": seed,
            "nll_delta": delta, "n_delta": n, "nll_intensity": 1.0, "n_intensity": n}


def test_merge_keeps_identical_reference_rows_once():
    a = {"rows": [row("Maraka_1", "B1_position", None, -0.5), row("Maraka_1", "GNN_V0", 0, -0.7)]}
    b = {"rows": [row("Maraka_1", "B1_position", None, -0.5), row("Maraka_1", "GNN_V0", 1, -0.8)]}
    rows = merge_results.merge([a, b])
    assert len(rows) == 3
    assert summarise(rows)["GNN_V0"]["runs"] == 2


def test_merge_rejects_conflicting_rows():
    a = {"rows": [row("Maraka_1", "GNN_V0", 0, -0.7)]}
    b = {"rows": [row("Maraka_1", "GNN_V0", 0, -0.6)]}
    with pytest.raises(ValueError, match="conflicting"):
        merge_results.merge([a, b])


def test_summary_weights_by_stroke_count():
    rows = [row("Maraka_1", "GNN_V0", 0, -1.0, n=300), row("Woloso_1", "GNN_V0", 0, 0.0, n=100)]
    assert summarise(rows)["GNN_V0"]["nll_delta"] == pytest.approx(-0.75)
