"""Leakage guards for the take-level splits.

The first group runs anywhere (no corpus needed). The last test builds the
event table and is skipped when the annotations are not on disk.
"""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import djembe_dataset as d  # noqa: E402


@pytest.fixture(scope="module")
def splits():
    return d.load_splits()


def test_splits_file_matches_generator(splits):
    assert splits == d.make_splits(), "splits.json is stale: run `python djembe_dataset.py splits`"


def test_every_take_tested_exactly_once(splits):
    tested = [t for fold in splits["folds"] for t in fold["test"]]
    assert sorted(tested) == sorted(d.TAKES)
    assert len(tested) == len(set(tested))


@pytest.mark.parametrize("fold", d.make_splits()["folds"], ids=lambda f: f["test"][0])
def test_train_and_test_are_disjoint_and_complete(fold):
    train, test = set(fold["train"]), set(fold["test"])
    assert not train & test
    assert train | test == set(d.TAKES)


def test_assert_no_leakage_catches_shared_take():
    train = pd.DataFrame({"take_id": ["Maraka_1", "Maraka_2"]})
    test = pd.DataFrame({"take_id": ["Maraka_2"]})
    with pytest.raises(AssertionError, match="Maraka_2"):
        d.assert_no_leakage(train, test)


def test_iter_folds_never_shares_a_take():
    frame = pd.DataFrame({"take_id": [t for t in d.TAKES for _ in range(3)], "x": 0.0})
    n = 0
    for _, train, test in d.iter_folds(frame):
        assert set(train["take_id"]).isdisjoint(test["take_id"])
        assert len(train) + len(test) == len(frame)
        n += 1
    assert n == len(d.TAKES)


@pytest.mark.skipif(not (d.DEFAULT_CORPUS / "MJ_Woloso" / "annotations").exists(), reason="corpus not downloaded")
def test_event_table_labels_match_files():
    events = d.build_events()
    assert sorted(events["take_id"].unique()) == sorted(d.TAKES)
    # Every row of a take carries that take's piece (guards against the
    # mislabelled MJ_Maraka_5_Onsets_Selected.csv on OSF).
    assert (events["take_id"].str.split("_").str[0] == events["piece"]).all()
    assert events["sd"].between(1, d.N_SUBDIVISIONS).all()
