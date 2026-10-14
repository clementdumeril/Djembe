"""Grid and graph: shapes, round trip with the event table, and the cycle symmetry."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from djembe import dataset as d  # noqa: E402
from djembe import grid as g  # noqa: E402


def edge_set(edges):
    return set(map(tuple, edges.T.tolist()))


@pytest.mark.parametrize("r", range(12))
def test_rotating_the_cycle_maps_every_relation_onto_itself(r):
    n_cycles = 3
    edges = g.graph_edges(n_cycles)
    perm = g.rotate_nodes(n_cycles, r)
    for name, e in edges.items():
        assert edge_set(perm[e]) == edge_set(e), name


def test_rotation_is_a_permutation_of_the_nodes():
    perm = g.rotate_nodes(2, 5)
    assert sorted(perm.tolist()) == list(range(2 * 12 * 4))
    # rotating by 5 then by 7 is the identity
    assert np.array_equal(g.rotate_nodes(2, 7)[perm], np.arange(perm.size))


@pytest.mark.parametrize("mode, invariant_shifts", [("none", range(12)), ("subpulse", (0, 3, 6, 9)), ("full", (0,))])
def test_position_features_keep_exactly_the_intended_rotations(mode, invariant_shifts):
    pos = g.position_features(mode, 2)
    for r in range(12):
        same = np.array_equal(pos[g.rotate_nodes(2, r)], pos)
        assert same == (r in invariant_shifts), (mode, r)


def test_edge_counts():
    e = g.graph_edges(2, instruments=(0, 2))       # two instruments only
    assert e["next_pulse"].shape[1] == 2 * 12 * 2
    assert e["same_pulse"].shape[1] == 2 * 12 * 2   # one partner each
    assert e["prev_cycle"].shape[1] == 1 * 12 * 2


def test_helix_follows_time_across_cycles():
    n_cycles = 3
    idx = g.node_index(n_cycles)
    ring = edge_set(g.graph_edges(n_cycles)["next_pulse"])
    helix = edge_set(g.graph_edges(n_cycles, helix=True)["next_pulse"])
    for i in range(4):
        assert (idx[1, 11, i], idx[1, 0, i]) in ring and (idx[1, 11, i], idx[1, 0, i]) not in helix
        assert (idx[0, 11, i], idx[1, 0, i]) in helix and (idx[1, 11, i], idx[2, 0, i]) in helix
        assert (idx[1, 4, i], idx[1, 5, i]) in helix
    # pulse 1 of the first cycle has no predecessor, pulse 12 of the last no successor
    assert len(helix) == len(ring) - 4
    # the other relations do not change
    for name in ("same_pulse", "prev_cycle"):
        assert edge_set(g.graph_edges(n_cycles)[name]) == edge_set(g.graph_edges(n_cycles, helix=True)[name])


def test_section_classes():
    assert g.section_class("heating up / finale") == "heating"
    assert g.section_class("basic pattern 1  w variations") == "basic"
    assert g.section_class("diverse patterns, variations, impros") == "impro"
    assert g.section_class(None) == "other"


CORPUS = Path(os.environ.get("DJEMBE_CORPUS", d.DEFAULT_CORPUS))
needs_corpus = pytest.mark.skipif(not (CORPUS / "MJ_Woloso" / "annotations").exists(), reason="corpus not downloaded")


@pytest.fixture(scope="module")
def events():
    return d.build_events(CORPUS)


@needs_corpus
def test_grid_holds_every_stroke_with_an_end_time(events):
    for take in d.TAKES:
        tg = g.take_grid(events, take)
        n = ((events["take_id"] == take) & events["cycle_dur_s"].notna()).sum()
        assert tg.x[..., 0].sum() == n
        assert tg.x.shape[1:] == (12, 4, 3)
        assert not tg.x[..., 0][:, :, ~tg.present].any()   # absent instruments never hit
        assert np.isfinite(tg.log_T).all()


@needs_corpus
def test_presence_matches_the_corpus(events):
    assert g.take_grid(events, "Maraka_5").present.tolist() == [True, False, True, False]
    assert g.take_grid(events, "Woloso_2").present.all()
