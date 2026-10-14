"""The graph model is exactly as symmetric as its positional input, and it can learn."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from djembe import gnn  # noqa: E402
from djembe.grid import EDGE_TYPES, TakeGrid, graph_edges, position_features, rotate_nodes  # noqa: E402

K = 3


def run(model, x_base, position):
    pos = position_features(position, K)
    e = graph_edges(K)
    edges = [torch.as_tensor(e[name]) for name in EDGE_TYPES]
    x = torch.as_tensor(np.concatenate([x_base, pos], axis=1))
    with torch.no_grad():
        return torch.stack(model(x, edges), dim=1).numpy()


@pytest.mark.parametrize("position, symmetric_shifts", [("none", range(12)), ("subpulse", (3, 6, 9)), ("full", ())])
def test_output_rotates_with_the_input_exactly_for_the_kept_symmetries(position, symmetric_shifts):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n_in = gnn.n_base_features() + position_features(position, 1).shape[1]
    model = gnn.CycleGNN(n_in, dim=16, layers=2).eval()
    x = rng.normal(size=(K * 12 * 4, gnn.n_base_features())).astype(np.float32)
    out = run(model, x, position)
    for r in range(1, 12):
        perm = rotate_nodes(K, r)
        x_rot = np.empty_like(x)
        x_rot[perm] = x                       # node n moves to perm[n]
        out_rot = run(model, x_rot, position)
        same = np.allclose(out_rot[perm], out, atol=1e-5)
        assert same == (r in symmetric_shifts), (position, r)


def synthetic_grid(rng, n_cycles=40):
    x = np.zeros((n_cycles, 12, 4, 3), np.float32)
    x[:, ::3, 0, 0] = 1.0
    x[:, ::3, 0, 2] = -0.2 + 0.02 * rng.normal(size=(n_cycles, 4))
    x[:, ::3, 0, 1] = rng.normal(size=(n_cycles, 4))
    return TakeGrid("Woloso_9", "Woloso", x, np.zeros(n_cycles, np.float32), np.zeros(n_cycles, int),
                    np.zeros(n_cycles, bool), np.array([True, False, False, False]))


def test_windows_mask_only_the_target_cycle():
    w = gnn.make_windows(synthetic_grid(np.random.default_rng(1)), K=K)
    assert len(w) == 40 - K + 1
    md = w[0].mask_delta.reshape(K, 12, 4)
    assert md[-1].sum() == 4 and md[:-1].sum() == 0
    known = w[0].x.reshape(K, 12, 4, -1)[..., 3]
    assert known[:-1].all() and not known[-1].any()


def test_model_learns_a_constant_feel():
    rng = np.random.default_rng(2)
    windows = gnn.make_windows(synthetic_grid(rng), K=K)
    model, hist = gnn.train(windows, windows[:8], K=K, position="subpulse", dim=16, layers=1, epochs=30,
                            batch_size=8, patience=30)
    res = gnn.evaluate(model, windows, K=K, position="subpulse")
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert res["nll_delta"] < -1.0          # offsets have sd 0.02 around -0.2: a good fit scores far below 0


def flat_prior(mu_d=-0.2, sd_d=0.05, mu_i=0.0, sd_i=1.0):
    p = np.zeros((12, 4, gnn.N_PRIOR), np.float32)
    p[..., 0], p[..., 1], p[..., 2], p[..., 3] = mu_d, np.log(sd_d), mu_i, np.log(sd_i)
    return {piece: p for piece in ("Manjanin", "Maraka", "Woloso")}


def test_untrained_residual_model_predicts_the_prior_exactly():
    torch.manual_seed(0)
    windows = gnn.make_windows(synthetic_grid(np.random.default_rng(3)), K=K)
    batch = gnn.collate(windows[:4], K, "full", helix=True, prior=flat_prior())
    n_in = gnn.n_base_features() + position_features("full", 1).shape[1] + gnn.N_PRIOR
    model = gnn.CycleGNN(n_in, dim=16, layers=2, residual=True).eval()
    with torch.no_grad():
        dm, dls, im, ils = model(batch["x"], batch["edges"], batch["prior"])
    assert torch.allclose(dm, torch.full_like(dm, -0.2)) and torch.allclose(dls.exp(), torch.full_like(dls, 0.05))
    assert torch.allclose(im, torch.zeros_like(im)) and torch.allclose(ils, torch.zeros_like(ils))


def test_residual_model_trains_every_epoch_when_keeping_the_last():
    rng = np.random.default_rng(4)
    windows = gnn.make_windows(synthetic_grid(rng), K=K)
    prior = flat_prior(sd_d=0.2)
    model, hist = gnn.train(windows, windows[:8], K=K, position="full", dim=16, layers=1, epochs=12,
                            batch_size=8, patience=1, select="last", helix=True, prior=prior)
    assert len(hist) == 12
    res = gnn.evaluate(model, windows, K=K, position="full", helix=True, prior=prior)
    assert res["nll_delta"] < -1.0          # starts at the wide prior (about -0.7) and sharpens
