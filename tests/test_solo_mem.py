"""J7: figure memory and groove encoder for the soloist."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip("torch")

from djembe import solo, solo_mem  # noqa: E402
from djembe.solo_seq import SoloGNN  # noqa: E402
from test_generate import synthetic_grid  # noqa: E402


def test_history_holds_only_cycles_before_the_window():
    g = synthetic_grid(n_cycles=50)
    w = solo_mem.mem_window(g, 40, K=4, H=32)
    assert w.mask.all() and w.age[-1] == 4 and w.age[0] == 35            # cycles 5 ... 36, the window is 37 ... 40
    fig = solo_mem._solo_figure(torch.as_tensor(w.hist)).numpy()
    assert np.allclose(fig[-1, :, 0], g.x[36, :, solo.SOLO, 0])
    assert np.allclose(fig[0, :, 1], g.x[5, :, solo.SOLO, 1])
    early = solo_mem.mem_window(g, 5, K=4, H=32)
    assert early.mask.sum() == 2 and not early.hist[~early.mask].any()   # only cycles 0 and 1 exist before


def test_without_memory_and_groove_it_is_the_j6_model():
    plain = solo_mem.SoloMemGNN(solo_mem.n_in(), use_memory=False, use_groove=False)
    assert sum(p.numel() for p in plain.parameters()) == sum(p.numel() for p in SoloGNN(solo_mem.n_in()).parameters())


def test_memory_starts_neutral_and_reads_pulse_aligned_figures():
    torch.manual_seed(0)
    g = synthetic_grid(n_cycles=50)
    w = solo_mem.mem_window(g, 40)
    m = solo_mem.SoloMemGNN(solo_mem.n_in(), use_memory=True, use_groove=False).eval()
    b = solo._collate([w.seq.reveal(3)], [w.seq.present], 4, "full", True, "cpu")
    args = (torch.as_tensor(w.hist[None]), torch.as_tensor(w.age[None]), torch.as_tensor(w.mask[None]), w.seq.x.shape[0])
    with torch.no_grad():
        base = m(b["x"], b["edges"], *args)[0]
        hist2 = args[0].clone()
        hist2[0, -1, :-1] = 0                                             # change the most recent figure
        same = m(b["x"], b["edges"], hist2, *args[1:])[0]
    assert torch.allclose(base, same)                                     # recall output starts at zero


@pytest.mark.parametrize("memory,groove", [(True, False), (False, True), (True, True)])
def test_generation_changes_only_the_soloist(memory, groove):
    g = synthetic_grid(n_cycles=12)
    w = [solo_mem.mem_window(g, t) for t in range(3, 12)]
    m = solo_mem.train(w, epochs=1, dim=16, layers=2, use_memory=memory, use_groove=groove)
    out = solo_mem.generate(g, m, rng=np.random.default_rng(0))
    assert np.array_equal(out.x[:, :, 1:], g.x[:, :, 1:]) and np.array_equal(out.x[:3], g.x[:3])
    assert solo_mem.score(m, w)["n_hit"] == 9 * 12


def test_groove_measures_on_a_repeating_line():
    g = synthetic_grid(n_cycles=20)
    g.x[:, :, solo.SOLO] = g.x[0, :, solo.SOLO][None]                    # one figure repeated
    m = solo.groove_measures(g)
    assert m["self_return"] == 1.0
