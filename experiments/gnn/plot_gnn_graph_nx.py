"""The graph the network runs on, laid out by networkx from ``djembe.grid.graph_edges``.

    python experiments/gnn/plot_gnn_graph_nx.py            # -> docs/figures/gnn_graph_nx.png and .svg
    python experiments/gnn/plot_gnn_graph_nx.py --html     # also docs/figures/gnn_graph_interactive.html (needs pyvis)

V5 setting: a window of 4 cycles, the 4 drums, helix edges. Nothing is placed by hand: the node
positions come from ``networkx.spring_layout(dim=3, seed=SEED)`` on the edge list. The 3D layout is
only turned rigidly (so that the cycle order runs upwards and the helix seam faces the reader) and
projected orthographically, with nodes and edges at the back drawn fainter. No corpus data is used.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

from djembe.dataset import INSTRUMENTS, N_SUBDIVISIONS as P
from djembe.grid import EDGE_TYPES, graph_edges, node_index

OUT = ROOT / "docs" / "figures"
K = 4                                           # cycles in a window, as in the J3 runs
NI = len(INSTRUMENTS)
SEED = 0
ELEV, TURN = 28, 35                             # camera elevation; azimuth offset from the seam [deg]

COLOR = ["#1f5fbf", "#74a9e6", "#c4501b", "#f0a050"]   # J1, J2 (jembes), D1, D2 (dunduns)
INK, MUTED, HELIX = "#1f1f1e", "#6b6a63", "#1a9e6e"
STYLE = {                                       # edge type -> line style
    "next_pulse": dict(color="#6f6e68", lw=1.1, ls="-"),
    "same_pulse": dict(color="#c9c8c1", lw=0.6, ls="-"),
    "prev_cycle": dict(color="#8d8c85", lw=0.8, ls=(0, (3, 2))),
    "helix": dict(color=HELIX, lw=2.2, ls="-"),
}

plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 10, "text.color": INK,
                     "svg.fonttype": "path"})


def decode(n):
    """Node number -> (cycle, pulse, instrument), matching ``node_index``."""
    return n // (P * NI), (n // NI) % P, n % NI


assert all(decode(n) == tuple(int(v) for v in np.argwhere(node_index(K) == n)[0]) for n in range(K * P * NI))


def build_graph() -> nx.DiGraph:
    """Directed graph with exactly the edges of ``graph_edges`` (V5: helix=True)."""
    e = graph_edges(K, helix=True)
    g = nx.DiGraph()
    for n in range(K * P * NI):
        c, k, i = decode(n)
        g.add_node(n, cycle=c + 1, pulse=k + 1, instrument=INSTRUMENTS[i])
    for t in EDGE_TYPES:
        for s, d in e[t].T:
            s, d = int(s), int(d)
            kind = "helix" if t == "next_pulse" and decode(s)[0] != decode(d)[0] else t
            g.add_edge(s, d, type=t, kind=kind)
    return g


def layout(g: nx.DiGraph) -> np.ndarray:
    """3D force-directed layout, then a rigid rotation: cycle order along +z, seam facing the camera."""
    pos = nx.spring_layout(g.to_undirected(), dim=3, seed=SEED, iterations=1000)
    X = np.array([pos[n] for n in range(g.number_of_nodes())])
    X -= X.mean(0)
    cyc = np.array([decode(n)[0] for n in range(len(X))], float)
    beta = np.linalg.lstsq(X, cyc - cyc.mean(), rcond=None)[0]
    z = beta / np.linalg.norm(beta)
    a = np.linalg.svd(X - np.outer(X @ z, z))[2][0]
    a = a - (a @ z) * z
    a /= np.linalg.norm(a)
    Y = X @ np.stack([a, np.cross(z, a), z]).T
    return Y / np.abs(Y[:, :2]).max()


def project(Y, seam_angle):
    """Orthographic camera at elevation ELEV, azimuth seam angle + TURN. Returns screen xy and depth."""
    az, el = seam_angle + np.radians(TURN), np.radians(ELEV)
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    toward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    up = np.cross(toward, right)
    return np.c_[Y @ right, Y @ up], Y @ toward


def seam(g, Y):
    """Mean angle (around the cycle axis) of the helix edges."""
    ends = [n for u, v, d in g.edges(data=True) if d["kind"] == "helix" for n in (u, v)]
    m = Y[ends, :2].mean(0)
    return np.arctan2(m[1], m[0])


def draw(g, Y):
    xy, depth = project(Y, seam(g, Y))
    fade = (depth - depth.min()) / np.ptp(depth)            # 0 = back, 1 = front
    alpha = lambda f: 0.18 + 0.82 * f
    fig, ax = plt.subplots(figsize=(8, 6.2), dpi=200, facecolor="white")
    fig.subplots_adjust(0.12, 0.09, 0.9, 0.86)
    ax.set_aspect("equal")
    ax.axis("off")

    drawn = set()
    for u, v, d in g.edges(data=True):
        if d["type"] == "same_pulse":                       # both directions exist; draw the pair once
            if (v, u) in drawn:
                continue
            drawn.add((u, v))
        st = STYLE[d["kind"]]
        f = (fade[u] + fade[v]) / 2
        ax.plot(*xy[[u, v]].T, color=st["color"], lw=st["lw"], ls=st["ls"], alpha=alpha(f),
                zorder=1 + f + (1 if d["kind"] == "helix" else 0), solid_capstyle="round")

    for n in np.argsort(depth):
        c, k, i = decode(n)
        last = c == K - 1
        ax.scatter(*xy[n], s=62 if last else 26, color=COLOR[i], alpha=alpha(fade[n]),
                   edgecolors=INK if last else "white", linewidths=1.3 if last else 0.4, zorder=3 + fade[n])

    # A few labels only.
    idx = node_index(K)
    span = np.ptp(xy, 0)

    def label(n, text, dx, dy, color=INK, ha="left"):
        x, y = xy[n]
        ax.annotate(text, (x, y), (x + dx * span[0], y + dy * span[1]), ha=ha, va="center", fontsize=10.5,
                    color=color, zorder=10, arrowprops=dict(arrowstyle="-", color=color, lw=0.7, shrinkA=2,
                                                            shrinkB=4),
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    label(idx[0, 0, 0], "J1, cycle 1, pulse 1", -0.30, 0.0, ha="right")
    label(idx[K - 1, P - 1, 0], f"J1, cycle {K}, pulse 12", -0.06, 0.13, ha="right")
    helix_edge = [(u, v) for u, v, d in g.edges(data=True) if d["kind"] == "helix" and decode(u)[2] == 0][0]
    mid = xy[list(helix_edge)].mean(0)
    ax.annotate("helix: pulse 12 feeds\npulse 1 of the next cycle", mid, mid + np.array([-0.20, -0.10]) * span,
                ha="right", va="center", fontsize=10.5, color=HELIX, zorder=10,
                arrowprops=dict(arrowstyle="-", color=HELIX, lw=0.8, shrinkA=2, shrinkB=2))
    for c in range(K):                          # cycle names next to the rightmost node of each ring
        ring = idx[c].ravel()
        n = ring[np.argmax(xy[ring, 0])]
        last = c == K - 1
        ax.text(xy[ring, 0].max() + 0.04 * span[0], xy[n, 1], f"cycle {c + 1}" + ("\npredicted" if last else ""),
                ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold" if last else "normal",
                linespacing=1.1)

    fig.text(0.02, 0.975, "The V5 graph as networkx lays it out", ha="left", va="top", fontsize=13,
             fontweight="bold")
    fig.text(0.02, 0.935, f"{g.number_of_nodes()} nodes (4 cycles × 12 pulses × 4 drums), "
             f"{g.number_of_edges()} directed edges.\nPositions from networkx spring_layout in 3D (seed {SEED}); "
             f"cycle {K} (outlined) is predicted.", ha="left", va="top", fontsize=10, color=MUTED,
             linespacing=1.3)
    counts = {t: sum(d["kind"] == t for *_, d in g.edges(data=True)) for t in STYLE}
    h = [Line2D([], [], marker="o", ls="", ms=6.5, mfc=COLOR[i], mec="white", label=name)
         for i, name in enumerate(INSTRUMENTS)]
    h += [Line2D([], [], color=STYLE["next_pulse"]["color"], lw=1.4, label=f"next pulse ({counts['next_pulse']})"),
          Line2D([], [], color=STYLE["helix"]["color"], lw=2.2, label=f"helix ({counts['helix']})"),
          Line2D([], [], color=STYLE["same_pulse"]["color"], lw=1.6,
                 label=f"same pulse ({counts['same_pulse'] // 2} pairs)"),
          Line2D([], [], color=STYLE["prev_cycle"]["color"], lw=1.2, ls=(0, (3, 2)),
                 label=f"previous cycle ({counts['prev_cycle']})")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=10, handlelength=1.8,
               columnspacing=1.4, handletextpad=0.5, bbox_to_anchor=(0.5, 0.0))
    return fig


def html(g, Y, path):
    """Interactive version (pyvis). Positions are the same 3D layout, projected."""
    from pyvis.network import Network

    xy, depth = project(Y, seam(g, Y))
    fade = (depth - depth.min()) / np.ptp(depth)
    net = Network(height="680px", width="100%", directed=True, bgcolor="#ffffff", font_color=INK,
                  cdn_resources="remote")
    for n, d in g.nodes(data=True):
        last = d["cycle"] == K
        net.add_node(n, label=" ", title=f"{d['instrument']}, cycle {d['cycle']}, pulse {d['pulse']}"
                     + (" (predicted)" if last else ""),
                     color={"background": COLOR[decode(n)[2]], "border": INK if last else "#ffffff"},
                     borderWidth=2 if last else 1, size=9 if last else 6, opacity=0.3 + 0.7 * float(fade[n]),
                     x=float(xy[n, 0] * 420), y=float(-xy[n, 1] * 420), physics=False)
    for u, v, d in g.edges(data=True):
        st = STYLE[d["kind"]]
        f = float(fade[u] + fade[v]) / 2
        net.add_edge(u, v, color={"color": st["color"], "opacity": 0.25 + 0.75 * f}, width=st["lw"] * 1.4,
                     dashes=d["kind"] == "prev_cycle", title=d["kind"].replace("_", " "),
                     arrows={"to": {"enabled": d["kind"] != "same_pulse", "scaleFactor": 0.35}})
    net.toggle_physics(False)
    net.set_options('{"interaction": {"hover": true, "tooltipDelay": 80}, "edges": {"smooth": false}}')
    sw = lambda c, line=False: (f'<span style="display:inline-block;width:{18 if line else 10}px;'
                                f'height:{3 if line else 10}px;border-radius:{0 if line else 5}px;background:{c};'
                                f'margin:0 5px 2px 12px;vertical-align:middle"></span>')
    head = ('<div style="font:14px Arial,sans-serif;color:#1f1f1e;padding:10px 12px 0">'
            f'<b>Djembe GNN, V5 graph</b>: {g.number_of_nodes()} nodes (4 cycles x 12 pulses x 4 drums), '
            f'{g.number_of_edges()} directed edges, positions from networkx spring_layout (3D, seed {SEED}). '
            f'Hover a node for its drum, cycle and pulse; cycle {K} (outlined) is predicted.<br>'
            + "".join(sw(COLOR[i]) + name for i, name in enumerate(INSTRUMENTS))
            + sw(STYLE["next_pulse"]["color"], True) + "next pulse" + sw(HELIX, True) + "helix"
            + sw(STYLE["same_pulse"]["color"], True) + "same pulse"
            + sw(STYLE["prev_cycle"]["color"], True) + "previous cycle (dashed)</div>")
    page = net.generate_html(notebook=False).replace("<body>", "<body>\n" + head, 1)
    page = page.replace("<head>", "<head>\n<title>Djembe GNN graph</title>", 1).replace("<h1></h1>", "")
    # Physics is off, so the loading bar would never close; fit the whole graph once the page is up.
    page = page.replace("</head>", "<style>#loadingBar{display:none !important}</style>\n</head>", 1)
    page = page.replace("</body>", "<script>window.addEventListener('load', function () "
                        "{ if (window.network) network.fit(); });</script>\n</body>", 1)
    path.write_text(page, encoding="utf-8")
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", action="store_true", help="also write the pyvis page")
    args = ap.parse_args()
    g = build_graph()
    Y = layout(g)
    fig = draw(g, Y)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "gnn_graph_nx.png", dpi=200, facecolor="white")
    fig.savefig(OUT / "gnn_graph_nx.svg", facecolor="white")
    print("saved", OUT / "gnn_graph_nx.png", OUT / "gnn_graph_nx.svg")
    if args.html:
        html(g, Y, OUT / "gnn_graph_interactive.html")


if __name__ == "__main__":
    main()
