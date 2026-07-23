"""Génère output/boussole_rythmes.html.

La Boussole des rythmes présente trois équations complexes, une par rythme.
Chaque équation combine les deux voix de dunun disponibles sur les prises
appariées D1/D2 :

    R_piece(s) = [D1_piece(s), D2_piece(s)]

Les coefficients peuvent être lus en mode original ou modifiés en mode synthé.
L'objectif est de comparer les signatures complexes puis de sculpter une boucle
rythmique à partir des mêmes paramètres.

Pourquoi 6 coefficients ? Pour 12 subdivisions réelles, c7..c11 sont les
miroirs conjugués de c5..c1. Les ondes indépendantes non triviales sont donc
c1..c6, plus c0 qui porte la densité moyenne.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import utils_V2 as u


COLORS = {
    "Manjanin": "#355C9A",
    "Maraka": "#B46A2A",
    "Woloso": "#2E7D62",
    "D1": "#214E8A",
    "D2": "#B46A2A",
}


def _paired_takes(piece: str) -> list[int]:
    paired = u.paired_dunun_takes()
    return paired.loc[paired["piece"] == piece, "take"].astype(int).tolist()


def _profile_from_takes(piece: str, code: str, takes: list[int]):
    vals = []
    used = []
    for take in sorted(set(takes)):
        profile = u.subdivision_profile(piece, int(take), code)
        if profile is not None:
            vals.append(profile.values.astype(float))
            used.append(int(take))

    if not vals:
        return np.zeros(12, dtype=float), []
    return np.mean(vals, axis=0), used


def _coefficients(motif: np.ndarray):
    """Coefficients Ck = Σ x_s exp(-i k θ_s), k=0..6, x normalisé."""
    density = float(motif.sum())
    x = motif / density if density > 0 else motif.copy()
    coeffs = []
    for k in range(7):
        z = sum(x[s] * np.exp(-2j * np.pi * k * s / 12) for s in range(12))
        coeffs.append({
            "k": k,
            "re": round(float(z.real), 8),
            "im": round(float(z.imag), 8),
            "mag": round(float(abs(z)), 8),
            "phase": round(float(np.angle(z)), 8),
        })
    return coeffs, density


def _pca_map(rhythms):
    vectors = []
    for rhythm in rhythms:
        row = []
        for voice in ("D1", "D2"):
            coeffs = rhythm["voices"][voice]["coeffs"]
            row.extend(v for k in range(1, 7) for v in (coeffs[k]["re"], coeffs[k]["im"]))
        vectors.append(row)

    X = np.asarray(vectors, dtype=float)
    if len(X) == 0:
        return []

    X = X - X.mean(axis=0, keepdims=True)
    if len(X) == 1 or np.allclose(X, 0):
        xy = np.zeros((len(X), 2))
    else:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        xy = X @ vt[:2].T
        if xy.shape[1] == 1:
            xy = np.c_[xy[:, 0], np.zeros(len(xy))]

    scale = float(np.max(np.abs(xy))) or 1.0
    xy = xy / scale
    return [
        {"piece": rhythm["piece"], "x": round(float(x), 4), "y": round(float(y), 4)}
        for rhythm, (x, y) in zip(rhythms, xy)
    ]


def _microtiming_profiles(rhythms, csv_path="output/dunun_frappes_V2.csv"):
    """Résume l'écart à la grille et sa dispersion sur les prises D1/D2 appariées."""
    if not os.path.exists(csv_path):
        return {}

    cols = ["piece", "take", "code", "SD", "rel", "dur"]
    try:
        strokes = pd.read_csv(csv_path, usecols=cols)
    except (OSError, ValueError):
        return {}

    out = {}
    for rhythm in rhythms:
        piece = rhythm["piece"]
        takes = set(rhythm["takes"])
        piece_data = {}
        for voice in ("D1", "D2"):
            s = strokes[
                (strokes["piece"] == piece)
                & (strokes["code"] == voice)
                & (strokes["take"].isin(takes))
            ].copy()
            if s.empty:
                continue

            s["dev_ms"] = (
                s["rel"] - (s["SD"].astype(float) - 1.0) / 12.0
            ) * s["dur"] * 1000.0
            mean_rel = s.groupby("SD")["rel"].transform("mean")
            precision = ((s["rel"] - mean_rel) * s["dur"] * 1000.0).std()
            grouped = s.groupby("SD")["dev_ms"]

            def value_at(series, subdivision, digits):
                value = series.get(subdivision, np.nan)
                return None if not np.isfinite(value) else round(float(value), digits)

            means = grouped.mean()
            stds = grouped.std()
            counts = grouped.size()
            piece_data[voice] = {
                "mean": [value_at(means, sd, 2) for sd in range(1, 13)],
                "std": [value_at(stds, sd, 2) for sd in range(1, 13)],
                "n": [int(counts.get(sd, 0)) for sd in range(1, 13)],
                "precision": (
                    None if not np.isfinite(precision) else round(float(precision), 2)
                ),
            }
        if piece_data:
            out[piece] = piece_data
    return out


def build_data():
    rhythms = []

    for piece in u.PIECES:
        takes = _paired_takes(piece)
        if not takes:
            continue

        voices = {}
        for code in ("D1", "D2"):
            motif, used = _profile_from_takes(piece, code, takes)
            if not used:
                continue
            coeffs, density = _coefficients(motif)
            voices[code] = {
                "voice": code,
                "motif": [round(float(x), 4) for x in motif],
                "takes": used,
                "density": round(density, 6),
                "coeffs": coeffs,
            }

        if {"D1", "D2"}.issubset(voices):
            rhythms.append({
                "piece": piece,
                "takes": sorted(set(voices["D1"]["takes"]) & set(voices["D2"]["takes"])),
                "voices": voices,
            })

    return {
        "pieces": [r["piece"] for r in rhythms],
        "colors": COLORS,
        "rhythms": rhythms,
        "map": _pca_map(rhythms),
        "microtiming": _microtiming_profiles(rhythms),
        "coefNames": {
            "1": "1 oscillation par cycle",
            "2": "2 oscillations par cycle",
            "3": "3 oscillations par cycle",
            "4": "4 oscillations par cycle",
            "5": "5 oscillations par cycle",
            "6": "alternance maximale",
        },
    }


HTML = r"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Boussole des rythmes</title>
<script>
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true
  },
  svg: {fontCache: "global"}
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
:root{
  --bg:#f2f1ed;--paper:#fbfaf7;--ink:#20211f;--muted:#77766f;
  --line:#d8d7d1;--line2:#e9e7e1;--soft:#f4f3ef;--accent:#b45d2c;
  --d1:#245d9b;--d2:#c17a2b;--live:#26815a;--shadow:0 8px 28px rgba(34,34,28,.055)
}
@media(prefers-color-scheme:dark){
  :root{--bg:#111315;--paper:#191c1e;--ink:#f0eee8;--muted:#aaa9a2;--line:#34383a;--line2:#292d2f;--soft:#202426;--accent:#df8b50;--d1:#6ca4df;--d2:#e0a458;--live:#52bc88;--shadow:0 12px 35px rgba(0,0,0,.22)}
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;line-height:1.45}
button,input,select{font:inherit}
button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid color-mix(in srgb,var(--accent) 35%,transparent);outline-offset:2px}
.page{max-width:1440px;margin:0 auto;padding:26px 24px 46px}
.masthead{display:flex;justify-content:space-between;gap:32px;align-items:flex-end;margin:0 2px 20px}
.titleBlock{min-width:max-content}
h1{font-family:Georgia,Times,serif;font-size:clamp(2.2rem,4vw,3.65rem);line-height:.95;margin:0;letter-spacing:-.045em}
.abstract{max-width:660px;color:var(--muted);font-size:.9rem;margin:0}
.abstract b{color:var(--ink)}
.surface{background:var(--paper);border:1px solid var(--line);border-radius:20px;box-shadow:var(--shadow)}
.toolbar{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:10px 12px;margin-bottom:16px}
.tabs{display:flex;gap:6px;flex-wrap:wrap}
.tab{appearance:none;border:1px solid transparent;background:transparent;color:var(--muted);border-radius:10px;padding:8px 13px;cursor:pointer;font-weight:800}
.tab:hover{background:var(--soft);color:var(--ink)}
.tab.on{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.transport{display:flex;gap:9px;align-items:center;flex-wrap:wrap;justify-content:flex-end}
.play{min-width:168px;appearance:none;border:0;background:var(--accent);color:#fff;border-radius:11px;padding:10px 15px;cursor:pointer;font-weight:900}
.play.playing{background:var(--ink)}
.voicePicker{display:flex;gap:3px;align-items:center;border:1px solid var(--line);background:var(--soft);border-radius:11px;padding:3px}
.voiceBtn{appearance:none;border:0;background:transparent;color:var(--muted);border-radius:8px;padding:6px 9px;cursor:pointer;font-weight:800;font-size:.78rem}
.voiceBtn.on{background:var(--ink);color:var(--paper)}
.liveStatus{display:inline-flex;align-items:center;gap:7px;color:var(--muted);font-size:.78rem;min-width:82px}
.liveStatus i{width:8px;height:8px;border-radius:50%;background:var(--line)}
.liveStatus.on{color:var(--live);font-weight:850}.liveStatus.on i{background:var(--live);box-shadow:0 0 0 5px color-mix(in srgb,var(--live) 14%,transparent)}
.playCaption{width:100%;text-align:right;color:var(--muted);font-size:.75rem}
.studioGrid{display:grid;grid-template-columns:minmax(520px,1.06fr) minmax(490px,.94fr);gap:16px;align-items:stretch}
.wheelPanel,.synthPanel,.curvePanel,.microPanel,.mapPanel{padding:18px}
.panelHead{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:4px}
.panelHead h2,.synthPanel h2,.curvePanel h2,.microPanel h2,.mapPanel h2{font-family:Georgia,Times,serif;font-size:1.35rem;margin:0;letter-spacing:-.025em}
.modeTag{border:1px solid var(--line);background:var(--soft);border-radius:999px;color:var(--muted);padding:5px 9px;font-size:.72rem;font-weight:800}
.modeTag.live{border-color:color-mix(in srgb,var(--live) 45%,var(--line));color:var(--live);background:color-mix(in srgb,var(--live) 7%,var(--paper))}
canvas{display:block;width:100%;height:auto}
.wheelPanel #circle{max-width:590px;margin:0 auto}
.caption,.note{font-size:.85rem;color:var(--muted)}
.caption b,.note b{color:var(--ink)}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:6px 0 2px;color:var(--muted);font-size:.8rem}
.swatch{display:inline-flex;align-items:center;gap:7px}.swatch i{width:13px;height:13px;border-radius:99px;display:inline-block;border:2px solid currentColor}
.d1{color:var(--d1)}.d2{color:var(--d2)}
.synthPanel{display:flex;flex-direction:column}
.synthIntro{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:12px}
.synthIntro .caption{margin:4px 0 0;max-width:420px}
.synthTop{display:flex;gap:7px;align-items:center;flex-wrap:wrap;padding:10px 0 12px;border-top:1px solid var(--line2);border-bottom:1px solid var(--line2);margin-bottom:10px}
.smallBtn{appearance:none;border:1px solid var(--line);background:var(--paper);color:var(--ink);border-radius:9px;padding:7px 10px;cursor:pointer;font-weight:800;font-size:.78rem}
.smallBtn:hover{background:var(--soft)}
.smallBtn.on{background:var(--ink);border-color:var(--ink);color:var(--paper)}
.resetBtn{margin-left:auto}
.transformBar{display:grid;grid-template-columns:.82fr 1.18fr;gap:7px;margin:0 0 10px}
.transformAction{display:grid;align-content:start;gap:6px;background:var(--soft);border:1px solid var(--line2);border-radius:11px;padding:7px}
.transformActionHead{display:flex;align-items:center;gap:6px}
.transformAction .smallBtn{flex:1;background:var(--paper)}
.transformAction select{max-width:98px;border:1px solid var(--line);background:var(--paper);color:var(--ink);border-radius:8px;padding:7px 6px;font-size:.73rem}
.symmetryGuide{margin:0;color:var(--muted);font-size:.71rem;line-height:1.35;font-variant-numeric:tabular-nums}
.symmetryGuide b{color:var(--ink)}
.transformFeedback{margin:-2px 2px 8px;color:var(--muted);font-size:.74rem}
.transformFeedback:empty{display:none}
.transformFeedback b{color:var(--ink)}
.coefControls{display:grid;gap:6px}
.coefRow{display:grid;grid-template-columns:92px 28px minmax(90px,1fr) 45px 24px minmax(90px,1fr) 43px;gap:7px;align-items:center;background:var(--soft);border:1px solid transparent;border-radius:11px;padding:8px 9px;font-size:.76rem;color:var(--muted)}
.coefRow:hover{border-color:var(--line)}
.coefRow b{color:var(--ink);font-size:.82rem}.coefRow small{display:block;font-size:.63rem;color:var(--muted);font-weight:600;line-height:1.15;margin-top:1px}
.coefRow input[type=range]{width:100%;height:4px;accent-color:var(--accent);cursor:ew-resize}
.coefRow output{font-variant-numeric:tabular-nums;color:var(--ink);font-size:.72rem;text-align:right}
.coefRow.density{grid-template-columns:92px 28px minmax(170px,1fr) 45px}
.curvePanel{margin-top:16px}
.curvePanel .panelHead{align-items:flex-end;margin-bottom:6px}
.curvePanel #curve{max-height:350px}
.lowerGrid{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(360px,.75fr);gap:16px;margin-top:16px}
.microControls{display:flex;align-items:center;gap:7px;flex-wrap:wrap}
.microScale{display:grid;grid-template-columns:auto 110px 38px;align-items:center;gap:6px;color:var(--muted);font-size:.72rem}
.microScale input{width:110px;accent-color:var(--accent)}
.microScale output{color:var(--ink);font-variant-numeric:tabular-nums;text-align:right}
.microPanel #microTiming,.mapPanel #map{max-height:330px}
.equationBox{border:1px solid var(--line);background:var(--soft);border-radius:14px;padding:14px;overflow:auto}
.math{font-size:.98rem}
.metaGrid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:12px 0 0}
.metric{border:1px solid var(--line2);background:var(--soft);border-radius:11px;padding:9px}
.metric span{display:block;color:var(--muted);font-size:.74rem;text-transform:uppercase;letter-spacing:.09em}
.metric b{font-size:1rem}
.coefTable{width:100%;border-collapse:separate;border-spacing:0;font-size:.82rem;overflow:hidden;border:1px solid var(--line);border-radius:12px}
.coefTable th,.coefTable td{padding:8px 9px;border-bottom:1px solid var(--line2);text-align:right;white-space:nowrap}
.coefTable th:first-child,.coefTable td:first-child{text-align:left}
.coefTable tr:last-child td{border-bottom:0}
.coefTable th{color:var(--muted);font-weight:800;background:var(--soft)}
.theory{margin-top:16px;padding:0;overflow:hidden}
.theory summary{cursor:pointer;list-style:none;padding:16px 18px;font-family:Georgia,Times,serif;font-size:1.1rem;font-weight:700}
.theory summary::-webkit-details-marker{display:none}
.theory summary::after{content:"+";float:right;color:var(--muted);font-family:Inter,sans-serif}
.theory[open] summary{border-bottom:1px solid var(--line2)}.theory[open] summary::after{content:"−"}
.theoryBody{padding:18px}
.theoryGrid{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(430px,.8fr);gap:16px;align-items:start}
.theoryGrid h3{font-size:.72rem;text-transform:uppercase;letter-spacing:.13em;color:var(--muted);margin:0 0 8px}
.researchNote{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:12px}
.researchNote div{border-left:3px solid var(--accent);padding:4px 0 4px 12px;color:var(--muted);font-size:.86rem}
@media(max-width:1080px){
  .masthead{align-items:flex-start}.studioGrid,.lowerGrid,.theoryGrid{grid-template-columns:1fr}
  .synthPanel{min-height:0}.wheelPanel #circle{max-width:620px}.curvePanel #curve{max-height:none}
}
@media(max-width:720px){
  .page{padding:18px 12px 34px}.masthead{display:block}.titleBlock{min-width:0}.abstract{margin-top:12px}
  .toolbar{align-items:flex-start;flex-direction:column}.transport{justify-content:flex-start}.playCaption{text-align:left}
  .studioGrid{grid-template-columns:minmax(0,1fr)}.wheelPanel,.synthPanel,.curvePanel,.microPanel,.mapPanel{padding:13px}
  .transformBar{grid-template-columns:1fr}
  .coefRow,.coefRow.density{grid-template-columns:78px 1fr 46px}.coefRow span{display:none}
  .coefRow input[data-kind="phase"]{grid-column:2}.coefRow output[data-out^="phase"]{grid-column:3}
  .metaGrid{grid-template-columns:1fr 1fr}.theoryBody{padding:13px}
}
</style></head><body><main class="page">
<header class="masthead">
  <div class="titleBlock">
    <h1>Boussole des rythmes</h1>
  </div>
  <p class="abstract">
    <b>Objet.</b> Trois rythmes, trois équations complexes. Chaque signature est
    calculée uniquement sur les prises où D1 et D2 sont tous les deux présents,
    pour comparer les deux voix sur la même base. Les coefficients sont des
    projections du motif sur les racines complexes du cycle.
  </p>
</header>

<section class="toolbar surface">
  <nav class="tabs" id="tabs"></nav>
  <div class="transport">
    <button class="play" id="playBtn">▶ Lancer la boucle</button>
    <div class="voicePicker" id="voicePicker"></div>
    <span class="liveStatus" id="liveStatus"><i></i><span>Prêt</span></span>
    <div class="playCaption" id="playCaption">Les modifications seront audibles en direct.</div>
  </div>
</section>

<section class="studioGrid">
  <article class="wheelPanel surface">
    <div class="panelHead">
      <div><h2>Cercle métrique · 12 subdivisions</h2></div>
      <span class="modeTag" id="circleMode">Original</span>
    </div>
    <canvas id="circle" width="620" height="620"></canvas>
    <div class="legend">
      <span class="swatch d1"><i></i>D1 — anneau extérieur</span>
      <span class="swatch d2"><i></i>D2 — anneau intérieur</span>
    </div>
    <p class="caption" id="circleCaption"></p>
  </article>

  <aside class="synthPanel surface">
    <div class="synthIntro">
      <div>
        <h2>Synthétiseur de coefficients</h2>
        <p class="caption">\(|c_k|\) règle la force de la répétition ; \(\varphi_k\) déplace cette forme dans le cycle.</p>
      </div>
      <span class="modeTag" id="synthLiveTag">Prêt</span>
    </div>
    <div class="synthTop">
      <button class="smallBtn" id="originalModeBtn">Original</button>
      <button class="smallBtn" id="synthModeBtn">Synthé</button>
      <div class="voicePicker" id="synthVoicePicker"></div>
      <button class="smallBtn resetBtn" id="resetSynthBtn">Réinitialiser</button>
    </div>
    <div class="transformBar">
      <div class="transformAction">
        <button class="smallBtn" id="orthogonalBtn">⊥ Orthogonal à l’autre voix</button>
      </div>
      <div class="transformAction symmetryAction">
        <div class="transformActionHead">
          <button class="smallBtn" id="mirrorBtn">Refléter la voix</button>
          <select id="symmetryAxis" aria-label="Axe de symétrie">
            <option value="0">axe 1–7</option>
            <option value="1">axe 2–8</option>
            <option value="2">axe 3–9</option>
            <option value="3">axe 4–10</option>
            <option value="4">axe 5–11</option>
            <option value="5">axe 6–12</option>
          </select>
        </div>
        <p class="symmetryGuide" id="symmetryGuide"></p>
      </div>
    </div>
    <div class="transformFeedback" id="transformFeedback"></div>
    <div class="coefControls" id="synthControls"></div>
  </aside>
</section>

<section class="curvePanel surface">
  <div class="panelHead">
    <div><h2>Courbes D1 et D2</h2></div>
    <span class="modeTag" id="curveMode">Points observés + modèle</span>
  </div>
  <canvas id="curve" width="1260" height="340"></canvas>
  <p class="caption">Abscisse : subdivision Polak du cycle. Ordonnée : intensité de frappe. En mode synthé, les carrés montrent les valeurs réellement jouées.</p>
</section>

<section class="lowerGrid">
  <article class="microPanel surface">
    <div class="panelHead">
      <div><h2>Micro-timing observé</h2></div>
      <div class="microControls">
        <button class="smallBtn" id="gridTimingBtn">Grille</button>
        <button class="smallBtn" id="observedTimingBtn">Feel observé</button>
        <label class="microScale">impact
          <input type="range" id="timingStrength" min="0" max="2" step="0.05" value="1">
          <output id="timingStrengthOut">×1.00</output>
        </label>
      </div>
    </div>
    <canvas id="microTiming" width="760" height="310"></canvas>
    <p class="caption" id="microCaption">Écart à la grille en millisecondes : négatif = avance, positif = retard. Les barres verticales montrent la dispersion.</p>
  </article>
  <article class="mapPanel surface">
    <div class="panelHead"><div><h2>Espace des signatures</h2></div></div>
    <canvas id="map" width="620" height="310"></canvas>
    <p class="caption">Axes : PC1 et PC2. Deux points proches correspondent à deux signatures de coefficients proches.</p>
  </article>
</section>

<details class="theory surface">
  <summary>Équation complexe et détail des coefficients</summary>
  <div class="theoryBody">
    <div class="theoryGrid">
      <section>
        <h3>Équation du rythme sélectionné</h3>
        <div class="equationBox"><div class="math" id="mainEquation"></div></div>
        <div class="metaGrid" id="metrics"></div>
      </section>
      <section>
        <h3>Table des coefficients</h3>
        <div id="coefTable"></div>
      </section>
    </div>
    <div class="researchNote">
      <div><b>D1/D2 comparables.</b><br>Seules les prises appariées où les deux voix sont présentes sont utilisées.</div>
      <div><b>Six coefficients ≠ six temps.</b><br>Chaque coefficient regarde les 12 subdivisions et teste une répétition complète.</div>
      <div><b>Calcul complexe.</b><br>\(z_s=e^{2\pi i s/12}\) et \(c_k\) additionne les frappes après rotation par \(z_s^{-k}\).</div>
      <div><b>Une couche = une onde test.</b><br>Pour \(k=3\), l'onde fait trois tours pendant le cycle.</div>
    </div>
  </div>
</details>
</main>

<script>
const DATA=__DATA__;
const TAU=Math.PI*2;
const state={
  piece:DATA.pieces[0],playVoice:"both",synthEnabled:false,synthVoice:"D1",
  synthData:{},timingMode:"grid",timingStrength:1,transformMessage:"",symmetryAxis:0
};
const playback={active:false,step:.155,lookaheadMs:25,scheduleAhead:.11,raf:null,ctx:null,master:null,timer:null,nextStepTime:0,stepIndex:0,startTime:0};

function rhythm(piece=state.piece){return DATA.rhythms.find(r=>r.piece===piece) || DATA.rhythms[0];}
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function phaseDeg(rad){let d=rad*180/Math.PI;while(d>180)d-=360;while(d<=-180)d+=360;return d;}
function signed(x,digits=2){return (x>=0?"+":"") + x.toFixed(digits);}
function phaseTex(rad){return signed(rad,2);}
function texVector(it){
  return String.raw`\left(${it.coeffs.slice(1).map(c=>`${c.mag.toFixed(2)}e^{${phaseTex(c.phase)}i}`).join(String.raw`,\; `)}\right)`;
}
function color(voice){return voice==="D1"?css("--d1"):css("--d2");}
function selectedVoices(){
  if(state.playVoice==="D1")return ["D1"];
  if(state.playVoice==="D2")return ["D2"];
  return ["D1","D2"];
}
function playVoiceLabel(){
  return state.playVoice==="both" ? "D1+D2" : state.playVoice;
}

function synthKey(piece=state.piece,voice=state.synthVoice){return `${piece}:${voice}`;}
function cloneSynthData(it){
  const d={density:it.density, coeffs:{}};
  for(let k=1;k<=5;k++) d.coeffs[k]={mag:it.coeffs[k].mag, phase:it.coeffs[k].phase};
  d.coeffs[6]={real:it.coeffs[6].re};
  return d;
}
function ensureSynthData(piece=state.piece,voice=state.synthVoice){
  const key=synthKey(piece,voice);
  if(!state.synthData[key]) state.synthData[key]=cloneSynthData(rhythm(piece).voices[voice]);
  return state.synthData[key];
}
function synthDataFromMotif(motif){
  const clean=motif.map(v=>Math.max(0,Number(v)||0));
  const density=clean.reduce((a,b)=>a+b,0);
  const x=density>1e-10 ? clean.map(v=>v/density) : clean.map(()=>0);
  const coeffs={};
  for(let k=1;k<=5;k++){
    let re=0,im=0;
    x.forEach((v,s)=>{
      const angle=TAU*k*s/12;
      re+=v*Math.cos(angle);
      im-=v*Math.sin(angle);
    });
    coeffs[k]={mag:Math.hypot(re,im),phase:Math.atan2(im,re)};
  }
  let real6=0;
  x.forEach((v,s)=>{real6+=v*Math.cos(Math.PI*s);});
  coeffs[6]={real:real6};
  return {density,coeffs};
}
function motifForTransform(voice){
  return state.synthEnabled ? synthMotif(voice) : rhythm().voices[voice].motif.slice();
}
function centered(values){
  const mean=values.reduce((a,b)=>a+b,0)/values.length;
  return values.map(v=>v-mean);
}
function centeredCosine(a,b){
  const ac=centered(a),bc=centered(b);
  const dot=ac.reduce((sum,v,i)=>sum+v*bc[i],0);
  const na=Math.sqrt(ac.reduce((sum,v)=>sum+v*v,0));
  const nb=Math.sqrt(bc.reduce((sum,v)=>sum+v*v,0));
  return na*nb>1e-10 ? dot/(na*nb) : 0;
}
function orthogonalComplement(source,reference){
  const a=centered(source),b=centered(reference);
  const bb=b.reduce((sum,v)=>sum+v*v,0);
  const ab=a.reduce((sum,v,i)=>sum+v*b[i],0);
  let q=bb>1e-10 ? a.map((v,i)=>v-ab/bb*b[i]) : a.slice();
  let norm=Math.sqrt(q.reduce((sum,v)=>sum+v*v,0));
  if(norm<1e-8){
    const candidate=Array.from({length:12},(_,s)=>Math.sin(TAU*s/12));
    const cb=candidate.reduce((sum,v,i)=>sum+v*b[i],0);
    q=bb>1e-10 ? candidate.map((v,i)=>v-cb/bb*b[i]) : candidate;
    norm=Math.sqrt(q.reduce((sum,v)=>sum+v*v,0))||1;
  }
  const sourceDensity=source.reduce((a,b)=>a+b,0);
  const floor=-Math.min(...q)+.012*norm;
  const playable=q.map(v=>v+floor);
  const total=playable.reduce((a,b)=>a+b,0)||1;
  return playable.map(v=>v*sourceDensity/total);
}
function mirroredMotif(source,axis){
  return Array.from({length:12},(_,s)=>source[(2*axis-s+120)%12]);
}
function symmetryPairs(axis){
  const fixed=[],pairs=[],seen=new Set();
  for(let s=0;s<12;s++){
    if(seen.has(s))continue;
    const target=(2*axis-s+120)%12;
    seen.add(s);seen.add(target);
    if(s===target)fixed.push(s+1);
    else pairs.push([s+1,target+1]);
  }
  return {fixed,pairs};
}
function renderSymmetryGuide(){
  const axis=state.symmetryAxis;
  const select=document.getElementById("symmetryAxis");
  const guide=document.getElementById("symmetryGuide");
  const mirrorBtn=document.getElementById("mirrorBtn");
  const {fixed,pairs}=symmetryPairs(axis);
  select.value=String(axis);
  mirrorBtn.textContent=`Refléter ${state.synthVoice}`;
  guide.innerHTML=
    `<b>A<sub>${axis+1}–${axis+7}</sub></b> : fixes {${fixed.join(", ")}} ; `+
    `permutation ${pairs.map(([a,b])=>`${a}↔${b}`).join(", ")} ; ρ′ = ρ.`;
  select.onchange=()=>{
    state.symmetryAxis=Number(select.value);
    state.transformMessage="";
    document.getElementById("transformFeedback").textContent="";
    renderSymmetryGuide();
    drawCircle();
  };
}
function applySynthMotif(motif,message){
  state.synthData[synthKey()]=synthDataFromMotif(motif);
  state.synthEnabled=true;
  state.transformMessage=message;
  renderSynthControls();
  refreshSynthViews();
}
function applyOrthogonalProjection(){
  const voice=state.synthVoice,other=voice==="D1"?"D2":"D1";
  const projected=orthogonalComplement(motifForTransform(voice),motifForTransform(other));
  const cos=centeredCosine(projected,motifForTransform(other));
  applySynthMotif(
    projected,
    `<b>${voice}</b> projeté dans le complément orthogonal de ${other} · cosinus centré ${cos.toFixed(3)}`
  );
}
function applyMirrorSymmetry(){
  const axis=state.symmetryAxis;
  const voice=state.synthVoice;
  applySynthMotif(
    mirroredMotif(motifForTransform(voice),axis),
    `<b>${voice}</b> réfléchi autour de l’axe ${axis+1}–${axis+7} · densité conservée, aucune frappe dupliquée`
  );
}
function synthCoeffArray(data){
  const out=[{k:0,re:1,im:0,mag:1,phase:0}];
  for(let k=1;k<=5;k++){
    const c=data.coeffs[k], re=c.mag*Math.cos(c.phase), im=c.mag*Math.sin(c.phase);
    out[k]={k,re,im,mag:c.mag,phase:c.phase};
  }
  const r=data.coeffs[6].real;
  out[6]={k:6,re:r,im:0,mag:Math.abs(r),phase:r>=0?0:Math.PI};
  return out;
}
function curveValueFromCoeffs(coeffs,density,t,clamp=true){
  const theta=TAU*t/12, N=12;
  let y=1;
  for(let k=1;k<=5;k++){
    const c=coeffs[k];
    y += 2*(c.re*Math.cos(k*theta)-c.im*Math.sin(k*theta));
  }
  y += coeffs[6].re*Math.cos(6*theta);
  const value=y/N*(density??1);
  return clamp?Math.max(0,value):value;
}
function curveValue(it,t){
  const C=it.coeffs, theta=TAU*t/12, N=12;
  let y=C[0].re;
  for(let k=1;k<=5;k++){
    const c=C[k];
    y += 2*(c.re*Math.cos(k*theta)-c.im*Math.sin(k*theta));
  }
  const c6=C[6];
  y += c6.re*Math.cos(6*theta)-c6.im*Math.sin(6*theta);
  return Math.max(0,y/N*(it.density??1));
}
function synthMotif(voice,piece=state.piece){
  const data=ensureSynthData(piece,voice), coeffs=synthCoeffArray(data);
  return Array.from({length:12},(_,s)=>curveValueFromCoeffs(coeffs,data.density,s,true));
}
function voiceView(voice,piece=state.piece){
  if(!state.synthEnabled) return rhythm(piece).voices[voice];
  const data=ensureSynthData(piece,voice);
  return {voice,density:data.density,motif:synthMotif(voice,piece),takes:rhythm(piece).voices[voice].takes,coeffs:synthCoeffArray(data),synthetic:true};
}
function curveValueForVoice(voice,t){
  if(!state.synthEnabled) return curveValue(rhythm().voices[voice],t);
  const data=ensureSynthData(state.piece,voice), coeffs=synthCoeffArray(data);
  return curveValueFromCoeffs(coeffs,data.density,t,true);
}
function motifForPlayback(voice){
  return state.synthEnabled ? synthMotif(voice) : rhythm().voices[voice].motif;
}
function playheadPosition(){
  if(!playback.active||!playback.ctx)return null;
  const elapsed=playback.ctx.currentTime-playback.startTime;
  if(elapsed<0)return 0;
  return (elapsed/playback.step)%12;
}

function latexForRhythm(r){
  const d1=voiceView("D1",r.piece), d2=voiceView("D2",r.piece);
  return String.raw`
\[
\begin{aligned}
\text{points du cercle:}\quad
z_s &= e^{2\pi i s/12}\\
\text{motif normalisé:}\quad
x_{v,s} &= \frac{\operatorname{D}_{v}(s)}{\rho_v},
\qquad \rho_v=\sum_{s=0}^{11}\operatorname{D}_{v}(s)\\
\text{coefficients:}\quad
c_{v,k} &= \sum_{s=0}^{11}x_{v,s}\,z_s^{-k},
\qquad k=0,\ldots,6\\
\operatorname{Rythme}_{\mathrm{${r.piece}}}(s)
  &= \begin{bmatrix}\operatorname{D1}_{\mathrm{${r.piece}}}(s)\\\operatorname{D2}_{\mathrm{${r.piece}}}(s)\end{bmatrix},
  \qquad s\in\{0,\ldots,11\}\\
\operatorname{D}_{v,\mathrm{${r.piece}}}(s)
  &= \frac{\rho_v}{12}\left[
      1+2\,\Re\!\left(\sum_{k=1}^{5}c_{v,k}z_s^k\right)
      + c_{v,6}(-1)^s
    \right]\\
\mathbf{c}_{D1}
  &= ${texVector(d1)}\\
\mathbf{c}_{D2}
  &= ${texVector(d2)}
\end{aligned}
\]`;
}

function queueMath(){
  if(window.MathJax?.typesetPromise){
    MathJax.typesetClear?.();
    MathJax.typesetPromise().catch(()=>{});
  }else{
    setTimeout(queueMath,120);
  }
}

function syncSynthModeButtons(){
  const original=document.getElementById("originalModeBtn"), synth=document.getElementById("synthModeBtn");
  if(!original||!synth)return;
  original.className="smallBtn"+(!state.synthEnabled?" on":"");
  synth.className="smallBtn"+(state.synthEnabled?" on":"");
  const circleMode=document.getElementById("circleMode");
  const curveMode=document.getElementById("curveMode");
  const liveTag=document.getElementById("synthLiveTag");
  if(circleMode){
    circleMode.textContent=state.synthEnabled?"Synthèse":"Original";
    circleMode.className="modeTag"+(state.synthEnabled?" live":"");
  }
  if(curveMode){
    curveMode.textContent=state.synthEnabled?"Courbe et pas synthétisés":"Points observés + modèle";
    curveMode.className="modeTag"+(state.synthEnabled?" live":"");
  }
  if(liveTag){
    liveTag.textContent=state.synthEnabled
      ? `${playback.active?"LIVE":"Édition"} · ${state.synthVoice}`
      : "Signal original";
    liveTag.className="modeTag"+(state.synthEnabled?" live":"");
  }
}

let equationRefreshTimer=null;
function scheduleEquationRefresh(){
  clearTimeout(equationRefreshTimer);
  equationRefreshTimer=setTimeout(()=>{
    renderEquation();
    renderCoefTable();
    queueMath();
  },140);
}

function refreshSynthViews(){
  syncSynthModeButtons();
  drawCircle();
  drawCurve();
  scheduleEquationRefresh();
}

function renderSynthVoicePicker(){
  const box=document.getElementById("synthVoicePicker");
  box.innerHTML="";
  ["D1","D2"].forEach(voice=>{
    const b=document.createElement("button");
    b.className="voiceBtn"+(state.synthVoice===voice?" on":"");
    b.textContent=`éditer ${voice}`;
    b.onclick=()=>{
      state.synthVoice=voice;
      state.transformMessage="";
      renderSynthControls();
      syncSynthModeButtons();
    };
    box.appendChild(b);
  });
}

function renderSynthControls(){
  renderSynthVoicePicker();
  syncSynthModeButtons();
  document.getElementById("originalModeBtn").onclick=()=>{state.synthEnabled=false;refreshSynthViews();};
  document.getElementById("synthModeBtn").onclick=()=>{state.synthEnabled=true;refreshSynthViews();};
  document.getElementById("resetSynthBtn").onclick=()=>{
    delete state.synthData[synthKey(state.piece,state.synthVoice)];
    ensureSynthData(state.piece,state.synthVoice);
    state.synthEnabled=true;
    state.transformMessage=`<b>${state.synthVoice}</b> revenu à ses coefficients observés`;
    renderSynthControls();
    refreshSynthViews();
  };
  const other=state.synthVoice==="D1"?"D2":"D1";
  const orthogonalBtn=document.getElementById("orthogonalBtn");
  orthogonalBtn.textContent=`⊥ ${state.synthVoice} orthogonal à ${other}`;
  orthogonalBtn.onclick=applyOrthogonalProjection;
  document.getElementById("mirrorBtn").onclick=applyMirrorSymmetry;
  renderSymmetryGuide();
  document.getElementById("transformFeedback").innerHTML=
    state.transformMessage||"";

  const data=ensureSynthData();
  const rows=[];
  rows.push(`<div class="coefRow density"><b>ρ<small>densité</small></b><span>val</span><input type="range" min="0" max="8" step="0.05" value="${data.density}" data-kind="density"><output data-out="density">${data.density.toFixed(2)}</output></div>`);
  for(let k=1;k<=5;k++){
    const c=data.coeffs[k], deg=phaseDeg(c.phase);
    rows.push(`<div class="coefRow"><b>c${k}<small>${DATA.coefNames[String(k)]}</small></b><span>|c|</span><input type="range" min="0" max="1.2" step="0.001" value="${c.mag}" data-kind="mag" data-k="${k}"><output data-out="mag-${k}">${c.mag.toFixed(3)}</output><span>φ</span><input type="range" min="-180" max="180" step="1" value="${deg}" data-kind="phase" data-k="${k}"><output data-out="phase-${k}">${deg.toFixed(0)}°</output></div>`);
  }
  rows.push(`<div class="coefRow density"><b>c6<small>alternance max.</small></b><span>val</span><input type="range" min="-1" max="1" step="0.001" value="${data.coeffs[6].real}" data-kind="real6"><output data-out="real6">${data.coeffs[6].real.toFixed(3)}</output></div>`);
  document.getElementById("synthControls").innerHTML=rows.join("");

  document.querySelectorAll("#synthControls input").forEach(input=>{
    input.oninput=()=>{
      const d=ensureSynthData(), kind=input.dataset.kind, k=Number(input.dataset.k), value=Number(input.value);
      state.synthEnabled=true;
      state.transformMessage="";
      if(kind==="density"){
        d.density=value;
        document.querySelector('[data-out="density"]').textContent=value.toFixed(2);
      }else if(kind==="mag"){
        d.coeffs[k].mag=value;
        document.querySelector(`[data-out="mag-${k}"]`).textContent=value.toFixed(3);
      }else if(kind==="phase"){
        d.coeffs[k].phase=value*Math.PI/180;
        document.querySelector(`[data-out="phase-${k}"]`).textContent=`${value.toFixed(0)}°`;
      }else if(kind==="real6"){
        d.coeffs[6].real=value;
        document.querySelector('[data-out="real6"]').textContent=value.toFixed(3);
      }
      refreshSynthViews();
    };
  });
}

function renderVoicePicker(){
  const box=document.getElementById("voicePicker");
  box.innerHTML="";
  [
    ["both","D1+D2"],
    ["D1","D1 seul"],
    ["D2","D2 seul"],
  ].forEach(([value,label])=>{
    const b=document.createElement("button");
    b.className="voiceBtn"+(state.playVoice===value?" on":"");
    b.textContent=label;
    b.onclick=()=>{
      state.playVoice=value;
      renderVoicePicker();
      drawCircle();
      drawCurve();
      drawMicroTiming();
      syncPlaybackUI();
    };
    box.appendChild(b);
  });
  document.getElementById("playCaption").textContent =
    playback.active
      ? `Boucle continue · ${playVoiceLabel()} · ${timingModeLabel()} · les changements entrent au prochain pas.`
      : `Écoute : ${playVoiceLabel()} · ${timingModeLabel()} · boucle continue jusqu'à l'arrêt.`;
}

function renderTabs(){
  const nav=document.getElementById("tabs");
  nav.innerHTML="";
  DATA.rhythms.forEach(r=>{
    const b=document.createElement("button");
    b.className="tab"+(r.piece===state.piece?" on":"");
    b.textContent=r.piece;
    b.onclick=()=>{state.piece=r.piece;state.transformMessage="";render();};
    nav.appendChild(b);
  });
}

function drawCircle(){
  const r=rhythm(), cv=document.getElementById("circle"), g=cv.getContext("2d");
  const W=cv.width, cx=W/2, cy=W/2, outer=218, inner=146;
  const p=playheadPosition(), currentStep=p===null?null:Math.floor(p)%12;
  const audibleVoices=new Set(selectedVoices());
  g.clearRect(0,0,W,W);
  g.save();
  g.translate(cx,cy);

  g.strokeStyle=css("--line");
  g.lineWidth=1.2;
  [outer,inner].forEach(rad=>{g.beginPath();g.arc(0,0,rad,0,TAU);g.stroke();});

  const axisAngle=state.symmetryAxis/12*TAU-Math.PI/2;
  const axisX=Math.cos(axisAngle),axisY=Math.sin(axisAngle),axisRadius=outer+34;
  g.save();
  g.setLineDash([7,6]);
  g.strokeStyle=css("--accent");
  g.globalAlpha=.48;
  g.lineWidth=1.7;
  g.beginPath();
  g.moveTo(-axisRadius*axisX,-axisRadius*axisY);
  g.lineTo(axisRadius*axisX,axisRadius*axisY);
  g.stroke();
  g.restore();
  g.fillStyle=css("--accent");
  g.globalAlpha=.78;
  g.font="700 11px Inter, system-ui, sans-serif";
  g.textAlign="center";g.textBaseline="middle";
  g.fillText(`A ${state.symmetryAxis+1}–${state.symmetryAxis+7}`,0,24);

  for(let s=0;s<12;s++){
    const a=s/12*TAU-Math.PI/2;
    const ca=Math.cos(a), sa=Math.sin(a);
    const fixed=s===state.symmetryAxis||s===state.symmetryAxis+6;
    g.fillStyle=fixed?css("--accent"):css("--muted");
    g.globalAlpha=fixed ? .9 : .55;
    g.beginPath();g.arc((outer+30)*ca,(outer+30)*sa,fixed?4:2.2,0,TAU);g.fill();
    g.globalAlpha=1;
    g.fillStyle=fixed?css("--accent"):css("--muted");
    g.font=fixed?"800 12px Inter, system-ui, sans-serif":"12px Inter, system-ui, sans-serif";
    g.textAlign="center";g.textBaseline="middle";
    g.fillText(String(s+1),(outer+49)*ca,(outer+49)*sa);
  }

  [["D1",outer],["D2",inner]].forEach(([voice,rad])=>{
    const it=voiceView(voice,r.piece), values=it.motif, mx=Math.max(...values,0.001);
    const audible=audibleVoices.has(voice);
    g.strokeStyle=color(voice);
    g.globalAlpha=audible ? .72 : .2;
    g.lineWidth=2.2;
    g.beginPath();g.arc(0,0,rad,0,TAU);g.stroke();
    values.forEach((v,s)=>{
      const a=s/12*TAU-Math.PI/2, ca=Math.cos(a), sa=Math.sin(a);
      const size=5+16*Math.sqrt(v/mx);
      if(audible&&s===currentStep&&v>.045){
        g.globalAlpha=.16;
        g.fillStyle=color(voice);
        g.beginPath();g.arc(rad*ca,rad*sa,size+10,0,TAU);g.fill();
      }
      g.globalAlpha=audible ? .96 : .28;
      g.fillStyle=color(voice);
      g.beginPath();g.arc(rad*ca,rad*sa,size,0,TAU);g.fill();
      g.globalAlpha=audible ? .95 : .3;
      g.strokeStyle=css("--paper");
      g.lineWidth=1.7;
      g.stroke();
    });
    g.globalAlpha=1;
    g.fillStyle=color(voice);
    g.font="700 15px Inter, system-ui, sans-serif";
    g.textAlign="left";g.textBaseline="middle";
    g.fillText(voice,rad+18,0);
  });

  if(p!==null){
    const a=p/12*TAU-Math.PI/2, ca=Math.cos(a), sa=Math.sin(a);
    let from=inner-42, to=outer+28;
    if(state.playVoice==="D1"){from=outer-42;to=outer+30;}
    if(state.playVoice==="D2"){from=inner-42;to=inner+30;}
    g.globalAlpha=1;
    g.strokeStyle=css("--ink");
    g.lineWidth=2.8;
    g.beginPath();g.moveTo(from*ca,from*sa);g.lineTo(to*ca,to*sa);g.stroke();
    g.fillStyle=css("--ink");
    g.beginPath();g.arc(to*ca,to*sa,5.2,0,TAU);g.fill();
    g.font="700 12px Inter, system-ui, sans-serif";
    g.textAlign="center";g.textBaseline="middle";
    g.fillText(playVoiceLabel(),0,0);
  }

  g.restore();
  const d1=voiceView("D1"),d2=voiceView("D2");
  document.getElementById("circleCaption").innerHTML =
    `<b>${r.piece}</b> — prises appariées D1/D2 : ${r.takes.join(", ")}. `+
    `${state.synthEnabled?"Densités synthétisées":"Densités moyennes"} : D1 ${d1.density.toFixed(2)}, D2 ${d2.density.toFixed(2)} frappes/cycle.`;
}

function drawCurve(){
  const r=rhythm(), cv=document.getElementById("curve"), g=cv.getContext("2d");
  const W=cv.width,H=cv.height,left=50,right=20,top=18,bottom=42,span=W-left-right;
  g.clearRect(0,0,W,H);
  const samples=[];
  ["D1","D2"].forEach(voice=>{
    const it=r.voices[voice];
    for(let i=0;i<=180;i++) samples.push(curveValueForVoice(voice,i/180*12));
    samples.push(...it.motif,...motifForPlayback(voice));
  });
  const ymax=Math.max(...samples,0.05)*1.14;
  const xOf=t=>left+(t/12)*span;
  const yOf=v=>H-bottom-(v/ymax)*(H-top-bottom);

  g.strokeStyle=css("--line");g.lineWidth=1;
  g.beginPath();g.moveTo(left,top);g.lineTo(left,H-bottom);g.lineTo(W-right,H-bottom);g.stroke();
  g.fillStyle=css("--muted");g.font="11px Inter, system-ui";g.textAlign="center";g.textBaseline="top";
  for(let s=0;s<12;s++){
    const x=xOf(s);
    g.globalAlpha=.45;g.strokeStyle=css("--line2");g.beginPath();g.moveTo(x,top);g.lineTo(x,H-bottom);g.stroke();
    g.globalAlpha=1;g.fillStyle=css("--muted");g.fillText(String(s+1),x,H-bottom+10);
  }
  g.save();
  g.translate(13,H/2);g.rotate(-Math.PI/2);
  g.textAlign="center";g.textBaseline="middle";g.fillStyle=css("--muted");
  g.fillText("probabilité de frappe",0,0);
  g.restore();
  g.textAlign="center";g.textBaseline="bottom";g.fillStyle=css("--muted");
  g.fillText("subdivision du cycle (1–12)",left+span/2,H-5);

  ["D1","D2"].forEach(voice=>{
    const it=r.voices[voice], col=color(voice);
    const audible=selectedVoices().includes(voice);
    g.strokeStyle=col;g.lineWidth=2.6;g.globalAlpha=audible ? .92 : .25;
    if(state.synthEnabled){g.setLineDash([8,5]);}else{g.setLineDash([]);}
    g.beginPath();
    for(let i=0;i<=180;i++){
      const t=i/180*12, x=xOf(t), y=yOf(curveValueForVoice(voice,t));
      if(i===0)g.moveTo(x,y);else g.lineTo(x,y);
    }
    g.stroke();
    g.setLineDash([]);
    it.motif.forEach((v,s)=>{
      g.globalAlpha=audible ? .95 : .25;g.fillStyle=col;
      g.beginPath();g.arc(xOf(s),yOf(v),5.2,0,TAU);g.fill();
      g.strokeStyle=css("--paper");g.lineWidth=1.4;g.stroke();
    });
    if(state.synthEnabled){
      motifForPlayback(voice).forEach((v,s)=>{
        g.globalAlpha=audible ? .72 : .2;g.strokeStyle=col;g.lineWidth=2;
        g.beginPath();g.rect(xOf(s)-4,yOf(v)-4,8,8);g.stroke();
      });
    }
  });

  const p=playheadPosition();
  if(p!==null){
    const x=xOf(p);
    g.globalAlpha=1;
    g.strokeStyle=css("--ink");
    g.lineWidth=2;
    g.beginPath();g.moveTo(x,top);g.lineTo(x,H-bottom);g.stroke();
    g.fillStyle=css("--ink");
    g.beginPath();g.arc(x,top+7,4.5,0,TAU);g.fill();
  }
  g.globalAlpha=1;
}

function microProfile(voice,piece=state.piece){
  return DATA.microtiming?.[piece]?.[voice]||null;
}
function timingOffsetMs(voice,stepIndex){
  if(state.timingMode!=="observed")return 0;
  const value=microProfile(voice)?.mean?.[stepIndex];
  if(!Number.isFinite(value))return 0;
  return Math.max(-60,Math.min(60,value*state.timingStrength));
}
function timingModeLabel(){
  return state.timingMode==="observed"
    ? `feel observé ×${state.timingStrength.toFixed(2)}`
    : "grille régulière";
}
function renderTimingControls(){
  const grid=document.getElementById("gridTimingBtn");
  const observed=document.getElementById("observedTimingBtn");
  const slider=document.getElementById("timingStrength");
  const output=document.getElementById("timingStrengthOut");
  grid.className="smallBtn"+(state.timingMode==="grid"?" on":"");
  observed.className="smallBtn"+(state.timingMode==="observed"?" on":"");
  slider.value=state.timingStrength;
  output.textContent=`×${state.timingStrength.toFixed(2)}`;
  grid.onclick=()=>{
    state.timingMode="grid";
    renderTimingControls();
    drawMicroTiming();
    syncPlaybackUI();
  };
  observed.onclick=()=>{
    state.timingMode="observed";
    renderTimingControls();
    drawMicroTiming();
    syncPlaybackUI();
  };
  slider.oninput=()=>{
    state.timingStrength=Number(slider.value);
    state.timingMode="observed";
    output.textContent=`×${state.timingStrength.toFixed(2)}`;
    renderTimingControls();
    drawMicroTiming();
    syncPlaybackUI();
  };
}
function drawMicroTiming(){
  const cv=document.getElementById("microTiming"),g=cv.getContext("2d");
  const W=cv.width,H=cv.height,left=54,right=18,top=30,bottom=42,span=W-left-right;
  const profiles=["D1","D2"].map(voice=>[voice,microProfile(voice)]).filter(([,p])=>p);
  g.clearRect(0,0,W,H);
  if(!profiles.length){
    g.fillStyle=css("--muted");g.font="13px Inter, system-ui";g.textAlign="center";
    g.fillText("Données de micro-timing indisponibles pour ce rythme.",W/2,H/2);
    return;
  }
  const observedValues=profiles.flatMap(([,p])=>p.mean.flatMap((m,i)=>{
    if(!Number.isFinite(m)||p.n[i]<3)return [];
    const sd=Number.isFinite(p.std[i])?p.std[i]:0;
    return [m-sd,m+sd];
  }));
  const maxAbs=Math.max(20,...observedValues.map(v=>Math.abs(v)));
  const limit=Math.min(100,Math.ceil(maxAbs/10)*10);
  const xOf=s=>left+s/11*span;
  const yOf=v=>top+(limit-v)/(2*limit)*(H-top-bottom);
  const p=playheadPosition(),activeStep=p===null?null:Math.floor(p)%12;

  if(activeStep!==null){
    const x=xOf(activeStep),cell=span/11;
    g.fillStyle="rgba(127,127,127,.07)";
    g.fillRect(x-cell*.42,top,cell*.84,H-top-bottom);
  }
  g.strokeStyle=css("--line2");g.lineWidth=1;
  for(let s=0;s<12;s++){
    const x=xOf(s);
    g.beginPath();g.moveTo(x,top);g.lineTo(x,H-bottom);g.stroke();
  }
  g.strokeStyle=css("--line");g.lineWidth=1.2;
  g.beginPath();g.moveTo(left,yOf(0));g.lineTo(W-right,yOf(0));g.stroke();
  g.fillStyle=css("--muted");g.font="11px Inter, system-ui";g.textAlign="center";g.textBaseline="top";
  for(let s=0;s<12;s++)g.fillText(String(s+1),xOf(s),H-bottom+10);
  g.textAlign="right";g.textBaseline="middle";
  [-limit,0,limit].forEach(v=>g.fillText(`${v>0?"+":""}${v} ms`,left-8,yOf(v)));
  g.save();g.translate(13,H/2);g.rotate(-Math.PI/2);g.textAlign="center";
  g.fillText("avance ← écart à la grille → retard",0,0);g.restore();

  profiles.forEach(([voice,profile],voiceIndex)=>{
    const col=color(voice),audible=selectedVoices().includes(voice);
    const points=[];
    profile.mean.forEach((mean,s)=>{
      if(!Number.isFinite(mean)||profile.n[s]<3)return;
      const sd=Number.isFinite(profile.std[s])?profile.std[s]:0;
      const x=xOf(s)+(voiceIndex===0?-3:3);
      const observedY=yOf(mean),applied=timingOffsetMs(voice,s);
      const appliedY=yOf(applied);
      g.globalAlpha=audible ? .34 : .12;g.strokeStyle=col;g.lineWidth=1.3;
      g.beginPath();g.moveTo(x,yOf(Math.min(limit,mean+sd)));g.lineTo(x,yOf(Math.max(-limit,mean-sd)));g.stroke();
      g.beginPath();g.moveTo(x-4,yOf(Math.min(limit,mean+sd)));g.lineTo(x+4,yOf(Math.min(limit,mean+sd)));g.stroke();
      g.beginPath();g.moveTo(x-4,yOf(Math.max(-limit,mean-sd)));g.lineTo(x+4,yOf(Math.max(-limit,mean-sd)));g.stroke();
      g.globalAlpha=audible ? .55 : .18;g.fillStyle=css("--paper");g.strokeStyle=col;g.lineWidth=1.5;
      g.beginPath();g.arc(x,observedY,4,0,TAU);g.fill();g.stroke();
      points.push([x,appliedY]);
    });
    g.globalAlpha=audible ? .82 : .2;g.strokeStyle=col;g.lineWidth=2;
    g.beginPath();points.forEach(([x,y],i)=>{if(i===0)g.moveTo(x,y);else g.lineTo(x,y);});g.stroke();
    points.forEach(([x,y])=>{g.fillStyle=col;g.beginPath();g.arc(x,y,4.5,0,TAU);g.fill();});
  });
  g.globalAlpha=1;
  const d1=microProfile("D1")?.precision,d2=microProfile("D2")?.precision;
  document.getElementById("microCaption").innerHTML=
    `Cercle vide + barre = moyenne et dispersion observées. Point plein = placement entendu (${timingModeLabel()}). `+
    `Précision résiduelle : D1 ${Number.isFinite(d1)?d1.toFixed(1):"—"} ms, D2 ${Number.isFinite(d2)?d2.toFixed(1):"—"} ms.`;
}

function drawMap(){
  const cv=document.getElementById("map"),g=cv.getContext("2d"),W=cv.width,H=cv.height,pad=36;
  g.clearRect(0,0,W,H);
  g.strokeStyle=css("--line");g.lineWidth=1;
  g.beginPath();g.moveTo(pad,H/2);g.lineTo(W-pad,H/2);g.moveTo(W/2,pad);g.lineTo(W/2,H-pad);g.stroke();
  g.fillStyle=css("--muted");g.font="11px Inter, system-ui";g.textAlign="right";g.textBaseline="top";
  g.fillText("axe x : PC1",W-pad,H/2+6);
  g.save();g.translate(W/2+8,pad);g.rotate(-Math.PI/2);g.textAlign="right";g.fillText("axe y : PC2",0,0);g.restore();
  DATA.map.forEach(pt=>{
    const selected=pt.piece===state.piece;
    const x=W/2+pt.x*(W/2-pad*1.6), y=H/2-pt.y*(H/2-pad*1.4);
    const col=DATA.colors[pt.piece]||css("--accent");
    g.globalAlpha=selected?1:.72;
    g.fillStyle=col;g.strokeStyle=css("--paper");g.lineWidth=selected?3:2;
    g.beginPath();g.arc(x,y,selected?13:9,0,TAU);g.fill();g.stroke();
    g.globalAlpha=selected?1:.78;
    g.fillStyle=css("--ink");
    g.font=selected?"700 12px Inter, system-ui":"12px Inter, system-ui";
    g.textAlign="center";g.textBaseline="bottom";
    g.fillText(pt.piece,x,y-15);
  });
  cv.onclick=(ev)=>{
    const rect=cv.getBoundingClientRect();
    const mx=(ev.clientX-rect.left)*cv.width/rect.width, my=(ev.clientY-rect.top)*cv.height/rect.height;
    let best=null, bd=1e9;
    DATA.map.forEach(pt=>{
      const x=W/2+pt.x*(W/2-pad*1.6), y=H/2-pt.y*(H/2-pad*1.4);
      const d=(x-mx)**2+(y-my)**2;
      if(d<bd){bd=d;best=pt;}
    });
    if(best&&bd<1100){state.piece=best.piece;render();}
  };
}

function renderEquation(){
  const r=rhythm();
  document.getElementById("mainEquation").textContent=latexForRhythm(r);
  const d1=voiceView("D1"), d2=voiceView("D2");
  document.getElementById("metrics").innerHTML=[
    ["prises D1/D2",r.takes.join(", ")],
    ["mode",state.synthEnabled?"synthé":"original"],
    ["densité D1",d1.density.toFixed(2)],
    ["densité D2",d2.density.toFixed(2)]
  ].map(([k,v])=>`<div class="metric"><span>${k}</span><b>${v}</b></div>`).join("");
}

function renderCoefTable(){
  const r=rhythm();
  const rows=[];
  for(let k=1;k<=6;k++){
    const a=voiceView("D1").coeffs[k], b=voiceView("D2").coeffs[k], name=DATA.coefNames[String(k)];
    rows.push(`<tr><td>c${k}<br><small>${name}</small></td>
      <td>${a.mag.toFixed(3)}</td><td>${phaseDeg(a.phase).toFixed(0)}°</td>
      <td>${b.mag.toFixed(3)}</td><td>${phaseDeg(b.phase).toFixed(0)}°</td></tr>`);
  }
  document.getElementById("coefTable").innerHTML=`<table class="coefTable">
    <thead><tr><th>onde</th><th>|D1|</th><th>arg D1</th><th>|D2|</th><th>arg D2</th></tr></thead>
    <tbody>${rows.join("")}</tbody></table>`;
}

function hit(ctx,time,voice,amp){
  const osc=ctx.createOscillator(), gain=ctx.createGain(), filter=ctx.createBiquadFilter();
  osc.type=voice==="D1"?"triangle":"sine";
  osc.frequency.setValueAtTime(voice==="D1"?118:235,time);
  osc.frequency.exponentialRampToValueAtTime(voice==="D1"?74:155,time+.12);
  filter.type="lowpass";filter.frequency.setValueAtTime(voice==="D1"?520:900,time);
  gain.gain.setValueAtTime(0.0001,time);
  gain.gain.exponentialRampToValueAtTime(Math.max(0.018,amp)*0.27,time+.006);
  gain.gain.exponentialRampToValueAtTime(0.0001,time+.18);
  osc.connect(filter);filter.connect(gain);gain.connect(playback.master||ctx.destination);
  osc.start(time);osc.stop(time+.2);
}

function scheduleSubdivision(stepIndex,time){
  selectedVoices().forEach(voice=>{
    const value=motifForPlayback(voice)[stepIndex]||0;
    if(value<.055)return;
    const amp=Math.min(1,Math.sqrt(value));
    hit(playback.ctx,time+timingOffsetMs(voice,stepIndex)/1000,voice,amp);
  });
}

function audioScheduler(){
  if(!playback.active||!playback.ctx)return;
  while(playback.nextStepTime<playback.ctx.currentTime+playback.scheduleAhead){
    scheduleSubdivision(playback.stepIndex,playback.nextStepTime);
    playback.nextStepTime+=playback.step;
    playback.stepIndex=(playback.stepIndex+1)%12;
  }
}

function syncPlaybackUI(){
  const btn=document.getElementById("playBtn");
  const status=document.getElementById("liveStatus");
  if(btn){
    btn.textContent=playback.active?"■ Arrêter la boucle":"▶ Lancer la boucle";
    btn.className="play"+(playback.active?" playing":"");
  }
  if(status){
    status.className="liveStatus"+(playback.active?" on":"");
    status.querySelector("span").textContent=playback.active?"LIVE":"Prêt";
  }
  const caption=document.getElementById("playCaption");
  if(caption){
    caption.textContent=playback.active
      ? `Boucle continue · ${playVoiceLabel()} · ${timingModeLabel()} · réglages audibles au prochain pas.`
      : `Écoute : ${playVoiceLabel()} · ${timingModeLabel()} · boucle continue jusqu'à l'arrêt.`;
  }
  syncSynthModeButtons();
}

function stopPlayback(){
  const ctx=playback.ctx;
  playback.active=false;
  if(playback.timer)clearInterval(playback.timer);
  if(playback.raf)cancelAnimationFrame(playback.raf);
  playback.timer=null;
  playback.raf=null;
  playback.ctx=null;
  playback.master=null;
  if(ctx)ctx.close().catch(()=>{});
  syncPlaybackUI();
  drawCircle();
  drawCurve();
  drawMicroTiming();
}

async function startPlayback(){
  const AudioCtx=window.AudioContext||window.webkitAudioContext;
  if(!AudioCtx){alert("Audio Web non disponible dans ce navigateur.");return;}
  const ctx=new AudioCtx();
  playback.ctx=ctx;
  await ctx.resume();
  playback.master=ctx.createGain();
  playback.master.gain.setValueAtTime(.82,ctx.currentTime);
  playback.master.connect(ctx.destination);
  playback.active=true;
  playback.stepIndex=0;
  playback.startTime=ctx.currentTime+.13;
  playback.nextStepTime=playback.startTime;
  audioScheduler();
  playback.timer=setInterval(audioScheduler,playback.lookaheadMs);
  syncPlaybackUI();
  animatePlayback();
}

async function togglePlayback(){
  if(playback.active){stopPlayback();return;}
  await startPlayback();
}
document.getElementById("playBtn").onclick=togglePlayback;

function animatePlayback(){
  drawCircle();
  drawCurve();
  drawMicroTiming();
  if(playback.active) playback.raf=requestAnimationFrame(animatePlayback);
  else playback.raf=null;
}

function render(){
  renderTabs();
  renderVoicePicker();
  renderSynthControls();
  renderTimingControls();
  drawCircle();
  drawCurve();
  drawMicroTiming();
  drawMap();
  renderEquation();
  renderCoefTable();
  syncPlaybackUI();
  queueMath();
}
window.addEventListener("pagehide",stopPlayback);
render();
</script></body></html>"""


def build(out_html="output/boussole_rythmes.html"):
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    html = HTML.replace("__DATA__", json.dumps(build_data(), ensure_ascii=False))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html, len(html)


if __name__ == "__main__":
    path, n = build()
    print(f"OK -> {path} ({n // 1000} Ko)")
