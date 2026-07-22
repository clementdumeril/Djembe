"""Génère output/boussole_rythmes.html.

La Boussole des rythmes présente trois équations complexes, une par rythme.
Chaque équation combine les deux voix de dunun disponibles sur les prises
appariées D1/D2 :

    R_piece(s) = [D1_piece(s), D2_piece(s)]

Les coefficients ne sont pas modifiables dans cette version. L'objectif est de
lire et comparer les signatures complexes des rythmes, avec un rendu HTML plus
proche d'une figure de recherche.

Pourquoi 6 coefficients ? Pour 12 subdivisions réelles, c7..c11 sont les
miroirs conjugués de c5..c1. Les couches indépendantes non triviales sont donc
c1..c6, plus c0 qui porte la densité moyenne.
"""
from __future__ import annotations

import json
import os

import numpy as np

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
        "coefNames": {
            "1": "gravité du cycle",
            "2": "balancement en deux pôles",
            "3": "trois appuis",
            "4": "quatre appuis",
            "5": "asymétrie fine",
            "6": "alternance rapide",
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
  --bg:#f6f4ee;--paper:#fffdf8;--ink:#1b1b18;--muted:#6f6a60;
  --line:#d8d0c3;--line2:#ebe5da;--accent:#8c5c2e;
  --d1:#214e8a;--d2:#b46a2a;--shadow:0 18px 55px rgba(40,32,20,.11)
}
@media(prefers-color-scheme:dark){
  :root{--bg:#111216;--paper:#191b21;--ink:#f2eadc;--muted:#aaa196;--line:#30333c;--line2:#242730;--accent:#d2a15d;--shadow:0 20px 60px rgba(0,0,0,.35)}
}
*{box-sizing:border-box}
body{margin:0;background:
  radial-gradient(circle at 10% -5%,rgba(180,106,42,.14),transparent 32rem),
  linear-gradient(180deg,var(--bg),color-mix(in srgb,var(--bg) 86%,#fff 14%));
  color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;line-height:1.45}
.page{max-width:1320px;margin:0 auto;padding:32px 22px 42px}
header{display:grid;grid-template-columns:1.05fr .95fr;gap:22px;align-items:end;margin-bottom:18px}
.kicker{font-size:.74rem;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);font-weight:800;margin:0 0 8px}
h1{font-family:Georgia,Times,serif;font-size:clamp(2rem,4vw,4rem);line-height:.95;margin:0;color:var(--ink);letter-spacing:-.045em}
.abstract{background:rgba(255,255,255,.36);border:1px solid var(--line);border-radius:18px;padding:14px 16px;color:var(--muted);font-size:.94rem}
.abstract b{color:var(--ink)}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}
.tab{appearance:none;border:1px solid var(--line);background:var(--paper);color:var(--ink);border-radius:999px;padding:9px 14px;cursor:pointer;font-weight:800;box-shadow:0 4px 14px rgba(0,0,0,.04)}
.tab.on{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.actionbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:-6px 0 18px}
.play{appearance:none;border:0;background:var(--accent);color:white;border-radius:999px;padding:10px 15px;cursor:pointer;font-weight:900;box-shadow:0 8px 22px rgba(140,92,46,.22)}
.actionbar span{color:var(--muted);font-size:.86rem}
.layout{display:grid;grid-template-columns:minmax(440px,.9fr) minmax(500px,1.1fr);gap:18px;align-items:start}
.panel{background:var(--paper);border:1px solid var(--line);border-radius:24px;box-shadow:var(--shadow);padding:18px}
.panel h2{font-family:Georgia,Times,serif;font-size:1.25rem;margin:.1rem 0 .85rem;letter-spacing:-.02em}
.panel h3{font-size:.82rem;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin:1rem 0 .55rem}
canvas{display:block;width:100%;height:auto}
.caption,.note{font-size:.85rem;color:var(--muted)}
.caption b,.note b{color:var(--ink)}
.legend{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0 2px;color:var(--muted);font-size:.82rem}
.swatch{display:inline-flex;align-items:center;gap:7px}.swatch i{width:13px;height:13px;border-radius:99px;display:inline-block;border:2px solid currentColor}
.d1{color:var(--d1)}.d2{color:var(--d2)}
.equationBox{border:1px solid var(--line);background:linear-gradient(180deg,rgba(255,255,255,.54),rgba(255,255,255,.2));border-radius:18px;padding:16px;overflow:auto}
.math{font-size:.98rem}
.metaGrid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:12px 0}
.metric{border:1px solid var(--line2);background:rgba(127,127,127,.045);border-radius:14px;padding:10px}
.metric span{display:block;color:var(--muted);font-size:.74rem;text-transform:uppercase;letter-spacing:.09em}
.metric b{font-size:1rem}
.coefTable{width:100%;border-collapse:separate;border-spacing:0;margin-top:10px;font-size:.84rem;overflow:hidden;border:1px solid var(--line);border-radius:14px}
.coefTable th,.coefTable td{padding:8px 9px;border-bottom:1px solid var(--line2);text-align:right;white-space:nowrap}
.coefTable th:first-child,.coefTable td:first-child{text-align:left}
.coefTable tr:last-child td{border-bottom:0}
.coefTable th{color:var(--muted);font-weight:800;background:rgba(127,127,127,.045)}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:18px}
.card{background:var(--paper);border:1px solid var(--line);border-radius:20px;padding:14px;box-shadow:0 10px 30px rgba(40,32,20,.07);cursor:pointer}
.card.on{border-color:var(--accent);box-shadow:0 0 0 2px color-mix(in srgb,var(--accent) 18%,transparent),0 10px 30px rgba(40,32,20,.08)}
.cardTop{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:8px}
.cardTop b{font-family:Georgia,Times,serif;font-size:1.15rem}.cardTop span{color:var(--muted);font-size:.78rem}
.miniEq{font-size:.78rem;max-height:155px;overflow:auto;border-top:1px solid var(--line2);padding-top:8px}
.researchNote{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:12px}
.researchNote div{border-left:3px solid var(--accent);padding:4px 0 4px 12px;color:var(--muted);font-size:.86rem}
.foot{color:var(--muted);font-size:.82rem;margin:18px 2px 0}
@media(max-width:980px){header,.layout,.researchNote{grid-template-columns:1fr}.cards{grid-template-columns:1fr}.metaGrid{grid-template-columns:1fr 1fr}}
</style></head><body><main class="page">
<header>
  <div>
    <p class="kicker">Complex rhythm signatures · dunun D1/D2</p>
    <h1>Boussole des rythmes</h1>
  </div>
  <div class="abstract">
    <b>Objet.</b> Trois rythmes, trois équations complexes. Chaque signature est
    calculée uniquement sur les prises où D1 et D2 sont tous les deux présents,
    pour comparer les deux voix sur la même base.
  </div>
</header>

<nav class="tabs" id="tabs"></nav>
<div class="actionbar">
  <button class="play" id="playBtn">▶ Écouter le rythme</button>
  <span>Synthèse courte : D1 grave, D2 plus aigu, deux cycles.</span>
</div>

<section class="layout">
  <article class="panel">
    <h2>Cercle métrique à 12 subdivisions</h2>
    <canvas id="circle" width="620" height="620"></canvas>
    <div class="legend">
      <span class="swatch d1"><i></i>D1 — anneau extérieur</span>
      <span class="swatch d2"><i></i>D2 — anneau intérieur</span>
    </div>
    <p class="caption" id="circleCaption"></p>
    <h3>Courbe du modèle</h3>
    <canvas id="curve" width="620" height="260"></canvas>
    <p class="caption">Axes : abscisse = subdivision Polak du cycle, ordonnée = probabilité moyenne de frappe. Les points sont observés ; les courbes viennent de l'équation complexe.</p>
    <h3>Espace des signatures</h3>
    <canvas id="map" width="620" height="250"></canvas>
    <p class="caption">Axes : x = composante principale 1, y = composante principale 2. Chaque point projette la signature complexe complète D1+D2 ; c'est une carte descriptive, pas une vérité absolue.</p>
  </article>

  <aside class="panel">
    <h2>Équation complexe</h2>
    <div class="equationBox"><div class="math" id="mainEquation"></div></div>
    <div class="metaGrid" id="metrics"></div>
    <h3>Table des coefficients</h3>
    <div id="coefTable"></div>
    <div class="researchNote">
      <div><b>D1/D2 comparables.</b><br>La figure utilise les mêmes prises pour D1 et D2 : uniquement les prises appariées où les deux voix sont présentes.</div>
      <div><b>Pourquoi six coefficients ?</b><br>Sur 12 subdivisions réelles, c7..c11 répètent l'information de c1..c5 en miroir conjugué. On garde donc c1..c6.</div>
      <div><b>Notation.</b><br><b>Rythme</b> désigne l'objet complet ; <b>D1</b> et <b>D2</b> sont les deux courbes/voix qui composent ce rythme.</div>
    </div>
  </aside>
</section>

<section class="cards" id="cards"></section>
<p class="foot">Les ronds indiquent les probabilités de frappe par subdivision. La figure reste volontairement minimale : deux anneaux, trois rythmes, trois équations.</p>
</main>

<script>
const DATA=__DATA__;
const TAU=Math.PI*2;
const state={piece:DATA.pieces[0]};

function rhythm(piece=state.piece){return DATA.rhythms.find(r=>r.piece===piece) || DATA.rhythms[0];}
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function phaseDeg(rad){let d=rad*180/Math.PI;while(d>180)d-=360;while(d<=-180)d+=360;return d;}
function signed(x,digits=2){return (x>=0?"+":"") + x.toFixed(digits);}
function phaseTex(rad){return signed(rad,2);}
function texVector(it){
  return String.raw`\left(${it.coeffs.slice(1).map(c=>`${c.mag.toFixed(2)}e^{${phaseTex(c.phase)}i}`).join(String.raw`,\; `)}\right)`;
}
function color(voice){return voice==="D1"?css("--d1"):css("--d2");}
function curveValue(it,t){
  const C=it.coeffs, theta=TAU*t/12, N=12;
  let y=C[0].re;
  for(let k=1;k<=5;k++){
    const c=C[k];
    y += 2*(c.re*Math.cos(k*theta)-c.im*Math.sin(k*theta));
  }
  const c6=C[6];
  y += c6.re*Math.cos(6*theta)-c6.im*Math.sin(6*theta);
  return Math.max(0,y/N*(it.density||1));
}

function latexForRhythm(r){
  const d1=r.voices.D1, d2=r.voices.D2;
  return String.raw`
\[
\begin{aligned}
\operatorname{Rythme}_{\mathrm{${r.piece}}}(s)
  &= \begin{bmatrix}\operatorname{D1}_{\mathrm{${r.piece}}}(s)\\\operatorname{D2}_{\mathrm{${r.piece}}}(s)\end{bmatrix},
  \qquad s\in\{0,\ldots,11\}\\
\operatorname{D}_{v,\mathrm{${r.piece}}}(s)
  &= \frac{\rho_v}{12}\left[
      1+2\sum_{k=1}^{5}|c_{v,k}|
      \cos\!\left(\frac{2\pi ks}{12}+\phi_{v,k}\right)
      + c_{v,6}\cos(\pi s)
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

function renderTabs(){
  const nav=document.getElementById("tabs");
  nav.innerHTML="";
  DATA.rhythms.forEach(r=>{
    const b=document.createElement("button");
    b.className="tab"+(r.piece===state.piece?" on":"");
    b.textContent=r.piece;
    b.onclick=()=>{state.piece=r.piece;render();};
    nav.appendChild(b);
  });
}

function drawCircle(){
  const r=rhythm(), cv=document.getElementById("circle"), g=cv.getContext("2d");
  const W=cv.width, cx=W/2, cy=W/2, outer=218, inner=146;
  g.clearRect(0,0,W,W);
  g.save();
  g.translate(cx,cy);

  g.strokeStyle=css("--line");
  g.lineWidth=1.2;
  [outer,inner].forEach(rad=>{g.beginPath();g.arc(0,0,rad,0,TAU);g.stroke();});

  for(let s=0;s<12;s++){
    const a=s/12*TAU-Math.PI/2;
    const ca=Math.cos(a), sa=Math.sin(a);
    g.fillStyle=css("--muted");
    g.globalAlpha=.55;
    g.beginPath();g.arc((outer+30)*ca,(outer+30)*sa,2.2,0,TAU);g.fill();
    g.globalAlpha=1;
    g.font="12px Inter, system-ui, sans-serif";
    g.textAlign="center";g.textBaseline="middle";
    g.fillText(String(s+1),(outer+49)*ca,(outer+49)*sa);
  }

  [["D1",outer],["D2",inner]].forEach(([voice,rad])=>{
    const it=r.voices[voice], values=it.motif, mx=Math.max(...values,0.001);
    g.strokeStyle=color(voice);
    g.globalAlpha=.7;
    g.lineWidth=2.2;
    g.beginPath();g.arc(0,0,rad,0,TAU);g.stroke();
    values.forEach((v,s)=>{
      const a=s/12*TAU-Math.PI/2, ca=Math.cos(a), sa=Math.sin(a);
      const size=5+16*Math.sqrt(v/mx);
      g.globalAlpha=.96;
      g.fillStyle=color(voice);
      g.beginPath();g.arc(rad*ca,rad*sa,size,0,TAU);g.fill();
      g.globalAlpha=.95;
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

  g.restore();
  document.getElementById("circleCaption").innerHTML =
    `<b>${r.piece}</b> — prises appariées D1/D2 : ${r.takes.join(", ")}. `+
    `Densités moyennes : D1 ${r.voices.D1.density.toFixed(2)}, D2 ${r.voices.D2.density.toFixed(2)} frappes/cycle.`;
}

function drawCurve(){
  const r=rhythm(), cv=document.getElementById("curve"), g=cv.getContext("2d");
  const W=cv.width,H=cv.height,left=50,right=20,top=18,bottom=42,span=W-left-right;
  g.clearRect(0,0,W,H);
  const samples=[];
  ["D1","D2"].forEach(voice=>{
    const it=r.voices[voice];
    for(let i=0;i<=180;i++) samples.push(curveValue(it,i/180*11));
    samples.push(...it.motif);
  });
  const ymax=Math.max(...samples,0.05)*1.14;
  const xOf=t=>left+(t/11)*span;
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
    g.strokeStyle=col;g.lineWidth=2.6;g.globalAlpha=.92;
    g.beginPath();
    for(let i=0;i<=180;i++){
      const t=i/180*11, x=xOf(t), y=yOf(curveValue(it,t));
      if(i===0)g.moveTo(x,y);else g.lineTo(x,y);
    }
    g.stroke();
    it.motif.forEach((v,s)=>{
      g.globalAlpha=.95;g.fillStyle=col;
      g.beginPath();g.arc(xOf(s),yOf(v),5.2,0,TAU);g.fill();
      g.strokeStyle=css("--paper");g.lineWidth=1.4;g.stroke();
    });
  });
  g.globalAlpha=1;
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
  document.getElementById("metrics").innerHTML=[
    ["prises D1/D2",r.takes.join(", ")],
    ["densité D1",r.voices.D1.density.toFixed(2)],
    ["densité D2",r.voices.D2.density.toFixed(2)]
  ].map(([k,v])=>`<div class="metric"><span>${k}</span><b>${v}</b></div>`).join("");
}

function renderCoefTable(){
  const r=rhythm();
  const rows=[];
  for(let k=1;k<=6;k++){
    const a=r.voices.D1.coeffs[k], b=r.voices.D2.coeffs[k], name=DATA.coefNames[String(k)];
    rows.push(`<tr><td>c${k}<br><small>${name}</small></td>
      <td>${a.mag.toFixed(3)}</td><td>${phaseDeg(a.phase).toFixed(0)}°</td>
      <td>${b.mag.toFixed(3)}</td><td>${phaseDeg(b.phase).toFixed(0)}°</td></tr>`);
  }
  document.getElementById("coefTable").innerHTML=`<table class="coefTable">
    <thead><tr><th>couche</th><th>|D1|</th><th>arg D1</th><th>|D2|</th><th>arg D2</th></tr></thead>
    <tbody>${rows.join("")}</tbody></table>`;
}

function renderCards(){
  const box=document.getElementById("cards");
  box.innerHTML="";
  DATA.rhythms.forEach(r=>{
    const div=document.createElement("article");
    div.className="card"+(r.piece===state.piece?" on":"");
    div.onclick=()=>{state.piece=r.piece;render();};
    div.innerHTML=`<div class="cardTop"><b>${r.piece}</b><span>prises ${r.takes.join(", ")}</span></div>
      <div class="miniEq">${latexForRhythm(r)}</div>`;
    box.appendChild(div);
  });
}

function hit(ctx,time,voice,amp){
  const osc=ctx.createOscillator(), gain=ctx.createGain(), filter=ctx.createBiquadFilter();
  osc.type=voice==="D1"?"triangle":"sine";
  osc.frequency.setValueAtTime(voice==="D1"?118:235,time);
  osc.frequency.exponentialRampToValueAtTime(voice==="D1"?74:155,time+.12);
  filter.type="lowpass";filter.frequency.setValueAtTime(voice==="D1"?520:900,time);
  gain.gain.setValueAtTime(0.0001,time);
  gain.gain.exponentialRampToValueAtTime(Math.max(0.025,amp)*0.28,time+.006);
  gain.gain.exponentialRampToValueAtTime(0.0001,time+.18);
  osc.connect(filter);filter.connect(gain);gain.connect(ctx.destination);
  osc.start(time);osc.stop(time+.2);
}

async function playRhythm(){
  const AudioCtx=window.AudioContext||window.webkitAudioContext;
  if(!AudioCtx){alert("Audio Web non disponible dans ce navigateur.");return;}
  const ctx=new AudioCtx();
  await ctx.resume();
  const r=rhythm(), step=.155, start=ctx.currentTime+.08;
  const mx=Math.max(...r.voices.D1.motif,...r.voices.D2.motif,0.001);
  for(let cycle=0;cycle<2;cycle++){
    ["D1","D2"].forEach(voice=>{
      r.voices[voice].motif.forEach((v,s)=>{
        if(v<0.04)return;
        hit(ctx,start+(cycle*12+s)*step,voice,v/mx);
      });
    });
  }
}
document.getElementById("playBtn").onclick=playRhythm;

function render(){renderTabs();drawCircle();drawCurve();drawMap();renderEquation();renderCoefTable();renderCards();queueMath();}
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
