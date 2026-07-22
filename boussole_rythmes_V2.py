"""Génère output/boussole_rythmes.html.

La Boussole des rythmes affiche chaque rythme comme :

- une forme sur un cercle à 12 subdivisions ;
- une équation complexe fixe ;
- un vecteur de coefficients c1..c6.

Les coefficients ne sont pas modifiables dans cette version. L'objectif est de
comparer les équations des rythmes, pas de composer de nouveaux motifs.

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
    "Manjanin": "#4a8cff",
    "Maraka": "#e0952e",
    "Woloso": "#2fbf6e",
    "D1": "#4a8cff",
    "D2": "#2fbf6e",
}


def _profiles(piece: str, code: str):
    tracks = u.list_dunun()
    if code == "D2":
        paired = u.paired_dunun_takes()
        takes = paired.loc[paired["piece"] == piece, "take"].astype(int).tolist()
    else:
        takes = tracks.loc[(tracks["piece"] == piece) & (tracks["code"] == code), "take"].astype(int).tolist()

    vals = []
    used = []
    for take in sorted(set(takes)):
        pr = u.subdivision_profile(piece, int(take), code)
        if pr is not None:
            vals.append(pr.values.astype(float))
            used.append(int(take))

    if not vals:
        return np.zeros(12, dtype=float), []
    return np.mean(vals, axis=0), used


def _coefficients(motif: np.ndarray):
    """Coefficients Ck = Σ x_s exp(-i k θ_s), k=0..6, x normalisé."""
    total = float(motif.sum())
    x = motif / total if total > 0 else motif.copy()
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
    return coeffs, total


def _pca_map(items):
    vectors = []
    for it in items:
        c = it["coeffs"]
        vectors.append([v for k in range(1, 7) for v in (c[k]["re"], c[k]["im"])])
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
    return [{"piece": it["piece"], "voice": it["voice"], "x": round(float(x), 4), "y": round(float(y), 4)}
            for it, (x, y) in zip(items, xy)]


def build_data():
    motifs = {}
    items = []
    paired = u.paired_dunun_takes()

    for piece in u.PIECES:
        motifs[piece] = {}
        for code in ("D1", "D2"):
            motif, takes = _profiles(piece, code)
            if not takes:
                continue
            coeffs, total = _coefficients(motif)
            info = {
                "piece": piece,
                "voice": code,
                "motif": [round(float(x), 4) for x in motif],
                "takes": takes,
                "total": round(total, 6),
                "coeffs": coeffs,
                "partial": code == "D2",
                "pairedTakes": paired.loc[paired["piece"] == piece, "take"].astype(int).tolist(),
            }
            motifs[piece][code] = info
            items.append(info)

    return {
        "pieces": u.PIECES,
        "colors": COLORS,
        "motifs": motifs,
        "map": _pca_map(items),
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
<title>Boussole des rythmes</title>
<style>
:root{--bg:#f4f1ec;--card:#fffaf2;--ink:#211d17;--muted:#70675b;--line:#ded5c8;--accent:#b0742a;--blue:#4a8cff;--green:#2fbf6e;--orange:#e0952e}
@media(prefers-color-scheme:dark){:root{--bg:#101116;--card:#181a22;--ink:#f3ecdf;--muted:#a79d8f;--line:#2c303c;--accent:#e0a94a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,Segoe UI,sans-serif;line-height:1.45}
.wrap{max-width:1260px;margin:0 auto;padding:24px 18px 34px}
h1{margin:.1em 0 .25em;font-size:1.6rem}.sub{max-width:94ch;color:var(--muted);font-size:.95rem;margin:0 0 16px}
.grid{display:grid;grid-template-columns:minmax(390px,1.08fr) minmax(390px,.92fr);gap:16px;align-items:start}
.panel{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:16px;box-shadow:0 8px 30px rgba(0,0,0,.08)}
.panel h2,.panel h3{margin:.1em 0 .7em;font-size:1rem}.controls{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 14px;align-items:center}
.pill{border:1px solid var(--line);background:transparent;color:var(--ink);border-radius:999px;padding:7px 12px;cursor:pointer;font-weight:700;font-size:.84rem}
.pill.on{background:var(--accent);border-color:var(--accent);color:white}.pill.soft.on{background:#293142;color:#fff;border-color:#293142}
label.check{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:.85rem;font-weight:650}
canvas{display:block;width:100%;height:auto}.caption{font-size:.82rem;color:var(--muted);margin-top:8px}
.equation{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;background:rgba(127,127,127,.08);border:1px solid var(--line);border-radius:12px;padding:12px;font-size:.84rem;overflow:auto}
.why{font-size:.86rem;color:var(--muted);border-left:4px solid var(--accent);padding-left:12px;margin:12px 0}
.cards{display:grid;gap:10px;margin-top:12px}.eqCard{border:1px solid var(--line);border-radius:14px;padding:11px;background:rgba(127,127,127,.035);cursor:pointer}
.eqCard.on{border-color:var(--accent);box-shadow:0 0 0 2px color-mix(in srgb,var(--accent) 18%,transparent)}
.eqTop{display:flex;align-items:baseline;justify-content:space-between;gap:10px}.eqTop b{font-size:.94rem}.eqTop span{font-size:.78rem;color:var(--muted)}
.coefGrid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:9px}.coef{border:1px solid var(--line);border-radius:10px;padding:7px;background:rgba(255,255,255,.05)}
.coef b{font-size:.78rem}.coef small{display:block;color:var(--muted);font-size:.72rem}.meter{height:6px;border-radius:999px;background:rgba(127,127,127,.14);overflow:hidden;margin-top:6px}.meter i{display:block;height:100%;background:var(--accent)}
.mapWrap{margin-top:16px}.foot{color:var(--muted);font-size:.82rem;margin-top:12px}
@media(max-width:900px){.grid{grid-template-columns:1fr}.coefGrid{grid-template-columns:repeat(2,1fr)}}
</style></head><body><div class="wrap">
<h1>🧭 Boussole des rythmes</h1>
<p class="sub">Chaque rythme est une forme sur le cercle et une équation complexe fixe. On compare les empreintes <b>c1..c6</b> des rythmes et des voix D1/D2, sans modifier les coefficients.</p>

<div class="grid">
 <section class="panel">
  <h2>Cercle du rythme</h2>
  <div class="controls" id="pieceControls"></div>
  <div class="controls" id="voiceControls"></div>
  <canvas id="circle" width="620" height="620"></canvas>
  <div class="caption" id="circleCaption"></div>
  <div class="mapWrap">
    <h3>Carte vectorielle des rythmes</h3>
    <canvas id="map" width="620" height="260"></canvas>
    <div class="caption">Chaque point = un vecteur complexe (Re/Im de c1..c6). Clique un point pour charger son équation.</div>
  </div>
 </section>

 <aside class="panel">
  <h2>Équation du rythme sélectionné</h2>
  <div class="equation" id="equation"></div>
  <div class="why"><b>Pourquoi 6 coefficients ?</b> Le motif a 12 subdivisions et il est réel. En Fourier discret, c7..c11 sont les miroirs conjugués de c5..c1. Les couches indépendantes sont donc c1..c6, plus c0 qui porte la densité moyenne.</div>
  <h3>Équations disponibles</h3>
  <div class="cards" id="equationCards"></div>
 </aside>
</div>

<p class="foot">Lecture : les points pleins sont le motif observé. Les traits translucides montrent le motif reconstruit par son équation complexe fixe. D2 est partiel : il n'est calculé que sur les prises appariées D1/D2.</p>
</div>

<script>
const DATA=__DATA__;
const TAU=Math.PI*2;
const state={piece:DATA.pieces[0],voice:"D1",showBoth:true};

function item(piece=state.piece,voice=state.voice){return DATA.motifs[piece]?.[voice] || DATA.motifs[piece]?.D1;}
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function colorFor(piece,voice){return voice==="D2"?DATA.colors.D2:DATA.colors[piece]||DATA.colors.D1;}
function fmtPhase(rad){let d=rad*180/Math.PI;while(d>180)d-=360;while(d<=-180)d+=360;return d.toFixed(0)+"°";}

function reconstruct(it){
  const out=[], N=12, total=it.total||1, C=it.coeffs;
  for(let s=0;s<N;s++){
    const th=TAU*s/N;
    let y=C[0].re;
    for(let k=1;k<=5;k++){
      const c=C[k];
      y += 2*(c.re*Math.cos(k*th)-c.im*Math.sin(k*th));
    }
    const c6=C[6];
    y += c6.re*Math.cos(6*th)-c6.im*Math.sin(6*th);
    out.push(Math.max(0,y/N*total));
  }
  return out;
}

function drawCircle(){
  const cv=document.getElementById("circle"), g=cv.getContext("2d"), W=cv.width, cx=W/2, cy=W/2;
  g.clearRect(0,0,W,W);
  const voices=state.showBoth?["D1","D2"].filter(v=>DATA.motifs[state.piece]?.[v]):[state.voice];
  const radii=voices.length===2?{D1:220,D2:148}:{[state.voice]:205};
  const maxR=252;
  g.strokeStyle=css("--line");g.lineWidth=1;
  for(let s=0;s<12;s++){
    const a=s/12*TAU-Math.PI/2;
    g.globalAlpha=.55;g.beginPath();g.moveTo(cx,cy);g.lineTo(cx+maxR*Math.cos(a),cy+maxR*Math.sin(a));g.stroke();
    g.globalAlpha=1;g.fillStyle=css("--muted");g.font="12px system-ui";g.textAlign="center";g.textBaseline="middle";
    g.fillText(String(s+1),cx+(maxR+18)*Math.cos(a),cy+(maxR+18)*Math.sin(a));
  }
  voices.forEach(voice=>{
    const it=DATA.motifs[state.piece][voice], base=radii[voice], col=colorFor(state.piece,voice);
    const rec=reconstruct(it), mx=Math.max(...it.motif,...rec,0.001), selected=voice===state.voice;
    g.globalAlpha=selected?1:.45;g.strokeStyle=col;g.lineWidth=selected?2.4:1.4;
    g.beginPath();g.arc(cx,cy,base,0,TAU);g.stroke();
    for(let s=0;s<12;s++){
      const a=s/12*TAU-Math.PI/2, ca=Math.cos(a), sa=Math.sin(a);
      const len=38*rec[s]/mx;
      g.globalAlpha=selected ? .25 : .12;g.strokeStyle=col;g.lineWidth=10;
      g.beginPath();g.moveTo(cx+(base-18)*ca,cy+(base-18)*sa);g.lineTo(cx+(base-18-len)*ca,cy+(base-18-len)*sa);g.stroke();
      const r=4+13*it.motif[s]/mx;
      g.globalAlpha=selected ? .96 : .55;g.fillStyle=col;g.beginPath();g.arc(cx+base*ca,cy+base*sa,r,0,TAU);g.fill();
      g.globalAlpha=selected ? .9 : .35;g.strokeStyle=css("--card");g.lineWidth=1.2;g.stroke();
    }
    g.globalAlpha=1;g.fillStyle=col;g.font="bold 14px system-ui";g.textAlign="center";
    g.fillText(`${voice}${it.partial?" · partiel":""}`,cx,cy-base+20);
  });
  const it=item();
  document.getElementById("circleCaption").innerHTML =
    `<b>${state.piece} · ${state.voice}</b> — prises utilisées : ${it.takes.join(", ")}${it.partial?" (D2 partiel, prises appariées seulement)":""}. Densité ≈ ${it.total.toFixed(2)} frappes/cycle.`;
}

function drawMap(){
  const cv=document.getElementById("map"),g=cv.getContext("2d"),W=cv.width,H=cv.height,pad=38;
  g.clearRect(0,0,W,H);g.strokeStyle=css("--line");g.lineWidth=1;
  g.beginPath();g.moveTo(pad,H/2);g.lineTo(W-pad,H/2);g.moveTo(W/2,pad/2);g.lineTo(W/2,H-pad/2);g.stroke();
  DATA.map.forEach(pt=>{
    const x=W/2+pt.x*(W/2-pad), y=H/2-pt.y*(H/2-pad);
    const selected=pt.piece===state.piece&&pt.voice===state.voice, col=colorFor(pt.piece,pt.voice);
    g.globalAlpha=selected?1:.75;g.fillStyle=pt.voice==="D2"?"transparent":col;g.strokeStyle=col;g.lineWidth=selected?3:2;
    g.beginPath();g.arc(x,y,selected?11:8,0,TAU);g.fill();g.stroke();
    g.fillStyle=css("--ink");g.globalAlpha=selected?1:.78;g.font=selected?"bold 12px system-ui":"11px system-ui";g.textAlign="center";
    g.fillText(`${pt.piece} ${pt.voice}`,x,y-15);
  });
  cv.onclick=(ev)=>{
    const rect=cv.getBoundingClientRect(), mx=(ev.clientX-rect.left)*cv.width/rect.width, my=(ev.clientY-rect.top)*cv.height/rect.height;
    let best=null, bd=1e9;
    DATA.map.forEach(pt=>{
      const x=W/2+pt.x*(W/2-pad), y=H/2-pt.y*(H/2-pad), d=(x-mx)**2+(y-my)**2;
      if(d<bd){bd=d;best=pt;}
    });
    if(best&&bd<900){state.piece=best.piece;state.voice=best.voice;render();}
  };
}

function equationText(it){
  const C=it.coeffs;
  const terms=[];
  for(let k=1;k<=6;k++){
    terms.push(`${C[k].mag.toFixed(2)}∠${fmtPhase(C[k].phase)}·e${k}`);
  }
  return `R_${it.piece}_${it.voice} = c0(${it.total.toFixed(2)}) + ` + terms.join(" + ");
}

function renderEquation(){
  const it=item(), C=it.coeffs;
  document.getElementById("equation").innerHTML =
    `${equationText(it)}<br><br>`+
    `Vecteur c1..c6 : (${C.slice(1).map(c=>c.mag.toFixed(2)+"∠"+fmtPhase(c.phase)).join(", ")})`;
}

function renderEquationCards(){
  const box=document.getElementById("equationCards");box.innerHTML="";
  DATA.pieces.forEach(piece=>{
    ["D1","D2"].forEach(voice=>{
      const it=DATA.motifs[piece]?.[voice]; if(!it) return;
      const div=document.createElement("div");
      div.className="eqCard"+(piece===state.piece&&voice===state.voice?" on":"");
      div.onclick=()=>{state.piece=piece;state.voice=voice;render();};
      div.innerHTML=`<div class="eqTop"><b>${piece} · ${voice}${it.partial?" · partiel":""}</b><span>${it.takes.length} prise(s), densité ${it.total.toFixed(2)}</span></div>
        <div class="equation" style="margin-top:8px">${equationText(it)}</div>
        <div class="coefGrid">${it.coeffs.slice(1).map(c=>`<div class="coef"><b>c${c.k}</b><small>|c|=${c.mag.toFixed(3)}</small><small>phase ${fmtPhase(c.phase)}</small><div class="meter"><i style="width:${Math.min(100,c.mag*140)}%"></i></div></div>`).join("")}</div>`;
      box.appendChild(div);
    });
  });
}

function renderControls(){
  const pc=document.getElementById("pieceControls"), vc=document.getElementById("voiceControls");
  pc.innerHTML="<b style='color:var(--muted);font-size:.85rem'>rythme</b>";
  DATA.pieces.forEach(p=>{const b=document.createElement("button");b.className="pill"+(p===state.piece?" on":"");b.textContent=p;b.onclick=()=>{state.piece=p;if(!DATA.motifs[p][state.voice])state.voice="D1";render();};pc.appendChild(b);});
  vc.innerHTML="<b style='color:var(--muted);font-size:.85rem'>voix</b>";
  ["D1","D2"].forEach(v=>{if(!DATA.motifs[state.piece]?.[v])return;const b=document.createElement("button");b.className="pill soft"+(v===state.voice?" on":"");b.textContent=v+(DATA.motifs[state.piece][v].partial?" · partiel":"");b.onclick=()=>{state.voice=v;render();};vc.appendChild(b);});
  const lab=document.createElement("label");lab.className="check";lab.innerHTML=`<input type="checkbox" ${state.showBoth?"checked":""}> afficher D1+D2`;lab.querySelector("input").onchange=e=>{state.showBoth=e.target.checked;render();};vc.appendChild(lab);
}

function render(){renderControls();drawCircle();drawMap();renderEquation();renderEquationCards();}
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
