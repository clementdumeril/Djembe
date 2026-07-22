"""Génère output/boussole_complexe.html.

Une visualisation interactive où un rythme de dunun devient :

- une forme sur un cercle à 12 subdivisions ;
- un vecteur de coefficients complexes c1..c6 ;
- une équation manipulable par curseurs.

Pourquoi 6 coefficients ? Pour 12 subdivisions réelles, c7..c11 sont les
miroirs conjugués de c5..c1. Les seules couches non triviales indépendantes
sont donc c1..c6, plus c0 qui porte la densité moyenne.
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
        comps = vt[:2]
        xy = X @ comps.T
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
            is_partial = code == "D2"
            info = {
                "piece": piece,
                "voice": code,
                "motif": [round(float(x), 4) for x in motif],
                "takes": takes,
                "total": round(total, 6),
                "coeffs": coeffs,
                "partial": is_partial,
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
<title>Boussole complexe des dununs</title>
<style>
:root{--bg:#f4f1ec;--card:#fffaf2;--ink:#211d17;--muted:#70675b;--line:#ded5c8;--accent:#b0742a;--blue:#4a8cff;--green:#2fbf6e;--orange:#e0952e}
@media(prefers-color-scheme:dark){:root{--bg:#101116;--card:#181a22;--ink:#f3ecdf;--muted:#a79d8f;--line:#2c303c;--accent:#e0a94a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,Segoe UI,sans-serif;line-height:1.45}
.wrap{max-width:1260px;margin:0 auto;padding:24px 18px 34px}
h1{margin:.1em 0 .25em;font-size:1.55rem}.sub{max-width:92ch;color:var(--muted);font-size:.95rem;margin:0 0 16px}
.grid{display:grid;grid-template-columns:minmax(390px,1.15fr) minmax(360px,.85fr);gap:16px;align-items:start}
.panel{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:16px;box-shadow:0 8px 30px rgba(0,0,0,.08)}
.panel h2,.panel h3{margin:.1em 0 .7em;font-size:1rem}.controls{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 14px;align-items:center}
.pill{border:1px solid var(--line);background:transparent;color:var(--ink);border-radius:999px;padding:7px 12px;cursor:pointer;font-weight:700;font-size:.84rem}
.pill.on{background:var(--accent);border-color:var(--accent);color:white}.pill.soft.on{background:#293142;color:#fff;border-color:#293142}
label.check{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:.85rem;font-weight:650}
canvas{display:block;width:100%;height:auto}.caption{font-size:.82rem;color:var(--muted);margin-top:8px}
.equation{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;background:rgba(127,127,127,.08);border:1px solid var(--line);border-radius:12px;padding:12px;font-size:.84rem;overflow:auto}
.why{font-size:.86rem;color:var(--muted);border-left:4px solid var(--accent);padding-left:12px;margin:12px 0}
.cards{display:grid;gap:10px;margin-top:12px}.coef{border:1px solid var(--line);border-radius:14px;padding:10px;background:rgba(127,127,127,.035)}
.coefHead{display:flex;align-items:baseline;justify-content:space-between;gap:10px}.coefHead b{font-size:.92rem}.coefHead span{font-size:.78rem;color:var(--muted)}
.coef small{color:var(--muted)}.row{display:grid;grid-template-columns:48px 1fr 46px;gap:8px;align-items:center;margin-top:7px;font-size:.78rem;color:var(--muted)}
input[type=range]{width:100%}.meter{height:7px;border-radius:999px;background:rgba(127,127,127,.14);overflow:hidden;margin-top:8px}.meter i{display:block;height:100%;background:var(--accent)}
.miniBtns{display:flex;gap:6px;margin-top:8px}.miniBtns button{font-size:.72rem;border:1px solid var(--line);background:transparent;color:var(--ink);border-radius:7px;padding:4px 7px;cursor:pointer}
.mapWrap{margin-top:16px}.foot{color:var(--muted);font-size:.82rem;margin-top:12px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<h1>🧭 Boussole complexe des dununs</h1>
<p class="sub">Un rythme devient une forme sur le cercle, puis une petite équation complexe. Tu peux modifier les couches <b>c1..c6</b> : le motif reconstruit bouge immédiatement. D1 est la voix complète ; D2 est affiché seulement à partir des prises appariées.</p>

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
  <h2>Équation manipulable</h2>
  <div class="equation" id="equation"></div>
  <div class="why"><b>Pourquoi 6 coefficients ?</b> Le motif a 12 subdivisions et il est réel. En Fourier discret, c7..c11 sont les miroirs conjugués de c5..c1. Les seules couches indépendantes sont donc c1..c6, plus c0 qui porte la densité moyenne.</div>
  <div class="controls">
    <button class="pill soft" id="resetBtn">Réinitialiser</button>
    <button class="pill soft" id="muteBtn">Tout couper</button>
  </div>
  <div class="cards" id="coefCards"></div>
 </aside>
</div>

<p class="foot">Lecture : les points pleins sont le motif réel ; les traits translucides sont la reconstruction par l'équation. Monter un coefficient renforce sa couche, tourner sa phase décale cette couche dans le cycle.</p>
</div>

<script>
const DATA=__DATA__;
const TAU=Math.PI*2;
const state={piece:DATA.pieces[0],voice:"D1",showBoth:true,weights:{},phases:{}};
for(let k=1;k<=6;k++){state.weights[k]=1;state.phases[k]=0;}

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
      const c=C[k], wk=state.weights[k], ph=state.phases[k]*Math.PI/180;
      const re=c.re*Math.cos(ph)-c.im*Math.sin(ph);
      const im=c.re*Math.sin(ph)+c.im*Math.cos(ph);
      y += 2*wk*(re*Math.cos(k*th)-im*Math.sin(k*th));
    }
    const k=6, c=C[6], wk=state.weights[6], ph=state.phases[6]*Math.PI/180;
    const re=c.re*Math.cos(ph)-c.im*Math.sin(ph);
    const im=c.re*Math.sin(ph)+c.im*Math.cos(ph);
    y += wk*(re*Math.cos(6*th)-im*Math.sin(6*th));
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

function renderEquation(){
  const it=item(), C=it.coeffs, terms=[];
  for(let k=1;k<=6;k++){
    const amp=C[k].mag.toFixed(2), ph=fmtPhase(C[k].phase);
    terms.push(`w${k}·${amp}∠${ph}·e^{i${k}θ}`);
  }
  document.getElementById("equation").innerHTML =
    `r(θ) = c0 + 2·Re(${terms.slice(0,5).join(" + ")}) + ${terms[5]}<br>`+
    `c0 = densité moyenne · ${state.piece}/${state.voice}<br>`+
    `Vecteur : (${C.slice(1).map(c=>c.mag.toFixed(2)+"∠"+fmtPhase(c.phase)).join(", ")})`;
}

function renderCoefCards(){
  const it=item(), box=document.getElementById("coefCards");box.innerHTML="";
  for(let k=1;k<=6;k++){
    const c=it.coeffs[k], name=DATA.coefNames[String(k)], div=document.createElement("div");
    div.className="coef";
    div.innerHTML=`<div class="coefHead"><b>c${k} · ${name}</b><span>|c${k}|=${c.mag.toFixed(3)} · phase ${fmtPhase(c.phase)}</span></div>
      <div class="meter"><i style="width:${Math.min(100,c.mag*140)}%"></i></div>
      <div class="row"><span>force</span><input data-kind="w" data-k="${k}" type="range" min="0" max="2" step="0.01" value="${state.weights[k]}"><b>${state.weights[k].toFixed(2)}</b></div>
      <div class="row"><span>phase</span><input data-kind="p" data-k="${k}" type="range" min="-180" max="180" step="1" value="${state.phases[k]}"><b>${state.phases[k]}°</b></div>
      <div class="miniBtns"><button data-solo="${k}">solo</button><button data-zero="${k}">couper</button></div>`;
    box.appendChild(div);
  }
  box.querySelectorAll("input").forEach(inp=>inp.oninput=()=>{
    const k=+inp.dataset.k;
    if(inp.dataset.kind==="w") state.weights[k]=+inp.value; else state.phases[k]=+inp.value;
    render();
  });
  box.querySelectorAll("button[data-solo]").forEach(btn=>btn.onclick=()=>{
    const solo=+btn.dataset.solo;for(let k=1;k<=6;k++)state.weights[k]=k===solo?1:0;render();
  });
  box.querySelectorAll("button[data-zero]").forEach(btn=>btn.onclick=()=>{
    state.weights[+btn.dataset.zero]=0;render();
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

function render(){renderControls();drawCircle();drawMap();renderEquation();renderCoefCards();}
document.getElementById("resetBtn").onclick=()=>{for(let k=1;k<=6;k++){state.weights[k]=1;state.phases[k]=0;}render();};
document.getElementById("muteBtn").onclick=()=>{for(let k=1;k<=6;k++)state.weights[k]=0;render();};
render();
</script></body></html>"""


def build(out_html="output/boussole_complexe.html"):
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    html = HTML.replace("__DATA__", json.dumps(build_data(), ensure_ascii=False))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html, len(html)


if __name__ == "__main__":
    path, n = build()
    print(f"OK -> {path} ({n // 1000} Ko)")
