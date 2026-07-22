"""
Génère output/carte_interactive.html — carte UMAP des frappes de jembé,
cliquable : chaque point joue sa vraie frappe (clips embarqués en base64).
Utilisé par 10_carte_interactive.ipynb.
"""
import base64, io, json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.preprocessing import StandardScaler
import umap
import djembe_utils as dj

CLIP_SR = 16000
CLIP_DUR = 0.22
N_PER_TYPE = 150
COLORS = {"bass": "#4a8cff", "tone": "#e05ce0", "slap": "#3ddb7a"}


def build_carte(out_html="output/carte_interactive.html", seed=0):
    strokes = pd.read_csv("output/all_jembe_strokes_typed.csv")
    # échantillon stratifié par type de frappe
    sample = (strokes.groupby("stroke", group_keys=False)
              .apply(lambda g: g.sample(min(N_PER_TYPE, len(g)), random_state=seed))
              .reset_index(drop=True))

    # UMAP sur l'échantillon
    X = StandardScaler().fit_transform(sample[dj.FAST_FEATS])
    emb = umap.UMAP(n_neighbors=15, min_dist=0.15, random_state=42).fit_transform(X)
    sample["x"], sample["y"] = emb[:, 0], emb[:, 1]

    # clips audio : charger chaque piste une seule fois (à 16 kHz)
    wl = int(CLIP_DUR * CLIP_SR)
    fade = np.linspace(1, 0, 300)
    points = []
    for (piece, take, code), g in sample.groupby(["piece", "take", "code"]):
        y, _ = librosa.load(f"{dj.CORPUS}/MJ_{piece}/MJ_{piece}_{take}_{code}.wav",
                            sr=CLIP_SR, mono=True)
        for _, r in g.iterrows():
            a = int(r["time"] * CLIP_SR)
            clip = y[a:a + wl].copy()
            if len(clip) < wl // 2:
                continue
            clip[-len(fade):] *= fade[:len(clip[-len(fade):])]
            buf = io.BytesIO()
            sf.write(buf, clip, CLIP_SR, format="WAV", subtype="PCM_16")
            b64 = base64.b64encode(buf.getvalue()).decode()
            points.append({"x": round(float(r["x"]), 3), "y": round(float(r["y"]), 3),
                           "s": r["stroke"], "p": piece, "r": r["role"].replace("jembe_", ""),
                           "a": b64})

    data = json.dumps(points)
    html = """<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"><title>Carte des frappes de jembé</title>
<style>
 body{margin:0;background:#0d0d0d;color:#eee;font-family:system-ui,sans-serif}
 #wrap{display:flex;flex-direction:column;height:100vh}
 header{padding:10px 16px}
 h1{font-size:16px;margin:0 0 4px} p{margin:0;font-size:12px;color:#999}
 #legend{padding:6px 16px;font-size:13px}
 .chip{display:inline-block;margin-right:14px;cursor:pointer;user-select:none}
 .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}
 canvas{flex:1;cursor:crosshair}
 #info{position:fixed;pointer-events:none;background:#222;border:1px solid #444;
       padding:4px 8px;border-radius:4px;font-size:12px;display:none}
</style></head><body><div id="wrap">
<header><h1>🥁 Carte des frappes de jembé — clique un point pour l'entendre</h1>
<p>Chaque point = une vraie frappe du corpus IEMP Malian Jembe (Polak). Position = ressemblance de timbre (UMAP). Survole pour l'info, clique pour le son.</p></header>
<div id="legend"></div>
<canvas id="cv"></canvas><div id="info"></div></div>
<script>
const PTS=__DATA__;
const COL=__COLORS__;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d'),info=document.getElementById('info');
let hidden={};
const xs=PTS.map(p=>p.x),ys=PTS.map(p=>p.y);
const [x0,x1,y0,y1]=[Math.min(...xs),Math.max(...xs),Math.min(...ys),Math.max(...ys)];
function sx(x){return 40+(x-x0)/(x1-x0)*(cv.width-80)}
function sy(y){return 30+(y-y0)/(y1-y0)*(cv.height-60)}
function draw(){
  cv.width=cv.clientWidth;cv.height=cv.clientHeight;
  ctx.clearRect(0,0,cv.width,cv.height);
  for(const p of PTS){ if(hidden[p.s])continue;
    ctx.beginPath();ctx.arc(sx(p.x),sy(p.y),4,0,7);ctx.fillStyle=COL[p.s];ctx.globalAlpha=.75;ctx.fill();}
  ctx.globalAlpha=1;
}
function nearest(mx,my){let best=null,bd=144;
  for(const p of PTS){ if(hidden[p.s])continue;
    const d=(sx(p.x)-mx)**2+(sy(p.y)-my)**2;
    if(d<bd){bd=d;best=p}} return best}
cv.onmousemove=e=>{const p=nearest(e.offsetX,e.offsetY);
  if(p){info.style.display='block';info.style.left=(e.clientX+12)+'px';info.style.top=(e.clientY+12)+'px';
    info.textContent=p.s.toUpperCase()+' · '+p.p+' · '+p.r;}else info.style.display='none';};
cv.onclick=e=>{const p=nearest(e.offsetX,e.offsetY);
  if(p) new Audio('data:audio/wav;base64,'+p.a).play();};
const leg=document.getElementById('legend');
for(const s of ['bass','tone','slap']){
  const el=document.createElement('span');el.className='chip';
  el.innerHTML='<span class="dot" style="background:'+COL[s]+'"></span>'+s;
  el.onclick=()=>{hidden[s]=!hidden[s];el.style.opacity=hidden[s]?0.35:1;draw()};
  leg.appendChild(el);}
new ResizeObserver(draw).observe(cv);draw();
</script></body></html>"""
    html = html.replace("__DATA__", data).replace("__COLORS__", json.dumps(COLORS))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html, len(points)


if __name__ == "__main__":
    path, n = build_carte()
    import os
    print(f"OK -> {path} ({n} frappes, {os.path.getsize(path)/1e6:.1f} Mo)")
