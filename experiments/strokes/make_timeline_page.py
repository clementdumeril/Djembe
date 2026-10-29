"""Listening check, continuous version: label each soloist stroke inside a passage.

Short clips mixed two or three strokes (a pulse lasts about 0.12 s), so this page plays
continuous passages of the soloist's isolated track with a marker at every annotated
onset. Selecting a marker plays that stroke alone (cut just before the next one);
keys 1/2/3/0 label it bass / tone / slap / unsure and move on. The whole passage can be
played at normal or reduced speed. Predictions are never shown; they stay in the key file.

Audio is embedded in the page (works offline from a double-click). It is corpus audio
(CC BY-NC-ND): the page lives in ``output/`` and is never published.

    python experiments/strokes/make_timeline_page.py      # -> output/stroke_timeline/index.html
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from djembe import dataset as d

ROOT = Path(__file__).resolve().parents[2]
SR_OUT = 24000

PAGE = r"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Stroke timeline</title>
<style>
:root{--bg:#fbfaf7;--ink:#1f1f1e;--muted:#6b6a63;--line:#e4e3dc;--wave:#9ec5f4;--sel:#eb6834;--b:#184f95;--t:#1baf7a;--s:#eda100}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#f1f0ea;--muted:#a9a89f;--line:#3a3a36;--wave:#3d6ea8}}
body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:0 auto;padding:16px}
h1{font-size:1.3rem;margin:.2em 0}.muted{color:var(--muted)}canvas{width:100%;height:190px;border:1px solid var(--line);border-radius:8px;cursor:pointer;display:block}
button,select{font:inherit;padding:6px 10px;border:1px solid var(--line);background:transparent;color:var(--ink);border-radius:6px;cursor:pointer}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 0}.k{display:inline-block;border:1px solid var(--line);border-radius:4px;padding:0 6px;font-size:.85em}
.leg span{margin-right:14px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}
</style></head><body>
<h1>Coups du soliste, en continu</h1>
<p class="muted">Clique sur un repère (ou <span class="k">←</span> <span class="k">→</span>) pour entendre <b>cette frappe seule</b>, puis <span class="k">1</span> basse, <span class="k">2</span> ton, <span class="k">3</span> claque, <span class="k">0</span> je ne sais pas : la page passe à la frappe suivante. <span class="k">Espace</span> rejoue la frappe, <span class="k">P</span> joue le passage depuis la frappe choisie. Tes réponses restent dans ce navigateur ; <b>Exporter</b> à la fin.</p>
<div class="row"><button id="prev">◀ Passage précédent</button><b id="title"></b><button id="next">Passage suivant ▶</button>
<button id="playall">▶ Écouter le passage</button><select id="rate"><option value="1">vitesse 1</option><option value="0.75">0,75</option><option value="0.5">0,5</option></select>
<button id="stop">■</button></div>
<canvas id="cv" width="1900" height="380"></canvas>
<div class="row leg"><span><i class="dot" style="background:var(--b)"></i>B basse</span><span><i class="dot" style="background:var(--t)"></i>T ton</span><span><i class="dot" style="background:var(--s)"></i>C claque</span><span>? je ne sais pas</span>
<span id="count" class="muted"></span><button id="export">Exporter</button><button id="reset">Tout effacer</button></div>
<script>
const P=__PASSAGES__, KEY="stroke-timeline-v1", LET={bass:"B",tone:"T",slap:"C",unsure:"?"};
let labels={}; try{labels=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let pi=0, si=0, ctx=null, buffers={}, src=null;
const cv=document.getElementById("cv"), g=cv.getContext("2d");
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
function save(){try{localStorage.setItem(KEY,JSON.stringify(labels))}catch(e){} count()}
function count(){const n=P.reduce((a,p)=>a+p.strokes.length,0);document.getElementById("count").textContent=Object.keys(labels).length+" / "+n+" frappes étiquetées"}
async function buf(i){ctx=ctx||new AudioContext();if(!buffers[i]){const b=Uint8Array.from(atob(P[i].audio),c=>c.charCodeAt(0)).buffer;buffers[i]=await ctx.decodeAudioData(b)}return buffers[i]}
function stop(){if(src){try{src.stop()}catch(e){}src=null}}
async function play(t0,dur,rate=1){stop();const b=await buf(pi);src=ctx.createBufferSource();src.buffer=b;src.playbackRate.value=rate;src.connect(ctx.destination);src.start(0,Math.max(0,t0),dur)}
function strokeWin(i){const s=P[pi].strokes,t=s[i].t,nx=i+1<s.length?s[i+1].t:t+0.25;return [t-0.012,Math.min(nx-0.006,t+0.25)-(t-0.012)]}
function playStroke(){const [a,b]=strokeWin(si);play(a,b);draw()}
function draw(){const p=P[pi],W=cv.width,H=cv.height,T=p.duration;g.clearRect(0,0,W,H);
 g.fillStyle=css("--wave");const n=p.env.length;for(let i=0;i<n;i++){const v=p.env[i]*H*0.42;g.fillRect(i/n*W,H*0.58-v,Math.max(1,W/n),2*v)}
 p.strokes.forEach((s,i)=>{const x=s.t/T*W,l=labels[s.id];g.strokeStyle=i===si?css("--sel"):"rgba(120,120,120,.55)";g.lineWidth=i===si?4:2;
  g.beginPath();g.moveTo(x,28);g.lineTo(x,H);g.stroke();
  if(l){g.fillStyle=l==="bass"?css("--b"):l==="tone"?css("--t"):l==="slap"?css("--s"):css("--muted");g.font="bold 26px system-ui";g.textAlign="center";g.fillText(LET[l],x,24)}});
 document.getElementById("title").textContent="Passage "+(pi+1)+" / "+P.length}
function go(i){stop();pi=(i+P.length)%P.length;si=0;draw()}
cv.onclick=e=>{const r=cv.getBoundingClientRect(),t=(e.clientX-r.left)/r.width*P[pi].duration;let best=0,bd=1e9;
 P[pi].strokes.forEach((s,i)=>{const dd=Math.abs(s.t-t);if(dd<bd){bd=dd;best=i}});si=best;playStroke()};
document.getElementById("prev").onclick=()=>go(pi-1);document.getElementById("next").onclick=()=>go(pi+1);
document.getElementById("playall").onclick=()=>play(0,P[pi].duration,+document.getElementById("rate").value);
document.getElementById("stop").onclick=stop;
document.addEventListener("keydown",e=>{const s=P[pi].strokes;
 if(e.key==="ArrowRight"){si=Math.min(si+1,s.length-1);playStroke()} else if(e.key==="ArrowLeft"){si=Math.max(si-1,0);playStroke()}
 else if(e.key===" "){e.preventDefault();playStroke()} else if(e.key==="p"||e.key==="P"){play(s[si].t-0.05,P[pi].duration,+document.getElementById("rate").value)}
 else{const m={"1":"bass","2":"tone","3":"slap","0":"unsure"}[e.key];if(m){labels[s[si].id]=m;save();if(si<s.length-1){si++;playStroke()}else draw()}}});
document.getElementById("export").onclick=()=>{const b=new Blob([JSON.stringify(labels,null,1)],{type:"application/json"});
 const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="stroke_labels.json";a.click()};
document.getElementById("reset").onclick=()=>{if(confirm("Effacer toutes les réponses ?")){labels={};save();draw()}};
count();draw();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strokes", default=str(ROOT / "output" / "strokes_J1.csv"))
    ap.add_argument("--out", default=str(ROOT / "output" / "stroke_timeline"))
    ap.add_argument("--passages", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import soundfile as sf

    table = pd.read_csv(args.strokes)
    events = d.load_events()
    player = {t: events[(events.take_id == t) & (events.instrument == "J1")].player.iloc[0] for t in d.TAKES}
    rng = np.random.default_rng(args.seed)
    # spread the passages over takes, alternating soloists
    by_player = {}
    for t in d.TAKES:
        by_player.setdefault(player[t], []).append(t)
    order = []
    while len(order) < args.passages:
        for takes in by_player.values():
            if len(order) < args.passages:
                order.append(takes[int(rng.integers(len(takes)))])
    passages, key = [], {}
    for n, take in enumerate(order):
        piece = take.split("_")[0]
        y, sr = sf.read(Path(d.DEFAULT_CORPUS) / f"MJ_{piece}" / f"MJ_{take}_J1.wav", dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        s = table[table.take_id == take].sort_values("onset_s")
        lo, hi = s.onset_s.quantile(0.15), s.onset_s.quantile(0.85) - args.seconds
        t0 = float(rng.uniform(lo, hi))
        clip = y[int(t0 * sr):int((t0 + args.seconds) * sr)]
        clip = resample_poly(clip, SR_OUT, sr).astype(np.float32)
        clip /= max(np.abs(clip).max(), 1e-9) / 0.8
        buf = io.BytesIO()
        sf.write(buf, clip, SR_OUT, format="WAV", subtype="PCM_16")
        env = np.abs(clip[: len(clip) // 800 * 800]).reshape(800, -1).max(1)
        inside = s[(s.onset_s >= t0 + 0.02) & (s.onset_s <= t0 + args.seconds - 0.05)]
        strokes = []
        for _, r in inside.iterrows():
            sid = f"p{n}_{int(round(r.onset_s * 1000))}"
            strokes.append({"id": sid, "t": round(float(r.onset_s - t0), 4)})
            key[sid] = {"predicted": r.stroke, "confidence": round(float(r.confidence), 3), "take": take,
                        "onset_s": round(float(r.onset_s), 4), "bleed": bool(r.bleed)}
        passages.append({"take": take, "duration": len(clip) / SR_OUT, "env": np.round(env / env.max(), 3).tolist(),
                         "strokes": strokes, "audio": base64.b64encode(buf.getvalue()).decode("ascii")})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(PAGE.replace("__PASSAGES__", json.dumps(passages)), encoding="utf-8")
    (out.parent / "stroke_check_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(passages)} passages, {len(key)} strokes -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
