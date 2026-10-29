"""Listening check, continuous version: label each soloist stroke inside a passage.

Short clips mixed two or three strokes (a pulse lasts about 0.12 s), so this page plays
continuous passages of the soloist's isolated track with a marker at every annotated
onset. Two ways to label: stroke by stroke (each stroke is played in context, the
previous and next strokes at reduced speed, then alone; buttons or keys 1/2/3/0 for
bass / tone / slap / unsure), or tapping along while the passage plays slowly, each tap
going to the stroke that just passed the playhead. Predictions are never shown; they
stay in the key file.

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
:root{--bg:#fbfaf7;--ink:#1f1f1e;--muted:#6b6a63;--line:#e4e3dc;--wave:#9ec5f4;--sel:#eb6834;--b:#184f95;--t:#1baf7a;--s:#c98500;--ph:#d55181}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#f1f0ea;--muted:#a9a89f;--line:#3a3a36;--wave:#3d6ea8}}
body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:0 auto;padding:16px}
h1{font-size:1.3rem;margin:.2em 0}h2{font-size:1rem;margin:1em 0 .3em}.muted{color:var(--muted)}
canvas{width:100%;height:200px;border:1px solid var(--line);border-radius:8px;cursor:pointer;display:block}
button,select{font:inherit;padding:6px 10px;border:1px solid var(--line);background:transparent;color:var(--ink);border-radius:6px;cursor:pointer}
button.big{font-size:1.1rem;padding:10px 18px;min-width:110px;font-weight:600}
button.on{background:var(--sel);color:#fff;border-color:var(--sel)}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 0}.k{display:inline-block;border:1px solid var(--line);border-radius:4px;padding:0 6px;font-size:.85em}
#status{font-weight:600}ol{margin:.2em 0 .5em 1.2em;padding:0}li{margin:.15em 0}
details{border:1px solid var(--line);border-radius:8px;padding:8px 12px}
</style></head><body>
<h1>Coups du soliste : basse, ton ou claque</h1>
<details open><summary><b>Mode d'emploi</b> (à lire une fois)</summary>
<p>Chaque trait vertical sur la forme d'onde est <b>une frappe du soliste</b>. Il faut dire pour chaque trait si c'est une basse, un ton ou une claque.</p>
<h2>Méthode 1 : frappe par frappe (la plus sûre)</h2>
<ol><li>Clique sur un trait (ou <span class="k">→</span> pour le suivant). La page joue <b>la frappe d'avant, celle-ci et celle d'après, au ralenti</b>, puis <b>celle-ci seule</b>. Le trait concerné est en orange, la barre rose montre ce qui joue.</li>
<li>Clique sur <b>Basse</b>, <b>Ton</b> ou <b>Claque</b> (ou <span class="k">1</span> <span class="k">2</span> <span class="k">3</span>). Si tu hésites : <b>?</b> (<span class="k">0</span>). La page passe toute seule au trait suivant et le joue.</li>
<li><span class="k">Espace</span> rejoue le contexte, <span class="k">A</span> rejoue la frappe seule.</li></ol>
<h2>Méthode 2 : au fil de l'écoute (plus rapide)</h2>
<ol><li>Clique sur <b>Mode au fil de l'écoute</b>, choisis la vitesse (0,5 conseillé), puis <b>▶ Lancer le passage</b> (ou <span class="k">P</span>).</li>
<li>Pendant que le passage défile, tape <span class="k">1</span> <span class="k">2</span> <span class="k">3</span> à chaque coup que tu entends, comme si tu le jouais. Chaque appui est attribué à la frappe qui vient de passer sous la barre rose.</li>
<li>Repasse ensuite en mode frappe par frappe (re-clique sur le bouton du mode) pour vérifier les traits douteux ou sans lettre.</li></ol>
<p class="muted">Tes réponses restent dans ce navigateur, même si tu fermes la page. Pas besoin de tout faire : une centaine de frappes suffit. À la fin, clique sur <b>Exporter</b>.</p></details>
<div class="row"><button id="prev">◀ Passage précédent</button><b id="title"></b><button id="next">Passage suivant ▶</button>
<button id="modebtn">Mode au fil de l'écoute</button><select id="rate"><option value="0.5">vitesse 0,5</option><option value="0.75" selected>vitesse 0,75</option><option value="1">vitesse 1</option></select>
<button id="playall">▶ Lancer le passage</button><button id="stop">■ Stop</button></div>
<canvas id="cv" width="1900" height="400"></canvas>
<div class="row"><span id="status"></span></div>
<div class="row"><button class="big" data-l="bass" style="border-color:var(--b)">Basse (1)</button><button class="big" data-l="tone" style="border-color:var(--t)">Ton (2)</button>
<button class="big" data-l="slap" style="border-color:var(--s)">Claque (3)</button><button class="big" data-l="unsure">? (0)</button>
<button id="ctxbtn">↻ Contexte</button><button id="alonebtn">Frappe seule</button><button id="back">◀ Frappe</button><button id="fwd">Frappe ▶</button></div>
<div class="row"><span id="count" class="muted"></span><button id="export">Exporter</button><button id="reset">Tout effacer</button></div>
<script>
const P=__PASSAGES__, KEY="stroke-timeline-v2", LET={bass:"B",tone:"T",slap:"C",unsure:"?"};
let labels={}; try{labels=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let pi=0, si=0, ctx=null, buffers={}, src=null, tap=false, playing=null, timer=null;
const cv=document.getElementById("cv"), g=cv.getContext("2d"), rateSel=document.getElementById("rate");
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const S=()=>P[pi].strokes;
function save(){try{localStorage.setItem(KEY,JSON.stringify(labels))}catch(e){} count()}
function count(){const n=P.reduce((a,p)=>a+p.strokes.length,0);document.getElementById("count").textContent=Object.keys(labels).length+" / "+n+" frappes étiquetées"}
async function buf(i){ctx=ctx||new AudioContext();if(ctx.state==="suspended")await ctx.resume();
 if(!buffers[i]){const b=Uint8Array.from(atob(P[i].audio),c=>c.charCodeAt(0)).buffer;buffers[i]=await ctx.decodeAudioData(b)}return buffers[i]}
function stop(){clearTimeout(timer);if(src){src.onended=null;try{src.stop()}catch(e){}src=null}playing=null;draw()}
async function seg(t0,t1,rate,then){stop();const b=await buf(pi);t0=Math.max(0,t0);t1=Math.min(t1,b.duration);
 src=ctx.createBufferSource();src.buffer=b;src.playbackRate.value=rate;src.connect(ctx.destination);
 playing={t0,t1,rate,start:ctx.currentTime};src.onended=()=>{playing=null;draw();if(then)timer=setTimeout(then,250)};
 src.start(0,t0,t1-t0);requestAnimationFrame(tick)}
function now(){return playing?playing.t0+(ctx.currentTime-playing.start)*playing.rate:null}
function tick(){if(!playing)return;draw();requestAnimationFrame(tick)}
function alone(){const s=S(),t=s[si].t,nx=si+1<s.length?s[si+1].t:t+0.25;seg(t-0.012,Math.min(nx-0.006,t+0.25),1)}
function context(){const s=S(),a=si>0?s[si-1].t-0.02:s[si].t-0.15,b=si+2<s.length?s[si+2].t-0.006:s[si].t+0.35;
 seg(a,b,+rateSel.value,alone)}
function select(i){si=Math.max(0,Math.min(i,S().length-1));status();context()}
function status(){const s=S(),l=labels[s[si].id];document.getElementById("status").textContent=tap?
 "Mode au fil de l'écoute : lance le passage et tape 1 / 2 / 3 à chaque coup.":
 "Frappe "+(si+1)+" / "+s.length+" du passage "+(pi+1)+(l?" : déjà notée "+LET[l]:" : pas encore notée")}
function draw(){const p=P[pi],W=cv.width,H=cv.height,T=p.duration;g.clearRect(0,0,W,H);
 g.fillStyle=css("--wave");const n=p.env.length;for(let i=0;i<n;i++){const v=p.env[i]*H*0.4;g.fillRect(i/n*W,H*0.6-v,Math.max(1,W/n),2*v)}
 p.strokes.forEach((s,i)=>{const x=s.t/T*W,l=labels[s.id],cur=!tap&&i===si;g.strokeStyle=cur?css("--sel"):"rgba(120,120,120,.5)";g.lineWidth=cur?6:2;
  g.beginPath();g.moveTo(x,40);g.lineTo(x,H);g.stroke();
  if(l){g.fillStyle=l==="bass"?css("--b"):l==="tone"?css("--t"):l==="slap"?css("--s"):css("--muted");g.font="bold 30px system-ui";g.textAlign="center";g.fillText(LET[l],x,32)}});
 const t=now();if(t!==null){g.strokeStyle=css("--ph");g.lineWidth=4;g.beginPath();g.moveTo(t/T*W,0);g.lineTo(t/T*W,H);g.stroke()}
 document.getElementById("title").textContent="Passage "+(pi+1)+" / "+P.length}
function label(l){const s=S();
 if(tap){const t=now();if(t===null)return;const target=t-0.18*playing.rate;let best=-1,bd=1e9;
  s.forEach((x,i)=>{if(x.t<=t+0.02){const dd=Math.abs(x.t-target);if(dd<bd){bd=dd;best=i}}});
  if(best>=0&&bd<0.3){labels[s[best].id]=l;save();draw()}return}
 labels[s[si].id]=l;save();if(si<s.length-1)select(si+1);else{status();draw()}}
function go(i){stop();pi=(i+P.length)%P.length;si=0;status();draw()}
cv.onclick=e=>{if(tap)return;const r=cv.getBoundingClientRect(),t=(e.clientX-r.left)/r.width*P[pi].duration;let best=0,bd=1e9;
 S().forEach((s,i)=>{const dd=Math.abs(s.t-t);if(dd<bd){bd=dd;best=i}});select(best)};
document.querySelectorAll("[data-l]").forEach(b=>b.onclick=()=>label(b.dataset.l));
document.getElementById("prev").onclick=()=>go(pi-1);document.getElementById("next").onclick=()=>go(pi+1);
document.getElementById("playall").onclick=()=>seg(0,P[pi].duration,+rateSel.value);
document.getElementById("stop").onclick=stop;document.getElementById("ctxbtn").onclick=context;document.getElementById("alonebtn").onclick=alone;
document.getElementById("back").onclick=()=>select(si-1);document.getElementById("fwd").onclick=()=>select(si+1);
document.getElementById("modebtn").onclick=e=>{tap=!tap;stop();e.target.classList.toggle("on",tap);if(tap)rateSel.value="0.5";status();draw()};
document.addEventListener("keydown",e=>{if(e.target.tagName==="SELECT")return;
 if(e.key==="ArrowRight"&&!tap)select(si+1);else if(e.key==="ArrowLeft"&&!tap)select(si-1);
 else if(e.key===" "){e.preventDefault();if(!tap)context()}else if(e.key==="a"||e.key==="A"){if(!tap)alone()}
 else if(e.key==="p"||e.key==="P"){playing?stop():seg(0,P[pi].duration,+rateSel.value)}
 else{const m={"1":"bass","2":"tone","3":"slap","0":"unsure"}[e.key];if(m)label(m)}});
document.getElementById("export").onclick=()=>{const b=new Blob([JSON.stringify(labels,null,1)],{type:"application/json"});
 const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="stroke_labels.json";a.click()};
document.getElementById("reset").onclick=()=>{if(confirm("Effacer toutes les réponses ?")){labels={};save();draw()}};
count();status();draw();
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
