"""Listening check of the stroke detector: is each detected stroke real, and which type?

Uses ``output/soloist_strokes_detected.csv`` (``djembe.onsets.soloist_strokes`` on every
take). Passages are chosen where the detector found strokes the corpus did not have
(fast ornaments), plus a few ordinary ones. Every detected stroke is a marker, green if
the corpus had it, orange if new. Each stroke is played in context then alone, cut just
before the next detected stroke; the listener answers bass / tone / slap / unsure, or
"not a stroke". The answers set the detector's thresholds and check the stroke types.

Audio is embedded (works offline); it is corpus audio (CC BY-NC-ND), so the page lives in
``output/`` and is never published.

    python experiments/strokes/make_detect_page.py      # -> output/stroke_detect/index.html
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
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Stroke check</title>
<style>
:root{--bg:#fbfaf7;--ink:#1f1f1e;--muted:#6b6a63;--line:#e4e3dc;--wave:#9ec5f4;--sel:#d55181;--old:#1baf7a;--new:#eb6834;--b:#184f95;--t:#0f7a54;--s:#b07200;--x:#8a1c1c}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#f1f0ea;--muted:#a9a89f;--line:#3a3a36;--wave:#3d6ea8;--b:#86b6ef;--t:#5fd4a6;--s:#f2c14e;--x:#ff8a80}}
body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:0 auto;padding:16px}
h1{font-size:1.3rem;margin:.2em 0}.muted{color:var(--muted)}
canvas{width:100%;height:210px;border:1px solid var(--line);border-radius:8px;cursor:pointer;display:block}
button,select{font:inherit;padding:6px 10px;border:1px solid var(--line);background:transparent;color:var(--ink);border-radius:6px;cursor:pointer}
button.big{font-size:1.05rem;padding:10px 16px;min-width:105px;font-weight:600}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 0}.k{display:inline-block;border:1px solid var(--line);border-radius:4px;padding:0 6px;font-size:.85em}
#status{font-weight:600}ol{margin:.2em 0 .5em 1.2em;padding:0}li{margin:.2em 0}
details{border:1px solid var(--line);border-radius:8px;padding:8px 12px}.sw{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:middle;margin-right:4px}
</style></head><body>
<h1>Frappes du soliste : vrai coup ? et lequel ?</h1>
<details open><summary><b>Mode d'emploi</b></summary>
<p>Chaque trait est une frappe trouvée automatiquement sur la piste du soliste.
<span class="sw" style="background:var(--old)"></span><b>vert</b> : le corpus l'avait déjà.
<span class="sw" style="background:var(--new)"></span><b>orange</b> : frappe nouvelle (souvent un coup rapide collé à un autre). C'est surtout sur les traits orange que ton avis compte.</p>
<ol><li>Clique sur un trait (ou <span class="k">→</span>). La page joue <b>la frappe d'avant, celle-ci et la suivante, au ralenti</b>, puis <b>celle-ci seule</b> (coupée juste avant la suivante). Le trait choisi est en rose.</li>
<li>Réponds : <b>Basse</b> <span class="k">1</span>, <b>Ton</b> <span class="k">2</span>, <b>Claque</b> <span class="k">3</span>, <b>?</b> <span class="k">0</span> si c'est bien un coup mais que tu hésites sur lequel, ou <b>Pas un coup</b> <span class="k">9</span> si c'est du bruit, la résonance du coup d'avant ou un autre tambour.</li>
<li>La page passe au trait suivant. <span class="k">Espace</span> rejoue le contexte, <span class="k">A</span> la frappe seule. <span class="k">N</span> saute directement au prochain trait orange.</li></ol>
<p class="muted">Pas besoin de tout faire : 2 ou 3 passages suffisent, en visant les traits orange. Tes réponses restent dans ce navigateur. À la fin : <b>Exporter</b>.</p></details>
<div class="row"><button id="prev">◀ Passage précédent</button><b id="title"></b><button id="next">Passage suivant ▶</button>
<select id="rate"><option value="0.5">contexte à 0,5</option><option value="0.75" selected>contexte à 0,75</option><option value="1">contexte à 1</option></select>
<button id="playall">▶ Écouter le passage</button><button id="stop">■ Stop</button></div>
<canvas id="cv" width="1900" height="420"></canvas>
<div class="row"><span id="status"></span></div>
<div class="row"><button class="big" data-l="bass">Basse (1)</button><button class="big" data-l="tone">Ton (2)</button><button class="big" data-l="slap">Claque (3)</button>
<button class="big" data-l="unsure">? (0)</button><button class="big" data-l="none" style="border-color:var(--x)">Pas un coup (9)</button></div>
<div class="row"><button id="ctxbtn">↻ Contexte</button><button id="alonebtn">Frappe seule</button><button id="back">◀ Frappe</button><button id="fwd">Frappe ▶</button><button id="nextnew">Prochain orange (N)</button>
<span id="count" class="muted"></span><button id="export">Exporter</button><button id="reset">Tout effacer</button></div>
<script>
const P=__PASSAGES__, KEY="stroke-detect-v1", LET={bass:"B",tone:"T",slap:"C",unsure:"?",none:"✕"};
let labels={}; try{labels=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let pi=0, si=0, ctx=null, buffers={}, src=null, playing=null, timer=null;
const cv=document.getElementById("cv"), g=cv.getContext("2d"), rateSel=document.getElementById("rate");
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const S=()=>P[pi].strokes;
function save(){try{localStorage.setItem(KEY,JSON.stringify(labels))}catch(e){} count()}
function count(){const all=P.flatMap(p=>p.strokes),nw=all.filter(s=>s.new);const ln=nw.filter(s=>labels[s.id]).length;
 document.getElementById("count").textContent=Object.keys(labels).length+" / "+all.length+" frappes notées (orange : "+ln+" / "+nw.length+")"}
async function buf(i){ctx=ctx||new AudioContext();if(ctx.state==="suspended")await ctx.resume();
 if(!buffers[i]){const b=Uint8Array.from(atob(P[i].audio),c=>c.charCodeAt(0)).buffer;buffers[i]=await ctx.decodeAudioData(b)}return buffers[i]}
function stop(){clearTimeout(timer);if(src){src.onended=null;try{src.stop()}catch(e){}src=null}playing=null;draw()}
async function seg(t0,t1,rate,then){stop();const b=await buf(pi);t0=Math.max(0,t0);t1=Math.min(t1,b.duration);
 src=ctx.createBufferSource();src.buffer=b;src.playbackRate.value=rate;src.connect(ctx.destination);
 playing={t0,rate,start:ctx.currentTime};src.onended=()=>{playing=null;draw();if(then)timer=setTimeout(then,250)};
 src.start(0,t0,t1-t0);requestAnimationFrame(tick)}
function now(){return playing?playing.t0+(ctx.currentTime-playing.start)*playing.rate:null}
function tick(){if(!playing)return;draw();requestAnimationFrame(tick)}
function alone(){const s=S(),t=s[si].t,nx=si+1<s.length?s[si+1].t:t+0.25;seg(t-0.008,Math.min(nx-0.004,t+0.25),1)}
function context(){const s=S(),a=si>0?s[si-1].t-0.02:s[si].t-0.15,b=si+2<s.length?s[si+2].t-0.004:s[si].t+0.35;seg(a,b,+rateSel.value,alone)}
function select(i){si=Math.max(0,Math.min(i,S().length-1));status();context()}
function status(){const s=S()[si],l=labels[s.id];document.getElementById("status").textContent="Frappe "+(si+1)+" / "+S().length+" du passage "+(pi+1)+
 (s.new?" (orange, nouvelle)":" (verte, déjà dans le corpus)")+(l?" : notée "+LET[l]:" : pas encore notée")}
function draw(){const p=P[pi],W=cv.width,H=cv.height,T=p.duration;g.clearRect(0,0,W,H);
 g.fillStyle=css("--wave");const n=p.env.length;for(let i=0;i<n;i++){const v=p.env[i]*H*0.38;g.fillRect(i/n*W,H*0.62-v,Math.max(1,W/n),2*v)}
 p.strokes.forEach((s,i)=>{const x=s.t/T*W,l=labels[s.id],cur=i===si;g.strokeStyle=cur?css("--sel"):(s.new?css("--new"):css("--old"));g.lineWidth=cur?7:3;
  g.beginPath();g.moveTo(x,46);g.lineTo(x,H);g.stroke();
  if(l){g.fillStyle=l==="bass"?css("--b"):l==="tone"?css("--t"):l==="slap"?css("--s"):l==="none"?css("--x"):css("--muted");g.font="bold 30px system-ui";g.textAlign="center";g.fillText(LET[l],x,34)}});
 const t=now();if(t!==null){g.strokeStyle=css("--ink");g.lineWidth=2;g.setLineDash([8,6]);g.beginPath();g.moveTo(t/T*W,0);g.lineTo(t/T*W,H);g.stroke();g.setLineDash([])}
 document.getElementById("title").textContent="Passage "+(pi+1)+" / "+P.length}
function label(l){const s=S();labels[s[si].id]=l;save();if(si<s.length-1)select(si+1);else{status();draw()}}
function nextNew(){const s=S();for(let i=si+1;i<s.length;i++)if(s[i].new){select(i);return}for(let q=1;q<=P.length;q++){const pj=(pi+q)%P.length;const k=P[pj].strokes.findIndex(x=>x.new&&!labels[x.id]);if(k>=0){stop();pi=pj;select(k);return}}}
function go(i){stop();pi=(i+P.length)%P.length;si=0;status();draw()}
cv.onclick=e=>{const r=cv.getBoundingClientRect(),t=(e.clientX-r.left)/r.width*P[pi].duration;let best=0,bd=1e9;
 S().forEach((s,i)=>{const dd=Math.abs(s.t-t);if(dd<bd){bd=dd;best=i}});select(best)};
document.querySelectorAll("[data-l]").forEach(b=>b.onclick=()=>label(b.dataset.l));
document.getElementById("prev").onclick=()=>go(pi-1);document.getElementById("next").onclick=()=>go(pi+1);
document.getElementById("playall").onclick=()=>seg(0,P[pi].duration,1);document.getElementById("stop").onclick=stop;
document.getElementById("ctxbtn").onclick=context;document.getElementById("alonebtn").onclick=alone;document.getElementById("nextnew").onclick=nextNew;
document.getElementById("back").onclick=()=>select(si-1);document.getElementById("fwd").onclick=()=>select(si+1);
document.addEventListener("keydown",e=>{if(e.target.tagName==="SELECT")return;
 if(e.key==="ArrowRight")select(si+1);else if(e.key==="ArrowLeft")select(si-1);
 else if(e.key===" "){e.preventDefault();context()}else if(e.key==="a"||e.key==="A")alone();else if(e.key==="n"||e.key==="N")nextNew();
 else{const m={"1":"bass","2":"tone","3":"slap","0":"unsure","9":"none"}[e.key];if(m)label(m)}});
document.getElementById("export").onclick=()=>{const b=new Blob([JSON.stringify(labels,null,1)],{type:"application/json"});
 const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="stroke_detect_labels.json";a.click()};
document.getElementById("reset").onclick=()=>{if(confirm("Effacer toutes les réponses ?")){labels={};save();draw()}};
count();status();draw();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--detected", default=str(ROOT / "output" / "soloist_strokes_detected.csv"))
    ap.add_argument("--out", default=str(ROOT / "output" / "stroke_detect"))
    ap.add_argument("--dense", type=int, default=6, help="passages chosen for their many new strokes")
    ap.add_argument("--ordinary", type=int, default=2, help="random passages")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import soundfile as sf

    det = pd.read_csv(args.detected)
    rng = np.random.default_rng(args.seed)
    chosen, used = [], set()
    # dense passages: the window with most new strokes, one per take, best takes first
    candidates = []
    for take, g in det.groupby("take_id"):
        new = g[~g.in_corpus].time.to_numpy()
        if len(new) == 0:
            continue
        counts = [(np.sum((new >= t) & (new < t + args.seconds)), t) for t in new]
        n, t0 = max(counts)
        candidates.append((n, take, max(t0 - 0.5, 0.0)))
    for n, take, t0 in sorted(candidates, reverse=True)[:args.dense]:
        chosen.append((take, t0))
        used.add(take)
    for take in rng.permutation([t for t in d.TAKES if t not in used])[:args.ordinary]:
        g = det[det.take_id == take]
        chosen.append((take, float(rng.uniform(g.time.quantile(0.2), g.time.quantile(0.8) - args.seconds))))

    passages, key = [], {}
    for n, (take, t0) in enumerate(chosen):
        piece = take.split("_")[0]
        y, sr = sf.read(Path(d.DEFAULT_CORPUS) / f"MJ_{piece}" / f"MJ_{take}_J1.wav", dtype="float32", always_2d=True)
        clip = resample_poly(y.mean(axis=1)[int(t0 * sr):int((t0 + args.seconds) * sr)], SR_OUT, sr).astype(np.float32)
        clip /= max(np.abs(clip).max(), 1e-9) / 0.8
        buf = io.BytesIO()
        sf.write(buf, clip, SR_OUT, format="WAV", subtype="PCM_16")
        env = np.abs(clip[: len(clip) // 900 * 900]).reshape(900, -1).max(1)
        g = det[(det.take_id == take) & (det.time >= t0 + 0.02) & (det.time <= t0 + args.seconds - 0.05)].sort_values("time")
        strokes = []
        for _, r in g.iterrows():
            sid = f"d{n}_{int(round(r.time * 1000))}"
            strokes.append({"id": sid, "t": round(float(r.time - t0), 4), "new": not bool(r.in_corpus)})
            key[sid] = {"take": take, "time": float(r.time), "in_corpus": bool(r.in_corpus), "level": float(r.level)}
        passages.append({"take": take, "duration": len(clip) / SR_OUT, "env": np.round(env / env.max(), 3).tolist(),
                         "strokes": strokes, "audio": base64.b64encode(buf.getvalue()).decode("ascii")})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(PAGE.replace("__PASSAGES__", json.dumps(passages)), encoding="utf-8")
    (out.parent / "stroke_detect_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    n_new = sum(s["new"] for p in passages for s in p["strokes"])
    print(f"{len(passages)} passages, {len(key)} strokes ({n_new} new) -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
