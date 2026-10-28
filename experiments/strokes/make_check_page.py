"""Listening check of the automatic stroke types: a local page to label strokes by ear.

Draws ``--per-class`` strokes of each predicted type (random takes and confidences),
cuts a short clip of each from the soloist's isolated track, and writes a page that
plays them in random order without showing the prediction. The listener labels each
one (bass, tone, slap, unsure) and exports ``stroke_labels.json``; ``agreement.py``
then compares the labels with the predictions kept in ``key.json``.

Clips are corpus audio (CC BY-NC-ND): the page lives in ``output/`` and is never published.

    python experiments/strokes/make_check_page.py            # -> output/stroke_check/index.html
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from djembe import dataset as d
from djembe import render

ROOT = Path(__file__).resolve().parents[2]
CLIP = (-0.03, 0.30)                     # seconds around the onset

PAGE = """<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Stroke check</title>
<style>
:root{--bg:#fbfaf7;--ink:#1f1f1e;--muted:#6b6a63;--line:#e4e3dc;--on:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#f1f0ea;--muted:#a9a89f;--line:#3a3a36;--on:#5598e7}}
body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;max-width:760px;margin:0 auto;padding:16px}
h1{font-size:1.3rem;margin:.2em 0}.muted{color:var(--muted)}
.ref{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0 16px}.ref div{border:1px solid var(--line);padding:8px;border-radius:8px}
.item{display:flex;align-items:center;gap:10px;padding:8px;border-bottom:1px solid var(--line)}
.item.cur{outline:2px solid var(--on);border-radius:6px}.n{width:3em;color:var(--muted)}
button{font:inherit;padding:6px 10px;border:1px solid var(--line);background:transparent;color:var(--ink);border-radius:6px;cursor:pointer}
button.sel{background:var(--on);color:#fff;border-color:var(--on)}
.bar{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center;flex-wrap:wrap}
</style></head><body>
<h1>Coups du soliste : basse, ton ou claque ?</h1>
<p class="muted">Écoute chaque frappe et choisis. Clavier : <b>Espace</b> rejoue, <b>1</b> basse, <b>2</b> ton, <b>3</b> claque, <b>0</b> je ne sais pas (passe à la suivante). Tes réponses sont gardées dans ce navigateur ; clique sur <b>Exporter</b> à la fin et envoie le fichier.</p>
<p class="muted">Références (un autre djembé, échantillons CC0) :</p>
<div class="ref"><div>Basse <audio controls src="ref_bass.mp3"></audio></div><div>Ton <audio controls src="ref_tone.mp3"></audio></div><div>Claque <audio controls src="ref_slap.mp3"></audio></div></div>
<div class="bar"><span id="count"></span><button id="export">Exporter</button><button id="reset">Tout effacer</button></div>
<div id="list"></div>
<script>
const ITEMS=__ITEMS__, KEY="stroke-check-v1", LAB={bass:"Basse",tone:"Ton",slap:"Claque",unsure:"?"};
let labels={}; try{labels=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let cur=0; const list=document.getElementById("list");
function save(){try{localStorage.setItem(KEY,JSON.stringify(labels))}catch(e){} count()}
function count(){document.getElementById("count").textContent=Object.keys(labels).length+" / "+ITEMS.length+" étiquetées"}
ITEMS.forEach((id,i)=>{const r=document.createElement("div");r.className="item";r.id="it"+i;
 r.innerHTML=`<span class="n">${i+1}</span><audio preload="auto" src="clips/${id}.wav"></audio><button data-p>▶</button>`+
 Object.entries(LAB).map(([k,v])=>`<button data-l="${k}">${v}</button>`).join("");
 r.querySelector("[data-p]").onclick=()=>{cur=i;play()};
 r.querySelectorAll("[data-l]").forEach(b=>b.onclick=()=>{cur=i;label(b.dataset.l)});list.appendChild(r)});
function paint(){ITEMS.forEach((id,i)=>{const r=document.getElementById("it"+i);r.classList.toggle("cur",i===cur);
 r.querySelectorAll("[data-l]").forEach(b=>b.classList.toggle("sel",labels[id]===b.dataset.l))})}
function play(){const a=document.querySelector("#it"+cur+" audio");a.currentTime=0;a.play();paint();
 document.getElementById("it"+cur).scrollIntoView({block:"center",behavior:"smooth"})}
function label(l){labels[ITEMS[cur]]=l;save();paint();if(cur<ITEMS.length-1){cur++;play()}}
document.addEventListener("keydown",e=>{if(e.key===" "){e.preventDefault();play()}
 const m={"1":"bass","2":"tone","3":"slap","0":"unsure"}[e.key];if(m)label(m)});
document.getElementById("export").onclick=()=>{const b=new Blob([JSON.stringify(labels,null,1)],{type:"application/json"});
 const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="stroke_labels.json";a.click()};
document.getElementById("reset").onclick=()=>{if(confirm("Effacer toutes les réponses ?")){labels={};save();paint()}};
count();paint();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strokes", default=str(ROOT / "output" / "strokes_J1.csv"))
    ap.add_argument("--out", default=str(ROOT / "output" / "stroke_check"))
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import soundfile as sf

    table = pd.read_csv(args.strokes)
    rng = np.random.default_rng(args.seed)
    pick = pd.concat([g.sample(args.per_class, random_state=args.seed) for _, g in table.groupby("stroke")])
    pick = pick.iloc[rng.permutation(len(pick))].reset_index(drop=True)
    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    key, ids, tracks = {}, [], {}
    for n, r in pick.iterrows():
        piece = r.take_id.split("_")[0]
        if r.take_id not in tracks:
            y, sr = sf.read(Path(d.DEFAULT_CORPUS) / f"MJ_{piece}" / f"MJ_{r.take_id}_J1.wav", dtype="float32", always_2d=True)
            tracks[r.take_id] = (y.mean(axis=1), sr)
        y, sr = tracks[r.take_id]
        clip = y[max(int((r.onset_s + CLIP[0]) * sr), 0):int((r.onset_s + CLIP[1]) * sr)].copy()
        fade = min(len(clip), int(0.02 * sr))
        clip[-fade:] *= np.linspace(1, 0, fade)
        clip /= max(np.abs(clip).max(), 1e-9) / 0.8
        cid = f"{n:03d}"
        sf.write(out / "clips" / f"{cid}.wav", clip, sr, subtype="PCM_16")
        ids.append(cid)
        key[cid] = {"predicted": r.stroke, "confidence": round(float(r.confidence), 3), "take": r.take_id,
                    "onset_s": round(float(r.onset_s), 4), "bleed": bool(r.bleed)}
    for name, fname in (("bass", "bass"), ("tone", "tone"), ("slap", "slap")):
        shutil.copyfile(render.SAMPLES_DIR / render.SAMPLE_FILES[fname], out / f"ref_{name}.mp3")
    (out / "index.html").write_text(PAGE.replace("__ITEMS__", json.dumps(ids)), encoding="utf-8")
    (out.parent / "stroke_check_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(ids)} clips -> {out / 'index.html'}  (key: {out.parent / 'stroke_check_key.json'})")


if __name__ == "__main__":
    main()
