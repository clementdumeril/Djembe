"""
Génère output/annotation_dunun.html — outil d'annotation À L'AVEUGLE des frappes de dunun.
Tu écoutes ~150 frappes (échantillon stratifié, ordre mélangé, sans voir ma prédiction),
tu étiquettes Ouvert / Fermé / Incertain, puis tu exportes tes labels (CSV).
Le validateur (05_dunun_supervision_V2) comparera tes labels au clustering.
"""
import base64, io, json
import numpy as np, pandas as pd, librosa, soundfile as sf
import utils_V2 as u

CLIP_SR = 16000
CLIP_DUR = 0.55          # assez long pour entendre la résonance (ouvert sonne plus longtemps)
PER_STRATE = 25          # 3 pièces × 2 groupes prédits × 25 = 150 frappes


def build(out_html="output/annotation_dunun.html", n_per=PER_STRATE, seed=0):
    df = pd.read_csv("output/dunun_frappes_V2.csv")
    # échantillon stratifié par pièce ET par groupe prédit (pour l'équilibre) — mais AVEUGLE à l'écoute
    sample = (df.groupby(["piece", "tonalite"], group_keys=False)
              .apply(lambda g: g.sample(min(n_per, len(g)), random_state=seed))
              .reset_index(drop=True))
    sample.insert(0, "id", range(len(sample)))
    sample.to_csv("output/dunun_annot_sample.csv", index=False)  # garde le lien id -> frappe (+ prédiction cachée)

    # clips audio
    wl = int(CLIP_DUR * CLIP_SR); fade = np.linspace(1, 0, 400)
    cache = {}
    items = []
    for _, r in sample.iterrows():
        key = (r["piece"], int(r["take"]), r["code"])
        if key not in cache:
            cache[key] = librosa.load(f"{u.CORPUS}/MJ_{r['piece']}/MJ_{r['piece']}_{int(r['take'])}_{r['code']}.wav",
                                      sr=CLIP_SR, mono=True)[0]
        y = cache[key]; a = int(r["time"] * CLIP_SR); clip = y[a:a + wl].copy()
        if len(clip) < wl // 2:
            continue
        clip = clip / (np.abs(clip).max() + 1e-9)
        clip[-len(fade):] *= fade[:len(clip[-len(fade):])]
        buf = io.BytesIO(); sf.write(buf, clip, CLIP_SR, format="WAV", subtype="PCM_16")
        items.append({"id": int(r["id"]), "a": base64.b64encode(buf.getvalue()).decode()})

    rng = np.random.RandomState(seed); rng.shuffle(items)   # ordre aléatoire = aveugle
    html = _TEMPLATE.replace("__DATA__", json.dumps(items))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html, len(items)


_TEMPLATE = r"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<title>Annotation des frappes de dunun</title><style>
 :root{--bg:#101319;--card:#1a1e28;--txt:#eef;--dim:#8b93a7;--open:#4a8cff;--closed:#e0952e;--unc:#6b7280}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);
   font-family:system-ui,-apple-system,Segoe UI,sans-serif;min-height:100vh;
   display:flex;flex-direction:column;align-items:center}
 header{padding:22px;text-align:center;max-width:720px}
 h1{margin:0;font-size:1.3rem}p.sub{color:var(--dim);font-size:.9rem;margin:.5em 0 0}
 .card{background:var(--card);border-radius:18px;padding:34px;margin:14px;width:min(560px,92vw);
   text-align:center;box-shadow:0 10px 40px rgba(0,0,0,.4)}
 .count{color:var(--dim);font-size:.85rem;font-variant-numeric:tabular-nums}
 .bar{height:6px;background:#242a36;border-radius:3px;margin:14px 0 26px;overflow:hidden}
 .bar>div{height:100%;background:var(--open);width:0;transition:width .2s}
 .play{width:96px;height:96px;border-radius:50%;border:0;background:#242a36;color:#fff;
   font-size:2rem;cursor:pointer;margin-bottom:8px}.play:hover{background:#2c3342}
 .hint{color:var(--dim);font-size:.8rem;margin:8px 0 22px}
 .labels{display:flex;gap:12px;justify-content:center;flex-wrap:wrap}
 .lab{flex:1;min-width:120px;border:0;border-radius:12px;padding:16px 10px;cursor:pointer;
   font-size:1rem;font-weight:650;color:#001}
 .lab small{display:block;font-weight:400;font-size:.72rem;opacity:.75;margin-top:3px}
 .lab.o{background:var(--open)}.lab.f{background:var(--closed)}.lab.i{background:var(--unc);color:#fff}
 .nav{display:flex;justify-content:space-between;margin-top:20px;font-size:.85rem}
 .nav button{background:none;border:1px solid #333;color:var(--dim);border-radius:8px;padding:7px 14px;cursor:pointer}
 .done{background:#1f9a58;color:#fff;border:0;border-radius:10px;padding:12px 22px;font-weight:650;cursor:pointer;font-size:.95rem}
 .export{margin:6px 0 40px;text-align:center}
</style></head><body>
<header><h1>🥁 Annote les frappes de dunun</h1>
<p class="sub">Écoute chaque frappe et dis si elle sonne <b>ouverte</b> (résonante, qui sonne longtemps) ou <b>fermée</b> (étouffée, mate). Si tu hésites : <b>incertain</b>. Tu ne vois aucune prédiction : c'est TOI la vérité terrain. Raccourcis clavier : <b>O</b> / <b>F</b> / <b>I</b>, <b>Espace</b> = rejouer.</p></header>
<div class="card" id="card"></div>
<div class="export"><button class="done" id="exp">💾 Exporter mes labels (CSV)</button>
<div class="count" id="expc" style="margin-top:8px"></div>
<textarea id="out" style="display:none;width:min(560px,92vw);height:140px;margin-top:12px;background:#0c0d11;color:#cde;border:1px solid #333;border-radius:8px;padding:10px;font-family:ui-monospace,monospace;font-size:.78rem"></textarea>
<div id="outh" class="count" style="display:none;margin-top:6px">Si le téléchargement ne part pas : copie ce texte dans un fichier <b>dunun_labels.csv</b>.</div></div>
<script>
const DATA=__DATA__;let i=0;const labels={};
const card=document.getElementById('card');
let cur=null;
function audio(){if(cur){cur.pause();}cur=new Audio('data:audio/wav;base64,'+DATA[i].a);cur.play();}
function render(){
 const done=Object.keys(labels).length;
 card.innerHTML=`<div class="count">frappe ${i+1} / ${DATA.length} · ${done} étiquetées</div>
  <div class="bar"><div style="width:${done/DATA.length*100}%"></div></div>
  <button class="play" id="pl">▶</button><div class="hint">clique pour (ré)écouter · Espace</div>
  <div class="labels">
   <button class="lab o" data-l="ouvert">Ouvert<small>résonant (O)</small></button>
   <button class="lab f" data-l="fermé">Fermé<small>étouffé (F)</small></button>
   <button class="lab i" data-l="incertain">Incertain<small>(I)</small></button></div>
  <div class="nav"><button id="prev">← précédent</button>
   <span class="count">${labels[DATA[i].id]?'→ '+labels[DATA[i].id]:'—'}</span>
   <button id="next">passer →</button></div>`;
 document.getElementById('pl').onclick=audio;
 document.querySelectorAll('.lab').forEach(b=>b.onclick=()=>choose(b.dataset.l));
 document.getElementById('prev').onclick=()=>{i=Math.max(0,i-1);render();audio();};
 document.getElementById('next').onclick=()=>advance();
 updateExport();
}
function choose(l){labels[DATA[i].id]=l;advance();}
function advance(){if(i<DATA.length-1){i++;render();audio();}else{render();}}
function updateExport(){document.getElementById('expc').textContent=
  Object.keys(labels).length+' / '+DATA.length+' étiquetées';}
document.addEventListener('keydown',e=>{const k=e.key.toLowerCase();
 if(k==='o')choose('ouvert');else if(k==='f')choose('fermé');else if(k==='i')choose('incertain');
 else if(k===' '){e.preventDefault();audio();}});
document.getElementById('exp').onclick=()=>{
 let csv='id,label\n';for(const id in labels)csv+=id+','+labels[id]+'\n';
 try{const b=new Blob([csv],{type:'text/csv'});const a=document.createElement('a');
  a.href=URL.createObjectURL(b);a.download='dunun_labels.csv';a.click();}catch(e){}
 const ta=document.getElementById('out');ta.style.display='block';ta.value=csv;
 document.getElementById('outh').style.display='block';ta.select();};
render();
</script></body></html>"""


if __name__ == "__main__":
    path, n = build()
    import os
    print(f"OK -> {path} ({n} frappes à annoter, {os.path.getsize(path)/1e6:.1f} Mo)")
