"""
Génère output/atome_vivant_V2.html — « l'atome vivant » du micro-timing du dunun.
Chaque rythme = un atome polaire : angle = position dans le cycle (12h),
rayon = écart de timing (avance vers l'intérieur / retard vers l'extérieur),
nuage = précision. Deux couches : D1 (externe) + D2 (interne).
Tête de lecture au tempo réel, orbitales qui pulsent quand elles sonnent,
sélection des dununs à entendre (audio séparé, Web Audio). Clic = son.
Utilisé par 07_atome_vivant_V2.ipynb.
"""
import base64, io, json, os
import numpy as np, pandas as pd, librosa, soundfile as sf
import utils_V2 as u

CLIP_SR = 16000
T0, DUR = 42.0, 12.0
VC = ["#4a8cff", "#2fbf6e"]     # D1, D2
PLAYED_MIN = 20


def _clip(piece, take, code):
    p = f"{u.CORPUS}/MJ_{piece}/MJ_{piece}_{take}_{code}.wav"
    if not os.path.exists(p):
        return None
    y = librosa.load(p, sr=CLIP_SR, mono=True)[0]
    return y[int(T0 * CLIP_SR):int((T0 + DUR) * CLIP_SR)]


def _b64(clip):
    buf = io.BytesIO(); sf.write(buf, clip, CLIP_SR, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def build(out_html="output/atome_vivant_V2.html"):
    df = pd.read_csv("output/dunun_frappes_V2.csv")
    df["dev"] = (df["rel"] - (df["SD"] - 1) / 12) * df["dur"] * 1000
    rhythms = []
    for piece in u.PIECES:
        take = u.representative_dunun_take(piece, prefer_paired=True)
        if take is None:
            continue

        clips = {c: _clip(piece, take, c) for c in ("D1", "D2")}
        clips = {c: y for c, y in clips.items() if y is not None}
        norm = max(np.abs(y).max() for y in clips.values()) + 1e-9

        voices = []
        for code in clips:
            s_all = df[(df["piece"] == piece) & (df["code"] == code)]
            orbitals = []
            for sd in range(1, 13):
                g = s_all[s_all["SD"] == sd]
                if len(g) < PLAYED_MIN:
                    continue
                cloud = [round(float(x), 1) for x in g["dev"].sample(min(70, len(g)), random_state=0)]
                orbitals.append({"sd": sd, "mean": round(float(g["dev"].mean()), 1), "cloud": cloud})
            sub = df[(df["piece"] == piece) & (df["take"] == take) & (df["code"] == code) &
                     (df["time"] >= T0) & (df["time"] < T0 + DUR)]
            onsets = [{"t": round(float(r.time - T0), 3), "sd": int(r.SD)} for r in sub.itertuples()]
            voices.append({"label": code, "orbitals": orbitals, "onsets": onsets, "audio": _b64(clips[code] / norm)})

        met = u.load_metre(piece, take)
        cyc = [round(float(t - T0), 3) for t in met["Time"].values if T0 <= t < T0 + DUR] if met is not None else []
        rhythms.append({"name": piece, "take": take, "dur": round(DUR, 2), "cycles": cyc, "voices": voices})

    html = _TEMPLATE.replace("__DATA__", json.dumps(rhythms)).replace("__VC__", json.dumps(VC))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html, sum(len(v["onsets"]) for r in rhythms for v in r["voices"])


_TEMPLATE = r"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<title>L'atome vivant du dunun</title><style>
 :root{--bg:#0b0c10;--card:#14161d;--txt:#eef;--dim:#8b93a7}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);
   font-family:system-ui,-apple-system,Segoe UI,sans-serif}
 header{padding:22px 26px 6px}h1{margin:0;font-size:1.35rem}
 p.sub{color:var(--dim);margin:.35em 0 0;font-size:.9rem;max-width:80ch}
 .row{display:flex;gap:18px;flex-wrap:wrap;padding:14px 26px 34px}
 .card{background:var(--card);border-radius:18px;padding:16px;flex:1;min-width:330px;
   box-shadow:0 10px 40px rgba(0,0,0,.45)}
 .card h2{margin:.1em 0 .5em;font-size:1.05rem}.card h2 small{color:var(--dim);font-weight:400;font-size:.75rem}
 .toggles{display:flex;gap:8px;margin-bottom:6px}
 .tog{display:flex;align-items:center;gap:7px;border:1px solid #262a35;border-radius:999px;
   padding:5px 12px;cursor:pointer;font-size:.82rem;font-weight:600;user-select:none;background:#0f1118}
 .tog .sw{width:11px;height:11px;border-radius:50%}.tog.off{opacity:.4}
 canvas.atom{display:block;margin:0 auto;cursor:pointer}
 .ctl{display:flex;align-items:center;gap:12px;margin-top:8px}
 button.play{background:#4a8cff;color:#001;border:0;border-radius:8px;padding:8px 16px;font-weight:650;cursor:pointer}
 button.play.pause{background:#e0952e}.meta{font-size:.8rem;color:var(--dim);font-variant-numeric:tabular-nums}
</style></head><body>
<header><h1>⚛️ L'atome vivant du dunun</h1>
<p class="sub">Chaque rythme est un « atome ». <b>Angle</b> = position dans le cycle (12 h). <b>Rayon</b> = micro-timing : une frappe <b>en avance</b> est tirée vers l'intérieur, <b>en retard</b> vers l'extérieur. Le <b>flou</b> d'une orbitale = sa précision. Deux couches : <b>D1</b> (externe) et <b>D2</b> (interne). La tête de lecture tourne au tempo réel ; une orbitale <b>pulse</b> quand elle sonne. Coche les dununs à entendre, clique pour lancer.</p></header>
<div class="row" id="row"></div>
<script>
const DATA=__DATA__;const VC=__VC__;const BASE=[104,60],SC=0.85;
const row=document.getElementById('row');
function b64buf(b){const s=atob(b);const u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u.buffer;}
function jit(i){return ((i*2654435761>>>0)%1000/1000-0.5)*0.09;}   // jitter angulaire deterministe

DATA.forEach((R,idx)=>{
 const card=document.createElement('div');card.className='card';
 card.innerHTML=`<h2>${R.name} <small>prise ${R.take}</small></h2>`;
 const tg=document.createElement('div');tg.className='toggles';const on=R.voices.map(()=>true);
 R.voices.forEach((v,vi)=>{const el=document.createElement('div');el.className='tog';
   el.innerHTML=`<span class="sw" style="background:${VC[vi]}"></span>${v.label}`;
   el.onclick=()=>{on[vi]=!on[vi];el.classList.toggle('off',!on[vi]);if(gains[vi])gains[vi].gain.value=on[vi]?1:0;};
   tg.appendChild(el);});
 const S=330,cv=document.createElement('canvas');cv.width=S;cv.height=S;cv.className='atom';
 const ctl=document.createElement('div');ctl.className='ctl';
 const btn=document.createElement('button');btn.className='play';btn.textContent='▶ Jouer';
 const meta=document.createElement('span');meta.className='meta';
 ctl.appendChild(btn);ctl.appendChild(meta);
 card.appendChild(tg);card.appendChild(cv);card.appendChild(ctl);row.appendChild(card);
 const g=cv.getContext('2d'),cx=S/2,cy=S/2,maxR=S/2-16;
 function pt(ang,r){return [cx+r*Math.sin(ang),cy-r*Math.cos(ang)];}
 function progress(t){const c=R.cycles;if(c.length<2)return (t/R.dur)*2*Math.PI;
   let i=0;while(i<c.length-1&&c[i+1]<=t)i++;const a0=c[i],a1=c[Math.min(c.length-1,i+1)];
   return (a1>a0?(t-a0)/(a1-a0):0)*2*Math.PI;}
 function draw(t){g.clearRect(0,0,S,S);const tt=t%R.dur;
   // graduations 12h
   g.strokeStyle='#181b23';g.lineWidth=1;
   for(let s=0;s<12;s++){const a=s/12*2*Math.PI;const [x,y]=pt(a,maxR);g.beginPath();g.moveTo(cx,cy);g.lineTo(x,y);g.stroke();}
   R.voices.forEach((v,vi)=>{const bR=BASE[vi],al=on[vi]?1:.28;
     g.strokeStyle='#2a2f3a';g.globalAlpha=al*.8;g.beginPath();g.arc(cx,cy,bR,0,7);g.setLineDash([3,4]);g.stroke();g.setLineDash([]);
     v.orbitals.forEach(o=>{const a0=(o.sd-1)/12*2*Math.PI;
       o.cloud.forEach((d,i)=>{const [x,y]=pt(a0+jit(i+o.sd*13),bR+SC*d);
         g.globalAlpha=al*.5;g.fillStyle=VC[vi];g.beginPath();g.arc(x,y,1.7,0,7);g.fill();});
       const near=on[vi]&&v.onsets.some(o2=>o2.sd===o.sd&&Math.abs(tt-o2.t)<0.09);
       const [mx,my]=pt(a0,bR+SC*o.mean);
       g.globalAlpha=1;g.fillStyle=near?'#fff':VC[vi];g.strokeStyle=VC[vi];g.lineWidth=2;
       g.beginPath();g.arc(mx,my,near?7:4,0,7);g.fill();
       if(near){g.beginPath();g.arc(mx,my,13,0,7);g.globalAlpha=.9;g.stroke();}});
     g.globalAlpha=1;});
   const pa=progress(tt);const [px,py]=pt(pa,maxR);
   g.strokeStyle='#fff';g.globalAlpha=.85;g.lineWidth=1.5;g.beginPath();g.moveTo(cx,cy);g.lineTo(px,py);g.stroke();
   g.globalAlpha=1;g.fillStyle='#fff';g.beginPath();g.arc(cx,cy,3,0,7);g.fill();}

 let ctx=null,bufs=null,gains=[],srcs=[],startT=0,raf=null,playing=false;
 async function ensure(){if(bufs)return;ctx=new (window.AudioContext||window.webkitAudioContext)();
   bufs=await Promise.all(R.voices.map(v=>ctx.decodeAudioData(b64buf(v.audio))));}
 async function play(){DATA.forEach((_,j)=>{if(j!==idx&&window['stopA'+j])window['stopA'+j]();});
   await ensure();if(ctx.state==='suspended')await ctx.resume();srcs=[];gains=[];startT=ctx.currentTime+0.03;
   R.voices.forEach((v,vi)=>{const s=ctx.createBufferSource();s.buffer=bufs[vi];s.loop=true;
     const gg=ctx.createGain();gg.gain.value=on[vi]?1:0;s.connect(gg).connect(ctx.destination);s.start(startT);srcs.push(s);gains.push(gg);});
   playing=true;btn.textContent='❚❚ Pause';btn.classList.add('pause');frame();}
 function stop(){if(srcs)srcs.forEach(s=>{try{s.stop();}catch(e){}});srcs=[];playing=false;
   btn.textContent='▶ Jouer';btn.classList.remove('pause');if(raf)cancelAnimationFrame(raf);}
 window['stopA'+idx]=stop;
 function frame(){const t=playing?(ctx.currentTime-startT):0;draw(t);
   meta.textContent=(t%R.dur).toFixed(1)+' / '+R.dur.toFixed(0)+' s';raf=requestAnimationFrame(frame);}
 btn.onclick=()=>playing?stop():play();cv.onclick=()=>playing?stop():play();
 draw(0);
});
</script></body></html>"""


if __name__ == "__main__":
    path, n = build()
    print(f"OK -> {path} ({n} onsets, {os.path.getsize(path)/1e6:.1f} Mo)")
