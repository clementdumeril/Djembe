"""
Génère output/dunun_viz_V2.html — visualisation dynamique des dununs des 3 rythmes.
Chaque roue montre les deux échelons de dunun en cercles concentriques (D1 grand, D2 petit).
Chaque dunun a son audio SÉPARÉ → on peut activer/couper D1 et D2 indépendamment
(Web Audio, synchro parfaite). Tête de lecture au tempo réel, timeline à 2 voies. Clic = son.
Utilisé par 04_dunun_visu_V2.ipynb.
"""
import base64, io, json, os
import numpy as np, pandas as pd, librosa, soundfile as sf
import utils_V2 as u

CLIP_SR = 16000
T0, DUR = 42.0, 12.0
VOICE_COLORS = ["#4a8cff", "#2fbf6e"]   # D1, D2


def _clip(piece, take, code):
    p = f"{u.CORPUS}/MJ_{piece}/MJ_{piece}_{take}_{code}.wav"
    if not os.path.exists(p):
        return None
    y = librosa.load(p, sr=CLIP_SR, mono=True)[0]
    return y[int(T0 * CLIP_SR):int((T0 + DUR) * CLIP_SR)]


def _b64(clip):
    buf = io.BytesIO(); sf.write(buf, clip, CLIP_SR, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def build(out_html="output/dunun_viz_V2.html"):
    df = pd.read_csv("output/dunun_frappes_V2.csv")
    rhythms = []
    for piece in u.PIECES:
        take = u.representative_dunun_take(piece, prefer_paired=True)
        if take is None:
            continue

        # charge les clips, normalise avec un facteur COMMUN (préserve l'équilibre D1/D2)
        clips = {c: _clip(piece, take, c) for c in ("D1", "D2")}
        clips = {c: y for c, y in clips.items() if y is not None}
        norm = max(np.abs(y).max() for y in clips.values()) + 1e-9

        voices = []
        for code, y in clips.items():
            prof = u.subdivision_profile(piece, take, code)
            motif = [round(float(x), 2) for x in prof.values] if prof is not None else [0] * 12
            sub = df[(df["piece"] == piece) & (df["take"] == take) & (df["code"] == code) &
                     (df["time"] >= T0) & (df["time"] < T0 + DUR)]
            onsets = [{"t": round(float(r.time - T0), 3), "sd": int(r.SD)} for r in sub.itertuples()]
            voices.append({"label": code, "motif": motif, "onsets": onsets, "audio": _b64(y / norm)})

        met = u.load_metre(piece, take)
        cyc = [round(float(t - T0), 3) for t in met["Time"].values if T0 <= t < T0 + DUR] if met is not None else []
        rhythms.append({"name": piece, "take": take, "dur": round(DUR, 2), "cycles": cyc, "voices": voices})

    html = _TEMPLATE.replace("__DATA__", json.dumps(rhythms)).replace("__VC__", json.dumps(VOICE_COLORS))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html, sum(len(v["onsets"]) for r in rhythms for v in r["voices"])


_TEMPLATE = r"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<title>Dununs — visualisation dynamique</title><style>
 :root{--bg:#0e0f13;--card:#171921;--txt:#eef;--dim:#8b93a7}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);
   font-family:system-ui,-apple-system,Segoe UI,sans-serif}
 header{padding:22px 26px 6px}h1{margin:0;font-size:1.35rem}
 p.sub{color:var(--dim);margin:.35em 0 0;font-size:.9rem;max-width:76ch}
 .row{display:flex;gap:18px;flex-wrap:wrap;padding:14px 26px 30px}
 .card{background:var(--card);border-radius:16px;padding:18px;flex:1;min-width:320px;
   box-shadow:0 8px 30px rgba(0,0,0,.35)}
 .card h2{margin:.1em 0 .4em;font-size:1.05rem}
 .card h2 small{color:var(--dim);font-weight:400;font-size:.75rem}
 .toggles{display:flex;gap:8px;margin-bottom:8px}
 .tog{display:flex;align-items:center;gap:7px;border:1px solid #2b2f3a;border-radius:999px;
   padding:5px 12px;cursor:pointer;font-size:.82rem;font-weight:600;user-select:none;background:#12141b}
 .tog .sw{width:11px;height:11px;border-radius:50%}
 .tog.off{opacity:.4}
 .wheel{display:block;margin:0 auto;cursor:pointer}
 .tl{width:100%;height:88px;display:block;margin-top:12px;background:#0c0d11;border-radius:8px}
 .ctl{display:flex;align-items:center;gap:12px;margin-top:12px}
 button.play{background:#4a8cff;color:#001;border:0;border-radius:8px;padding:8px 16px;
   font-weight:650;cursor:pointer;font-size:.9rem}
 button.play.pause{background:#e0952e}
 .meta{font-size:.8rem;color:var(--dim);font-variant-numeric:tabular-nums}
</style></head><body>
<header><h1>🥁 Les dununs des 3 rythmes — deux échelons</h1>
<p class="sub">Grand cercle = <b>D1</b> (dunun principal), petit cercle = <b>D2</b> (2ᵉ dunun, souvent plus grand/grave). Coche les dununs que tu veux <b>entendre</b> (audio séparé, synchro). La tête de lecture tourne au tempo réel, chaque frappe s'allume ; la timeline (2 voies) défile avec l'audio. Clique une roue ou « Jouer ».</p></header>
<div class="row" id="row"></div>
<script>
const DATA=__DATA__;const VC=__VC__;
const row=document.getElementById('row');
function b64buf(b64){const s=atob(b64);const u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u.buffer;}

DATA.forEach((R,idx)=>{
 const card=document.createElement('div');card.className='card';
 card.innerHTML=`<h2>${R.name} <small>prise ${R.take}</small></h2>`;
 const tg=document.createElement('div');tg.className='toggles';
 const on=R.voices.map(()=>true);   // état activé par voix
 R.voices.forEach((v,vi)=>{const el=document.createElement('div');el.className='tog';
   el.innerHTML=`<span class="sw" style="background:${VC[vi]}"></span>${v.label}`;
   el.onclick=()=>{on[vi]=!on[vi];el.classList.toggle('off',!on[vi]);if(gains[vi])gains[vi].gain.value=on[vi]?1:0;};
   tg.appendChild(el);});
 const W=300,cv=document.createElement('canvas');cv.width=W;cv.height=W;cv.className='wheel';
 const tl=document.createElement('canvas');tl.className='tl';tl.width=560;tl.height=88;
 const ctl=document.createElement('div');ctl.className='ctl';
 const btn=document.createElement('button');btn.className='play';btn.textContent='▶ Jouer';
 const meta=document.createElement('span');meta.className='meta';
 ctl.appendChild(btn);ctl.appendChild(meta);
 card.appendChild(tg);card.appendChild(cv);card.appendChild(tl);card.appendChild(ctl);row.appendChild(card);

 const wx=cv.getContext('2d'),tx=tl.getContext('2d');
 const cx=W/2,cy=W/2,radii=[W/2-34,(W/2-34)*0.60];
 function ang(sd){return (sd-1)/12*2*Math.PI-Math.PI/2;}
 function progress(t){const c=R.cycles;if(c.length<2)return (t/R.dur)*2*Math.PI-Math.PI/2;
   let i=0;while(i<c.length-1&&c[i+1]<=t)i++;const a0=c[i],a1=c[Math.min(c.length-1,i+1)];
   const f=a1>a0?(t-a0)/(a1-a0):0;return f*2*Math.PI-Math.PI/2;}
 function drawWheel(t){wx.clearRect(0,0,W,W);const tt=t%R.dur;
   R.voices.forEach((v,vi)=>{const rad=radii[vi];const al=on[vi]?1:.28;
     wx.strokeStyle='#242732';wx.lineWidth=1.3;wx.globalAlpha=al;wx.beginPath();wx.arc(cx,cy,rad,0,7);wx.stroke();
     for(let sd=1;sd<=12;sd++){if(v.motif[sd-1]<0.15)continue;
       const a=ang(sd),x=cx+rad*Math.cos(a),y=cy+rad*Math.sin(a);
       const near=on[vi]&&v.onsets.some(o=>o.sd===sd&&Math.abs(tt-o.t)<0.09);
       wx.beginPath();wx.fillStyle=VC[vi];wx.globalAlpha=near?1:.5*al;
       wx.arc(x,y,near?11:5+5*v.motif[sd-1],0,7);wx.fill();
       if(near){wx.globalAlpha=1;wx.strokeStyle=VC[vi];wx.lineWidth=2;wx.beginPath();wx.arc(x,y,15,0,7);wx.stroke();}}
     wx.globalAlpha=al;wx.fillStyle='#6a7186';wx.font='10px system-ui';wx.textAlign='center';wx.fillText(v.label,cx,cy-rad+13);
   });wx.globalAlpha=1;
   const pa=progress(tt);wx.strokeStyle='#fff';wx.lineWidth=2;wx.beginPath();
   wx.moveTo(cx,cy);wx.lineTo(cx+radii[0]*Math.cos(pa),cy+radii[0]*Math.sin(pa));wx.stroke();
   wx.fillStyle='#fff';wx.beginPath();wx.arc(cx,cy,3.5,0,7);wx.fill();}
 function drawTL(t){const w=tl.width,h=tl.height,pad=8,uw=(w-2*pad)/R.dur,tt=t%R.dur;tx.clearRect(0,0,w,h);
   R.voices.forEach((v,vi)=>{const yc=h*(vi+1)/(R.voices.length+1),al=on[vi]?1:.3;
     tx.globalAlpha=al;tx.strokeStyle='#20232d';tx.beginPath();tx.moveTo(pad,yc);tx.lineTo(w-pad,yc);tx.stroke();
     tx.fillStyle='#6a7186';tx.font='9px system-ui';tx.textAlign='left';tx.fillText(v.label,2,yc-14);
     v.onsets.forEach(o=>{const x=pad+o.t*uw,nn=on[vi]&&Math.abs(tt-o.t)<0.09;
       tx.fillStyle=VC[vi];tx.globalAlpha=nn?1:.5*al;const hh=nn?26:16;tx.fillRect(x-2,yc-hh/2,4,hh);});
     tx.globalAlpha=1;});
   const cxp=pad+tt*uw;tx.strokeStyle='#fff';tx.lineWidth=1.4;tx.beginPath();tx.moveTo(cxp,4);tx.lineTo(cxp,h-4);tx.stroke();}

 let ctx=null,bufs=null,gains=[],srcs=[],startT=0,raf=null,playing=false;
 async function ensure(){if(bufs)return;ctx=new (window.AudioContext||window.webkitAudioContext)();
   bufs=await Promise.all(R.voices.map(v=>ctx.decodeAudioData(b64buf(v.audio))));}
 async function play(){DATA.forEach((_,j)=>{if(j!==idx&&window['stop'+j])window['stop'+j]();});
   await ensure();if(ctx.state==='suspended')await ctx.resume();
   srcs=[];gains=[];startT=ctx.currentTime+0.03;
   R.voices.forEach((v,vi)=>{const s=ctx.createBufferSource();s.buffer=bufs[vi];s.loop=true;
     const g=ctx.createGain();g.gain.value=on[vi]?1:0;s.connect(g).connect(ctx.destination);
     s.start(startT);srcs.push(s);gains.push(g);});
   playing=true;btn.textContent='❚❚ Pause';btn.classList.add('pause');frame();}
 function stop(){if(srcs)srcs.forEach(s=>{try{s.stop();}catch(e){}});srcs=[];playing=false;
   btn.textContent='▶ Jouer';btn.classList.remove('pause');if(raf)cancelAnimationFrame(raf);}
 window['stop'+idx]=stop;
 function frame(){const t=playing?(ctx.currentTime-startT):0;drawWheel(t);drawTL(t);
   meta.textContent=((t%R.dur)).toFixed(1)+' / '+R.dur.toFixed(0)+' s';raf=requestAnimationFrame(frame);}
 btn.onclick=()=>playing?stop():play();cv.onclick=()=>playing?stop():play();
 drawWheel(0);drawTL(0);
});
</script></body></html>"""


if __name__ == "__main__":
    path, n = build()
    print(f"OK -> {path} ({n} frappes, {os.path.getsize(path)/1e6:.1f} Mo)")
