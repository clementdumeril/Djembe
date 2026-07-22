"""
Génère output/atome_3d_V2.html — la « tour 3D » du dunun.
Chaque cycle est empilé en hauteur (le temps monte), l'angle = position dans le cycle,
le rayon = micro-timing (avance→intérieur / retard→extérieur). D1 et D2 en couleurs.
Rotation 3D libre (souris) + rotation auto, tête de lecture qui monte la tour au tempo réel,
sélection des dununs (audio séparé, Web Audio), clic sur le fond = play/pause.
Projection 3D faite main en Canvas (aucune librairie externe). 07b via 08_atome_3d_V2.ipynb.
"""
import base64, io, json, os
import numpy as np, pandas as pd, librosa, soundfile as sf
import utils_V2 as u

CLIP_SR = 16000
T0, W = 30.0, 22.0            # fenêtre (s) -> hauteur de la tour
VC = ["#4a8cff", "#2fbf6e"]   # D1, D2


def _clip(piece, take, code):
    p = f"{u.CORPUS}/MJ_{piece}/MJ_{piece}_{take}_{code}.wav"
    if not os.path.exists(p):
        return None
    y = librosa.load(p, sr=CLIP_SR, mono=True)[0]
    return y[int(T0 * CLIP_SR):int((T0 + W) * CLIP_SR)]


def _b64(clip):
    buf = io.BytesIO(); sf.write(buf, clip, CLIP_SR, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def build(out_html="output/atome_3d_V2.html"):
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
            sub = df[(df["piece"] == piece) & (df["take"] == take) & (df["code"] == code) &
                     (df["time"] >= T0) & (df["time"] < T0 + W)]
            pts = [{"a": round((r.SD - 1) / 12 * 2 * np.pi, 3), "dev": round(float(r.dev), 1),
                    "z": round(float(r.time - T0), 3), "sd": int(r.SD)} for r in sub.itertuples()]
            voices.append({"label": code, "points": pts, "audio": _b64(clips[code] / norm)})

        met = u.load_metre(piece, take)
        cyc = [round(float(t - T0), 3) for t in met["Time"].values if T0 <= t < T0 + W] if met is not None else []
        rhythms.append({"name": piece, "take": take, "dur": round(W, 2), "cycles": cyc, "voices": voices})

    html = _TEMPLATE.replace("__DATA__", json.dumps(rhythms)).replace("__VC__", json.dumps(VC))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html, sum(len(v["points"]) for r in rhythms for v in r["voices"])


_TEMPLATE = r"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<title>La tour 3D du dunun</title><style>
 :root{--bg:#0a0b0f;--card:#13151c;--txt:#eef;--dim:#8b93a7}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);
   font-family:system-ui,-apple-system,Segoe UI,sans-serif}
 header{padding:22px 26px 6px}h1{margin:0;font-size:1.35rem}
 p.sub{color:var(--dim);margin:.35em 0 0;font-size:.9rem;max-width:82ch}
 .row{display:flex;gap:16px;flex-wrap:wrap;padding:14px 22px 34px}
 .card{background:var(--card);border-radius:16px;padding:14px;flex:1;min-width:320px;box-shadow:0 10px 40px rgba(0,0,0,.45)}
 .card h2{margin:.1em 0 .4em;font-size:1rem}.card h2 small{color:var(--dim);font-weight:400;font-size:.72rem}
 .toggles{display:flex;gap:8px;margin-bottom:6px}
 .tog{display:flex;align-items:center;gap:7px;border:1px solid #262a35;border-radius:999px;padding:4px 11px;
   cursor:pointer;font-size:.8rem;font-weight:600;user-select:none;background:#0e1017}.tog .sw{width:10px;height:10px;border-radius:50%}.tog.off{opacity:.4}
 canvas.d3{display:block;margin:0 auto;cursor:grab;touch-action:none}canvas.d3:active{cursor:grabbing}
 .ctl{display:flex;align-items:center;gap:10px;margin-top:8px}
 button.play{background:#4a8cff;color:#001;border:0;border-radius:8px;padding:7px 15px;font-weight:650;cursor:pointer}
 button.play.pause{background:#e0952e}.meta{font-size:.78rem;color:var(--dim);font-variant-numeric:tabular-nums}
</style></head><body>
<header><h1>🗼 La tour 3D du dunun</h1>
<p class="sub">Le temps monte : chaque <b>cycle</b> est un étage empilé. <b>Angle</b> = position dans le cycle, <b>rayon</b> = micro-timing (avance→intérieur, retard→extérieur). D1 (bleu) / D2 (vert). <b>Tourne à la souris</b> (glisser), la tête de lecture monte la tour au tempo réel et les frappes s'allument. Coche les dununs à entendre ; clique le fond pour jouer/pauser.</p></header>
<div class="row" id="row"></div>
<script>
const DATA=__DATA__;const VC=__VC__;const BASE=[70,44],SC=0.62,ZS=14,F=430;
const row=document.getElementById('row');
function b64buf(b){const s=atob(b);const u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u.buffer;}
DATA.forEach((R,idx)=>{
 const card=document.createElement('div');card.className='card';
 card.innerHTML=`<h2>${R.name} <small>prise ${R.take} · glisser pour tourner</small></h2>`;
 const tg=document.createElement('div');tg.className='toggles';const on=R.voices.map(()=>true);
 R.voices.forEach((v,vi)=>{const el=document.createElement('div');el.className='tog';
   el.innerHTML=`<span class="sw" style="background:${VC[vi]}"></span>${v.label}`;
   el.onclick=()=>{on[vi]=!on[vi];el.classList.toggle('off',!on[vi]);if(gains[vi])gains[vi].gain.value=on[vi]?1:0;};
   tg.appendChild(el);});
 const S=340,H=380,cv=document.createElement('canvas');cv.width=S;cv.height=H;cv.className='d3';
 const ctl=document.createElement('div');ctl.className='ctl';
 const btn=document.createElement('button');btn.className='play';btn.textContent='▶ Jouer';
 const meta=document.createElement('span');meta.className='meta';ctl.appendChild(btn);ctl.appendChild(meta);
 card.appendChild(tg);card.appendChild(cv);card.appendChild(ctl);row.appendChild(card);
 const g=cv.getContext('2d'),cx=S/2,cys=H/2;
 let yaw=0.6,tilt=0.5,dragging=false,lx=0,ly=0,autoYaw=true;
 const Hz=R.dur*ZS;   // hauteur totale
 function proj(X,Y,Z){ // Y = hauteur ; rotation yaw autour de Y, puis tilt autour de X
   const cx1=Math.cos(yaw),sx1=Math.sin(yaw);let x=X*cx1+Z*sx1,z=-X*sx1+Z*cx1,y=Y;
   const ct=Math.cos(tilt),st=Math.sin(tilt);let y2=y*ct-z*st,z2=y*st+z*ct;
   const sc=F/(F+z2+260);return [cx+x*sc,cys-y2*sc,z2,sc];}
 function allPts(t){const arr=[];R.voices.forEach((v,vi)=>{if(!on[vi]&&false)return;
   v.points.forEach(p=>{const r=BASE[vi]+SC*p.dev;const X=r*Math.cos(p.a),Z=r*Math.sin(p.a);
     const Y=p.z*ZS-Hz/2;const pr=proj(X,Y,Z);
     arr.push({sx:pr[0],sy:pr[1],depth:pr[2],sc:pr[3],vi,near:Math.abs(t-p.z)<0.10,vis:on[vi]});});});
   return arr;}
 function draw(t){g.clearRect(0,0,S,H);
   // axe central + cerclages de reperes (bas/milieu/haut)
   const a0=proj(0,-Hz/2,0),a1=proj(0,Hz/2,0);
   g.strokeStyle='#1a1d26';g.lineWidth=1;g.beginPath();g.moveTo(a0[0],a0[1]);g.lineTo(a1[0],a1[1]);g.stroke();
   // plan de lecture (hauteur courante)
   const pts=allPts(t).sort((p,q)=>p.depth-q.depth);
   const yPlay=(t*ZS-Hz/2);
   for(const p of pts){if(!p.vis)continue;
     g.globalAlpha=Math.max(.15,Math.min(1,(p.sc-0.5)*1.4))*(p.near?1:.8);
     g.fillStyle=p.near?'#fff':VC[p.vi];
     const rr=(p.near?4.5:2.6)*p.sc;g.beginPath();g.arc(p.sx,p.sy,rr,0,7);g.fill();
     if(p.near){g.globalAlpha=.8;g.strokeStyle=VC[p.vi];g.lineWidth=1.5;g.beginPath();g.arc(p.sx,p.sy,rr+5,0,7);g.stroke();}}
   g.globalAlpha=1;
   // anneau tete de lecture
   g.strokeStyle='#ffffff';g.globalAlpha=.5;g.beginPath();
   for(let k=0;k<=48;k++){const a=k/48*2*Math.PI;const pr=proj(90*Math.cos(a),yPlay,90*Math.sin(a));
     if(k===0)g.moveTo(pr[0],pr[1]);else g.lineTo(pr[0],pr[1]);}g.stroke();g.globalAlpha=1;}
 // interaction souris
 cv.onpointerdown=e=>{dragging=true;autoYaw=false;lx=e.offsetX;ly=e.offsetY;cv.setPointerCapture(e.pointerId);};
 cv.onpointermove=e=>{if(!dragging)return;yaw+=(e.offsetX-lx)*0.01;tilt=Math.max(0.05,Math.min(1.2,tilt+(e.offsetY-ly)*0.005));lx=e.offsetX;ly=e.offsetY;if(!playing)draw(curT);};
 cv.onpointerup=e=>{dragging=false;};
 let ctx=null,bufs=null,gains=[],srcs=[],startT=0,raf=null,playing=false,curT=0;
 async function ensure(){if(bufs)return;ctx=new (window.AudioContext||window.webkitAudioContext)();
   bufs=await Promise.all(R.voices.map(v=>ctx.decodeAudioData(b64buf(v.audio))));}
 async function play(){DATA.forEach((_,j)=>{if(j!==idx&&window['stop3'+j])window['stop3'+j]();});
   await ensure();if(ctx.state==='suspended')await ctx.resume();srcs=[];gains=[];startT=ctx.currentTime+0.03;
   R.voices.forEach((v,vi)=>{const s=ctx.createBufferSource();s.buffer=bufs[vi];s.loop=true;const gg=ctx.createGain();
     gg.gain.value=on[vi]?1:0;s.connect(gg).connect(ctx.destination);s.start(startT);srcs.push(s);gains.push(gg);});
   playing=true;btn.textContent='❚❚ Pause';btn.classList.add('pause');frame();}
 function stop(){if(srcs)srcs.forEach(s=>{try{s.stop();}catch(e){}});srcs=[];playing=false;
   btn.textContent='▶ Jouer';btn.classList.remove('pause');if(raf)cancelAnimationFrame(raf);}
 window['stop3'+idx]=stop;
 function frame(){curT=playing?((ctx.currentTime-startT)%R.dur):0;if(autoYaw)yaw+=0.004;draw(curT);
   meta.textContent=curT.toFixed(1)+' / '+R.dur.toFixed(0)+' s';raf=requestAnimationFrame(frame);}
 btn.onclick=()=>playing?stop():play();
 // rotation auto au repos
 (function idle(){if(!playing){if(autoYaw)yaw+=0.003;draw(curT);}requestAnimationFrame(idle);})();
});
</script></body></html>"""


if __name__ == "__main__":
    path, n = build()
    print(f"OK -> {path} ({n} points, {os.path.getsize(path)/1e6:.1f} Mo)")
