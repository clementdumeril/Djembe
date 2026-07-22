"""Génère output/explain_fourier.html — explicatif interactif :
un coefficient de Fourier = la somme des petites flèches (une par frappe)."""
import json
import os
import numpy as np
import utils_V2 as u


def motif(p, dd=None):
    if dd is None:
        dd = u.list_dunun()
    vals = []
    for t in dd.query('piece==@p and code=="D1"')["take"].unique():
        pr = u.subdivision_profile(p, int(t), "D1")
        if pr is not None:
            vals.append(pr.values)
    if not vals:
        return [0.0] * 12
    return [round(float(x), 3) for x in np.mean(vals, axis=0)]


HTML = r"""<style>
 :root{--bg:#f4f1ec;--card:#fff;--txt:#211d17;--dim:#6c655b;--bd:#e2ddd3;--accent:#b0742a;--arrow:#4a8cff;--sum:#e0952e}
 @media(prefers-color-scheme:dark){:root{--bg:#12131a;--card:#1b1d26;--txt:#ece7dd;--dim:#9a9182;--bd:#2b2f3a;--accent:#e0a94a;--arrow:#5b8ff0;--sum:#e8a94a}}
 :root[data-theme="dark"]{--bg:#12131a;--card:#1b1d26;--txt:#ece7dd;--dim:#9a9182;--bd:#2b2f3a;--accent:#e0a94a;--arrow:#5b8ff0;--sum:#e8a94a}
 :root[data-theme="light"]{--bg:#f4f1ec;--card:#fff;--txt:#211d17;--dim:#6c655b;--bd:#e2ddd3;--accent:#b0742a;--arrow:#4a8cff;--sum:#e0952e}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font-family:system-ui,sans-serif;line-height:1.55}
 .wrap{max-width:960px;margin:0 auto;padding:26px 18px}
 h1{font-size:1.4rem;margin:0 0 .2em}p.sub{color:var(--dim);font-size:.92rem;margin:.3em 0 1.2em;max-width:72ch}
 .ctrls{display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin-bottom:14px}
 .grp{display:flex;gap:6px;align-items:center}.grp b{font-size:.8rem;color:var(--dim);margin-right:4px}
 button{border:1px solid var(--bd);background:var(--card);color:var(--txt);border-radius:8px;padding:6px 12px;cursor:pointer;font-weight:600;font-size:.85rem}
 button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
 .stage{display:flex;gap:16px;flex-wrap:wrap}
 .panel{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:14px;flex:1;min-width:300px}
 .panel h3{margin:.1em 0 .5em;font-size:.95rem}.panel .cap{font-size:.8rem;color:var(--dim)}
 canvas{width:100%;height:auto;display:block}
 .verdict{margin-top:14px;background:var(--card);border:1px solid var(--bd);border-left:4px solid var(--accent);border-radius:12px;padding:14px 18px;font-size:.95rem}
 .bars{display:flex;gap:8px;align-items:flex-end;height:90px;margin-top:10px}
 .bar{flex:1;background:var(--arrow);border-radius:4px 4px 0 0;position:relative;opacity:.5;transition:opacity .2s}
 .bar.on{opacity:1;background:var(--sum)}.bar span{position:absolute;bottom:-18px;left:0;right:0;text-align:center;font-size:.72rem;color:var(--dim)}
</style>
<div class="wrap">
 <h1>🧭 Un coefficient de Fourier = la somme des flèches</h1>
 <p class="sub">Chaque frappe devient une <b>petite flèche</b>. On les additionne bout à bout. Si elles s'alignent → grande flèche = <b>gros coefficient</b> (le rythme est régulier à ce niveau). Si elles s'éparpillent → elles s'annulent → <b>coefficient ≈ 0</b>. Choisis un rythme et un niveau <b>n</b>.</p>
 <div class="ctrls">
   <div class="grp" id="rgrp"><b>rythme</b></div>
   <div class="grp" id="ngrp"><b>n =</b></div>
 </div>
 <div class="stage">
   <div class="panel"><h3>Le rythme sur le cycle</h3><canvas id="clock" width="340" height="340"></canvas>
     <div class="cap">Les points = les frappes du dunun (taille = fréquence).</div></div>
   <div class="panel"><h3>Les flèches s'additionnent → le coefficient</h3><canvas id="arr" width="340" height="340"></canvas>
     <div class="cap">Bleu = une flèche par frappe (tournée n fois). Orange = leur somme = |Mₙ|.</div></div>
 </div>
 <div class="verdict" id="verdict"></div>
 <div class="panel" style="margin-top:16px"><h3>L'empreinte : régularité pour chaque n</h3>
   <div class="bars" id="bars"></div></div>
</div>
<script>
const MOT=__MOTIFS__;const PIECES=Object.keys(MOT);
let piece=PIECES[0],n=3,anim=0,raf=null;
const TAU=2*Math.PI;
function sumv(m){let s=0;for(const v of m)s+=v;return s;}
function coef(m,nn){let re=0,im=0,arr=[];for(let k=0;k<12;k++){if(m[k]<0.05)continue;
  const a=-nn*(k/12)*TAU;re+=m[k]*Math.cos(a);im+=m[k]*Math.sin(a);arr.push({a,len:m[k]});}
  return {re,im,mag:Math.hypot(re,im),arr};}
// controls
const rg=document.getElementById('rgrp'),ng=document.getElementById('ngrp');
PIECES.forEach(p=>{const b=document.createElement('button');b.textContent=p;b.onclick=()=>{piece=p;refresh();};b.dataset.r=p;rg.appendChild(b);});
for(let i=1;i<=6;i++){const b=document.createElement('button');b.textContent=i;b.onclick=()=>{n=i;refresh();};b.dataset.n=i;ng.appendChild(b);}
function setActive(){[...rg.querySelectorAll('button')].forEach(b=>b.classList.toggle('on',b.dataset.r===piece));
  [...ng.querySelectorAll('button')].forEach(b=>b.classList.toggle('on',+b.dataset.n===n));}
// clock
function drawClock(){const c=document.getElementById('clock'),g=c.getContext('2d'),W=c.width,cx=W/2,cy=W/2,R=W/2-30;
  g.clearRect(0,0,W,W);const m=MOT[piece];
  g.strokeStyle=css('--bd');g.lineWidth=1;g.beginPath();g.arc(cx,cy,R,0,TAU);g.stroke();
  for(let k=0;k<12;k++){const a=k/12*TAU-Math.PI/2;const x=cx+R*Math.cos(a),y=cy+R*Math.sin(a);
    g.fillStyle=css('--dim');g.font='11px system-ui';g.textAlign='center';g.textBaseline='middle';
    g.fillText(k+1,cx+(R+16)*Math.cos(a),cy+(R+16)*Math.sin(a));
    if(m[k]>=0.05){g.fillStyle=css('--arrow');g.globalAlpha=.85;g.beginPath();g.arc(x,y,4+9*m[k],0,TAU);g.fill();g.globalAlpha=1;}}}
// arrows
function drawArrows(){const c=document.getElementById('arr'),g=c.getContext('2d'),W=c.width,cx=W/2,cy=W/2;
  const m=MOT[piece],C=coef(m,n),S=sumv(m),R=W/2-24,scale=R/S;
  g.clearRect(0,0,W,W);
  g.strokeStyle=css('--bd');g.setLineDash([3,4]);g.beginPath();g.arc(cx,cy,R,0,TAU);g.stroke();g.setLineDash([]);
  const shown=Math.max(1,Math.round(anim*C.arr.length));
  let x=cx,y=cy;
  for(let i=0;i<Math.min(shown,C.arr.length);i++){const ar=C.arr[i];
    const dx=ar.len*Math.cos(ar.a)*scale,dy=-ar.len*Math.sin(ar.a)*scale;
    arrow(g,x,y,x+dx,y+dy,css('--arrow'),1.6);x+=dx;y+=dy;}
  if(shown>=C.arr.length){arrow(g,cx,cy,x,y,css('--sum'),3);
    g.fillStyle=css('--sum');g.font='bold 13px system-ui';g.textAlign='left';
    g.fillText('|M'+n+'| = '+(C.mag/S).toFixed(2),12,18);}}
function arrow(g,x1,y1,x2,y2,col,w){g.strokeStyle=col;g.fillStyle=col;g.lineWidth=w;
  g.beginPath();g.moveTo(x1,y1);g.lineTo(x2,y2);g.stroke();
  const a=Math.atan2(y2-y1,x2-x1),h=6+w;g.beginPath();g.moveTo(x2,y2);
  g.lineTo(x2-h*Math.cos(a-.4),y2-h*Math.sin(a-.4));g.lineTo(x2-h*Math.cos(a+.4),y2-h*Math.sin(a+.4));g.closePath();g.fill();}
function drawBars(){const m=MOT[piece],S=sumv(m),bars=document.getElementById('bars');bars.innerHTML='';
  for(let i=1;i<=6;i++){const v=coef(m,i).mag/S;const d=document.createElement('div');d.className='bar'+(i===n?' on':'');
    d.style.height=(8+v*80)+'px';d.innerHTML='<span>n='+i+'</span>';d.onclick=()=>{n=i;refresh();};bars.appendChild(d);}}
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v)||'#888';}
function verdict(){const m=MOT[piece],S=sumv(m),v=coef(m,n).mag/S;const el=document.getElementById('verdict');
  let t;if(v>0.6)t='<b>Fort.</b> Les flèches s\'alignent → le rythme <b>'+piece+'</b> est nettement régulier en <b>'+n+'</b> coups par cycle.';
  else if(v>0.3)t='<b>Moyen.</b> Les flèches se compensent en partie → régularité modérée en <b>'+n+'</b>.';
  else t='<b>Faible.</b> Les flèches s\'annulent → le rythme n\'est presque pas structuré en <b>'+n+'</b>.';
  el.innerHTML=t+' &nbsp;(coefficient normalisé = '+v.toFixed(2)+', entre 0 et 1)';}
function refresh(){setActive();drawClock();drawBars();verdict();anim=0;if(raf)cancelAnimationFrame(raf);
  (function step(){anim=Math.min(1,anim+0.045);drawArrows();if(anim<1)raf=requestAnimationFrame(step);})();}
new MutationObserver(()=>{drawClock();drawArrows();drawBars();}).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
refresh();
</script>"""


def build(out_html="output/explain_fourier.html"):
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    dd = u.list_dunun()
    motifs = {p: motif(p, dd) for p in u.PIECES}
    html = HTML.replace("__MOTIFS__", json.dumps(motifs))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html, len(html)


if __name__ == "__main__":
    path, n = build()
    print("OK ->", path, n // 1000, "Ko")
