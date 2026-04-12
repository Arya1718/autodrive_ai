import os
import streamlit as st
import json
import time
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AutoDrive AI — Self-Driving Car Platform",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── DARK THEME CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
  html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #0b0f1a !important; color: #c8d0e0;
  }
  [data-testid="stSidebar"] { background: #0e1420 !important; }
  .block-container { padding: 1rem 1.5rem 2rem !important; }
  h1,h2,h3 { color: #e2e8f0 !important; font-family: 'JetBrains Mono', monospace !important; }
  .stButton > button {
    background: #1e4d6e; border: 1px solid #2d7ab5; color: #a8d4f5;
    font-family: 'JetBrains Mono', monospace; font-size: 13px;
    border-radius: 6px; padding: 0.5rem 1rem; width: 100%;
    transition: all 0.2s;
  }
  .stButton > button:hover { background: #2d6fa3; border-color: #4da3e0; color: #fff; }
  .stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background: #141c2e !important; color: #c8d0e0 !important;
    border: 1px solid #2a3a55 !important; font-family: 'JetBrains Mono', monospace;
    border-radius: 6px;
  }
  .stSelectbox > div > div { background: #141c2e !important; border: 1px solid #2a3a55 !important; }
  .metric-card {
    background: #111827; border: 1px solid #1f2d45; border-radius: 8px;
    padding: 14px 16px; margin: 6px 0;
  }
  .metric-label { font-size: 11px; color: #6b7fa3; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { font-size: 22px; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #38bdf8; }
  .ai-card {
    background: #0d1626; border: 1px solid #1e3a5a; border-radius: 8px;
    padding: 14px; margin: 8px 0; font-size: 13px; line-height: 1.6;
  }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-family: 'JetBrains Mono',monospace; margin: 2px; }
  .tag-green { background: #0f2d1f; border: 1px solid #16a34a; color: #4ade80; }
  .tag-red { background: #2d0f0f; border: 1px solid #dc2626; color: #f87171; }
  .tag-yellow { background: #2d260f; border: 1px solid #ca8a04; color: #fbbf24; }
  .tag-blue { background: #0f1e2d; border: 1px solid #2563eb; color: #60a5fa; }
  .section-header { color: #64748b; font-size: 11px; font-family: 'JetBrains Mono',monospace; text-transform: uppercase; letter-spacing: 1.5px; margin: 12px 0 6px; border-bottom: 1px solid #1f2d45; padding-bottom: 4px; }
  .stSlider > div > div > div { background: #1e4d6e !important; }
  div[data-testid="stExpander"] { background: #111827 !important; border: 1px solid #1f2d45 !important; border-radius: 8px; }
  .stAlert { background: #0d1626 !important; border-color: #1e3a5a !important; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────────────────────
for k, v in {
  "env_config": {
        "obstacle_count": 8, "car_speed": 2.5, "environment_name": "Urban Circuit",
        "weather": "clear", "track_complexity": "moderate", "fog_alpha": 0.0,
        "description": "Standard urban circuit. Navigate to destination avoiding obstacles."
    },
    "ai_analysis": "", "last_crash": None, "env_generated": False,
    "total_score": 0, "total_crashes": 0, "episodes": 0, "agent_initialized": False
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── SIMULATION HTML ─────────────────────────────────────────────────────────
def build_simulation(cfg: dict) -> str:
    cfg_json = json.dumps(cfg)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#0b0f1a;display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;overflow:hidden}}
  canvas{{border:2px solid #1e3a5a;border-radius:8px;display:block}}
  #status{{color:#64748b;font-family:'JetBrains Mono',monospace;font-size:11px;margin-top:6px;text-align:center}}
</style>
</head>
<body>
<canvas id="sim"></canvas>
<div id="status">⬤ SIMULATION RUNNING — AI AGENT ACTIVE</div>
<script>
const CFG = {cfg_json};
const canvas = document.getElementById('sim');
const ctx = canvas.getContext('2d');
const W = 860, H = 520;
canvas.width = W; canvas.height = H;

// ── Track Control Points (oval) ──
const TP = [
  {{x:90, y:260}},{{x:160,y:105}},{{x:330,y:68}},{{x:530,y:68}},
  {{x:710,y:105}},{{x:780,y:260}},{{x:710,y:415}},{{x:530,y:452}},
  {{x:330,y:452}},{{x:160,y:415}}
];
const TW = 78;

function catmull(p0,p1,p2,p3,t){{
  const t2=t*t,t3=t2*t;
  return{{
    x:0.5*((2*p1.x)+(-p0.x+p2.x)*t+(2*p0.x-5*p1.x+4*p2.x-p3.x)*t2+(-p0.x+3*p1.x-3*p2.x+p3.x)*t3),
    y:0.5*((2*p1.y)+(-p0.y+p2.y)*t+(2*p0.y-5*p1.y+4*p2.y-p3.y)*t2+(-p0.y+3*p1.y-3*p2.y+p3.y)*t3)
  }};
}}
function smoothTrack(pts, seg=24){{
  const R=[]; const n=pts.length;
  for(let i=0;i<n;i++){{
    const p0=pts[(i-1+n)%n],p1=pts[i],p2=pts[(i+1)%n],p3=pts[(i+2)%n];
    for(let t=0;t<1;t+=1/seg) R.push(catmull(p0,p1,p2,p3,t));
  }}
  return R;
}}
const TRACK = smoothTrack(TP, 24);
const DEST_IDX = Math.floor(TRACK.length/2);

// Spatial grid for fast road check
const GRID_SZ = 30;
const roadGrid = new Set();
TRACK.forEach(p=>{{
  const gx=Math.floor(p.x/GRID_SZ), gy=Math.floor(p.y/GRID_SZ);
  roadGrid.add(gx+','+gy);
  roadGrid.add((gx-1)+','+(gy-1)); roadGrid.add(gx+','+(gy-1));
  roadGrid.add((gx+1)+','+(gy-1)); roadGrid.add((gx-1)+','+gy);
  roadGrid.add((gx+1)+','+gy); roadGrid.add((gx-1)+','+(gy+1));
  roadGrid.add(gx+','+(gy+1)); roadGrid.add((gx+1)+','+(gy+1));
}});

function isOnRoad(x,y){{
  const gx=Math.floor(x/GRID_SZ), gy=Math.floor(y/GRID_SZ);
  if(!roadGrid.has(gx+','+gy)) return false;
  for(const p of TRACK){{
    const dx=x-p.x,dy=y-p.y;
    if(dx*dx+dy*dy<(TW/2)*(TW/2)) return true;
  }}
  return false;
}}

// ── Obstacles ──
let obstacles=[];
function placeObstacles(){{
  obstacles=[];
  let attempts=0;
  const count=Math.min(20,Math.max(2, CFG.obstacle_count||8));
  while(obstacles.length<count && attempts<2000){{
    attempts++;
    const tidx=Math.floor(Math.random()*TRACK.length);
    const p=TRACK[tidx];
    const next=TRACK[(tidx+1)%TRACK.length];
    const prev=TRACK[(tidx-1+TRACK.length)%TRACK.length];
    const tx=next.x-prev.x,ty=next.y-prev.y;
    const len=Math.sqrt(tx*tx+ty*ty)||1;
    const side=(Math.random()>0.5?1:-1)*(14+Math.random()*18);
    const ox=p.x+(-ty/len)*side, oy=p.y+(tx/len)*side;
    const startPt=TRACK[0];
    const dStart=Math.sqrt((ox-startPt.x)**2+(oy-startPt.y)**2);
    if(dStart<90) continue;
    if(!isOnRoad(ox,oy)) continue;
    let tooClose=false;
    for(const o of obstacles){{
      if(Math.sqrt((ox-o.x)**2+(oy-o.y)**2)<22){{tooClose=true;break;}}
    }}
    if(tooClose) continue;
    obstacles.push({{x:ox,y:oy,r:11,type:Math.random()>0.45?'cone':'barrel',angle:Math.random()*Math.PI*2}});
  }}
}}

// ── Car ──
const CAR = {{
  x:TRACK[0].x, y:TRACK[0].y,
  angle:Math.atan2(TRACK[1].y-TRACK[0].y,TRACK[1].x-TRACK[0].x),
  speed:0, steer:0, alive:true,
  sensors:new Array(7).fill(1.0),
  trail:[], targetIdx:5,
  score:0, laps:0, episodes:0, totalCrashes:0
}};

// ── Sensor Raycasting ──
function castRay(fx,fy,angle,maxD=140){{
  const stepSize=4;
  for(let d=8;d<=maxD;d+=stepSize){{
    const rx=fx+Math.cos(angle)*d, ry=fy+Math.sin(angle)*d;
    for(const obs of obstacles){{
      const ddx=rx-obs.x,ddy=ry-obs.y;
      if(ddx*ddx+ddy*ddy<obs.r*obs.r) return d/maxD;
    }}
    if(!isOnRoad(rx,ry)) return d/maxD;
  }}
  return 1.0;
}}
function updateSensors(){{
  const angles=[-90,-50,-25,0,25,50,90];
  CAR.sensors=angles.map(a=>castRay(CAR.x,CAR.y,CAR.angle+a*Math.PI/180));
}}

// ── Agent Logic ──
function agentStep(){{
  if(!CAR.alive) return;
  // Advance target waypoint
  for(let i=0;i<TRACK.length;i++){{
    const idx=(CAR.targetIdx+i)%TRACK.length;
    const dx=TRACK[idx].x-CAR.x, dy=TRACK[idx].y-CAR.y;
    if(Math.sqrt(dx*dx+dy*dy)<28){{
      CAR.targetIdx=(idx+1)%TRACK.length;
      CAR.score+=2;
      if(CAR.targetIdx===0){{CAR.laps++;CAR.score+=200;}}
    }}
    if(i>5 && Math.sqrt((TRACK[idx].x-CAR.x)**2+(TRACK[idx].y-CAR.y)**2)>40) break;
  }}
  const tgt=TRACK[CAR.targetIdx];
  const dx=tgt.x-CAR.x, dy=tgt.y-CAR.y;
  let targetAngle=Math.atan2(dy,dx);
  let angleDiff=targetAngle-CAR.angle;
  while(angleDiff>Math.PI) angleDiff-=2*Math.PI;
  while(angleDiff<-Math.PI) angleDiff+=2*Math.PI;

  const [s0,s1,s2,s3,s4,s5,s6]=CAR.sensors;
  let avoid=0;
  if(s3<0.32){{
    avoid=(s1+s2)<(s4+s5)?1.1:-1.1;
  }} else {{
    if(s0<0.55) avoid+=0.45*(1-s0);
    if(s1<0.55) avoid+=0.32*(1-s1);
    if(s2<0.45) avoid+=0.18*(1-s2);
    if(s4<0.45) avoid-=0.18*(1-s4);
    if(s5<0.55) avoid-=0.32*(1-s5);
    if(s6<0.55) avoid-=0.45*(1-s6);
  }}
  CAR.steer=Math.max(-1,Math.min(1, angleDiff*1.3+avoid*0.85));
  const minS=Math.min(s1,s2,s3,s4,s5);
  const maxSpd=Math.min(4.5,Math.max(1.5,CFG.car_speed||2.5));
  if(minS<0.28) CAR.speed=Math.max(0.4,CAR.speed-0.18);
  else if(minS<0.5) CAR.speed=Math.min(maxSpd*0.55,CAR.speed+0.04);
  else CAR.speed=Math.min(maxSpd,CAR.speed+0.07);
  if(!isOnRoad(CAR.x,CAR.y)) CAR.speed*=0.88;
}}

let particles=[], resetTimer=0, successAnim=0, flashAlpha=0;

function triggerCrash(){{
  if(!CAR.alive) return;
  CAR.alive=false; CAR.totalCrashes++; CAR.episodes++;
  flashAlpha=1;
  for(let i=0;i<35;i++){{
    const a=Math.random()*Math.PI*2,sp=Math.random()*5+1;
    particles.push({{x:CAR.x,y:CAR.y,vx:Math.cos(a)*sp,vy:Math.sin(a)*sp,
      life:1,color:`hsl(${{10+Math.random()*50}},100%,${{50+Math.random()*30}}%)`}});
  }}
  resetTimer=160;
  try{{ window.parent.postMessage(JSON.stringify({{
    type:'crash',x:Math.round(CAR.x),y:Math.round(CAR.y),
    speed:CAR.speed.toFixed(2),score:CAR.score,laps:CAR.laps,
    crashes:CAR.totalCrashes,
    sensor_front:CAR.sensors[3].toFixed(2),
    sensor_left:CAR.sensors[1].toFixed(2),
    sensor_right:CAR.sensors[5].toFixed(2)
  }}),'*'); }}catch(e){{}}
}}

function triggerSuccess(){{
  if(successAnim>0) return;
  successAnim=220; CAR.score+=600;
  for(let i=0;i<50;i++){{
    const a=Math.random()*Math.PI*2,sp=Math.random()*6+2;
    particles.push({{x:CAR.x,y:CAR.y,vx:Math.cos(a)*sp,vy:Math.sin(a)*sp,
      life:1,color:`hsl(${{100+Math.random()*60}},100%,60%)`}});
  }}
}}

function updateCar(){{
  if(!CAR.alive) return;
  CAR.angle+=CAR.steer*CAR.speed*0.046;
  CAR.x+=Math.cos(CAR.angle)*CAR.speed;
  CAR.y+=Math.sin(CAR.angle)*CAR.speed;
  CAR.trail.push({{x:CAR.x,y:CAR.y,spd:CAR.speed}});
  if(CAR.trail.length>90) CAR.trail.shift();
  for(const obs of obstacles){{
    const dx=CAR.x-obs.x,dy=CAR.y-obs.y;
    if(Math.sqrt(dx*dx+dy*dy)<obs.r+8){{triggerCrash();return;}}
  }}
  if(CAR.x<4||CAR.x>W-4||CAR.y<4||CAR.y>H-4){{triggerCrash();return;}}
  const dest=TRACK[DEST_IDX];
  if(Math.sqrt((dest.x-CAR.x)**2+(dest.y-CAR.y)**2)<28) triggerSuccess();
}}

function resetCar(){{
  CAR.x=TRACK[0].x; CAR.y=TRACK[0].y;
  CAR.angle=Math.atan2(TRACK[1].y-TRACK[0].y,TRACK[1].x-TRACK[0].x);
  CAR.speed=0; CAR.steer=0; CAR.alive=true; CAR.trail=[]; CAR.targetIdx=5;
  placeObstacles();
}}

// ── RENDERING ──
function drawGrass(){{
  ctx.fillStyle='#12261a';
  ctx.fillRect(0,0,W,H);
  // Grid lines (subtle)
  ctx.strokeStyle='rgba(30,60,35,0.4)';
  ctx.lineWidth=1;
  for(let x=0;x<W;x+=40){{ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}}
  for(let y=0;y<H;y+=40){{ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}}
}}

function drawRoad(){{
  // Outer shadow
  ctx.save();
  ctx.shadowBlur=18; ctx.shadowColor='rgba(0,0,0,0.7)';
  ctx.strokeStyle='#1e2430'; ctx.lineWidth=TW+10;
  ctx.lineCap='round'; ctx.lineJoin='round';
  ctx.beginPath(); ctx.moveTo(TRACK[0].x,TRACK[0].y);
  for(let i=1;i<TRACK.length;i++) ctx.lineTo(TRACK[i].x,TRACK[i].y);
  ctx.closePath(); ctx.stroke();
  ctx.restore();
  // Asphalt
  ctx.strokeStyle='#2c3340'; ctx.lineWidth=TW;
  ctx.lineCap='round'; ctx.lineJoin='round';
  ctx.beginPath(); ctx.moveTo(TRACK[0].x,TRACK[0].y);
  for(let i=1;i<TRACK.length;i++) ctx.lineTo(TRACK[i].x,TRACK[i].y);
  ctx.closePath(); ctx.stroke();
  // Road edges (white)
  ctx.save();
  for(let sign=-1;sign<=1;sign+=2){{
    const offset=sign*(TW/2-4);
    // Build offset path for edges
    ctx.strokeStyle='rgba(255,255,255,0.55)'; ctx.lineWidth=2;
    ctx.setLineDash([]);
    ctx.beginPath();
    for(let i=0;i<TRACK.length;i++){{
      const curr=TRACK[i],next=TRACK[(i+1)%TRACK.length];
      const tx=next.x-curr.x,ty=next.y-curr.y;
      const len=Math.sqrt(tx*tx+ty*ty)||1;
      const px=-ty/len*offset, py=tx/len*offset;
      if(i===0) ctx.moveTo(curr.x+px,curr.y+py);
      else ctx.lineTo(curr.x+px,curr.y+py);
    }}
    ctx.closePath(); ctx.stroke();
  }}
  ctx.restore();
  // Centre dashed line (yellow)
  ctx.strokeStyle='rgba(255,220,50,0.6)'; ctx.lineWidth=1.5;
  ctx.setLineDash([18,14]);
  ctx.beginPath(); ctx.moveTo(TRACK[0].x,TRACK[0].y);
  for(let i=1;i<TRACK.length;i++) ctx.lineTo(TRACK[i].x,TRACK[i].y);
  ctx.closePath(); ctx.stroke();
  ctx.setLineDash([]);
}}

function drawTrail(){{
  for(let i=1;i<CAR.trail.length;i++){{
    const alpha=(i/CAR.trail.length)*0.45;
    const spd=CAR.trail[i].spd||1;
    const hue=Math.min(120,spd*25);
    ctx.strokeStyle=`hsla(${{hue}},90%,55%,${{alpha}})`;
    ctx.lineWidth=2.5;
    ctx.beginPath();
    ctx.moveTo(CAR.trail[i-1].x,CAR.trail[i-1].y);
    ctx.lineTo(CAR.trail[i].x,CAR.trail[i].y);
    ctx.stroke();
  }}
}}

function drawDestination(){{
  const dest=TRACK[DEST_IDX];
  const t=Date.now()/1000;
  const pulse=(Math.sin(t*3)+1)/2;
  // Outer glow rings
  for(let r=1;r<=3;r++){{
    ctx.strokeStyle=`rgba(0,255,136,${{(0.15+pulse*0.3)*(4-r)/3}})`;
    ctx.lineWidth=2;
    ctx.beginPath();
    ctx.arc(dest.x,dest.y,22+r*8+pulse*6,0,Math.PI*2);
    ctx.stroke();
  }}
  // Core dot
  const grd=ctx.createRadialGradient(dest.x,dest.y,0,dest.x,dest.y,14);
  grd.addColorStop(0,'#00ff88'); grd.addColorStop(1,'rgba(0,200,80,0)');
  ctx.fillStyle=grd;
  ctx.beginPath(); ctx.arc(dest.x,dest.y,14,0,Math.PI*2); ctx.fill();
  // Checkered flag
  const fx=dest.x+2, fy=dest.y-38;
  ctx.strokeStyle='#00ff88'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(fx,fy+28); ctx.lineTo(fx,fy+2); ctx.stroke();
  const sz=5;
  for(let row=0;row<3;row++) for(let col=0;col<3;col++){{
    ctx.fillStyle=((row+col)%2===0)?'#fff':'#000';
    ctx.fillRect(fx+col*sz,fy+2+row*sz,sz,sz);
  }}
  // START label
  const start=TRACK[0];
  ctx.fillStyle='rgba(0,0,0,0.7)'; ctx.fillRect(start.x-18,start.y-18,36,14);
  ctx.fillStyle='#fbbf24'; ctx.font='bold 10px JetBrains Mono,monospace';
  ctx.textAlign='center'; ctx.fillText('START',start.x,start.y-7);
  // GOAL label
  ctx.fillStyle='rgba(0,0,0,0.7)'; ctx.fillRect(dest.x-18,dest.y+20,36,14);
  ctx.fillStyle='#4ade80'; ctx.font='bold 10px JetBrains Mono,monospace';
  ctx.fillText('GOAL',dest.x,dest.y+31);
}}

function drawObstacles(){{
  for(const obs of obstacles){{
    ctx.save();
    ctx.translate(obs.x,obs.y); ctx.rotate(obs.angle||0);
    if(obs.type==='cone'){{
      // Traffic cone: orange with white stripe
      ctx.shadowBlur=6; ctx.shadowColor='rgba(255,100,0,0.5)';
      ctx.fillStyle='#f97316';
      ctx.beginPath(); ctx.arc(0,0,obs.r,0,Math.PI*2); ctx.fill();
      ctx.fillStyle='#fff';
      ctx.fillRect(-obs.r,obs.r*0.05,obs.r*2,obs.r*0.3);
      ctx.fillStyle='rgba(0,0,0,0.3)';
      ctx.beginPath(); ctx.arc(0,0,obs.r,0,Math.PI*2); ctx.stroke();
      // Top pip
      ctx.fillStyle='#fff'; ctx.beginPath(); ctx.arc(0,-obs.r*0.4,2,0,Math.PI*2); ctx.fill();
    }} else {{
      // Barrel: red/black striped
      ctx.shadowBlur=6; ctx.shadowColor='rgba(200,0,0,0.5)';
      ctx.fillStyle='#dc2626';
      ctx.beginPath(); ctx.arc(0,0,obs.r,0,Math.PI*2); ctx.fill();
      // Stripes
      ctx.save(); ctx.beginPath(); ctx.arc(0,0,obs.r,0,Math.PI*2); ctx.clip();
      ctx.fillStyle='#1a0000';
      for(let i=-2;i<3;i++) ctx.fillRect(-obs.r,i*5,obs.r*2,3);
      ctx.restore();
      ctx.strokeStyle='#7f1d1d'; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.arc(0,0,obs.r,0,Math.PI*2); ctx.stroke();
    }}
    ctx.restore();
  }}
}}

function drawCar(){{
  if(!CAR.alive) return;
  const CW=16, CH=28;
  ctx.save();
  ctx.translate(CAR.x,CAR.y); ctx.rotate(CAR.angle);
  // Shadow
  ctx.save(); ctx.translate(3,3);
  ctx.fillStyle='rgba(0,0,0,0.35)';
  ctx.beginPath(); ctx.roundRect(-CW/2,-CH/2,CW,CH,4); ctx.fill();
  ctx.restore();
  // Body
  const bodyGrd=ctx.createLinearGradient(-CW/2,-CH/2,CW/2,CH/2);
  bodyGrd.addColorStop(0,'#38bdf8'); bodyGrd.addColorStop(0.5,'#0ea5e9'); bodyGrd.addColorStop(1,'#0369a1');
  ctx.fillStyle=bodyGrd;
  ctx.beginPath(); ctx.roundRect(-CW/2,-CH/2,CW,CH,4); ctx.fill();
  // Windshield
  ctx.fillStyle='rgba(186,230,253,0.7)';
  ctx.beginPath(); ctx.roundRect(-CW/2+2,-CH/2+5,CW-4,8,2); ctx.fill();
  // Roof
  ctx.fillStyle='#075985';
  ctx.beginPath(); ctx.roundRect(-CW/2+2,-CH/2+5,CW-4,9,2); ctx.fill();
  // Headlights (front = top because angle=0 faces right, but let's do standard top-down)
  ctx.fillStyle=CAR.speed>0.3?'#fef08a':'#713f12';
  ctx.beginPath(); ctx.roundRect(-CW/2,-CH/2,5,3,1); ctx.fill();
  ctx.beginPath(); ctx.roundRect(CW/2-5,-CH/2,5,3,1); ctx.fill();
  // Headlight glow
  if(CAR.speed>0.3){{
    ctx.save();
    ctx.shadowBlur=14; ctx.shadowColor='rgba(255,255,100,0.8)';
    ctx.fillStyle='#fef08a';
    ctx.beginPath(); ctx.arc(-CW/2+2,-CH/2+1,2,0,Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(CW/2-2,-CH/2+1,2,0,Math.PI*2); ctx.fill();
    ctx.restore();
  }}
  // Brake lights
  const braking=CAR.speed<1.0;
  ctx.fillStyle=braking?'#ef4444':'#450a0a';
  ctx.beginPath(); ctx.roundRect(-CW/2,CH/2-4,5,4,1); ctx.fill();
  ctx.beginPath(); ctx.roundRect(CW/2-5,CH/2-4,5,4,1); ctx.fill();
  if(braking){{
    ctx.save(); ctx.shadowBlur=12; ctx.shadowColor='rgba(255,0,0,0.8)';
    ctx.fillStyle='#ef4444';
    ctx.beginPath(); ctx.arc(-CW/2+2,CH/2-2,2,0,Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(CW/2-2,CH/2-2,2,0,Math.PI*2); ctx.fill();
    ctx.restore();
  }}
  // Wheels
  ctx.fillStyle='#1e293b';
  [[-CW/2-1,-CH/2+4],[CW/2-3,-CH/2+4],[-CW/2-1,CH/2-8],[CW/2-3,CH/2-8]].forEach(([wx,wy])=>{{
    ctx.beginPath(); ctx.roundRect(wx,wy,4,8,1); ctx.fill();
  }});
  ctx.restore();
}}

function drawSensors(){{
  if(!CAR.alive) return;
  const angles=[-90,-50,-25,0,25,50,90];
  angles.forEach((a,i)=>{{
    const rad=CAR.angle+a*Math.PI/180;
    const dist=CAR.sensors[i]*140;
    const hue=Math.round(CAR.sensors[i]*120);
    ctx.save();
    ctx.shadowBlur=3; ctx.shadowColor=`hsl(${{hue}},100%,60%)`;
    ctx.strokeStyle=`hsla(${{hue}},100%,60%,0.5)`;
    ctx.lineWidth=1;
    ctx.setLineDash([4,4]);
    ctx.beginPath();
    ctx.moveTo(CAR.x,CAR.y);
    ctx.lineTo(CAR.x+Math.cos(rad)*dist,CAR.y+Math.sin(rad)*dist);
    ctx.stroke();
    // Endpoint dot
    ctx.setLineDash([]);
    ctx.fillStyle=`hsl(${{hue}},100%,70%)`;
    ctx.beginPath();
    ctx.arc(CAR.x+Math.cos(rad)*dist,CAR.y+Math.sin(rad)*dist,2,0,Math.PI*2);
    ctx.fill();
    ctx.restore();
  }});
}}

function drawParticles(){{
  for(let i=particles.length-1;i>=0;i--){{
    const p=particles[i];
    ctx.save(); ctx.globalAlpha=p.life;
    ctx.shadowBlur=6; ctx.shadowColor=p.color;
    ctx.fillStyle=p.color;
    ctx.beginPath(); ctx.arc(p.x,p.y,3.5*p.life,0,Math.PI*2); ctx.fill();
    ctx.restore();
    p.x+=p.vx; p.y+=p.vy; p.vy+=0.12; p.vx*=0.97; p.life-=0.024;
    if(p.life<=0) particles.splice(i,1);
  }}
}}

function drawHUD(){{
  const speedKmh=(CAR.speed*(CFG.car_speed||2.5)*12).toFixed(0);
  const dest=TRACK[DEST_IDX];
  const distToGoal=Math.sqrt((dest.x-CAR.x)**2+(dest.y-CAR.y)**2).toFixed(0);
  // HUD background
  ctx.fillStyle='rgba(11,15,26,0.82)';
  ctx.beginPath(); ctx.roundRect(8,8,188,108,6); ctx.fill();
  ctx.strokeStyle='#1e3a5a'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.roundRect(8,8,188,108,6); ctx.stroke();
  // Content
  ctx.font='bold 11px JetBrains Mono,monospace';
  ctx.textAlign='left';
  const rows=[
    ['SPEED',speedKmh+' km/h','#38bdf8'],
    ['SCORE',CAR.score,'#4ade80'],
    ['LAPS',CAR.laps,'#fbbf24'],
    ['CRASHES',CAR.totalCrashes,'#f87171'],
    ['DIST TO GOAL',distToGoal+'px','#c084fc'],
  ];
  rows.forEach(([label,val,color],i)=>{{
    const y=24+i*18;
    ctx.fillStyle='#475569'; ctx.fillText(label,18,y);
    ctx.fillStyle=color;
    ctx.textAlign='right'; ctx.fillText(val,190,y);
    ctx.textAlign='left';
  }});
  // Env name
  ctx.fillStyle='rgba(11,15,26,0.82)';
  ctx.beginPath(); ctx.roundRect(W-200,8,192,22,4); ctx.fill();
  ctx.fillStyle='#64748b'; ctx.font='10px JetBrains Mono,monospace';
  ctx.textAlign='center';
  ctx.fillText((CFG.environment_name||'Circuit').toUpperCase().substring(0,24),W-104,23);
  ctx.textAlign='left';
}}

function drawFog(){{
  const alpha=CFG.fog_alpha||0;
  if(alpha<=0) return;
  ctx.fillStyle=`rgba(180,200,220,${{alpha}})`;
  ctx.fillRect(0,0,W,H);
}}

function drawCrashOverlay(){{
  ctx.save();
  ctx.fillStyle=`rgba(220,38,38,${{flashAlpha*0.35}})`;
  ctx.fillRect(0,0,W,H);
  const prog=1-(resetTimer/160);
  ctx.fillStyle=`rgba(220,38,38,${{0.9*Math.max(0,1-prog*2)}})`;
  ctx.font=`bold 36px JetBrains Mono,monospace`;
  ctx.textAlign='center';
  ctx.shadowBlur=20; ctx.shadowColor='#ef4444';
  ctx.fillText('COLLISION DETECTED',W/2,H/2-20);
  ctx.shadowBlur=0;
  ctx.fillStyle='rgba(255,255,255,0.7)'; ctx.font='14px JetBrains Mono,monospace';
  ctx.fillText(`Resetting in ${{Math.ceil(resetTimer/60)}}s... (Episode ${{CAR.episodes}})`,W/2,H/2+18);
  ctx.restore();
}}

function drawSuccessOverlay(){{
  if(successAnim<=0) return;
  const a=successAnim/220;
  ctx.save();
  ctx.fillStyle=`rgba(74,222,128,${{a*0.2}})`;
  ctx.fillRect(0,0,W,H);
  ctx.fillStyle=`rgba(74,222,128,${{a}})`;
  ctx.font=`bold 32px JetBrains Mono,monospace`;
  ctx.textAlign='center';
  ctx.shadowBlur=24; ctx.shadowColor='#4ade80';
  ctx.fillText('🏁 DESTINATION REACHED!',W/2,H/2);
  ctx.restore();
  successAnim-=2;
  if(successAnim<=0){{
    setTimeout(()=>resetCar(),800);
  }}
}}

// ── MAIN LOOP ──
let animId;
function loop(){{
  drawGrass();
  drawRoad();
  drawTrail();
  drawDestination();
  drawObstacles();
  updateSensors();
  agentStep();
  updateCar();
  drawSensors();
  drawCar();
  drawParticles();
  drawFog();
  drawHUD();
  if(!CAR.alive){{
    resetTimer--;
    flashAlpha=Math.max(0,flashAlpha-0.04);
    drawCrashOverlay();
    if(resetTimer<=0) resetCar();
  }}
  drawSuccessOverlay();
  animId=requestAnimationFrame(loop);
}}

placeObstacles();
loop();
</script>
</body>
</html>"""

# ─── GROQ HELPERS ────────────────────────────────────────────────────────────
@st.cache_resource
def get_agent():
    from groq_agent import GroqAgent
    return GroqAgent()

# ─── LAYOUT ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;gap:14px;margin-bottom:18px;padding-bottom:14px;border-bottom:1px solid #1f2d45'>
      <div style='background:#0e2238;border:2px solid #2d7ab5;border-radius:10px;width:44px;height:44px;display:flex;align-items:center;justify-content:center;font-size:24px;flex-shrink:0'>🚗</div>
      <div>
    <div style='font-family:JetBrains Mono,monospace;font-size:18px;font-weight:700;color:#e2e8f0;line-height:1'>AutoDrive AI</div>
    <div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#4a6fa3;margin-top:2px'>Adaptive Self-Driving Car Training Platform · Powered by Groq LLaMA 3.1</div>
      </div>
</div>
""", unsafe_allow_html=True)

left_col, sim_col, right_col = st.columns([1.6, 3.6, 1.8])

# ────────────────────────────── LEFT COLUMN ──────────────────────────────────
with left_col:
    # API Key
    st.markdown('<div class="section-header"> Groq API</div>', unsafe_allow_html=True)
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        st.markdown('<div class="ai-card" style="border-color:#854d0e"><span class="tag tag-yellow"> NO KEY</span> Set GROQ_API_KEY in your .env file to enable AI features. Simulation runs without it.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ai-card"><span class="tag tag-green"> CONNECTED</span> AI features active.</div>', unsafe_allow_html=True)

    # Environment Generator
    st.markdown('<div class="section-header"> Environment Generator</div>', unsafe_allow_html=True)
    env_desc = st.text_area("Describe scenario", placeholder="e.g. rainy highway with debris and barricades",
                            height=80, label_visibility="collapsed")
    if st.button(" Generate with AI", disabled=not api_key):
        if env_desc.strip():
            with st.spinner("Generating..."):
                agent = get_agent()
                cfg = agent.generate_environment(env_desc.strip())
                st.session_state.env_config = cfg
                st.session_state.env_generated = True
                st.session_state.ai_analysis = f"**Generated:** {cfg.get('environment_name','Environment')}\n\n{cfg.get('description','')}"
            st.rerun()

    # Manual Controls
    st.markdown('<div class="section-header"> Manual Config</div>', unsafe_allow_html=True)
    obs_count = st.slider("Obstacles", 2, 18, st.session_state.env_config.get("obstacle_count", 8), label_visibility="visible")
    car_speed = st.slider("Car Speed", 1.0, 4.5, float(st.session_state.env_config.get("car_speed", 2.5)), 0.1, label_visibility="visible")
    weather = st.selectbox("Weather", ["clear", "foggy", "rainy"], index=["clear","foggy","rainy"].index(st.session_state.env_config.get("weather","clear")))
    fog_map = {"clear": 0.0, "foggy": 0.28, "rainy": 0.18}

    if st.button(" Apply Config"):
        st.session_state.env_config.update({
            "obstacle_count": obs_count,
            "car_speed": car_speed,
            "weather": weather,
            "fog_alpha": fog_map[weather]
        })
        st.rerun()

    # Adaptive Difficulty
    st.markdown('<div class="section-header"> Adaptive AI</div>', unsafe_allow_html=True)
    if st.button(" Auto-Adjust Difficulty", disabled=not api_key):
        with st.spinner("Analyzing performance..."):
            agent = get_agent()
            perf = {
                "score": st.session_state.total_score,
                "crashes": st.session_state.total_crashes,
                "episodes": st.session_state.episodes
            }
            new_cfg = agent.get_adaptive_config(perf)
            st.session_state.env_config.update({
                "obstacle_count": new_cfg.get("obstacle_count", 8),
                "car_speed": new_cfg.get("car_speed", 2.5)
            })
            st.session_state.ai_analysis = (
                f"**Adaptive Difficulty: {new_cfg.get('difficulty_label','Medium')}**\n\n"
                f"{new_cfg.get('reasoning','Difficulty adjusted.')}"
            )
        st.rerun()

# ────────────────────────────── SIMULATION ──────────────────────────────────
with sim_col:
    cfg = st.session_state.env_config
    sim_html = build_simulation(cfg)
    st.components.v1.html(sim_html, height=560, scrolling=False)

    # Config badge row
    weather_icon = {"clear": "☀", "foggy": "🌫", "rainy": "🌧"}.get(cfg.get("weather","clear"),"☀")
    st.markdown(f"""
    <div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:8px'>
      <span class="tag tag-blue">🗺 {cfg.get("environment_name","Circuit")}</span>
      <span class="tag tag-yellow">🚧 {cfg.get("obstacle_count",8)} obstacles</span>
      <span class="tag tag-green">⚡ {cfg.get("car_speed",2.5):.1f}x speed</span>
      <span class="tag tag-blue">{weather_icon} {cfg.get("weather","clear").upper()}</span>
      <span class="tag tag-blue">📐 {cfg.get("track_complexity","moderate").upper()}</span>
    </div>
    """, unsafe_allow_html=True)

    # Description
    if cfg.get("description"):
        st.markdown(f'<div class="ai-card" style="margin-top:8px;font-size:12px;color:#94a3b8">📍 {cfg["description"]}</div>', unsafe_allow_html=True)

# ────────────────────────────── RIGHT COLUMN ────────────────────────────────
with right_col:
    st.markdown('<div class="section-header">📊 Session Stats</div>', unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Score</div><div class="metric-value">{st.session_state.total_score}</div></div>', unsafe_allow_html=True)
    with r2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Crashes</div><div class="metric-value" style="color:#f87171">{st.session_state.total_crashes}</div></div>', unsafe_allow_html=True)

    # AI Insights
    st.markdown('<div class="section-header">🤖 AI Insights</div>', unsafe_allow_html=True)
    if st.session_state.ai_analysis:
        st.markdown(f'<div class="ai-card">{st.session_state.ai_analysis}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ai-card" style="color:#4a6fa3">AI analysis will appear here after generating an environment or analyzing a failure.</div>', unsafe_allow_html=True)

    # Manual crash analysis
    st.markdown('<div class="section-header">🔬 Failure Analysis</div>', unsafe_allow_html=True)
    if st.button("🔍 Analyze Last Failure", disabled=not api_key):
        crash_data = {
            "position": {"x": 450, "y": 260},
            "speed": 2.8, "laps_completed": 1,
            "sensor_front": 0.18, "sensor_left": 0.62, "sensor_right": 0.55,
            "obstacle_count": cfg.get("obstacle_count", 8),
            "weather": cfg.get("weather", "clear"),
            "track_complexity": cfg.get("track_complexity", "moderate")
        }
        with st.spinner("Analyzing..."):
            agent = get_agent()
            analysis = agent.analyze_failure(crash_data)
        st.session_state.ai_analysis = f"**Failure Analysis:**\n\n{analysis}"
        st.rerun()

    st.markdown('<div class="section-header"> Sensor Explainer</div>', unsafe_allow_html=True)
    if st.button(" Explain Sensor State", disabled=not api_key):
        with st.spinner("Interpreting..."):
            agent = get_agent()
            sensors_sample = [0.95, 0.72, 0.45, 0.28, 0.61, 0.88, 1.0]
            explanation = agent.explain_sensor_state(sensors_sample, 2.4)
        st.session_state.ai_analysis = f"**Sensor Interpretation:**\n\n{explanation}"
        st.rerun()

    # How it works
    with st.expander("ℹ How it works"):
        st.markdown("""
**Simulation:**
- Smooth Catmull-Rom spline oval track
- 7 LiDAR-style sensor rays (140px range)
- Pure Pursuit + obstacle avoidance agent
- Real-time physics at 60 FPS

**AI (Groq LLaMA 3.1):**
- Natural language → environment config
- Causal failure analysis
- Adaptive difficulty tuning
- Sensor state explanation

**Controls:** Use the sliders on the left to manually tune obstacle density and car speed, then click *Apply Config* to reload the simulation.
        """)

