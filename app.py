import os
import json
from dataclasses import asdict

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from adaptive_ai import AdaptiveDirector
from rl_agent import RLAgent
from simulation_core import SimulationConfig, run_episode, train_agent
from ui_components import inject_theme, render_hero, render_config_summary, render_metrics, render_episode_card

load_dotenv()

st.set_page_config(
    page_title="AutoDrive AI | Adaptive Self-Driving Lab",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_theme()
render_hero()

DEFAULT_CONFIG = {
    "environment_name": "Urban Circuit",
    "weather": "clear",
    "track_complexity": "moderate",
    "obstacle_count": 8,
    "car_speed": 2.8,
    "fog_alpha": 0.0,
    "description": "Standard city loop with a single destination zone and scattered road obstacles.",
}

STATE_DEFAULTS = {
    "config": DEFAULT_CONFIG.copy(),
    "agent": RLAgent(),
    "history_standard": [],
    "history_adaptive": [],
    "analysis_text": "AI analysis will appear here after you run training or adaptive co-evolution.",
    "failure_text": "Failure analysis will appear here after an episode with a collision.",
    "sensor_text": "Sensor explanation will appear here after a simulation run.",
    "adaptive_text": "Adaptive AI will suggest harder or easier environments here.",
    "last_episode": None,
    "groq_ready": bool(os.getenv("GROQ_API_KEY")),
}

for key, value in STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def normalize_track_complexity(value: str) -> str:
    lookup = {
        "low": "simple",
        "easy": "simple",
        "medium": "moderate",
        "normal": "moderate",
        "high": "complex",
        "hard": "complex",
    }
    normalized = (value or "").strip().lower()
    if normalized in {"simple", "moderate", "complex"}:
        return normalized
    return lookup.get(normalized, "moderate")


def build_simulation(cfg: dict) -> str:
        cfg_json = json.dumps(cfg)
        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{background:#0b0f1a;display:flex;flex-direction:column;align-items:center;justify-content:flex-start;min-height:100vh;padding:10px;overflow:auto}}
    canvas{{border:2px solid #1e3a5a;border-radius:8px;display:block;max-width:100%;height:auto;aspect-ratio:860/520}}
    #status{{color:#64748b;font-family:'JetBrains Mono',monospace;font-size:11px;margin-top:6px;text-align:center;width:100%}}
</style>
</head>
<body>
<canvas id="sim"></canvas>
<div id="status">⬤ SIMULATION RUNNING - AI AGENT ACTIVE</div>
<script>
const CFG = {cfg_json};
const canvas = document.getElementById('sim');
const ctx = canvas.getContext('2d');
const W = 860, H = 520;
canvas.width = W; canvas.height = H;

// Track Control Points (oval)
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

const CAR = {{
    x:TRACK[0].x, y:TRACK[0].y,
    angle:Math.atan2(TRACK[1].y-TRACK[0].y,TRACK[1].x-TRACK[0].x),
    speed:0, steer:0, alive:true,
    sensors:new Array(7).fill(1.0),
    trail:[], targetIdx:5,
    score:0, laps:0, episodes:0, totalCrashes:0
}};

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

function agentStep(){{
    if(!CAR.alive) return;
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

function drawGrass(){{
    ctx.fillStyle='#12261a';
    ctx.fillRect(0,0,W,H);
    ctx.strokeStyle='rgba(30,60,35,0.4)';
    ctx.lineWidth=1;
    for(let x=0;x<W;x+=40){{ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}}
    for(let y=0;y<H;y+=40){{ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}}
}}

function drawRoad(){{
    ctx.save();
    ctx.shadowBlur=18; ctx.shadowColor='rgba(0,0,0,0.7)';
    ctx.strokeStyle='#1e2430'; ctx.lineWidth=TW+10;
    ctx.lineCap='round'; ctx.lineJoin='round';
    ctx.beginPath(); ctx.moveTo(TRACK[0].x,TRACK[0].y);
    for(let i=1;i<TRACK.length;i++) ctx.lineTo(TRACK[i].x,TRACK[i].y);
    ctx.closePath(); ctx.stroke();
    ctx.restore();
    ctx.strokeStyle='#2c3340'; ctx.lineWidth=TW;
    ctx.lineCap='round'; ctx.lineJoin='round';
    ctx.beginPath(); ctx.moveTo(TRACK[0].x,TRACK[0].y);
    for(let i=1;i<TRACK.length;i++) ctx.lineTo(TRACK[i].x,TRACK[i].y);
    ctx.closePath(); ctx.stroke();
    ctx.save();
    for(let sign=-1;sign<=1;sign+=2){{
        const offset=sign*(TW/2-4);
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
    for(let r=1;r<=3;r++){{
        ctx.strokeStyle=`rgba(0,255,136,${{(0.15+pulse*0.3)*(4-r)/3}})`;
        ctx.lineWidth=2;
        ctx.beginPath();
        ctx.arc(dest.x,dest.y,22+r*8+pulse*6,0,Math.PI*2);
        ctx.stroke();
    }}
    const grd=ctx.createRadialGradient(dest.x,dest.y,0,dest.x,dest.y,14);
    grd.addColorStop(0,'#00ff88'); grd.addColorStop(1,'rgba(0,200,80,0)');
    ctx.fillStyle=grd;
    ctx.beginPath(); ctx.arc(dest.x,dest.y,14,0,Math.PI*2); ctx.fill();
    const fx=dest.x+2, fy=dest.y-38;
    ctx.strokeStyle='#00ff88'; ctx.lineWidth=2;
    ctx.beginPath(); ctx.moveTo(fx,fy+28); ctx.lineTo(fx,fy+2); ctx.stroke();
    const sz=5;
    for(let row=0;row<3;row++) for(let col=0;col<3;col++){{
        ctx.fillStyle=((row+col)%2===0)?'#fff':'#000';
        ctx.fillRect(fx+col*sz,fy+2+row*sz,sz,sz);
    }}
    const start=TRACK[0];
    ctx.fillStyle='rgba(0,0,0,0.7)'; ctx.fillRect(start.x-18,start.y-18,36,14);
    ctx.fillStyle='#fbbf24'; ctx.font='bold 10px JetBrains Mono,monospace';
    ctx.textAlign='center'; ctx.fillText('START',start.x,start.y-7);
    ctx.fillStyle='rgba(0,0,0,0.7)'; ctx.fillRect(dest.x-18,dest.y+20,36,14);
    ctx.fillStyle='#4ade80'; ctx.font='bold 10px JetBrains Mono,monospace';
    ctx.fillText('GOAL',dest.x,dest.y+31);
}}

function drawObstacles(){{
    for(const obs of obstacles){{
        ctx.save();
        ctx.translate(obs.x,obs.y); ctx.rotate(obs.angle||0);
        if(obs.type==='cone'){{
            ctx.shadowBlur=6; ctx.shadowColor='rgba(255,100,0,0.5)';
            ctx.fillStyle='#f97316';
            ctx.beginPath(); ctx.arc(0,0,obs.r,0,Math.PI*2); ctx.fill();
            ctx.fillStyle='#fff';
            ctx.fillRect(-obs.r,obs.r*0.05,obs.r*2,obs.r*0.3);
            ctx.fillStyle='rgba(0,0,0,0.3)';
            ctx.beginPath(); ctx.arc(0,0,obs.r,0,Math.PI*2); ctx.stroke();
            ctx.fillStyle='#fff'; ctx.beginPath(); ctx.arc(0,-obs.r*0.4,2,0,Math.PI*2); ctx.fill();
        }} else {{
            ctx.shadowBlur=6; ctx.shadowColor='rgba(200,0,0,0.5)';
            ctx.fillStyle='#dc2626';
            ctx.beginPath(); ctx.arc(0,0,obs.r,0,Math.PI*2); ctx.fill();
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
    ctx.save(); ctx.translate(3,3);
    ctx.fillStyle='rgba(0,0,0,0.35)';
    ctx.beginPath(); ctx.roundRect(-CW/2,-CH/2,CW,CH,4); ctx.fill();
    ctx.restore();
    const bodyGrd=ctx.createLinearGradient(-CW/2,-CH/2,CW/2,CH/2);
    bodyGrd.addColorStop(0,'#38bdf8'); bodyGrd.addColorStop(0.5,'#0ea5e9'); bodyGrd.addColorStop(1,'#0369a1');
    ctx.fillStyle=bodyGrd;
    ctx.beginPath(); ctx.roundRect(-CW/2,-CH/2,CW,CH,4); ctx.fill();
    ctx.fillStyle='rgba(186,230,253,0.7)';
    ctx.beginPath(); ctx.roundRect(-CW/2+2,-CH/2+5,CW-4,8,2); ctx.fill();
    ctx.fillStyle='#075985';
    ctx.beginPath(); ctx.roundRect(-CW/2+2,-CH/2+5,CW-4,9,2); ctx.fill();
    ctx.fillStyle=CAR.speed>0.3?'#fef08a':'#713f12';
    ctx.beginPath(); ctx.roundRect(-CW/2,-CH/2,5,3,1); ctx.fill();
    ctx.beginPath(); ctx.roundRect(CW/2-5,-CH/2,5,3,1); ctx.fill();
    if(CAR.speed>0.3){{
        ctx.save();
        ctx.shadowBlur=14; ctx.shadowColor='rgba(255,255,100,0.8)';
        ctx.fillStyle='#fef08a';
        ctx.beginPath(); ctx.arc(-CW/2+2,-CH/2+1,2,0,Math.PI*2); ctx.fill();
        ctx.beginPath(); ctx.arc(CW/2-2,-CH/2+1,2,0,Math.PI*2); ctx.fill();
        ctx.restore();
    }}
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
    ctx.fillStyle='rgba(11,15,26,0.82)';
    ctx.beginPath(); ctx.roundRect(8,8,188,108,6); ctx.fill();
    ctx.strokeStyle='#1e3a5a'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.roundRect(8,8,188,108,6); ctx.stroke();
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
    ctx.fillText('DESTINATION REACHED!',W/2,H/2);
    ctx.restore();
    successAnim-=2;
    if(successAnim<=0){{
        setTimeout(()=>resetCar(),800);
    }}
}}

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

adaptive_director = AdaptiveDirector()

with st.sidebar:
    st.markdown("## Control Center")
    st.session_state.config["track_complexity"] = normalize_track_complexity(
        st.session_state.config.get("track_complexity", "moderate")
    )
    prompt = st.text_area(
        "Environment prompt",
        value="Urban training route with medium traffic pressure, roadside cones, and occasional fog.",
        height=110,
    )
    weather = st.selectbox("Weather", ["clear", "rainy", "foggy"], index=["clear", "rainy", "foggy"].index(st.session_state.config["weather"]))
    complexity = st.selectbox("Track complexity", ["simple", "moderate", "complex"], index=["simple", "moderate", "complex"].index(st.session_state.config["track_complexity"]))
    obstacles = st.slider("Obstacles", 2, 18, int(st.session_state.config["obstacle_count"]))
    speed = st.slider("Car speed", 1.0, 4.5, float(st.session_state.config["car_speed"]), 0.1)

    if st.button("Generate Environment", width="stretch"):
        generated = adaptive_director.generate_environment(prompt)
        generated["weather"] = weather
        generated["track_complexity"] = normalize_track_complexity(complexity)
        generated["obstacle_count"] = obstacles
        generated["car_speed"] = speed
        st.session_state.config = generated
        st.session_state.adaptive_text = f"Environment generated: {generated['environment_name']} | {generated['description']}"
        st.rerun()

st.session_state.config["track_complexity"] = normalize_track_complexity(
    st.session_state.config.get("track_complexity", "moderate")
)
cfg = SimulationConfig(**st.session_state.config)

tab_standard, tab_adaptive = st.tabs(["Standard RL Training", "Adaptive Co-Evolution"])

with tab_standard:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        render_config_summary(cfg, adaptive=False)
        col_a, col_b = st.columns(2)
        run_train = col_a.button("Run RL Training", width="stretch")
        run_eval = col_b.button("Evaluate Current Agent", width="stretch")

        if run_train:
            history = train_agent(st.session_state.agent, cfg, episodes=30)
            st.session_state.history_standard.extend(history)
            latest = history[-1]
            st.session_state.last_episode = latest
            st.session_state.sensor_text = adaptive_director.explain_sensor_state(latest["sensor_snapshot"], latest["avg_speed"])
            if latest["collision"]:
                st.session_state.failure_text = adaptive_director.analyze_failure(latest)
            st.session_state.analysis_text = adaptive_director.summarize_training_block(history, adaptive=False)
            st.rerun()

        if run_eval:
            episode = run_episode(st.session_state.agent, cfg, learn=False)
            st.session_state.history_standard.append(episode)
            st.session_state.last_episode = episode
            st.session_state.sensor_text = adaptive_director.explain_sensor_state(episode["sensor_snapshot"], episode["avg_speed"])
            if episode["collision"]:
                st.session_state.failure_text = adaptive_director.analyze_failure(episode)
            st.session_state.analysis_text = adaptive_director.summarize_training_block([episode], adaptive=False)
            st.rerun()

        sim_html = build_simulation(st.session_state.config)
        st.components.v1.html(sim_html, height=620, scrolling=False)

    with right:
        std_df = pd.DataFrame(st.session_state.history_standard)
        render_metrics(std_df)
        st.markdown("### 🤖 AI Insights")
        st.info(st.session_state.analysis_text)
        st.markdown("### 🔬 Failure Analysis")
        st.warning(st.session_state.failure_text)
        st.markdown("### 🛰 Sensor Explainer")
        st.success(st.session_state.sensor_text)

        if not std_df.empty:
            chart_df = std_df.copy()
            chart_df["episode_index"] = range(1, len(chart_df) + 1)
            fig = px.line(chart_df, x="episode_index", y=["score", "path_efficiency"], markers=True, template="plotly_dark")
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(chart_df[["episode_index", "score", "collision", "reached_goal", "path_efficiency", "avg_speed"]].tail(10), width="stretch")

with tab_adaptive:
    left2, right2 = st.columns([1.1, 0.9], gap="large")

    with left2:
        render_config_summary(cfg, adaptive=True)
        evolve = st.button("Run Adaptive Loop", width="stretch")

        if evolve:
            local_cfg = cfg
            block = []
            for _ in range(12):
                history = train_agent(st.session_state.agent, local_cfg, episodes=3)
                latest = history[-1]
                block.extend(history)
                proposal = adaptive_director.propose_adaptation(local_cfg, latest)
                proposal["config"]["track_complexity"] = normalize_track_complexity(
                    proposal["config"].get("track_complexity", "moderate")
                )
                local_cfg = SimulationConfig(**proposal["config"])
                st.session_state.adaptive_text = proposal["analysis"]
            st.session_state.config = asdict(local_cfg)
            st.session_state.history_adaptive.extend(block)
            st.session_state.last_episode = block[-1]
            st.session_state.analysis_text = adaptive_director.summarize_training_block(block, adaptive=True)
            if block[-1]["collision"]:
                st.session_state.failure_text = adaptive_director.analyze_failure(block[-1])
            st.session_state.sensor_text = adaptive_director.explain_sensor_state(block[-1]["sensor_snapshot"], block[-1]["avg_speed"])
            st.rerun()

        sim_html = build_simulation(st.session_state.config)
        st.components.v1.html(sim_html, height=620, scrolling=False)

    with right2:
        adf = pd.DataFrame(st.session_state.history_adaptive)
        render_metrics(adf)
        st.markdown("### 🧠 Adaptive Director")
        st.info(st.session_state.adaptive_text)
        st.markdown("### 🤖 AI Insights")
        st.success(st.session_state.analysis_text)

        if not adf.empty:
            adf = adf.copy()
            adf["episode_index"] = range(1, len(adf) + 1)
            fig2 = px.line(adf, x="episode_index", y=["score", "difficulty"], markers=True, template="plotly_dark")
            fig2.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig2, width="stretch")
            render_episode_card(adf.iloc[-1].to_dict())
