import json

import streamlit as st
import streamlit.components.v1 as components


def inject_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background:#08111f;color:#d9e4f2;}
        .block-container {padding-top:1rem;padding-bottom:1.5rem;max-width:1450px;}
        [data-testid="stSidebar"] {background:#0d1729;border-right:1px solid rgba(90,132,188,0.18);}
        .hero-wrap {padding:16px 20px;border:1px solid rgba(68,120,196,0.28);background:linear-gradient(180deg,#0d1b31 0%,#091321 100%);border-radius:20px;margin-bottom:14px;}
        .hero-title {font-family:'JetBrains Mono', monospace;font-size:28px;font-weight:700;color:#e8f2ff;}
        .hero-sub {color:#8db8e8;font-size:14px;margin-top:6px;}
        .card {background:#0d1729;border:1px solid rgba(88,128,181,.18);padding:14px 16px;border-radius:18px;}
        .pill {display:inline-block;padding:5px 10px;border-radius:999px;background:#10243f;border:1px solid rgba(97,165,255,.18);color:#93c5fd;font-size:12px;margin:4px 6px 0 0;}
        .kpi {background:#091321;border:1px solid rgba(88,128,181,.18);border-radius:18px;padding:14px;}
        .kpi-label {color:#7ea5cf;font-size:12px;text-transform:uppercase;letter-spacing:.08em;}
        .kpi-value {font-family:'JetBrains Mono', monospace;font-size:28px;font-weight:700;color:#f8fbff;}
        .stTabs [data-baseweb="tab-list"] {gap:8px;}
        .stTabs [data-baseweb="tab"] {background:#0b1627;border-radius:12px;padding:10px 16px;border:1px solid rgba(88,128,181,.15);}
        .stButton button {border-radius:14px;background:linear-gradient(180deg,#1e60c6 0%,#194c9a 100%);color:white;border:none;padding:.75rem 1rem;font-weight:600;}
        .stSelectbox label, .stSlider label, .stTextArea label {color:#9ec4ee !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-title">🚗 AutoDrive AI Adaptive Self-Driving Car Training Platform</div>
            <div class="hero-sub">Powered by Groq LLaMA 3.1 · RL training, adaptive co-evolution, failure analysis, and professional simulation UI.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_config_summary(cfg, adaptive=False):
    mode = "Adaptive AI enabled" if adaptive else "Adaptive AI disabled"
    st.markdown("### Environment")
    st.markdown(
        f"""
        <div class="card">
            <span class="pill">🗺 {cfg.environment_name}</span>
            <span class="pill">🚧 {cfg.obstacle_count} obstacles</span>
            <span class="pill">⚡ {cfg.car_speed:.1f}x speed</span>
            <span class="pill">☀ {cfg.weather.upper()}</span>
            <span class="pill">📐 {cfg.track_complexity.upper()}</span>
            <span class="pill">🤖 {mode}</span>
            <p style="margin-top:12px;color:#dbeafe;">📍 {cfg.description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(df):
    if df.empty:
        score = 0
        crashes = 0
        goals = 0
        eps = 0
    else:
        score = round(df["score"].iloc[-1], 2)
        crashes = int(df["collision"].sum())
        goals = int(df["reached_goal"].sum())
        eps = len(df)
    cols = st.columns(4)
    data = [("Score", score), ("Crashes", crashes), ("Goals", goals), ("Episodes", eps)]
    for col, (label, value) in zip(cols, data):
        col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)


def render_episode_card(ep: dict):
    st.markdown("### Latest Episode")
    adaptation = ep.get('adaptation') or {}
    next_cfg = adaptation.get('next_config') or ep.get('next_config') or {}
    directional = ep.get('directional_snapshot') or {}
    st.markdown(
        f"""
        <div class="card">
            <p><strong>Score:</strong> {ep.get('score')}</p>
            <p><strong>Reached goal:</strong> {ep.get('reached_goal')}</p>
            <p><strong>Collision:</strong> {ep.get('collision')}</p>
            <p><strong>Path efficiency:</strong> {ep.get('path_efficiency')}</p>
            <p><strong>Average speed:</strong> {ep.get('avg_speed')}</p>
            <p><strong>Difficulty:</strong> {ep.get('difficulty')}</p>
            <p><strong>Goal angle:</strong> {directional.get('goal_angle', 'n/a')}</p>
            <p><strong>Heading error:</strong> {directional.get('heading_error', 'n/a')}</p>
            <p><strong>Turn bias:</strong> {directional.get('turn_bias', 'n/a')}</p>
            <p><strong>Goal distance:</strong> {directional.get('goal_distance', 'n/a')}</p>
            <p><strong>Next obstacles:</strong> {next_cfg.get('obstacle_count', 'n/a')}</p>
            <p><strong>Next speed:</strong> {next_cfg.get('car_speed', 'n/a')}</p>
            <p><strong>Adaptation:</strong> {adaptation.get('analysis', ep.get('adaptation_summary', 'n/a'))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_simulation(episode):
    st.markdown("### Simulation")
    if not episode:
        st.info("Run training or evaluation to render the animated self-driving playback.")
        return

    payload = {
        "track": episode["track_points"],
        "obstacles": episode["obstacles"],
        "path": episode["path"],
        "destination": episode["destination"],
        "config": episode["config"],
    }

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8" />
    <style>
    body {{ margin:0; background:#07101c; font-family:Inter,sans-serif; }}
    .wrap {{ padding:10px; }}
    canvas {{ width:100%; height:520px; border-radius:18px; background:#081522; border:1px solid rgba(76,126,196,.25); }}
    .hud {{ color:#9ec4ee; display:flex; gap:12px; flex-wrap:wrap; margin:8px 2px 0; font-size:13px; }}
    .chip {{ background:#0d1b31; padding:6px 10px; border-radius:999px; border:1px solid rgba(76,126,196,.25); }}
    </style>
    </head>
    <body>
    <div class="wrap">
        <canvas id="cv" width="920" height="520"></canvas>
        <div class="hud">
            <div class="chip">Environment: {payload['config']['environment_name']}</div>
            <div class="chip">Weather: {payload['config']['weather']}</div>
            <div class="chip">Obstacles: {payload['config']['obstacle_count']}</div>
            <div class="chip">Top speed: {payload['config']['car_speed']}</div>
        </div>
    </div>
    <script>
    const data = {json.dumps(payload)};
    const canvas = document.getElementById('cv');
    const ctx = canvas.getContext('2d');
    const track = data.track;
    const obstacles = data.obstacles;
    const path = data.path;
    const destination = data.destination;
    let idx = 0;

    function drawBackground() {{
        ctx.fillStyle = '#16351f';
        ctx.fillRect(0,0,canvas.width,canvas.height);
        ctx.strokeStyle = 'rgba(255,255,255,0.04)';
        for (let x=0; x<canvas.width; x+=40) {{ ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,canvas.height); ctx.stroke(); }}
        for (let y=0; y<canvas.height; y+=40) {{ ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(canvas.width,y); ctx.stroke(); }}
    }}

    function drawRoad() {{
        ctx.save();
        ctx.lineJoin = 'round'; ctx.lineCap = 'round';
        ctx.strokeStyle = '#1b2432'; ctx.lineWidth = 88;
        ctx.beginPath(); ctx.moveTo(track[0][0], track[0][1]);
        for (let i=1;i<track.length;i++) ctx.lineTo(track[i][0], track[i][1]);
        ctx.closePath(); ctx.stroke();
        ctx.strokeStyle = '#384557'; ctx.lineWidth = 74;
        ctx.beginPath(); ctx.moveTo(track[0][0], track[0][1]);
        for (let i=1;i<track.length;i++) ctx.lineTo(track[i][0], track[i][1]);
        ctx.closePath(); ctx.stroke();
        ctx.strokeStyle = 'rgba(255,217,87,.7)'; ctx.lineWidth = 2; ctx.setLineDash([18,12]);
        ctx.beginPath(); ctx.moveTo(track[0][0], track[0][1]);
        for (let i=1;i<track.length;i++) ctx.lineTo(track[i][0], track[i][1]);
        ctx.closePath(); ctx.stroke();
        ctx.restore();
    }}

    function drawObstacles() {{
        obstacles.forEach((o, i) => {{
            const [x,y,r] = o;
            ctx.fillStyle = i % 2 === 0 ? '#ef4444' : '#f59e0b';
            ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI*2); ctx.fill();
            ctx.fillStyle = 'rgba(255,255,255,.3)';
            ctx.beginPath(); ctx.arc(x-3, y-3, r*0.35, 0, Math.PI*2); ctx.fill();
        }});
    }}

    function drawDestination() {{
        const [x,y] = destination;
        const pulse = 8 + Math.sin(Date.now()/250) * 3;
        ctx.strokeStyle = 'rgba(16,185,129,.8)';
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(x,y,24+pulse,0,Math.PI*2); ctx.stroke();
        ctx.fillStyle = '#22c55e';
        ctx.beginPath(); ctx.arc(x,y,10,0,Math.PI*2); ctx.fill();
    }}

    function drawPathTrail(upTo) {{
        ctx.strokeStyle = 'rgba(56,189,248,.5)';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(path[0][0], path[0][1]);
        for (let i=1;i<upTo;i++) ctx.lineTo(path[i][0], path[i][1]);
        ctx.stroke();
    }}

    function drawCar(i) {{
        const p = path[Math.min(i, path.length-1)];
        const q = path[Math.min(i+1, path.length-1)];
        const angle = Math.atan2(q[1]-p[1], q[0]-p[0]);
        ctx.save();
        ctx.translate(p[0], p[1]);
        ctx.rotate(angle);
        ctx.shadowBlur = 16;
        ctx.shadowColor = 'rgba(14,165,233,.45)';
        ctx.fillStyle = '#1d4ed8';
        ctx.beginPath();
        ctx.roundRect(-16,-10,32,20,7);
        ctx.fill();
        ctx.fillStyle = '#dbeafe';
        ctx.fillRect(-7,-8,14,6);
        ctx.fillStyle = '#111827';
        ctx.fillRect(-13,-12,7,4); ctx.fillRect(6,-12,7,4); ctx.fillRect(-13,8,7,4); ctx.fillRect(6,8,7,4);
        ctx.restore();
    }}

    function frame() {{
        drawBackground();
        drawRoad();
        drawObstacles();
        drawDestination();
        drawPathTrail(Math.max(2, idx));
        drawCar(idx);
        idx = (idx + 2) % path.length;
        requestAnimationFrame(frame);
    }}
    frame();
    </script>
    </body>
    </html>
    """
    components.html(html, height=590, scrolling=False)
