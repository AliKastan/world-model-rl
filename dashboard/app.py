"""Main Streamlit dashboard for Think Before You Act.

Launch::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys

# Ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

st.set_page_config(
    page_title="Think Before You Act",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom dark theme CSS ──────────────────────────────────────────────

_CSS = """
<style>
/* ── Global ── */
:root {
    --bg-primary: #16161e;
    --bg-card: #1e1e2e;
    --bg-card-hover: #252538;
    --accent: #6366f1;
    --accent-light: #818cf8;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --border: #2e2e42;
    --success: #10b981;
    --danger: #ef4444;
    --warning: #f59e0b;
}

[data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-family: system-ui, 'Segoe UI', Roboto, sans-serif;
}
[data-testid="stSidebar"] {
    background-color: var(--bg-card);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: var(--text-primary);
}
[data-testid="stHeader"] {
    background-color: var(--bg-primary);
}

/* ── Cards ── */
.dash-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.dash-card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 12px rgba(99, 102, 241, 0.12);
}
.dash-card h3 {
    margin: 0 0 0.6rem 0;
    font-size: 1rem;
    color: var(--accent-light);
}
.dash-card .big-number {
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.1;
}
.dash-card .sub {
    color: var(--text-secondary);
    font-size: 0.85rem;
}

/* ── Thought bubble ── */
.thought-bubble {
    background: var(--bg-card);
    border: 1px solid var(--accent);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    font-family: 'Cascadia Code', 'Fira Code', monospace;
    font-size: 0.88rem;
    line-height: 1.55;
    color: var(--text-primary);
    box-shadow: 0 2px 16px rgba(99, 102, 241, 0.10);
}
.thought-bubble .header {
    font-weight: 600;
    color: var(--accent-light);
    margin-bottom: 0.5rem;
}
.thought-bubble .strategy-ok { color: var(--success); }
.thought-bubble .strategy-fail { color: var(--danger); }
.thought-bubble .strategy-warn { color: var(--warning); }

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background-color: var(--accent);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
    transition: background-color 0.2s, transform 0.1s;
}
[data-testid="stButton"] > button:hover {
    background-color: var(--accent-light);
    transform: translateY(-1px);
}
[data-testid="stButton"] > button:active {
    transform: translateY(0);
}

/* ── Metrics ── */
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
}

/* ── Slider / Select ── */
.stSlider [data-baseweb="slider"] div {
    background: var(--accent) !important;
}
.stSelectbox [data-baseweb="select"] {
    background-color: var(--bg-card);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Plotly chart background override ── */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* ── Grid image ── */
.grid-frame {
    border-radius: 10px;
    border: 2px solid var(--border);
    image-rendering: pixelated;
}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# \U0001f9e0 Think Before You Act")
    st.caption("World-Model RL Dashboard")
    st.divider()

    mode = st.radio(
        "Mode",
        [
            "\U0001f3ae Play",
            "\U0001f916 Watch AI",
            "\U0001f4ca Compare",
            "\U0001f9e0 Thought Analysis",
        ],
        index=1,
    )

    st.divider()

    difficulty = st.slider("Difficulty", 1, 10, 3, key="difficulty")

    agent_choice = st.selectbox(
        "Agent",
        ["Thinker (Perfect)", "Thinker (Learned)", "PPO"],
        key="agent",
    )

    speed = st.slider("Speed", 0.25, 4.0, 1.0, step=0.25, key="speed")

    st.divider()

    if st.button("\U0001f504 New Level", use_container_width=True):
        # Clear cached world to trigger regeneration
        for k in list(st.session_state.keys()):
            if k.startswith("world_") or k.startswith("episode_"):
                del st.session_state[k]
        st.rerun()

    # File picker for compare mode
    if "\U0001f4ca" in mode:
        uploaded = st.file_uploader(
            "Load Experiment (metrics.json)", type=["json"]
        )
        if uploaded is not None:
            st.session_state["uploaded_metrics"] = uploaded.read().decode("utf-8")


# ── Main area routing ──────────────────────────────────────────────────

if "\U0001f3ae" in mode:
    from dashboard.live_view import render_play_mode
    render_play_mode(difficulty, speed)

elif "\U0001f916" in mode:
    from dashboard.live_view import render_watch_mode
    render_watch_mode(difficulty, agent_choice, speed)

elif "\U0001f4ca" in mode:
    from dashboard.metrics import render_comparison
    render_comparison()

elif "\U0001f9e0" in mode:
    from dashboard.thought_viz import render_analysis
    render_analysis(difficulty, agent_choice)
