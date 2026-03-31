"""Metrics comparison dashboard — interactive experiment analysis."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

_PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#1e1e2e",
    font=dict(color="#e2e8f0", family="system-ui, Segoe UI, sans-serif", size=12),
    margin=dict(l=50, r=30, t=50, b=50),
)

COLORS = {
    "PPO": "#EF4444",
    "ThinkerAgent-Perfect": "#10B981",
    "ThinkerAgent-Learned": "#6366F1",
}
SHORT = {
    "PPO": "PPO",
    "ThinkerAgent-Perfect": "Thinker-Perfect",
    "ThinkerAgent-Learned": "Thinker-Learned",
}


# ── Data loading ───────────────────────────────────────────────────────

def _load_metrics() -> Optional[Dict[str, Any]]:
    """Load metrics from session state or experiments directory."""
    # Check uploaded file first
    if "uploaded_metrics" in st.session_state:
        try:
            return json.loads(st.session_state["uploaded_metrics"])
        except json.JSONDecodeError:
            st.error("Invalid JSON file")
            return None

    # Try default experiment path
    default_path = os.path.join("experiments", "results", "main_experiment", "metrics.json")
    alt_path = os.path.join("experiments", "results", "quick_test", "metrics.json")

    for path in [default_path, alt_path]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)

    return None


def _gather_curves(
    data: Dict[str, Any], agent_name: str, diff: str, metric: str
) -> tuple:
    """Gather metric curves across seeds → (episodes, means, stds)."""
    seeds_data = data.get(agent_name, [])
    all_series: Dict[int, List[float]] = defaultdict(list)
    for sd in seeds_data:
        pts = sd.get("eval_points", {}).get(diff, [])
        for pt in pts:
            val = pt.get(metric)
            if val is not None:
                all_series[pt["episode"]].append(val)
    if not all_series:
        return [], [], []
    eps = sorted(all_series.keys())
    means = [np.mean(all_series[e]) for e in eps]
    stds = [np.std(all_series[e]) for e in eps]
    return eps, means, stds


# ── Summary cards ──────────────────────────────────────────────────────

def _render_summary_cards(data: Dict[str, Any]) -> None:
    """Top-row summary cards."""
    all_diffs = set()
    for agent_data in data.values():
        for sd in agent_data:
            all_diffs.update(sd.get("eval_points", {}).keys())
    primary_diff = "3" if "3" in all_diffs else (sorted(all_diffs)[0] if all_diffs else "1")

    # Compute episodes-to-80 for each agent
    e80 = {}
    for agent_name in COLORS:
        vals = []
        for sd in data.get(agent_name, []):
            v = sd.get("episodes_to_80", {}).get(primary_diff)
            if v is not None:
                vals.append(v)
        e80[agent_name] = int(np.mean(vals)) if vals else None

    # Final solve rates
    final_rates = {}
    final_deadlock = {}
    for agent_name in COLORS:
        rates = []
        dlr = []
        for sd in data.get(agent_name, []):
            pts = sd.get("eval_points", {}).get(primary_diff, [])
            if pts:
                rates.append(pts[-1]["solve_rate"])
                dlr.append(pts[-1].get("deadlock_rate", 0))
        final_rates[agent_name] = np.mean(rates) if rates else 0
        final_deadlock[agent_name] = np.mean(dlr) if dlr else 0

    cols = st.columns(3)

    # PPO card
    ppo_e80 = e80.get("PPO")
    with cols[0]:
        st.markdown(
            f'<div class="dash-card">'
            f"<h3>PPO (Baseline)</h3>"
            f'<div class="big-number">{ppo_e80 or "never"}</div>'
            f'<div class="sub">episodes to 80% solve</div><br>'
            f'Solve rate: {final_rates.get("PPO", 0):.0%}<br>'
            f'Deadlock rate: {final_deadlock.get("PPO", 0):.0%}'
            f"</div>",
            unsafe_allow_html=True,
        )

    # Thinker card
    best_thinker = "ThinkerAgent-Perfect"
    if e80.get("ThinkerAgent-Learned") is not None:
        best_thinker = "ThinkerAgent-Learned"
    thinker_e80 = e80.get(best_thinker) or e80.get("ThinkerAgent-Perfect")
    with cols[1]:
        st.markdown(
            f'<div class="dash-card">'
            f"<h3>Thinker (Ours)</h3>"
            f'<div class="big-number">{thinker_e80 or "never"}</div>'
            f'<div class="sub">episodes to 80% solve</div><br>'
            f'Solve rate: {final_rates.get(best_thinker, 0):.0%}<br>'
            f'Deadlock rate: {final_deadlock.get(best_thinker, 0):.0%}'
            f"</div>",
            unsafe_allow_html=True,
        )

    # Improvement card
    with cols[2]:
        speedup = "N/A"
        if ppo_e80 and thinker_e80 and thinker_e80 > 0:
            speedup = f"{ppo_e80 / thinker_e80:.0f}x faster"
        elif ppo_e80 is None and thinker_e80:
            speedup = "PPO never reached 80%"

        dl_ppo = final_deadlock.get("PPO", 0)
        dl_thinker = final_deadlock.get(best_thinker, 0)
        dl_reduction = f"{(1 - dl_thinker / max(dl_ppo, 0.001)) * 100:.0f}% fewer" if dl_ppo > 0 else "N/A"

        st.markdown(
            f'<div class="dash-card">'
            f"<h3>Improvement</h3>"
            f'<div class="big-number">{speedup}</div>'
            f'<div class="sub">sample efficiency gain</div><br>'
            f"{dl_reduction} deadlocks"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Charts ─────────────────────────────────────────────────────────────

def _chart_sample_efficiency(data: Dict[str, Any], diff: str) -> go.Figure:
    """Chart 1: Learning curves — the hero chart."""
    fig = go.Figure()
    for agent_name, color in COLORS.items():
        eps, means, stds = _gather_curves(data, agent_name, diff, "solve_rate")
        if not eps:
            continue
        means_pct = [m * 100 for m in means]
        stds_pct = [s * 100 for s in stds]

        fig.add_trace(go.Scatter(
            x=eps, y=means_pct,
            mode="lines", line=dict(color=color, width=2.5),
            name=SHORT.get(agent_name, agent_name),
        ))
        fig.add_trace(go.Scatter(
            x=eps + eps[::-1],
            y=[m + s for m, s in zip(means_pct, stds_pct)] +
              [max(0, m - s) for m, s in zip(means_pct[::-1], stds_pct[::-1])],
            fill="toself", fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba") if "rgb" in color else color + "1a",
            line=dict(width=0), showlegend=False, hoverinfo="none",
        ))

    fig.add_hline(y=80, line_dash="dot", line_color="#94a3b8", opacity=0.5,
                  annotation_text="80% threshold")
    fig.update_layout(
        xaxis_title="Training Episodes",
        yaxis_title="Solve Rate (%)",
        yaxis=dict(range=[-5, 105]),
        height=400,
        legend=dict(x=0.7, y=0.15, bgcolor="rgba(30,30,46,0.8)"),
        **_PLOTLY_DARK,
    )
    return fig


def _chart_difficulty_bars(data: Dict[str, Any]) -> go.Figure:
    """Chart 2: Grouped bar chart across difficulties."""
    all_diffs = sorted({d for ad in data.values() for sd in ad for d in sd.get("eval_points", {})})
    fig = go.Figure()

    for agent_name, color in COLORS.items():
        rates = []
        errs = []
        for d in all_diffs:
            vals = []
            for sd in data.get(agent_name, []):
                pts = sd.get("eval_points", {}).get(d, [])
                if pts:
                    vals.append(pts[-1]["solve_rate"] * 100)
            rates.append(np.mean(vals) if vals else 0)
            errs.append(np.std(vals) if vals else 0)

        fig.add_trace(go.Bar(
            name=SHORT.get(agent_name, agent_name),
            x=[f"Diff {d}" for d in all_diffs],
            y=rates,
            error_y=dict(type="data", array=errs, visible=True),
            marker_color=color, opacity=0.85,
        ))

    fig.update_layout(
        barmode="group",
        yaxis_title="Solve Rate (%)",
        yaxis=dict(range=[0, 110]),
        height=350,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(30,30,46,0.8)"),
        **_PLOTLY_DARK,
    )
    return fig


def _chart_step_efficiency(data: Dict[str, Any], diff: str) -> go.Figure:
    """Chart 3: Optimality ratio over time."""
    fig = go.Figure()
    for agent_name, color in COLORS.items():
        eps, means, stds = _gather_curves(data, agent_name, diff, "optimality_ratio")
        if not eps:
            continue
        # Filter out infinities
        valid = [(e, m, s) for e, m, s in zip(eps, means, stds) if m < 100]
        if not valid:
            continue
        ve, vm, vs = zip(*valid)
        fig.add_trace(go.Scatter(
            x=list(ve), y=list(vm),
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5),
            name=SHORT.get(agent_name, agent_name),
        ))

    fig.add_hline(y=1.0, line_dash="dot", line_color="#10b981", opacity=0.5,
                  annotation_text="Optimal")
    fig.update_layout(
        xaxis_title="Training Episodes",
        yaxis_title="Steps / Optimal Steps",
        height=350,
        legend=dict(bgcolor="rgba(30,30,46,0.8)"),
        **_PLOTLY_DARK,
    )
    return fig


def _chart_deadlock(data: Dict[str, Any], diff: str) -> go.Figure:
    """Chart 4: Deadlock rates over time."""
    fig = go.Figure()
    for agent_name in ["PPO", "ThinkerAgent-Learned", "ThinkerAgent-Perfect"]:
        color = COLORS.get(agent_name)
        eps, means, stds = _gather_curves(data, agent_name, diff, "deadlock_rate")
        if not eps:
            continue
        means_pct = [m * 100 for m in means]
        stds_pct = [s * 100 for s in stds]

        fig.add_trace(go.Scatter(
            x=eps, y=means_pct,
            mode="lines", line=dict(color=color, width=2),
            name=SHORT.get(agent_name, agent_name),
        ))
        fig.add_trace(go.Scatter(
            x=eps + eps[::-1],
            y=[m + s for m, s in zip(means_pct, stds_pct)] +
              [max(0, m - s) for m, s in zip(means_pct[::-1], stds_pct[::-1])],
            fill="toself", fillcolor=color + "1a" if not color.startswith("rgb") else color,
            line=dict(width=0), showlegend=False, hoverinfo="none",
        ))

    fig.update_layout(
        xaxis_title="Training Episodes",
        yaxis_title="Deadlock Rate (%)",
        yaxis=dict(range=[-5, 105]),
        height=350,
        legend=dict(bgcolor="rgba(30,30,46,0.8)"),
        **_PLOTLY_DARK,
    )
    return fig


def _chart_world_model(data: Dict[str, Any], diff: str) -> go.Figure:
    """Chart 5: World model prediction accuracy."""
    fig = go.Figure()
    agent_name = "ThinkerAgent-Learned"
    color = COLORS[agent_name]

    for metric, label, dash in [
        ("model_1step_acc", "1-step accuracy", "solid"),
        ("model_5step_acc", "5-step accuracy", "dash"),
    ]:
        eps, means, stds = _gather_curves(data, agent_name, diff, metric)
        if not eps:
            continue
        means_pct = [m * 100 for m in means]
        fig.add_trace(go.Scatter(
            x=eps, y=means_pct,
            mode="lines+markers",
            line=dict(color=color, width=2, dash=dash),
            marker=dict(size=4),
            name=label,
        ))

    fig.update_layout(
        xaxis_title="Training Episodes",
        yaxis_title="Prediction Accuracy (%)",
        yaxis=dict(range=[-5, 105]),
        height=350,
        legend=dict(bgcolor="rgba(30,30,46,0.8)"),
        **_PLOTLY_DARK,
    )
    return fig


def _chart_computation(data: Dict[str, Any]) -> go.Figure:
    """Chart 6: Thinking time vs solve rate."""
    fig = go.Figure()
    for agent_name, color in COLORS.items():
        times = []
        rates = []
        for sd in data.get(agent_name, []):
            for d, pts in sd.get("eval_points", {}).items():
                if pts:
                    last = pts[-1]
                    times.append(last.get("planning_overhead_ms", 0))
                    rates.append(last["solve_rate"] * 100)
        if times:
            fig.add_trace(go.Scatter(
                x=times, y=rates,
                mode="markers",
                marker=dict(size=10, color=color, opacity=0.8,
                            line=dict(width=1, color="#1e1e2e")),
                name=SHORT.get(agent_name, agent_name),
            ))

    fig.update_layout(
        xaxis_title="Avg Thinking Time per Episode (ms)",
        yaxis_title="Solve Rate (%)",
        yaxis=dict(range=[-5, 105]),
        height=350,
        legend=dict(bgcolor="rgba(30,30,46,0.8)"),
        **_PLOTLY_DARK,
    )
    return fig


# ── Main render function ───────────────────────────────────────────────

def render_comparison() -> None:
    """Load experiment results and display interactive comparison."""
    st.markdown("## \U0001f4ca Experiment Comparison")

    data = _load_metrics()
    if data is None:
        st.info(
            "No experiment data found. Either:\n"
            "- Upload a `metrics.json` via the sidebar\n"
            "- Run `python -m training.compare --quick` to generate data"
        )
        return

    # Summary cards
    _render_summary_cards(data)
    st.divider()

    # Difficulty selector for learning curve charts
    all_diffs = sorted({d for ad in data.values() for sd in ad for d in sd.get("eval_points", {})})
    if not all_diffs:
        st.warning("No evaluation data in metrics file.")
        return

    selected_diff = st.selectbox(
        "Select difficulty for learning curves",
        all_diffs,
        index=0,
        key="metrics_diff",
    )

    # Chart selection
    chart_options = [
        "Sample Efficiency",
        "Difficulty Progression",
        "Step Efficiency",
        "Deadlock Analysis",
        "World Model Learning",
        "Computation Tradeoff",
    ]
    selected_charts = st.multiselect(
        "Charts to display",
        chart_options,
        default=chart_options[:4],
        key="chart_select",
    )

    # Render selected charts in a 2-column layout
    if selected_charts:
        chart_pairs = [selected_charts[i:i + 2] for i in range(0, len(selected_charts), 2)]

        for pair in chart_pairs:
            cols = st.columns(len(pair))
            for col, chart_name in zip(cols, pair):
                with col:
                    st.markdown(f"#### {chart_name}")
                    if chart_name == "Sample Efficiency":
                        fig = _chart_sample_efficiency(data, selected_diff)
                    elif chart_name == "Difficulty Progression":
                        fig = _chart_difficulty_bars(data)
                    elif chart_name == "Step Efficiency":
                        fig = _chart_step_efficiency(data, selected_diff)
                    elif chart_name == "Deadlock Analysis":
                        fig = _chart_deadlock(data, selected_diff)
                    elif chart_name == "World Model Learning":
                        fig = _chart_world_model(data, selected_diff)
                    elif chart_name == "Computation Tradeoff":
                        fig = _chart_computation(data)
                    else:
                        continue
                    st.plotly_chart(fig, use_container_width=True)

    # Raw data expander
    with st.expander("Raw Experiment Data"):
        st.json(data)
