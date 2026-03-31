"""Thought analysis visualizations — deep dive into agent reasoning."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from env.level_generator import LevelGenerator, solve as bfs_solve
from env.objects import Box, Door, Key, Target
from env.puzzle_world import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_NAMES,
    ACTION_RIGHT,
    ACTION_UP,
    TYPE_IDS,
    PuzzleWorld,
)
from agents.model_based.affordance import ContextualAffordance, SceneAnalysis
from agents.model_based.mental_sim import MentalSimulator, ThoughtStep
from agents.model_based.planner import _is_goal
from agents.model_based.thinker_agent import ThinkerAgent
from agents.model_based.world_model import PerfectWorldModel

_PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#1e1e2e",
    font=dict(color="#e2e8f0", family="system-ui, Segoe UI, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=40),
)


# ── Grid rendering with affordance overlays ────────────────────────────

_TYPE_COLORS = {
    0: (30, 30, 46), 1: (75, 85, 99), 2: (249, 115, 22), 3: (99, 102, 241),
    4: (52, 211, 153), 5: (251, 146, 60), 6: (239, 68, 68), 7: (74, 222, 128),
    8: (147, 197, 253), 9: (168, 85, 247), 10: (192, 132, 252),
    11: (120, 53, 15), 12: (45, 45, 58), 13: (234, 179, 8),
}


def _render_affordance_grid(
    world: PuzzleWorld,
    analysis: SceneAnalysis,
    cell_size: int = 48,
) -> np.ndarray:
    """Render grid with affordance color overlays."""
    h, w = world.height, world.width
    cs = cell_size
    img = np.zeros((h * cs, w * cs, 3), dtype=np.uint8)
    obs = world.get_observation()
    grid = obs["grid"]

    for y in range(h):
        for x in range(w):
            tid = int(grid[y, x])
            color = _TYPE_COLORS.get(tid, (30, 30, 46))
            img[y * cs:(y + 1) * cs, x * cs:(x + 1) * cs] = color
            # Grid lines
            img[y * cs, x * cs:(x + 1) * cs] = tuple(min(255, c + 12) for c in color)
            img[y * cs:(y + 1) * cs, x * cs] = tuple(min(255, c + 12) for c in color)

    # Danger zones (red overlay)
    for dx, dy in analysis.danger_zones:
        if 0 <= dy < h and 0 <= dx < w:
            region = img[dy * cs:(dy + 1) * cs, dx * cs:(dx + 1) * cs]
            red = np.array([200, 40, 40], dtype=np.uint8)
            img[dy * cs:(dy + 1) * cs, dx * cs:(dx + 1) * cs] = (
                (region.astype(np.float32) * 0.55 + red * 0.45).astype(np.uint8)
            )

    # Affordance borders around objects
    for obj_info in analysis.objects:
        ox, oy = obj_info.pos
        if not (0 <= ox < w and 0 <= oy < h):
            continue
        aff = obj_info.affordance
        # Determine border color
        if aff.risk_score > 0.5:
            border = (239, 68, 68)    # red - dangerous
        elif aff.utility_score > 0.6:
            border = (52, 211, 153)   # green - helpful
        else:
            border = (251, 191, 36)   # yellow - neutral
        # Draw 3px border
        bw = 3
        img[oy * cs:oy * cs + bw, ox * cs:(ox + 1) * cs] = border
        img[(oy + 1) * cs - bw:(oy + 1) * cs, ox * cs:(ox + 1) * cs] = border
        img[oy * cs:(oy + 1) * cs, ox * cs:ox * cs + bw] = border
        img[oy * cs:(oy + 1) * cs, (ox + 1) * cs - bw:(ox + 1) * cs] = border

    # Agent
    ax, ay = world.agent_pos
    pad = cs // 5
    img[ay * cs + pad:(ay + 1) * cs - pad, ax * cs + pad:(ax + 1) * cs - pad] = (96, 165, 250)

    return img


# ── Relationship graph ─────────────────────────────────────────────────

def _build_relationship_graph(analysis: SceneAnalysis) -> go.Figure:
    """Build a Plotly node-edge diagram of object relationships."""
    nodes = {}
    for obj in analysis.objects:
        label = f"{obj.obj_type}@{obj.pos}"
        nodes[label] = obj

    node_labels = list(nodes.keys())
    if not node_labels:
        fig = go.Figure()
        fig.update_layout(title="No objects to graph", **_PLOTLY_DARK)
        return fig

    # Position nodes in a circle
    n = len(node_labels)
    angles = [2 * np.pi * i / max(n, 1) for i in range(n)]
    node_x = [np.cos(a) for a in angles]
    node_y = [np.sin(a) for a in angles]
    node_idx = {label: i for i, label in enumerate(node_labels)}

    # Node colors based on type
    type_colors = {
        "box": "#f97316", "target": "#6366f1", "key": "#fbbf24",
        "door": "#ef4444", "switch": "#a855f7", "switch_wall": "#78350f",
        "ice": "#93c5fd",
    }
    colors = [type_colors.get(nodes[l].obj_type, "#94a3b8") for l in node_labels]

    # Edges
    edge_x, edge_y = [], []
    edge_mid_x, edge_mid_y, edge_text = [], [], []
    edge_colors = []

    rel_colors = {
        "unlocks": "#fbbf24", "can_reach_target": "#10b981",
        "deadlock_if_pushed": "#ef4444", "path_blocked": "#f59e0b",
        "toggles": "#a855f7", "ice_slide": "#93c5fd",
    }

    for rel in analysis.relationships:
        src_label = f"{rel.source.obj_type}@{rel.source.pos}"
        tgt_label = f"{rel.target.obj_type}@{rel.target.pos}"
        if src_label not in node_idx or tgt_label not in node_idx:
            continue
        si, ti = node_idx[src_label], node_idx[tgt_label]
        edge_x.extend([node_x[si], node_x[ti], None])
        edge_y.extend([node_y[si], node_y[ti], None])
        edge_mid_x.append((node_x[si] + node_x[ti]) / 2)
        edge_mid_y.append((node_y[si] + node_y[ti]) / 2)
        edge_text.append(rel.relation)
        edge_colors.append(rel_colors.get(rel.relation, "#64748b"))

    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="#475569", width=1.5),
        hoverinfo="none",
    ))

    # Edge labels
    if edge_mid_x:
        fig.add_trace(go.Scatter(
            x=edge_mid_x, y=edge_mid_y, mode="text",
            text=edge_text,
            textfont=dict(size=9, color="#94a3b8"),
            hoverinfo="none",
        ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=28, color=colors, line=dict(width=2, color="#1e1e2e")),
        text=node_labels,
        textposition="bottom center",
        textfont=dict(size=10, color="#e2e8f0"),
        hovertext=[
            f"{l}<br>interact={nodes[l].affordance.can_interact:.2f}<br>"
            f"risk={nodes[l].affordance.risk_score:.2f}<br>"
            f"utility={nodes[l].affordance.utility_score:.2f}"
            for l in node_labels
        ],
        hoverinfo="text",
    ))

    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=350,
        **_PLOTLY_DARK,
    )
    return fig


# ── Main analysis view ─────────────────────────────────────────────────

def render_analysis(difficulty: int, agent_choice: str) -> None:
    """Deep dive into ThinkerAgent reasoning."""
    st.markdown("## \U0001f9e0 Thought Analysis")

    # Generate world + run thinking
    if "analysis_world" not in st.session_state:
        gen = LevelGenerator()
        seed = int(time.time() * 1000) % (2**31)
        world = gen.generate(difficulty=difficulty, seed=seed)
        st.session_state["analysis_world"] = world
    world: PuzzleWorld = st.session_state["analysis_world"]

    if "analysis_thought" not in st.session_state:
        sim = MentalSimulator()
        obs = world.get_observation()
        action, thought = sim.think(obs, world)
        st.session_state["analysis_thought"] = thought
        st.session_state["analysis_scene"] = sim.analyze(world)
        st.session_state["analysis_action"] = action
    thought: ThoughtStep = st.session_state["analysis_thought"]
    analysis: SceneAnalysis = st.session_state["analysis_scene"]

    # ── Panel 1: Scene Understanding ──
    st.markdown("### \U0001f441\ufe0f Scene Understanding")
    p1_grid, p1_graph = st.columns([1, 1])

    with p1_grid:
        img = _render_affordance_grid(world, analysis, cell_size=52)
        st.image(img, use_container_width=True, caption="Affordance overlay (Green=helpful, Red=dangerous, Yellow=neutral)")

        # Affordance details
        with st.expander("Object Affordances", expanded=False):
            for obj_info in analysis.objects:
                a = obj_info.affordance
                st.markdown(
                    f"**{obj_info.obj_type}@{obj_info.pos}** &mdash; "
                    f"interact={a.can_interact:.2f} push={a.push_score:.2f} "
                    f"collect={a.collect_score:.2f} risk={a.risk_score:.2f} "
                    f"utility={a.utility_score:.2f}"
                )

    with p1_graph:
        st.plotly_chart(
            _build_relationship_graph(analysis),
            use_container_width=True,
        )
        st.caption(f"Danger zones: {len(analysis.danger_zones)} positions")
        if analysis.required_sequence:
            st.markdown("**Required sequence:**")
            for i, step in enumerate(analysis.required_sequence):
                st.markdown(f"{i + 1}. {step}")

    # ── Panel 2: Strategy Comparison ──
    st.markdown("### \U0001f914 Strategy Comparison")

    if thought.simulations:
        n_strats = len(thought.simulations)
        cols = st.columns(min(n_strats, 4))
        for i, sim_result in enumerate(thought.simulations):
            with cols[i % len(cols)]:
                s = sim_result.strategy
                is_chosen = thought.chosen_strategy and s.name == thought.chosen_strategy.name

                border_color = "#10b981" if is_chosen else "#2e2e42"
                mark = "\u2705" if sim_result.success else ("\u274c" if sim_result.failure_reason else "\u26a0\ufe0f")

                st.markdown(
                    f'<div class="dash-card" style="border-color: {border_color}">'
                    f"<h3>{mark} {s.name}</h3>"
                    f'<div class="sub">{s.description}</div><br>'
                    f"<b>Reward:</b> {sim_result.final_reward:.1f}<br>"
                    f"<b>Risk:</b> {'LOW' if s.risk_level < 0.3 else ('MED' if s.risk_level < 0.6 else 'HIGH')}<br>"
                    f"<b>Steps:</b> {sim_result.steps_simulated}<br>"
                    + (f'<br><span class="sub">{sim_result.failure_reason}</span>' if sim_result.failure_reason else "")
                    + ("<br><b style='color: #10b981'>\u2190 CHOSEN</b>" if is_chosen else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )

                # Mini reward bar
                max_r = max(abs(sr.final_reward) for sr in thought.simulations) or 1
                bar_pct = max(0, min(100, (sim_result.final_reward / max_r) * 50 + 50))
                bar_color = "#10b981" if sim_result.final_reward > 0 else "#ef4444"
                st.progress(bar_pct / 100)
    else:
        st.info("No strategies generated for this state.")

    # ── Panel 3: Mental Simulation Replay ──
    st.markdown("### \U0001f4ad Mental Simulation Replay")

    if thought.simulations and thought.chosen_strategy:
        chosen_sim = next(
            (s for s in thought.simulations if s.strategy is thought.chosen_strategy),
            thought.simulations[0],
        )

        if chosen_sim.imagined_states:
            sim_steps = len(chosen_sim.imagined_states)
            step_slider = st.slider(
                "Simulation step", 0, sim_steps - 1, 0,
                key="sim_replay_step",
            )

            col_real, col_imagined = st.columns(2)

            with col_real:
                st.markdown("**Real World (current)**")
                real_img = _render_affordance_grid(world, analysis, cell_size=40)
                st.image(real_img, use_container_width=True)

            with col_imagined:
                st.markdown(f"**Imagined (step {step_slider + 1}/{sim_steps})**")
                # Show the imagined state info
                state_info = chosen_sim.imagined_states[step_slider]
                st.markdown(
                    f'<div class="dash-card">'
                    f'Agent pos: {state_info["agent_pos"]}<br>'
                    f'Boxes: {state_info["boxes_on_targets"]}/{state_info["total_targets"]}<br>'
                    f'Reward this step: {state_info["reward"]:.1f}<br>'
                    f'Cumulative: {state_info["cumulative_reward"]:.1f}'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Reward accumulation graph
            steps_x = list(range(1, sim_steps + 1))
            cum_rewards = [s["cumulative_reward"] for s in chosen_sim.imagined_states]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=steps_x, y=cum_rewards,
                mode="lines+markers",
                line=dict(color="#6366f1", width=2),
                marker=dict(size=6),
                name="Cumulative Reward",
            ))
            fig.update_layout(
                xaxis_title="Simulation Step",
                yaxis_title="Cumulative Reward",
                height=250,
                **_PLOTLY_DARK,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No imagined states for the chosen strategy.")
    else:
        st.info("Run analysis on a level with strategies to see mental simulation.")

    # ── Panel 4: Episode Timeline ──
    st.markdown("### \U0001f4c8 Episode Timeline")

    # Run a full episode and plot timeline
    if st.button("Run Full Episode & Visualize", use_container_width=True):
        with st.spinner("Running episode..."):
            agent = ThinkerAgent(mode="perfect", planner_type="astar")
            agent.set_world(world.clone())
            episode_data = _run_full_episode_with_thoughts(world.clone(), agent)
            st.session_state["timeline_data"] = episode_data

    if "timeline_data" in st.session_state:
        data = st.session_state["timeline_data"]
        _render_episode_timeline(data)


def _run_full_episode_with_thoughts(
    world: PuzzleWorld, agent: ThinkerAgent
) -> Dict[str, Any]:
    """Run a full episode collecting per-step data."""
    world = world.clone()
    agent.set_world(world)

    steps_data = []
    sim = MentalSimulator()
    cumulative_reward = 0.0

    for step in range(world.max_steps):
        obs = world.get_observation()
        action, thought = sim.think(obs, world)
        _obs_next, reward, terminated, truncated, _info = world.step(action)
        cumulative_reward += reward

        steps_data.append({
            "step": step + 1,
            "action": action,
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "thought_summary": (
                f"{thought.chosen_strategy.name}" if thought.chosen_strategy else "explore"
            ),
            "confidence": thought.confidence,
            "n_strategies": len(thought.strategies_considered),
            "is_replan": step > 0,  # simplified
        })

        agent.set_world(world)
        if terminated or truncated:
            break

    return {
        "steps": steps_data,
        "solved": world.solved or _is_goal(world),
        "total_steps": len(steps_data),
    }


def _render_episode_timeline(data: Dict[str, Any]) -> None:
    """Render interactive episode timeline."""
    steps = data["steps"]
    if not steps:
        st.info("No steps recorded.")
        return

    x = [s["step"] for s in steps]
    y = [s["cumulative_reward"] for s in steps]
    rewards = [s["reward"] for s in steps]

    # Color code: green = positive reward, red = negative, yellow = near zero
    colors = []
    for r in rewards:
        if r > 1.0:
            colors.append("#10b981")  # green
        elif r < -0.2:
            colors.append("#ef4444")  # red
        else:
            colors.append("#fbbf24")  # yellow

    hover_text = [
        f"Step {s['step']}<br>"
        f"Action: {ACTION_NAMES[s['action']]}<br>"
        f"Reward: {s['reward']:.1f}<br>"
        f"Strategy: {s['thought_summary']}<br>"
        f"Confidence: {s['confidence']:.0%}<br>"
        f"Strategies considered: {s['n_strategies']}"
        for s in steps
    ]

    fig = go.Figure()

    # Cumulative reward line
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color="#6366f1", width=2),
        name="Cumulative Reward",
        hoverinfo="none",
    ))

    # Per-step markers
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(size=8, color=colors, line=dict(width=1, color="#1e1e2e")),
        text=hover_text,
        hoverinfo="text",
        name="Steps",
    ))

    # Mark solve point
    if data["solved"]:
        fig.add_vline(
            x=data["total_steps"], line_dash="dash",
            line_color="#10b981", opacity=0.7,
            annotation_text="SOLVED",
            annotation_font_color="#10b981",
        )

    fig.update_layout(
        xaxis_title="Step",
        yaxis_title="Cumulative Reward",
        height=350,
        showlegend=False,
        **_PLOTLY_DARK,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{'Solved' if data['solved'] else 'Unsolved'} in {data['total_steps']} steps  |  "
        f"Green=good move, Red=bad move, Yellow=neutral"
    )
