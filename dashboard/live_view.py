"""Live gameplay and AI observation views."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st

from env.level_generator import LevelGenerator, solve as bfs_solve
from env.puzzle_world import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_NAMES,
    ACTION_RIGHT,
    ACTION_UP,
    PuzzleWorld,
    TYPE_IDS,
)
from agents.model_based.mental_sim import MentalSimulator, ThoughtStep
from agents.model_based.thinker_agent import ThinkerAgent, run_episode
from agents.model_based.planner import AStarPlanner, _is_goal
from agents.model_based.world_model import PerfectWorldModel


# ── Grid rendering (pure numpy, no pygame) ─────────────────────────────

_TYPE_COLORS: Dict[int, tuple] = {
    0: (30, 30, 46),      # floor
    1: (75, 85, 99),      # wall
    2: (249, 115, 22),    # box
    3: (99, 102, 241),    # target
    4: (52, 211, 153),    # box_on_target
    5: (251, 146, 60),    # key
    6: (239, 68, 68),     # door_locked
    7: (74, 222, 128),    # door_unlocked
    8: (147, 197, 253),   # ice
    9: (168, 85, 247),    # switch_inactive
    10: (192, 132, 252),  # switch_active
    11: (120, 53, 15),    # switch_wall_closed
    12: (45, 45, 58),     # switch_wall_open
    13: (234, 179, 8),    # oneway
}

_AGENT_COLOR = (96, 165, 250)
_HINT_COLOR = (99, 102, 241, 160)


def _render_grid_numpy(
    world: PuzzleWorld,
    cell_size: int = 48,
    hint_action: Optional[int] = None,
    danger_zones: Optional[list] = None,
) -> np.ndarray:
    """Render the world to an RGB numpy array without pygame."""
    h, w = world.height, world.width
    cs = cell_size
    img = np.zeros((h * cs, w * cs, 3), dtype=np.uint8)

    obs = world.get_observation()
    grid = obs["grid"]

    # Draw cells
    for y in range(h):
        for x in range(w):
            tid = int(grid[y, x])
            color = _TYPE_COLORS.get(tid, (30, 30, 46))
            img[y * cs:(y + 1) * cs, x * cs:(x + 1) * cs] = color

            # Grid lines
            img[y * cs, x * cs:(x + 1) * cs] = tuple(
                min(255, c + 15) for c in color
            )
            img[y * cs:(y + 1) * cs, x * cs] = tuple(
                min(255, c + 15) for c in color
            )

    # Danger zones overlay
    if danger_zones:
        for dx, dy in danger_zones:
            if 0 <= dy < h and 0 <= dx < w:
                region = img[dy * cs:(dy + 1) * cs, dx * cs:(dx + 1) * cs]
                overlay = np.array([180, 40, 40], dtype=np.uint8)
                img[dy * cs:(dy + 1) * cs, dx * cs:(dx + 1) * cs] = (
                    (region.astype(np.float32) * 0.6 + overlay * 0.4).astype(np.uint8)
                )

    # Draw agent
    ax, ay = world.agent_pos
    pad = cs // 5
    img[ay * cs + pad:(ay + 1) * cs - pad, ax * cs + pad:(ax + 1) * cs - pad] = _AGENT_COLOR

    # Hint arrow
    if hint_action is not None:
        _draw_hint_arrow(img, ax, ay, hint_action, cs)

    return img


def _draw_hint_arrow(img: np.ndarray, ax: int, ay: int, action: int, cs: int) -> None:
    """Draw a semi-transparent directional hint arrow."""
    deltas = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}
    dx, dy = deltas.get(action, (0, 0))
    tx, ty = ax + dx, ay + dy
    h, w = img.shape[0] // cs, img.shape[1] // cs
    if 0 <= tx < w and 0 <= ty < h:
        pad = cs // 4
        region = img[ty * cs + pad:(ty + 1) * cs - pad, tx * cs + pad:(tx + 1) * cs - pad]
        hint = np.array([99, 102, 241], dtype=np.uint8)
        img[ty * cs + pad:(ty + 1) * cs - pad, tx * cs + pad:(tx + 1) * cs - pad] = (
            (region.astype(np.float32) * 0.5 + hint * 0.5).astype(np.uint8)
        )


# ── Session state helpers ──────────────────────────────────────────────

def _get_world(difficulty: int) -> PuzzleWorld:
    key = f"world_{difficulty}"
    if key not in st.session_state:
        gen = LevelGenerator()
        seed = int(time.time() * 1000) % (2**31)
        st.session_state[key] = gen.generate(difficulty=difficulty, seed=seed)
        st.session_state["episode_steps"] = 0
        st.session_state["episode_reward"] = 0.0
        st.session_state["episode_history"] = []
    return st.session_state[key]


def _get_optimal(world: PuzzleWorld) -> Optional[List[int]]:
    key = "optimal_solution"
    if key not in st.session_state:
        sol = bfs_solve(world, max_states=200_000, timeout_seconds=3.0)
        st.session_state[key] = sol
    return st.session_state[key]


# ── Play mode ──────────────────────────────────────────────────────────

def render_play_mode(difficulty: int, speed: float) -> None:
    """Human plays the puzzle interactively."""
    st.markdown("## \U0001f3ae Play Mode")

    world = _get_world(difficulty)
    steps = st.session_state.get("episode_steps", 0)
    total_reward = st.session_state.get("episode_reward", 0.0)

    col_grid, col_info = st.columns([2, 1])

    with col_grid:
        # Render grid
        img = _render_grid_numpy(world, cell_size=52)
        st.image(img, use_container_width=True, caption=f"{world.width}x{world.height} grid")

        # Direction buttons
        bcols = st.columns([1, 1, 1, 1, 1])
        actions = {1: ACTION_LEFT, 2: ACTION_UP, 3: ACTION_DOWN, 4: ACTION_RIGHT}
        labels = {1: "\u2190", 2: "\u2191", 3: "\u2193", 4: "\u2192"}
        action_taken = None

        for i, bcol in enumerate(bcols):
            if i in actions:
                with bcol:
                    if st.button(labels[i], key=f"btn_{i}", use_container_width=True):
                        action_taken = actions[i]

    # Process action
    if action_taken is not None and not (world.solved or _is_goal(world)):
        _obs, reward, terminated, truncated, _info = world.step(action_taken)
        st.session_state["episode_steps"] = steps + 1
        st.session_state["episode_reward"] = total_reward + reward
        st.session_state["episode_history"] = st.session_state.get("episode_history", []) + [
            {"action": action_taken, "reward": reward}
        ]
        st.rerun()

    with col_info:
        obs = world.get_observation()

        # Info cards
        st.markdown(
            f'<div class="dash-card"><h3>Steps</h3>'
            f'<div class="big-number">{steps}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="dash-card"><h3>Boxes Placed</h3>'
            f'<div class="big-number">{obs["boxes_on_targets"]} / {obs["total_targets"]}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="dash-card"><h3>Keys Collected</h3>'
            f'<div class="big-number">{len(world.inventory)}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="dash-card"><h3>Reward</h3>'
            f'<div class="big-number">{total_reward:.1f}</div></div>',
            unsafe_allow_html=True,
        )

        # Solved state
        if world.solved or _is_goal(world):
            optimal = _get_optimal(world)
            opt_len = len(optimal) if optimal else "?"
            eff = f"{(opt_len / max(steps, 1)) * 100:.0f}%" if isinstance(opt_len, int) else "N/A"
            st.success(f"Solved in {steps} steps! (Optimal: {opt_len}, Efficiency: {eff})")

        st.divider()

        # Hint button
        if st.button("\U0001f4a1 Hint", use_container_width=True):
            sim = MentalSimulator()
            action, thought = sim.think(obs, world)
            st.session_state["hint_action"] = action
            st.session_state["hint_thought"] = thought
            st.rerun()

        if "hint_action" in st.session_state:
            hint_a = st.session_state["hint_action"]
            st.info(f"Suggested move: **{ACTION_NAMES[hint_a]}**")
            if "hint_thought" in st.session_state:
                t = st.session_state["hint_thought"]
                if t.chosen_strategy:
                    st.caption(f"Strategy: {t.chosen_strategy.name}")
                    st.caption(f"Confidence: {t.confidence:.0%}")

        # Solve button
        if st.button("\u2728 Auto-Solve", use_container_width=True):
            optimal = _get_optimal(world)
            if optimal:
                st.session_state["autosolve_actions"] = optimal
                st.session_state["autosolve_step"] = 0
                st.rerun()
            else:
                st.error("No solution found within budget")

        # Auto-solve animation
        if "autosolve_actions" in st.session_state:
            actions_list = st.session_state["autosolve_actions"]
            idx = st.session_state.get("autosolve_step", 0)
            if idx < len(actions_list):
                _obs, reward, _t, _tr, _i = world.step(actions_list[idx])
                st.session_state["autosolve_step"] = idx + 1
                st.session_state["episode_steps"] = steps + 1
                st.session_state["episode_reward"] = total_reward + reward
                time.sleep(0.15 / max(speed, 0.25))
                st.rerun()
            else:
                del st.session_state["autosolve_actions"]
                del st.session_state["autosolve_step"]


# ── Watch AI mode ──────────────────────────────────────────────────────

def render_watch_mode(
    difficulty: int, agent_choice: str, speed: float
) -> None:
    """Watch AI agent solve puzzles."""
    st.markdown("## \U0001f916 Watch AI Solve")

    # Initialize episode state
    if "watch_world" not in st.session_state:
        _reset_watch_episode(difficulty)

    world: PuzzleWorld = st.session_state["watch_world"]
    agent: ThinkerAgent = st.session_state["watch_agent"]
    history: list = st.session_state.get("watch_history", [])
    step_idx: int = st.session_state.get("watch_step", 0)
    paused: bool = st.session_state.get("watch_paused", True)

    col_grid, col_thought = st.columns([3, 2])

    with col_grid:
        # Render current state
        img = _render_grid_numpy(world, cell_size=52)
        st.image(img, use_container_width=True)

        # Controls
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            if st.button("\u25b6\ufe0f Play" if paused else "\u23f8\ufe0f Pause", use_container_width=True):
                st.session_state["watch_paused"] = not paused
                st.rerun()
        with c2:
            if st.button("\u23ed\ufe0f Step", use_container_width=True):
                _watch_take_step(world, agent, speed)
                st.rerun()
        with c3:
            if st.button("\U0001f504 Reset", use_container_width=True):
                _reset_watch_episode(difficulty)
                st.rerun()
        with c4:
            st.metric("Step", step_idx)
        with c5:
            obs = world.get_observation()
            st.metric("Boxes", f"{obs['boxes_on_targets']}/{obs['total_targets']}")

        # Episode stats row
        reward_total = sum(h["reward"] for h in history)
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Total Reward", f"{reward_total:.1f}")
        sc2.metric("Solved", "\u2705" if world.solved or _is_goal(world) else "\u274c")
        sc3.metric("Deadlock", "\u26a0\ufe0f" if world.is_deadlock() else "\u2705 Safe")

    with col_thought:
        # Thought bubble
        thoughts = st.session_state.get("watch_thoughts", [])
        if thoughts:
            latest = thoughts[-1]
            _render_thought_bubble(latest)
        else:
            st.markdown(
                '<div class="thought-bubble">'
                '<div class="header">\U0001f9e0 Waiting for first step...</div>'
                "Press Play or Step to begin."
                "</div>",
                unsafe_allow_html=True,
            )

        # Episode log
        if history:
            st.markdown("### Episode Log")
            for i, h in enumerate(history[-10:]):  # last 10 steps
                idx = len(history) - 10 + i if len(history) > 10 else i
                color = "strategy-ok" if h["reward"] > 0 else ("strategy-fail" if h["reward"] < -0.2 else "")
                st.markdown(
                    f'<span class="{color}">Step {idx + 1}: '
                    f'{ACTION_NAMES[h["action"]]} (r={h["reward"]:.1f})</span>',
                    unsafe_allow_html=True,
                )

    # Auto-play loop
    if not paused and not (world.solved or _is_goal(world) or world.is_deadlock()):
        time.sleep(0.3 / max(speed, 0.25))
        _watch_take_step(world, agent, speed)
        st.rerun()


def _reset_watch_episode(difficulty: int) -> None:
    gen = LevelGenerator()
    seed = int(time.time() * 1000) % (2**31)
    world = gen.generate(difficulty=difficulty, seed=seed)
    agent_choice = st.session_state.get("agent", "Thinker (Perfect)")

    if "Perfect" in agent_choice:
        agent = ThinkerAgent(mode="perfect", planner_type="astar")
    elif "Learned" in agent_choice:
        agent = ThinkerAgent(mode="learned", planner_type="beam")
    else:
        agent = ThinkerAgent(mode="no_thinking")

    agent.set_world(world)

    st.session_state["watch_world"] = world
    st.session_state["watch_agent"] = agent
    st.session_state["watch_history"] = []
    st.session_state["watch_thoughts"] = []
    st.session_state["watch_step"] = 0
    st.session_state["watch_paused"] = True


def _watch_take_step(world: PuzzleWorld, agent: ThinkerAgent, speed: float) -> None:
    if world.solved or _is_goal(world):
        return

    obs = world.get_observation()
    agent.set_world(world)

    # Run mental sim for thought log
    sim = MentalSimulator()
    action, thought = sim.think(obs, world)

    _obs_next, reward, terminated, truncated, _info = world.step(action)

    history = st.session_state.get("watch_history", [])
    history.append({"action": action, "reward": reward})
    st.session_state["watch_history"] = history

    thoughts = st.session_state.get("watch_thoughts", [])
    thoughts.append(thought)
    st.session_state["watch_thoughts"] = thoughts

    st.session_state["watch_step"] = len(history)

    # Update agent
    agent.set_world(world)
    next_obs = world.get_observation()
    agent.learn(
        state=obs, action=action, next_state=next_obs,
        reward=reward, done=terminated or truncated, world=world,
    )


def _render_thought_bubble(thought: ThoughtStep) -> None:
    """Render a thought step as a styled thought bubble."""
    lines = [
        f'<div class="thought-bubble">',
        f'<div class="header">\U0001f9e0 Agent is thinking...</div>',
        f"<div>{thought.perception}</div><br>",
    ]

    if thought.simulations:
        lines.append("<b>Strategies:</b><br>")
        for i, sim in enumerate(thought.simulations):
            s = sim.strategy
            if sim.success:
                cls = "strategy-ok"
                mark = "\u2705"
            elif sim.failure_reason and ("deadlock" in sim.failure_reason or sim.final_reward < 0):
                cls = "strategy-fail"
                mark = "\u274c"
            else:
                cls = "strategy-warn"
                mark = "\u26a0\ufe0f"
            chosen = " \u2190 CHOSEN" if thought.chosen_strategy and s.name == thought.chosen_strategy.name else ""
            lines.append(
                f'<span class="{cls}">{i + 1}. {mark} {s.name}{chosen}</span><br>'
            )
            if sim.failure_reason:
                lines.append(f'<span class="sub">   {sim.failure_reason}</span><br>')

    lines.append(f"<br><b>Confidence:</b> {thought.confidence:.0%}")

    if thought.chosen_strategy:
        first_action = thought.chosen_strategy.action_sequence[0] if thought.chosen_strategy.action_sequence else None
        if first_action is not None:
            lines.append(f"<br><b>Next move:</b> {ACTION_NAMES[first_action]}")

    lines.append(f'<br><br><span class="sub">{thought.reasoning}</span>')
    lines.append("</div>")

    st.markdown("\n".join(lines), unsafe_allow_html=True)
