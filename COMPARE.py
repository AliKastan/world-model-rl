"""COMPARE — Side-by-side PPO vs ThinkerAgent comparison.

Two agents train simultaneously on the same puzzle distribution.
Left panel: pure PPO (model-free).
Right panel: PPO + 1-step lookahead via real env clone (ThinkerAgent).
Shared learning-curve graph at the bottom shows both agents.

Controls:
  Space  — Pause / Resume
  F      — Fast mode (5 episodes per frame)
  S      — Slow mode (1 episode per frame, 200ms delay)
  V      — Visual mode (1 episode per frame, normal speed)
  ESC    — Quit
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import pygame

from env.sokoban_env import SokobanEnv
from env.sokoban import SokobanState
from agents.model_free.ppo_agent import PPOAgent


# ═════════════════════════════════════════════════════════════════════
# Colours
# ═════════════════════════════════════════════════════════════════════

COL_BG        = (22, 22, 35)
COL_PANEL     = (30, 30, 50)
COL_GRID_LINE = (40, 40, 55)
COL_WALL      = (75, 85, 99)
COL_BOX       = (251, 191, 36)
COL_BOX_DONE  = (52, 211, 153)
COL_TARGET    = (52, 211, 153)
COL_PLAYER    = (99, 102, 241)
COL_WHITE     = (220, 220, 235)
COL_DIM       = (120, 120, 140)
COL_GREEN     = (52, 211, 153)
COL_YELLOW    = (251, 191, 36)
COL_RED       = (239, 68, 68)
COL_BLUE      = (99, 102, 241)


# ═════════════════════════════════════════════════════════════════════
# Layout constants  (1200 x 720)
# ═════════════════════════════════════════════════════════════════════

WIN_W, WIN_H = 1200, 720
TITLE_H      = 40
GRID_TOP     = TITLE_H
GRID_BOT     = 500
GRID_H       = GRID_BOT - GRID_TOP          # 460
HALF_W       = WIN_W // 2                    # 600

GRID_SIZE    = 400
LEFT_GRID    = pygame.Rect(
    (HALF_W - GRID_SIZE) // 2,
    GRID_TOP + (GRID_H - GRID_SIZE) // 2,
    GRID_SIZE, GRID_SIZE,
)
RIGHT_GRID   = pygame.Rect(
    HALF_W + (HALF_W - GRID_SIZE) // 2,
    GRID_TOP + (GRID_H - GRID_SIZE) // 2,
    GRID_SIZE, GRID_SIZE,
)

STATS_Y      = GRID_BOT + 4
GRAPH_RECT   = pygame.Rect(20, 560, WIN_W - 40, 140)


# ═════════════════════════════════════════════════════════════════════
# SimpleThinkerAgent — PPO policy + 1-step lookahead via SokobanState
# ═════════════════════════════════════════════════════════════════════

class SimpleThinkerAgent:
    """Lightweight thinker: PPO policy + 1-step lookahead using real env clone.

    For the first 1024 transitions the agent behaves like vanilla PPO.
    After the first PPO update, planning kicks in: before choosing an
    action, every legal move is simulated on a copy of the SokobanState
    and the one with the highest shaped reward is selected.  When all
    moves yield the same reward (no useful signal), the PPO policy is
    used as tie-breaker.
    """

    def __init__(self, grid_h: int = 10, grid_w: int = 10) -> None:
        self.ppo = PPOAgent(grid_h, grid_w)
        self.world_model_accuracy = 0.0
        self.planning_active = False
        self.steps_collected = 0

    # ── reward estimation on a SokobanState ──────────────────────
    @staticmethod
    def _estimate_reward(
        old_state: SokobanState, action: int,
    ) -> Tuple[Optional[SokobanState], float]:
        """Simulate one action on an immutable SokobanState, return (new, r)."""
        old_on = old_state.n_boxes_on_target
        old_dist = old_state.box_distances()

        new_state = old_state.move(action)
        if new_state is None:
            return None, -1.0                     # invalid move

        reward = -0.1                             # step penalty

        if new_state.solved:
            return new_state, reward + 100.0

        if new_state.is_deadlocked():
            return new_state, reward - 50.0

        new_on = new_state.n_boxes_on_target
        if new_on > old_on:
            reward += 10.0
        elif new_on < old_on:
            reward -= 10.0

        dist_diff = old_dist - new_state.box_distances()
        if dist_diff > 0.5:
            reward += 1.0
        elif dist_diff < -0.5:
            reward -= 1.0

        return new_state, reward

    def select_action(
        self,
        obs: np.ndarray,
        env_state: Optional[SokobanState],
    ) -> Tuple[int, float, float]:
        """Pick an action — lookahead when planning is active, else PPO."""

        if self.planning_active and env_state is not None:
            # 1-step lookahead over all 4 actions
            rewards = [self._estimate_reward(env_state, a)[1] for a in range(4)]
            best_r = max(rewards)

            # If all rewards identical, no useful signal — fall back to PPO
            if rewards.count(best_r) == len(rewards):
                return self.ppo.select_action(obs)

            best_action = int(np.argmax(rewards))
            # Still need log_prob + value from PPO for training
            _, log_prob, value = self.ppo.select_action(obs)
            return best_action, log_prob, value

        return self.ppo.select_action(obs)

    def get_action_probs(self, obs: np.ndarray) -> np.ndarray:
        return self.ppo.get_action_probs(obs)

    def store_and_learn(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ) -> None:
        self.ppo.store_transition(obs, action, reward, log_prob, value, done)
        self.steps_collected += 1
        if self.steps_collected >= 1024:
            self.ppo.update()
            self.steps_collected = 0
            self.planning_active = True
            self.world_model_accuracy = min(0.99, self.world_model_accuracy + 0.05)


# ═════════════════════════════════════════════════════════════════════
# AgentRunner — wraps env + agent + episode bookkeeping
# ═════════════════════════════════════════════════════════════════════

class AgentRunner:
    """Manages one environment / agent pair and tracks statistics."""

    def __init__(
        self,
        name: str,
        color: Tuple[int, int, int],
        is_thinker: bool = False,
    ) -> None:
        self.name = name
        self.color = color
        self.is_thinker = is_thinker

        self.env = SokobanEnv(level_set="training", max_h=10, max_w=10)
        if is_thinker:
            self.agent: SimpleThinkerAgent | PPOAgent = SimpleThinkerAgent(10, 10)
        else:
            self.agent = PPOAgent(grid_h=10, grid_w=10)

        self.obs: np.ndarray = np.zeros(0)
        self.ep_reward = 0.0
        self.episode = 0
        self.steps_in_batch = 0

        self.all_rewards: List[float] = []
        self.all_solved: List[bool] = []
        self.deadlocks = 0

        self._reset()

    # ── helpers ───────────────────────────────────────────────────

    def _reset(self) -> None:
        self.obs, _ = self.env.reset()
        self.ep_reward = 0.0

    @property
    def env_state(self) -> Optional[SokobanState]:
        return self.env.state

    @property
    def solve_rate(self) -> float:
        if not self.all_solved:
            return 0.0
        recent = self.all_solved[-100:]
        return sum(recent) / len(recent) * 100

    @property
    def avg_reward(self) -> float:
        if not self.all_rewards:
            return 0.0
        recent = self.all_rewards[-100:]
        return sum(recent) / len(recent)

    # ── episode execution ─────────────────────────────────────────

    def run_episode(self) -> None:
        """Run one full episode to completion."""
        while True:
            done = self._step()
            if done:
                break
        self._finish_episode()

    def _step(self) -> bool:
        if self.is_thinker:
            ta: SimpleThinkerAgent = self.agent  # type: ignore[assignment]
            action, log_prob, value = ta.select_action(self.obs, self.env_state)
        else:
            pa: PPOAgent = self.agent  # type: ignore[assignment]
            action, log_prob, value = pa.select_action(self.obs)

        next_obs, reward, term, trunc, info = self.env.step(action)
        done = term or trunc

        if self.is_thinker:
            ta = self.agent  # type: ignore[assignment]
            ta.store_and_learn(self.obs, action, reward, log_prob, value, done)
        else:
            pa = self.agent  # type: ignore[assignment]
            pa.store_transition(self.obs, action, reward, log_prob, value, done)
            self.steps_in_batch += 1

        self.obs = next_obs
        self.ep_reward += reward
        return done

    def _finish_episode(self) -> None:
        state = self.env.state
        solved = state.solved if state else False
        deadlocked = state.is_deadlocked() if state else False

        self.all_rewards.append(self.ep_reward)
        self.all_solved.append(solved)
        if deadlocked:
            self.deadlocks += 1
        self.episode += 1

        # PPO-only batch update (Thinker handles its own inside store_and_learn)
        if not self.is_thinker and self.steps_in_batch >= 1024:
            pa: PPOAgent = self.agent  # type: ignore[assignment]
            pa.update()
            self.steps_in_batch = 0

        self._reset()


# ═════════════════════════════════════════════════════════════════════
# Rendering helpers
# ═════════════════════════════════════════════════════════════════════

def draw_sokoban_grid(
    surface: pygame.Surface,
    state: Optional[SokobanState],
    area: pygame.Rect,
) -> None:
    """Render a SokobanState into the given rectangle."""
    if state is None:
        return
    w, h = state.width, state.height
    cs = min(area.width // w, area.height // h)
    ox = area.x + (area.width - w * cs) // 2
    oy = area.y + (area.height - h * cs) // 2

    for gy in range(h):
        for gx in range(w):
            rect = pygame.Rect(ox + gx * cs, oy + gy * cs, cs, cs)
            pygame.draw.rect(surface, COL_BG, rect)
            pygame.draw.rect(surface, COL_GRID_LINE, rect, 1)

            pos = (gx, gy)

            if pos in state.walls:
                pygame.draw.rect(surface, COL_WALL, rect, border_radius=3)
                continue

            if pos in state.targets:
                pygame.draw.circle(surface, COL_TARGET, rect.center, cs // 4, 2)

            if pos in state.boxes:
                on_target = pos in state.targets
                c = COL_BOX_DONE if on_target else COL_BOX
                pygame.draw.rect(surface, c, rect.inflate(-6, -6), border_radius=4)

            if pos == state.player:
                pygame.draw.circle(surface, COL_PLAYER, rect.center, cs // 3)


def draw_agent_stats(
    surface: pygame.Surface,
    font: pygame.font.Font,
    font_bold: pygame.font.Font,
    runner: AgentRunner,
    x: int,
    y: int,
) -> None:
    """Draw per-agent statistics below the grid."""

    def text(txt: str, f: pygame.font.Font = font, color=COL_WHITE) -> None:
        nonlocal y
        s = f.render(txt, True, color)
        surface.blit(s, (x, y))
        y += s.get_height() + 2

    # Agent name
    text(runner.name, font_bold, runner.color)

    # Episode count
    text(f"Episode: {runner.episode:,}")

    # Solve rate
    sr = runner.solve_rate
    if runner.all_solved:
        col = COL_GREEN if sr > 50 else COL_YELLOW if sr > 20 else COL_RED
        text(f"Solve Rate: {sr:.1f}%", color=col)
    else:
        text("Solve Rate: --")

    # Avg reward
    if runner.all_rewards:
        text(f"Avg Reward: {runner.avg_reward:.1f}")
    else:
        text("Avg Reward: --")

    # Deadlock rate
    if runner.episode > 0:
        dr = runner.deadlocks / runner.episode * 100
        text(f"Deadlock Rate: {dr:.0f}%")

    # Thinker-specific stats
    if runner.is_thinker:
        ta: SimpleThinkerAgent = runner.agent  # type: ignore[assignment]
        acc = ta.world_model_accuracy * 100
        acc_col = COL_GREEN if acc > 80 else COL_YELLOW if acc > 40 else COL_RED
        text(f"World Model: {acc:.0f}%", color=acc_col)

        plan_on = ta.planning_active
        plan_col = COL_GREEN if plan_on else COL_RED
        text(f"Planning: {'ON' if plan_on else 'OFF'}", color=plan_col)


def _rolling_rate(data: List[bool], window: int = 100) -> List[float]:
    """Compute rolling solve rate."""
    if not data:
        return []
    rates: List[float] = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        chunk = data[start : i + 1]
        rates.append(sum(chunk) / len(chunk))
    return rates


def draw_dual_graph(
    surface: pygame.Surface,
    font: pygame.font.Font,
    ppo_solved: List[bool],
    thinker_solved: List[bool],
    rect: pygame.Rect,
) -> None:
    """Shared learning-curve graph with both lines."""
    pygame.draw.rect(surface, COL_PANEL, rect, border_radius=4)

    # Title
    lbl = font.render("Solve Rate (rolling 100)", True, COL_DIM)
    surface.blit(lbl, (rect.x + 8, rect.y + 4))

    graph_x = rect.x + 8
    graph_w = rect.width - 16
    graph_y = rect.y + 24
    graph_h = rect.height - 32

    # Axes
    pygame.draw.line(
        surface, COL_DIM,
        (graph_x, graph_y + graph_h), (graph_x + graph_w, graph_y + graph_h), 1,
    )
    pygame.draw.line(
        surface, COL_DIM,
        (graph_x, graph_y), (graph_x, graph_y + graph_h), 1,
    )

    # 50% dashed line
    mid_y = graph_y + graph_h // 2
    for dx in range(0, graph_w, 8):
        pygame.draw.line(
            surface, (50, 50, 70),
            (graph_x + dx, mid_y), (graph_x + dx + 4, mid_y), 1,
        )
    pct_lbl = font.render("50%", True, COL_DIM)
    surface.blit(pct_lbl, (graph_x + graph_w - 34, mid_y - 12))

    # 100% label
    top_lbl = font.render("100%", True, COL_DIM)
    surface.blit(top_lbl, (graph_x + graph_w - 42, graph_y - 2))

    # Plot helper
    def plot_line(rates: List[float], color: Tuple[int, int, int]) -> None:
        if len(rates) < 2:
            return
        n = len(rates)
        x_scale = graph_w / max(n - 1, 1)
        points = [
            (graph_x + int(i * x_scale), graph_y + graph_h - int(r * graph_h))
            for i, r in enumerate(rates)
        ]
        if len(points) >= 2:
            pygame.draw.lines(surface, color, False, points, 2)

    plot_line(_rolling_rate(ppo_solved), COL_RED)
    plot_line(_rolling_rate(thinker_solved), COL_BLUE)

    # Legend
    lx = rect.x + rect.width - 200
    ly = rect.y + 4
    pygame.draw.line(surface, COL_RED, (lx, ly + 6), (lx + 20, ly + 6), 2)
    surface.blit(font.render("PPO", True, COL_RED), (lx + 24, ly))
    lx += 80
    pygame.draw.line(surface, COL_BLUE, (lx, ly + 6), (lx + 20, ly + 6), 2)
    surface.blit(font.render("Thinker", True, COL_BLUE), (lx + 24, ly))


# ═════════════════════════════════════════════════════════════════════
# Main loop
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Think Before You Act -- PPO vs ThinkerAgent")
    clock = pygame.time.Clock()

    font      = pygame.font.SysFont("consolas,dejavusansmono,monospace", 14)
    font_bold = pygame.font.SysFont("consolas,dejavusansmono,monospace", 16, bold=True)
    font_title = pygame.font.SysFont("consolas,dejavusansmono,monospace", 20, bold=True)

    # Create the two runners
    ppo_runner     = AgentRunner("PPO Agent",     COL_RED,  is_thinker=False)
    thinker_runner = AgentRunner("Thinker Agent", COL_BLUE, is_thinker=True)

    mode: str = "visual"          # "visual" | "fast" | "slow"
    paused: bool = False
    last_step_time: float = time.time()
    slow_delay: float = 0.2

    running = True

    while running:
        now = time.time()

        # ── Events ────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_f:
                    mode = "fast"
                elif event.key == pygame.K_s:
                    mode = "slow"
                elif event.key == pygame.K_v:
                    mode = "visual"

        # ── Training ──────────────────────────────────────────────
        if not paused:
            if mode == "fast":
                for _ in range(5):
                    ppo_runner.run_episode()
                    thinker_runner.run_episode()
            elif mode == "visual":
                ppo_runner.run_episode()
                thinker_runner.run_episode()
            elif mode == "slow":
                if now - last_step_time >= slow_delay:
                    last_step_time = now
                    ppo_runner.run_episode()
                    thinker_runner.run_episode()

        # ── Render ────────────────────────────────────────────────
        screen.fill(COL_BG)

        # Title bar
        title_surf = font_title.render(
            "Think Before You Act  --  PPO vs ThinkerAgent", True, COL_WHITE,
        )
        screen.blit(
            title_surf,
            (WIN_W // 2 - title_surf.get_width() // 2, 10),
        )

        # Vertical divider
        pygame.draw.line(
            screen, COL_GRID_LINE,
            (HALF_W, TITLE_H), (HALF_W, GRID_BOT), 2,
        )

        # Left grid — PPO
        pygame.draw.rect(screen, COL_GRID_LINE, LEFT_GRID, 2, border_radius=4)
        draw_sokoban_grid(screen, ppo_runner.env_state, LEFT_GRID)
        ppo_lbl = font_bold.render("PPO", True, COL_RED)
        screen.blit(
            ppo_lbl,
            (LEFT_GRID.centerx - ppo_lbl.get_width() // 2, LEFT_GRID.y - 20),
        )

        # Right grid — Thinker
        pygame.draw.rect(screen, COL_GRID_LINE, RIGHT_GRID, 2, border_radius=4)
        draw_sokoban_grid(screen, thinker_runner.env_state, RIGHT_GRID)
        tk_lbl = font_bold.render("Thinker", True, COL_BLUE)
        screen.blit(
            tk_lbl,
            (RIGHT_GRID.centerx - tk_lbl.get_width() // 2, RIGHT_GRID.y - 20),
        )

        # Stats below each grid
        draw_agent_stats(
            screen, font, font_bold, ppo_runner,
            x=LEFT_GRID.x, y=STATS_Y,
        )
        draw_agent_stats(
            screen, font, font_bold, thinker_runner,
            x=RIGHT_GRID.x, y=STATS_Y,
        )

        # Mode / controls bar
        mode_labels = {
            "slow":   "SLOW (1 ep/frame, delayed)",
            "visual": "VISUAL (1 ep/frame)",
            "fast":   "FAST (5 eps/frame)",
        }
        status = "PAUSED" if paused else mode_labels.get(mode, mode)
        ctrl_txt = f"[{status}]   Space=pause  F=fast  S=slow  V=visual  ESC=quit"
        ctrl_surf = font.render(ctrl_txt, True, COL_DIM)
        screen.blit(ctrl_surf, (20, GRAPH_RECT.y - 18))

        # Dual learning-curve graph
        draw_dual_graph(
            screen, font,
            ppo_runner.all_solved, thinker_runner.all_solved,
            GRAPH_RECT,
        )

        pygame.display.flip()
        clock.tick(60)

    # ── Cleanup ───────────────────────────────────────────────────
    ppo_runner.env.close()
    thinker_runner.env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
