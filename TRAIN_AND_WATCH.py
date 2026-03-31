"""TRAIN_AND_WATCH -- Watch a PPO agent learn Sokoban in real-time.

Opens a PyGame window (920x640). Left side shows the Sokoban grid,
right side shows training stats, a learning curve, and action probs.

Controls:
  Space  -- Pause / Resume
  F      -- Fast mode   (10 episodes per frame, skip rendering)
  S      -- Slow mode   (1 step per frame, 150ms delay)
  V      -- Visual mode (1 episode per frame, render final state)
  ESC    -- Quit
"""

from __future__ import annotations

import sys
import time
from typing import List

import numpy as np
import pygame

from env.sokoban_env import SokobanEnv
from agents.model_free.ppo_agent import PPOAgent


# ── Colours ───────────────────────────────────────────────────────────
COL_BG         = (22, 22, 35)
COL_PANEL      = (30, 30, 50)
COL_GRID_LINE  = (40, 40, 55)
COL_WALL       = (75, 85, 99)
COL_BOX        = (251, 191, 36)
COL_BOX_DONE   = (52, 211, 153)
COL_TARGET     = (52, 211, 153)
COL_PLAYER     = (99, 102, 241)
COL_WHITE      = (220, 220, 235)
COL_DIM        = (120, 120, 140)
COL_GREEN      = (52, 211, 153)
COL_YELLOW     = (251, 191, 36)
COL_RED        = (239, 68, 68)
COL_GRAPH_BG   = (30, 30, 50)
COL_GRAPH_LINE = (52, 211, 153)
COL_DASH       = (50, 50, 70)

# ── Layout ────────────────────────────────────────────────────────────
WIN_W, WIN_H = 920, 640
GRID_AREA    = pygame.Rect(10, 10, 560, 560)
STATS_X      = 590
STATS_W      = 320
GRAPH_RECT   = pygame.Rect(STATS_X, 380, STATS_W, 180)
PROBS_RECT   = pygame.Rect(STATS_X, 570, STATS_W, 60)

# ── Training constants ────────────────────────────────────────────────
N_EPISODES       = 10_000
STEPS_PER_UPDATE = 1024
SLOW_DELAY       = 0.15


# ══════════════════════════════════════════════════════════════════════
# Grid Renderer
# ══════════════════════════════════════════════════════════════════════

def draw_grid(surface: pygame.Surface, state, area: pygame.Rect) -> None:
    """Draw the Sokoban grid from a SokobanState."""
    if state is None:
        return

    gw, gh = state.width, state.height
    cs = min(area.width // gw, area.height // gh)
    ox = area.x + (area.width  - gw * cs) // 2
    oy = area.y + (area.height - gh * cs) // 2

    # Background cells and grid lines
    for gy in range(gh):
        for gx in range(gw):
            rect = pygame.Rect(ox + gx * cs, oy + gy * cs, cs, cs)
            pygame.draw.rect(surface, COL_BG, rect)
            pygame.draw.rect(surface, COL_GRID_LINE, rect, 1)

    # Walls
    for (wx, wy) in state.walls:
        rect = pygame.Rect(ox + wx * cs, oy + wy * cs, cs, cs)
        pygame.draw.rect(surface, COL_WALL, rect, border_radius=3)

    # Targets (circle outline)
    for (tx, ty) in state.targets:
        cx = ox + tx * cs + cs // 2
        cy = oy + ty * cs + cs // 2
        pygame.draw.circle(surface, COL_TARGET, (cx, cy), cs // 4, 2)

    # Boxes
    boxes_on_target = state.boxes & state.targets
    for (bx, by) in state.boxes:
        rect = pygame.Rect(ox + bx * cs, oy + by * cs, cs, cs)
        colour = COL_BOX_DONE if (bx, by) in boxes_on_target else COL_BOX
        pygame.draw.rect(surface, colour, rect.inflate(-6, -6), border_radius=4)

    # Player (filled circle)
    px, py = state.player
    pcx = ox + px * cs + cs // 2
    pcy = oy + py * cs + cs // 2
    pygame.draw.circle(surface, COL_PLAYER, (pcx, pcy), cs // 3)


# ══════════════════════════════════════════════════════════════════════
# Stats Panel
# ══════════════════════════════════════════════════════════════════════

def draw_stats(
    surface: pygame.Surface,
    font: pygame.font.Font,
    font_big: pygame.font.Font,
    episode: int,
    all_rewards: List[float],
    all_solved: List[bool],
    deadlocks: int,
    avg_steps: float,
    mode: str,
    paused: bool,
) -> None:
    x = STATS_X
    y = 15

    def text(txt, f=font, color=COL_WHITE):
        nonlocal y
        s = f.render(txt, True, color)
        surface.blit(s, (x, y))
        y += s.get_height() + 4

    # Episode counter
    text(f"Episode: {episode} / {N_EPISODES}", font_big)

    # Solve rate
    if all_solved:
        recent = all_solved[-100:]
        sr = sum(recent) / len(recent) * 100
        col = COL_GREEN if sr > 50 else COL_YELLOW if sr > 20 else COL_RED
        text(f"Solve Rate: {sr:.1f}%", color=col)
    else:
        text("Solve Rate: --%")

    # Average reward
    if all_rewards:
        ar = sum(all_rewards[-100:]) / len(all_rewards[-100:])
        text(f"Avg Reward: {ar:.1f}")
    else:
        text("Avg Reward: --")

    # Deadlock rate
    if episode > 0:
        dr = deadlocks / episode * 100
        text(f"Deadlock Rate: {dr:.0f}%")
    else:
        text("Deadlock Rate: --%")

    # Average steps
    text(f"Avg Steps: {avg_steps:.0f}" if avg_steps > 0 else "Avg Steps: --")

    y += 8

    # Mode indicator
    mode_map = {
        "visual": "VISUAL",
        "fast":   "FAST",
        "slow":   "SLOW",
    }
    label = "PAUSED" if paused else mode_map.get(mode, mode.upper())
    text(f"Mode: {label}", color=COL_DIM)


# ══════════════════════════════════════════════════════════════════════
# Learning Curve Graph
# ══════════════════════════════════════════════════════════════════════

def draw_graph(
    surface: pygame.Surface,
    font: pygame.font.Font,
    all_solved: List[bool],
    rect: pygame.Rect,
) -> None:
    pygame.draw.rect(surface, COL_GRAPH_BG, rect, border_radius=4)

    lbl = font.render("Solve Rate (rolling 100)", True, COL_DIM)
    surface.blit(lbl, (rect.x + 4, rect.y + 2))

    graph_x = rect.x + 4
    graph_w = rect.width - 8
    graph_y = rect.y + 20
    graph_h = rect.height - 28

    # Axis lines
    pygame.draw.line(
        surface, COL_DIM,
        (graph_x, graph_y + graph_h),
        (graph_x + graph_w, graph_y + graph_h), 1,
    )
    pygame.draw.line(
        surface, COL_DIM,
        (graph_x, graph_y),
        (graph_x, graph_y + graph_h), 1,
    )

    # Dashed 50% line
    mid_y = graph_y + graph_h // 2
    for dx in range(0, graph_w, 6):
        pygame.draw.line(
            surface, COL_DASH,
            (graph_x + dx, mid_y),
            (graph_x + dx + 3, mid_y), 1,
        )

    if len(all_solved) < 2:
        return

    # Compute rolling solve rate
    window = 100
    rates: List[float] = []
    for i in range(len(all_solved)):
        start = max(0, i - window + 1)
        chunk = all_solved[start : i + 1]
        rates.append(sum(chunk) / len(chunk))

    # Build point list
    n = len(rates)
    x_scale = graph_w / max(n - 1, 1)
    points = []
    for i, r in enumerate(rates):
        px = graph_x + int(i * x_scale)
        py = graph_y + graph_h - int(r * graph_h)
        points.append((px, py))

    if len(points) >= 2:
        pygame.draw.lines(surface, COL_GRAPH_LINE, False, points, 2)


# ══════════════════════════════════════════════════════════════════════
# Action Probability Bars
# ══════════════════════════════════════════════════════════════════════

def draw_action_probs(
    surface: pygame.Surface,
    font: pygame.font.Font,
    probs: np.ndarray,
    rect: pygame.Rect,
) -> None:
    pygame.draw.rect(surface, COL_PANEL, rect, border_radius=4)

    labels = ["UP", "DN", "LT", "RT"]
    bar_h = 10
    bar_max_w = rect.width - 70
    y = rect.y + 6

    for lbl, p in zip(labels, probs):
        txt = font.render(f"{lbl} {p:.2f}", True, COL_DIM)
        surface.blit(txt, (rect.x + 4, y))
        bw = int(p * bar_max_w)
        bar_rect = pygame.Rect(rect.x + 50, y + 2, max(bw, 0), bar_h)
        pygame.draw.rect(surface, COL_PLAYER, bar_rect, border_radius=2)
        y += bar_h + 4


# ══════════════════════════════════════════════════════════════════════
# Training Loop Helpers
# ══════════════════════════════════════════════════════════════════════

class Trainer:
    """Encapsulates the PPO training state."""

    def __init__(self) -> None:
        self.env = SokobanEnv(level_set="training", max_h=10, max_w=10, max_steps=200)
        self.agent = PPOAgent(grid_h=10, grid_w=10)

        self.obs, _ = self.env.reset()
        self.ep_reward = 0.0
        self.ep_steps = 0

        self.episode = 0
        self.total_steps = 0
        self.deadlocks = 0

        self.all_rewards: List[float] = []
        self.all_solved:  List[bool]  = []
        self.all_ep_steps: List[int]  = []
        self.action_probs = np.array([0.25, 0.25, 0.25, 0.25])

    @property
    def avg_steps(self) -> float:
        if not self.all_ep_steps:
            return 0.0
        recent = self.all_ep_steps[-100:]
        return sum(recent) / len(recent)

    # ── Single step ───────────────────────────────────────────────────
    def do_one_step(self) -> bool:
        """Execute one environment step. Returns True if episode ended."""
        self.action_probs[:] = self.agent.get_action_probs(self.obs)
        action, log_prob, value = self.agent.select_action(self.obs)
        next_obs, reward, term, trunc, info = self.env.step(action)
        done = term or trunc

        self.agent.store_transition(self.obs, action, reward, log_prob, value, done)
        self.obs = next_obs
        self.ep_reward += reward
        self.ep_steps += 1
        self.total_steps += 1

        if done:
            self._finish_episode(info)

        # Trigger PPO update every STEPS_PER_UPDATE collected steps
        if self.total_steps % STEPS_PER_UPDATE == 0 and self.total_steps > 0:
            self.agent.update()

        return done

    def _finish_episode(self, info: dict) -> None:
        self.all_rewards.append(self.ep_reward)
        self.all_solved.append(info.get("solved", False))
        self.all_ep_steps.append(self.ep_steps)
        if info.get("deadlock", False):
            self.deadlocks += 1
        self.episode += 1

        self.obs, _ = self.env.reset()
        self.ep_reward = 0.0
        self.ep_steps = 0

    # ── Full episode ──────────────────────────────────────────────────
    def run_episode(self) -> None:
        """Run one complete episode."""
        while not self.do_one_step():
            pass

    @property
    def state(self):
        """Current SokobanState for rendering."""
        return self.env.state

    def close(self) -> None:
        self.env.close()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("TRAIN & WATCH -- PPO Learning Sokoban")
    clock = pygame.time.Clock()

    font     = pygame.font.SysFont("consolas,dejavusansmono,monospace", 15)
    font_big = pygame.font.SysFont("consolas,dejavusansmono,monospace", 18, bold=True)

    trainer = Trainer()

    mode = "visual"   # "visual", "fast", "slow"
    paused = False
    last_step_time = time.time()
    running = True

    while running:
        now = time.time()

        # ── Event handling ────────────────────────────────────────────
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

        # ── Training tick ─────────────────────────────────────────────
        if not paused and trainer.episode < N_EPISODES:
            if mode == "fast":
                for _ in range(10):
                    if trainer.episode >= N_EPISODES:
                        break
                    trainer.run_episode()

            elif mode == "visual":
                trainer.run_episode()

            elif mode == "slow":
                if now - last_step_time >= SLOW_DELAY:
                    last_step_time = now
                    trainer.do_one_step()

        # ── Render ────────────────────────────────────────────────────
        screen.fill(COL_BG)

        # Grid
        draw_grid(screen, trainer.state, GRID_AREA)
        pygame.draw.rect(screen, COL_GRID_LINE, GRID_AREA, 2, border_radius=4)

        # Title under grid
        title = font_big.render("PPO Agent -- Learning Sokoban", True, COL_WHITE)
        screen.blit(
            title,
            (GRID_AREA.centerx - title.get_width() // 2, GRID_AREA.bottom + 8),
        )

        # Stats panel
        draw_stats(
            screen, font, font_big,
            trainer.episode,
            trainer.all_rewards,
            trainer.all_solved,
            trainer.deadlocks,
            trainer.avg_steps,
            mode,
            paused,
        )

        # Learning curve graph
        draw_graph(screen, font, trainer.all_solved, GRAPH_RECT)

        # Action probability bars
        draw_action_probs(screen, font, trainer.action_probs, PROBS_RECT)

        pygame.display.flip()
        clock.tick(60)

    trainer.close()
    pygame.quit()


if __name__ == "__main__":
    main()
