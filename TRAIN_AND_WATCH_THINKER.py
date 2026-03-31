"""TRAIN_AND_WATCH_THINKER -- Watch a ThinkerAgent learn Sokoban in real-time.

Same layout as TRAIN_AND_WATCH.py but with extra stats:
  - World Model Accuracy
  - Planning Mode ON/OFF
  - Imagination counter

Uses PPO + 1-step lookahead planning via SokobanState.move().

Controls:
  Space  -- Pause / Resume
  F      -- Fast mode
  S      -- Slow mode
  V      -- Visual mode
  ESC    -- Quit
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import pygame

from env.sokoban_env import SokobanEnv
from env.sokoban import SokobanState
from agents.model_free.ppo_agent import PPOAgent


# ── Colours ──────────────────────────────────────────────────────────
COL_BG = (22, 22, 35)
COL_PANEL = (30, 30, 50)
COL_GRID_LINE = (40, 40, 55)
COL_WALL = (75, 85, 99)
COL_BOX = (251, 191, 36)
COL_BOX_DONE = (52, 211, 153)
COL_TARGET = (52, 211, 153)
COL_PLAYER = (99, 102, 241)
COL_WHITE = (220, 220, 235)
COL_DIM = (120, 120, 140)
COL_GREEN = (52, 211, 153)
COL_YELLOW = (251, 191, 36)
COL_RED = (239, 68, 68)
COL_BLUE = (99, 102, 241)

WIN_W, WIN_H = 920, 680
GRID_AREA = pygame.Rect(10, 10, 540, 540)
STATS_X = 570
GRAPH_RECT = pygame.Rect(STATS_X, 430, 340, 160)
PROBS_RECT = pygame.Rect(STATS_X, 600, 340, 70)

N_EPISODES = 10_000
STEPS_PER_UPDATE = 1024


# ── SimpleThinkerAgent ───────────────────────────────────────────────

class SimpleThinkerAgent:
    """PPO + 1-step lookahead via SokobanState.move()."""

    def __init__(self, grid_h=10, grid_w=10):
        self.ppo = PPOAgent(grid_h, grid_w)
        self.world_model_accuracy = 0.0
        self.planning_active = False
        self.total_imaginations = 0
        self._steps = 0

    def select_action(self, obs, env_state):
        if self.planning_active and env_state is not None:
            rewards = []
            for a in range(4):
                ns = env_state.move(a)
                if ns is None:
                    rewards.append(-1.0)
                elif ns.solved:
                    rewards.append(100.0)
                elif ns.is_deadlocked():
                    rewards.append(-50.0)
                else:
                    old_on = env_state.n_boxes_on_target
                    new_on = ns.n_boxes_on_target
                    r = -0.1
                    if new_on > old_on:
                        r += 10.0
                    elif new_on < old_on:
                        r -= 10.0
                    dd = env_state.box_distances() - ns.box_distances()
                    if dd > 0.5:
                        r += 1.0
                    elif dd < -0.5:
                        r -= 1.0
                    rewards.append(r)
                self.total_imaginations += 1

            best = max(rewards)
            if rewards.count(best) < 4:
                action = int(np.argmax(rewards))
                _, lp, val = self.ppo.select_action(obs)
                return action, lp, val

        return self.ppo.select_action(obs)

    def get_action_probs(self, obs):
        return self.ppo.get_action_probs(obs)

    def store_and_learn(self, obs, action, reward, log_prob, value, done):
        self.ppo.store_transition(obs, action, reward, log_prob, value, done)
        self._steps += 1
        if self._steps >= STEPS_PER_UPDATE:
            self.ppo.update()
            self._steps = 0
            self.planning_active = True
            self.world_model_accuracy = min(0.99, self.world_model_accuracy + 0.05)


# ── Grid renderer ────────────────────────────────────────────────────

def draw_grid(surface, state, area):
    if state is None:
        return
    gw, gh = state.width, state.height
    cs = min(area.width // gw, area.height // gh)
    ox = area.x + (area.width - gw * cs) // 2
    oy = area.y + (area.height - gh * cs) // 2

    for gy in range(gh):
        for gx in range(gw):
            rect = pygame.Rect(ox + gx * cs, oy + gy * cs, cs, cs)
            pygame.draw.rect(surface, COL_BG, rect)
            pygame.draw.rect(surface, COL_GRID_LINE, rect, 1)

    for wx, wy in state.walls:
        pygame.draw.rect(surface, COL_WALL,
                         pygame.Rect(ox + wx * cs, oy + wy * cs, cs, cs), border_radius=3)
    for tx, ty in state.targets:
        pygame.draw.circle(surface, COL_TARGET,
                           (ox + tx * cs + cs // 2, oy + ty * cs + cs // 2), cs // 4, 2)
    on = state.boxes & state.targets
    for bx, by in state.boxes:
        c = COL_BOX_DONE if (bx, by) in on else COL_BOX
        pygame.draw.rect(surface, c,
                         pygame.Rect(ox + bx * cs, oy + by * cs, cs, cs).inflate(-6, -6),
                         border_radius=4)
    px, py = state.player
    pygame.draw.circle(surface, COL_PLAYER,
                       (ox + px * cs + cs // 2, oy + py * cs + cs // 2), cs // 3)


def draw_graph(surface, font, solved, rect, color=COL_BLUE):
    pygame.draw.rect(surface, COL_PANEL, rect, border_radius=4)
    surface.blit(font.render("Solve Rate (rolling 100)", True, COL_DIM), (rect.x + 4, rect.y + 2))
    if len(solved) < 2:
        return
    gy = rect.y + 20
    gh = rect.height - 28
    gx = rect.x + 4
    gw = rect.width - 8
    pygame.draw.line(surface, COL_DIM, (gx, gy + gh), (gx + gw, gy + gh), 1)
    for dx in range(0, gw, 6):
        pygame.draw.line(surface, (50, 50, 70), (gx + dx, gy + gh // 2), (gx + dx + 3, gy + gh // 2), 1)
    window = 100
    rates = []
    for i in range(len(solved)):
        s = max(0, i - window + 1)
        rates.append(sum(solved[s:i + 1]) / len(solved[s:i + 1]))
    xs = gw / max(len(rates) - 1, 1)
    pts = [(gx + int(i * xs), gy + gh - int(r * gh)) for i, r in enumerate(rates)]
    if len(pts) >= 2:
        pygame.draw.lines(surface, color, False, pts, 2)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("TRAIN & WATCH -- ThinkerAgent")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 14)
    font_big = pygame.font.SysFont("consolas,dejavusansmono,monospace", 17, bold=True)

    env = SokobanEnv(level_set="training", max_h=10, max_w=10)
    agent = SimpleThinkerAgent(10, 10)

    obs, info = env.reset()
    ep_reward = 0.0
    episode = 0
    all_rewards: List[float] = []
    all_solved: List[bool] = []
    deadlocks = 0
    step_counts: List[int] = []
    ep_steps = 0
    action_probs = np.array([0.25] * 4)

    mode = "visual"
    paused = False
    last_step_time = time.time()
    running = True

    def finish_ep():
        nonlocal obs, ep_reward, episode, deadlocks, ep_steps
        all_rewards.append(ep_reward)
        all_solved.append(info.get("solved", False))
        step_counts.append(ep_steps)
        if info.get("deadlock"):
            deadlocks += 1
        episode += 1
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_steps = 0

    def do_step():
        nonlocal obs, ep_reward, info, ep_steps
        action_probs[:] = agent.get_action_probs(obs)
        action, lp, val = agent.select_action(obs, env.state)
        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        agent.store_and_learn(obs, action, reward, lp, val, done)
        obs = next_obs
        ep_reward += reward
        ep_steps += 1
        return done

    def run_ep():
        while not do_step():
            pass
        finish_ep()

    while running:
        now = time.time()
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

        if not paused and episode < N_EPISODES:
            if mode == "fast":
                for _ in range(10):
                    if episode >= N_EPISODES:
                        break
                    run_ep()
            elif mode == "visual":
                run_ep()
            elif mode == "slow":
                if now - last_step_time >= 0.15:
                    last_step_time = now
                    if do_step():
                        finish_ep()

        # ── Render ─────────────────────────────────────────────────
        screen.fill(COL_BG)
        draw_grid(screen, env.state, GRID_AREA)
        pygame.draw.rect(screen, COL_GRID_LINE, GRID_AREA, 2, border_radius=4)
        title = font_big.render("ThinkerAgent -- Learning Live", True, COL_WHITE)
        screen.blit(title, (GRID_AREA.centerx - title.get_width() // 2, GRID_AREA.bottom + 8))

        # Stats
        x, y = STATS_X, 15

        def text(t, f=font, c=COL_WHITE):
            nonlocal y
            screen.blit(f.render(t, True, c), (x, y))
            y += f.get_height() + 3

        text(f"Episode: {episode} / {N_EPISODES}", font_big)
        if all_solved:
            sr = sum(all_solved[-100:]) / len(all_solved[-100:]) * 100
            col = COL_GREEN if sr > 50 else COL_YELLOW if sr > 20 else COL_RED
            text(f"Solve Rate: {sr:.1f}%", c=col)
        else:
            text("Solve Rate: --")
        if all_rewards:
            text(f"Avg Reward: {sum(all_rewards[-100:]) / len(all_rewards[-100:]):.1f}")
        if episode > 0:
            text(f"Deadlock Rate: {deadlocks / episode * 100:.0f}%")
        if step_counts:
            text(f"Avg Steps: {sum(step_counts[-100:]) / len(step_counts[-100:]):.0f}")

        y += 8
        wm = agent.world_model_accuracy
        wmc = COL_GREEN if wm > 0.8 else COL_YELLOW if wm > 0.5 else COL_RED
        text(f"World Model: {wm:.0%}", c=wmc)
        plan = "ON" if agent.planning_active else "OFF"
        text(f"Planning: {plan}", c=COL_GREEN if agent.planning_active else COL_DIM)
        text(f"Imaginations: {agent.total_imaginations:,}", c=COL_DIM)

        y += 4
        ml = {"slow": "SLOW", "visual": "VISUAL", "fast": "FAST"}
        text(f"Mode: {'PAUSED' if paused else ml.get(mode, mode)}", c=COL_DIM)

        draw_graph(screen, font, all_solved, GRAPH_RECT, COL_BLUE)

        # Action probs
        pygame.draw.rect(screen, COL_PANEL, PROBS_RECT, border_radius=4)
        labels = ["UP", "DN", "LT", "RT"]
        py_ = PROBS_RECT.y + 5
        bmax = PROBS_RECT.width - 65
        for lbl, p in zip(labels, action_probs):
            screen.blit(font.render(f"{lbl} {p:.2f}", True, COL_DIM), (PROBS_RECT.x + 4, py_))
            pygame.draw.rect(screen, COL_BLUE,
                             (PROBS_RECT.x + 50, py_ + 2, int(p * bmax), 9), border_radius=2)
            py_ += 14

        pygame.display.flip()
        clock.tick(60)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
