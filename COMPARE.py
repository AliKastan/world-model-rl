"""COMPARE — PPO vs ThinkerAgent side by side.

Both agents train on the SAME puzzles simultaneously.
Watch PPO struggle while ThinkerAgent plans ahead and learns faster.

Controls:
  Space  — Pause / Resume
  F      — Fast mode (train many episodes, render periodically)
  S      — Slow mode (watch every step)
  V      — Visual mode (show every episode)
  1-3    — Change difficulty
  ESC    — Quit
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import pygame

from env.gym_env import PuzzleEnv
from env.objects import Box, Door, Floor, IceTile, Key, PressureSwitch, SwitchWall, Target, Wall
from agents.model_free.ppo_agent import PPOAgent
from agents.model_based.thinker_agent import ThinkerAgent


# ── Colours ────────────────────────────────────────────────────────────
COL_BG = (22, 22, 35)
COL_PANEL = (30, 30, 50)
COL_GRID_LINE = (40, 40, 55)
COL_WALL = (75, 85, 99)
COL_BOX = (251, 191, 36)
COL_BOX_DONE = (52, 211, 153)
COL_TARGET = (52, 211, 153)
COL_AGENT = (99, 102, 241)
COL_KEY = (251, 146, 60)
COL_DOOR_LOCKED = (239, 68, 68)
COL_DOOR_OPEN = (74, 222, 128)
COL_ICE = (147, 197, 253)
COL_SWITCH = (168, 85, 247)
COL_WHITE = (220, 220, 235)
COL_DIM = (120, 120, 140)
COL_GREEN = (52, 211, 153)
COL_YELLOW = (251, 191, 36)
COL_RED = (239, 68, 68)
COL_BLUE = (99, 102, 241)

WIN_W, WIN_H = 1200, 720
LEFT_GRID = pygame.Rect(10, 50, 400, 400)
RIGHT_GRID = pygame.Rect(620, 50, 400, 400)
GRAPH_RECT = pygame.Rect(30, 540, 1140, 160)


def draw_grid(surface, world, area):
    if world is None:
        return
    cs = min(area.width // world.width, area.height // world.height)
    ox = area.x + (area.width - world.width * cs) // 2
    oy = area.y + (area.height - world.height * cs) // 2
    for gy in range(world.height):
        for gx in range(world.width):
            rect = pygame.Rect(ox + gx * cs, oy + gy * cs, cs, cs)
            pygame.draw.rect(surface, COL_BG, rect)
            pygame.draw.rect(surface, COL_GRID_LINE, rect, 1)
            cell = world.get_cell(gx, gy)
            for obj in cell:
                if isinstance(obj, Wall):
                    pygame.draw.rect(surface, COL_WALL, rect, border_radius=3)
                elif isinstance(obj, Target):
                    pygame.draw.circle(surface, COL_TARGET, rect.center, cs // 4, 2)
                elif isinstance(obj, Box):
                    c = COL_BOX_DONE if obj.on_target else COL_BOX
                    pygame.draw.rect(surface, c, rect.inflate(-6, -6), border_radius=4)
                elif isinstance(obj, Key):
                    pygame.draw.circle(surface, COL_KEY, rect.center, cs // 5)
                elif isinstance(obj, Door):
                    c = COL_DOOR_LOCKED if obj.locked else COL_DOOR_OPEN
                    pygame.draw.rect(surface, c, rect.inflate(-4, -4), border_radius=4)
                elif isinstance(obj, IceTile):
                    s = pygame.Surface((cs, cs), pygame.SRCALPHA)
                    pygame.draw.rect(s, (*COL_ICE, 60), (0, 0, cs, cs), border_radius=3)
                    surface.blit(s, rect)
                elif isinstance(obj, PressureSwitch):
                    pad = cs // 4
                    pygame.draw.rect(surface, COL_SWITCH, rect.inflate(-pad*2, -pad*2), border_radius=3)
            if (gx, gy) == world.agent_pos:
                pygame.draw.circle(surface, COL_AGENT, rect.center, cs // 3)


# ── Per-agent training state ───────────────────────────────────────────
class AgentState:
    def __init__(self, name: str, env: PuzzleEnv, agent: Any, color: tuple,
                 is_thinker: bool = False):
        self.name = name
        self.env = env
        self.agent = agent
        self.color = color
        self.is_thinker = is_thinker

        self.obs, self.info = env.reset()
        self.ep_reward = 0.0
        self.episode = 0
        self.all_rewards: List[float] = []
        self.all_solved: List[bool] = []
        self.deadlocks = 0

    def run_episode(self):
        obs = self.obs
        ep_reward = 0.0
        done = False

        while not done:
            if self.is_thinker:
                action, _ = self.agent.select_action(obs)
            else:
                action, log_prob, value = self.agent.select_action(obs)

            next_obs, reward, term, trunc, info = self.env.step(action)
            done = term or trunc

            if self.is_thinker:
                self.agent.store_experience(obs, action, next_obs, reward, done)
            else:
                self.agent.memory.store(obs, action, reward, log_prob, value, done)

            obs = next_obs
            ep_reward += reward

        self.info = info
        self.all_rewards.append(ep_reward)
        self.all_solved.append(info.get("solved", False))
        if info.get("is_deadlock", False):
            self.deadlocks += 1
        self.episode += 1

        # PPO update
        if not self.is_thinker and self.episode % 10 == 0:
            self.agent.update()

        # Thinker learning
        if self.is_thinker:
            if self.episode % 10 == 0:
                self.agent.learn_world_model(train_steps=5)
            if self.episode % 50 == 0:
                self.agent.dream_and_learn(n_episodes=20, max_steps=20)

        self.obs, _ = self.env.reset()
        self.ep_reward = 0.0

    @property
    def solve_rate(self):
        if not self.all_solved:
            return 0.0
        recent = self.all_solved[-100:]
        return sum(recent) / len(recent) * 100

    @property
    def avg_reward(self):
        if not self.all_rewards:
            return 0.0
        recent = self.all_rewards[-100:]
        return sum(recent) / len(recent)


def draw_dual_graph(surface, font, ppo_solved, thinker_solved, rect):
    pygame.draw.rect(surface, COL_PANEL, rect, border_radius=4)
    lbl = font.render("Learning Curve  (PPO = red, Thinker = blue)", True, COL_DIM)
    surface.blit(lbl, (rect.x + 8, rect.y + 4))

    gy = rect.y + 22
    gh = rect.height - 30
    gx = rect.x + 8
    gw = rect.width - 16

    pygame.draw.line(surface, COL_DIM, (gx, gy + gh), (gx + gw, gy + gh), 1)
    for dx in range(0, gw, 6):
        pygame.draw.line(surface, (50, 50, 70), (gx + dx, gy + gh // 2),
                         (gx + dx + 3, gy + gh // 2), 1)

    def plot(solved_list, color):
        if len(solved_list) < 2:
            return
        window = 100
        rates = []
        for i in range(len(solved_list)):
            s = max(0, i - window + 1)
            c = solved_list[s:i + 1]
            rates.append(sum(c) / len(c))
        xs = gw / max(len(rates) - 1, 1)
        pts = [(gx + int(i * xs), gy + gh - int(r * gh)) for i, r in enumerate(rates)]
        if len(pts) >= 2:
            pygame.draw.lines(surface, color, False, pts, 2)

    plot(ppo_solved, COL_RED)
    plot(thinker_solved, COL_BLUE)

    # Legend
    lx = rect.right - 200
    ly = rect.y + 6
    pygame.draw.line(surface, COL_RED, (lx, ly + 6), (lx + 20, ly + 6), 2)
    surface.blit(font.render("PPO", True, COL_RED), (lx + 24, ly))
    pygame.draw.line(surface, COL_BLUE, (lx + 70, ly + 6), (lx + 90, ly + 6), 2)
    surface.blit(font.render("Thinker", True, COL_BLUE), (lx + 94, ly))


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("COMPARE — PPO vs ThinkerAgent")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 14)
    font_big = pygame.font.SysFont("consolas,dejavusansmono,monospace", 17, bold=True)
    font_title = pygame.font.SysFont("consolas,dejavusansmono,monospace", 20, bold=True)

    difficulty = 1

    def make_states():
        ppo_env = PuzzleEnv(difficulty=difficulty, max_steps=200)
        thinker_env = PuzzleEnv(difficulty=difficulty, max_steps=200)
        ppo = AgentState("PPO (Brute Force)", ppo_env, PPOAgent(), COL_RED)
        thinker = AgentState("ThinkerAgent (Think First)", thinker_env, ThinkerAgent(),
                             COL_BLUE, is_thinker=True)
        return ppo, thinker

    ppo_state, thinker_state = make_states()

    mode = "visual"
    paused = False
    running = True
    slow_delay = 0.2
    last_step_time = time.time()

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
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    new_d = event.key - pygame.K_0
                    if new_d != difficulty:
                        difficulty = new_d
                        ppo_state.env.close()
                        thinker_state.env.close()
                        ppo_state, thinker_state = make_states()

        if not paused:
            if mode == "fast":
                for _ in range(5):
                    ppo_state.run_episode()
                    thinker_state.run_episode()
            elif mode == "visual":
                ppo_state.run_episode()
                thinker_state.run_episode()
            elif mode == "slow":
                if now - last_step_time >= slow_delay:
                    last_step_time = now
                    ppo_state.run_episode()
                    thinker_state.run_episode()

        # ── Render ─────────────────────────────────────────────────────
        screen.fill(COL_BG)

        # Title bar
        title = font_title.render("Think Before You Act  --  PPO vs ThinkerAgent", True, COL_WHITE)
        screen.blit(title, (WIN_W // 2 - title.get_width() // 2, 10))

        # Divider
        pygame.draw.line(screen, COL_GRID_LINE, (WIN_W // 2, 40), (WIN_W // 2, 530), 1)

        # Grids
        draw_grid(screen, getattr(ppo_state.env, "_world", None), LEFT_GRID)
        pygame.draw.rect(screen, COL_RED, LEFT_GRID, 2, border_radius=4)
        draw_grid(screen, getattr(thinker_state.env, "_world", None), RIGHT_GRID)
        pygame.draw.rect(screen, COL_BLUE, RIGHT_GRID, 2, border_radius=4)

        # Labels
        def draw_agent_stats(state: AgentState, x: int, is_thinker: bool = False):
            y = 460
            def txt(t, f=font, c=COL_WHITE):
                nonlocal y
                screen.blit(f.render(t, True, c), (x, y))
                y += 16

            txt(state.name, font_big, state.color)
            txt(f"Episode: {state.episode:,}")
            sr = state.solve_rate
            col = COL_GREEN if sr > 50 else COL_YELLOW if sr > 20 else COL_RED
            txt(f"Solve Rate: {sr:.1f}%", color=col)
            txt(f"Avg Reward: {state.avg_reward:.1f}")
            if state.episode > 0:
                txt(f"Deadlocks: {state.deadlocks / state.episode * 100:.0f}%")
            if is_thinker:
                wm = state.agent.world_model_accuracy
                wm_col = COL_GREEN if wm > 0.8 else COL_YELLOW if wm > 0.5 else COL_RED
                txt(f"World Model: {wm:.0%}", color=wm_col)
                plan = "ON" if state.agent.planning_active else "OFF"
                txt(f"Planning: {plan}", color=COL_GREEN if state.agent.planning_active else COL_DIM)

        draw_agent_stats(ppo_state, 20)
        draw_agent_stats(thinker_state, 630, is_thinker=True)

        # Mode
        mode_txt = f"Mode: {'PAUSED' if paused else mode.upper()} | Difficulty: {difficulty}"
        screen.blit(font.render(mode_txt, True, COL_DIM), (WIN_W // 2 - 100, WIN_H - 18))

        # Dual graph
        draw_dual_graph(screen, font, ppo_state.all_solved, thinker_state.all_solved, GRAPH_RECT)

        pygame.display.flip()
        clock.tick(60)

    ppo_state.env.close()
    thinker_state.env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
