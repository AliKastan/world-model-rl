"""TRAIN_AND_WATCH — Watch a PPO agent learn to solve puzzles in real-time.

Opens a PyGame window. Left side shows the puzzle, right side shows stats.
The agent starts DUMB (random moves) and you WATCH it get smarter.

Controls:
  Space  — Pause / Resume
  F      — Fast mode (skip rendering, train fast)
  S      — Slow mode (watch every step, 200ms delay)
  V      — Visual mode (show every episode, normal speed)
  1-5    — Change difficulty (resets training)
  ESC    — Quit
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional

import numpy as np
import pygame

from env.gym_env import PuzzleEnv
from env.objects import Box, Door, Floor, IceTile, Key, PressureSwitch, SwitchWall, Target, Wall
from agents.model_free.ppo_agent import PPOAgent


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

WIN_W, WIN_H = 920, 640
GRID_AREA = pygame.Rect(10, 10, 560, 560)
STATS_X = 590
STATS_W = 320
GRAPH_RECT = pygame.Rect(STATS_X, 380, STATS_W, 180)
PROBS_RECT = pygame.Rect(STATS_X, 570, STATS_W, 60)


# ── Grid renderer ──────────────────────────────────────────────────────
def draw_grid(surface: pygame.Surface, world, area: pygame.Rect) -> None:
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


# ── Stats renderer ─────────────────────────────────────────────────────
def draw_stats(surface: pygame.Surface, font: pygame.font.Font, font_big: pygame.font.Font,
               episode: int, n_episodes: int, difficulty: int,
               all_rewards: List[float], all_solved: List[bool],
               deadlocks: int, mode: str, paused: bool, level_up_msg: str) -> None:
    x = STATS_X
    y = 15

    def text(txt, f=font, color=COL_WHITE, yoff=0):
        nonlocal y
        s = f.render(txt, True, color)
        surface.blit(s, (x, y + yoff))
        y += s.get_height() + 4

    text(f"Episode: {episode:,} / {n_episodes:,}", font_big)
    text(f"Difficulty: {difficulty}")

    # Solve rate
    if all_solved:
        recent = all_solved[-100:]
        sr = sum(recent) / len(recent) * 100
        col = COL_GREEN if sr > 50 else COL_YELLOW if sr > 20 else COL_RED
        tag = "[+++]" if sr > 50 else "[++ ]" if sr > 20 else "[+  ]"
        text(f"Solve Rate: {sr:.1f}% {tag}", color=col)
    else:
        text("Solve Rate: --")

    # Avg reward
    if all_rewards:
        ar = sum(all_rewards[-100:]) / len(all_rewards[-100:])
        text(f"Avg Reward: {ar:.1f}")
    else:
        text("Avg Reward: --")

    # Deadlocks
    if episode > 0:
        dr = deadlocks / episode * 100
        text(f"Deadlock Rate: {dr:.0f}%")

    y += 8
    mode_labels = {"slow": "SLOW (every step)", "visual": "VISUAL (every episode)", "fast": "FAST (skip render)"}
    status = "PAUSED" if paused else mode_labels.get(mode, mode)
    text(f"Mode: {status}", color=COL_DIM)

    if all_solved:
        recent = all_solved[-200:]
        sr200 = sum(recent) / len(recent) * 100
        if sr200 > 80:
            text("Status: MASTERED!", color=COL_GREEN)
        elif sr200 > 50:
            text("Status: LEARNING!", color=COL_YELLOW)
        elif sr200 > 20:
            text("Status: IMPROVING...", color=COL_YELLOW)
        else:
            text("Status: EXPLORING...", color=COL_RED)

    if level_up_msg:
        text(level_up_msg, font_big, COL_GREEN)


def draw_graph(surface: pygame.Surface, font: pygame.font.Font,
               all_solved: List[bool], rect: pygame.Rect) -> None:
    pygame.draw.rect(surface, COL_PANEL, rect, border_radius=4)
    lbl = font.render("Solve Rate (rolling 100)", True, COL_DIM)
    surface.blit(lbl, (rect.x + 4, rect.y + 2))

    if len(all_solved) < 2:
        return

    # Compute rolling solve rate
    window = 100
    rates = []
    for i in range(len(all_solved)):
        start = max(0, i - window + 1)
        chunk = all_solved[start:i + 1]
        rates.append(sum(chunk) / len(chunk))

    n = len(rates)
    graph_y = rect.y + 20
    graph_h = rect.height - 28
    graph_x = rect.x + 4
    graph_w = rect.width - 8

    # Axis lines
    pygame.draw.line(surface, COL_DIM, (graph_x, graph_y + graph_h), (graph_x + graph_w, graph_y + graph_h), 1)
    pygame.draw.line(surface, COL_DIM, (graph_x, graph_y), (graph_x, graph_y + graph_h), 1)

    # 50% line
    mid_y = graph_y + graph_h // 2
    for dx in range(0, graph_w, 6):
        pygame.draw.line(surface, (50, 50, 70), (graph_x + dx, mid_y), (graph_x + dx + 3, mid_y), 1)

    # Plot
    x_scale = graph_w / max(n - 1, 1)
    points = []
    for i, r in enumerate(rates):
        px = graph_x + int(i * x_scale)
        py = graph_y + graph_h - int(r * graph_h)
        points.append((px, py))

    if len(points) >= 2:
        pygame.draw.lines(surface, COL_GREEN, False, points, 2)


def draw_action_probs(surface: pygame.Surface, font: pygame.font.Font,
                      probs: np.ndarray, rect: pygame.Rect) -> None:
    pygame.draw.rect(surface, COL_PANEL, rect, border_radius=4)
    labels = ["UP", "DN", "LT", "RT"]
    bar_h = 10
    bar_max_w = rect.width - 70
    y = rect.y + 6
    for i, (lbl, p) in enumerate(zip(labels, probs)):
        txt = font.render(f"{lbl} {p:.2f}", True, COL_DIM)
        surface.blit(txt, (rect.x + 4, y))
        bw = int(p * bar_max_w)
        bar_rect = pygame.Rect(rect.x + 50, y + 2, bw, bar_h)
        pygame.draw.rect(surface, COL_AGENT, bar_rect, border_radius=2)
        y += bar_h + 4


# ── Main ───────────────────────────────────────────────────────────────
def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("TRAIN & WATCH — PPO Learning Live")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 15)
    font_big = pygame.font.SysFont("consolas,dejavusansmono,monospace", 18, bold=True)

    difficulty = 1
    n_episodes = 10_000
    update_every = 10

    def make_env_agent():
        e = PuzzleEnv(difficulty=difficulty, max_steps=200)
        a = PPOAgent()
        return e, a

    env, agent = make_env_agent()

    # Training state
    obs, info = env.reset()
    ep_reward = 0.0
    episode = 0
    all_rewards: List[float] = []
    all_solved: List[bool] = []
    deadlocks = 0
    action_probs = np.array([0.25, 0.25, 0.25, 0.25])

    # Display state
    mode = "visual"  # "slow", "visual", "fast"
    paused = False
    level_up_msg = ""
    level_up_time = 0.0

    running = True
    slow_delay = 0.15
    last_step_time = time.time()

    def finish_episode():
        nonlocal obs, ep_reward, episode, deadlocks
        all_rewards.append(ep_reward)
        all_solved.append(info.get("solved", False))
        if info.get("is_deadlock", False):
            deadlocks += 1
        episode += 1
        if episode % update_every == 0:
            agent.update()
        obs, _ = env.reset()
        ep_reward = 0.0

    def do_one_step():
        nonlocal obs, ep_reward, info
        action_probs[:] = agent.get_action_probs(obs)
        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        agent.memory.store(obs, action, reward, log_prob, value, done)
        obs = next_obs
        ep_reward += reward
        return done

    def run_episode():
        while not do_one_step():
            pass
        finish_episode()

    def check_level_up():
        nonlocal difficulty, level_up_msg, level_up_time, env, agent
        nonlocal obs, ep_reward, episode, all_rewards, all_solved, deadlocks
        if len(all_solved) >= 200:
            recent = all_solved[-200:]
            sr = sum(recent) / len(recent)
            if sr > 0.80 and difficulty < 5:
                difficulty += 1
                level_up_msg = f"LEVEL UP! Difficulty {difficulty}"
                level_up_time = time.time()
                env.close()
                env, agent = make_env_agent()
                obs, _ = env.reset()
                ep_reward = 0.0
                episode = 0
                all_rewards.clear()
                all_solved.clear()
                deadlocks = 0

    while running:
        now = time.time()

        # Clear level-up message after 3 seconds
        if level_up_msg and now - level_up_time > 3.0:
            level_up_msg = ""

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
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    new_diff = event.key - pygame.K_0
                    if new_diff != difficulty:
                        difficulty = new_diff
                        env.close()
                        env, agent = make_env_agent()
                        obs, _ = env.reset()
                        ep_reward = 0.0
                        episode = 0
                        all_rewards.clear()
                        all_solved.clear()
                        deadlocks = 0
                        level_up_msg = ""

        if not paused and episode < n_episodes:
            if mode == "fast":
                # Run many episodes without rendering each step
                for _ in range(10):
                    if episode >= n_episodes:
                        break
                    run_episode()
                check_level_up()
            elif mode == "visual":
                run_episode()
                check_level_up()
            elif mode == "slow":
                if now - last_step_time >= slow_delay:
                    last_step_time = now
                    done = do_one_step()
                    if done:
                        finish_episode()
                        check_level_up()

        # ── Render ─────────────────────────────────────────────────────
        screen.fill(COL_BG)

        # Grid
        world = getattr(env, "_world", None)
        draw_grid(screen, world, GRID_AREA)

        # Grid border
        pygame.draw.rect(screen, COL_GRID_LINE, GRID_AREA, 2, border_radius=4)

        # Title
        title = font_big.render("PPO Agent — Learning Live", True, COL_WHITE)
        screen.blit(title, (GRID_AREA.centerx - title.get_width() // 2, GRID_AREA.bottom + 8))

        # Stats
        draw_stats(screen, font, font_big, episode, n_episodes, difficulty,
                   all_rewards, all_solved, deadlocks, mode, paused, level_up_msg)

        # Graph
        draw_graph(screen, font, all_solved, GRAPH_RECT)

        # Action probs
        draw_action_probs(screen, font, action_probs, PROBS_RECT)

        pygame.display.flip()
        clock.tick(60)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
