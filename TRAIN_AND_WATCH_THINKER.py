"""TRAIN_AND_WATCH_THINKER — Watch the ThinkerAgent learn with world model + planning.

Same layout as TRAIN_AND_WATCH.py but with extra stats:
  - World Model Accuracy
  - Planning Mode ON/OFF
  - Imagination counter (dreamed episodes)

Controls:
  Space  — Pause / Resume
  F      — Fast mode
  S      — Slow mode
  V      — Visual mode
  1-5    — Change difficulty
  ESC    — Quit
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
import pygame

from env.gym_env import PuzzleEnv
from env.objects import Box, Door, Floor, IceTile, Key, PressureSwitch, SwitchWall, Target, Wall
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

WIN_W, WIN_H = 920, 680
GRID_AREA = pygame.Rect(10, 10, 540, 540)
STATS_X = 570
STATS_W = 340
GRAPH_RECT = pygame.Rect(STATS_X, 420, STATS_W, 170)
PROBS_RECT = pygame.Rect(STATS_X, 600, STATS_W, 70)


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


def draw_graph(surface, font, all_solved, rect, color=COL_GREEN, label="Solve Rate"):
    pygame.draw.rect(surface, COL_PANEL, rect, border_radius=4)
    lbl = font.render(f"{label} (rolling 100)", True, COL_DIM)
    surface.blit(lbl, (rect.x + 4, rect.y + 2))
    if len(all_solved) < 2:
        return
    window = 100
    rates = []
    for i in range(len(all_solved)):
        start = max(0, i - window + 1)
        chunk = all_solved[start:i + 1]
        rates.append(sum(chunk) / len(chunk))
    gy = rect.y + 20
    gh = rect.height - 28
    gx = rect.x + 4
    gw = rect.width - 8
    pygame.draw.line(surface, COL_DIM, (gx, gy + gh), (gx + gw, gy + gh), 1)
    for dx in range(0, gw, 6):
        pygame.draw.line(surface, (50, 50, 70), (gx + dx, gy + gh // 2), (gx + dx + 3, gy + gh // 2), 1)
    xs = gw / max(len(rates) - 1, 1)
    pts = [(gx + int(i * xs), gy + gh - int(r * gh)) for i, r in enumerate(rates)]
    if len(pts) >= 2:
        pygame.draw.lines(surface, color, False, pts, 2)


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("TRAIN & WATCH — ThinkerAgent Learning Live")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 14)
    font_big = pygame.font.SysFont("consolas,dejavusansmono,monospace", 17, bold=True)

    difficulty = 1
    n_episodes = 5_000
    wm_train_every = 10
    dream_every = 50

    def make_env_agent():
        return PuzzleEnv(difficulty=difficulty, max_steps=200), ThinkerAgent()

    env, agent = make_env_agent()
    obs, info = env.reset()
    ep_reward = 0.0
    episode = 0
    all_rewards: List[float] = []
    all_solved: List[bool] = []
    deadlocks = 0
    action_probs = np.array([0.25] * 4)

    mode = "visual"
    paused = False
    level_up_msg = ""
    level_up_time = 0.0
    slow_delay = 0.15
    last_step_time = time.time()
    running = True

    def finish_episode():
        nonlocal obs, ep_reward, episode, deadlocks
        all_rewards.append(ep_reward)
        all_solved.append(info.get("solved", False))
        if info.get("is_deadlock", False):
            deadlocks += 1
        episode += 1
        if episode % wm_train_every == 0:
            agent.learn_world_model(train_steps=5)
        if episode % dream_every == 0:
            agent.dream_and_learn(n_episodes=20, max_steps=20)
        obs, _ = env.reset()
        ep_reward = 0.0

    def do_one_step():
        nonlocal obs, ep_reward, info
        action_probs[:] = agent.get_action_probs(obs)
        action, thought = agent.select_action(obs)
        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        agent.store_experience(obs, action, next_obs, reward, done)
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
            sr = sum(all_solved[-200:]) / 200
            if sr > 0.80 and difficulty < 5:
                difficulty += 1
                level_up_msg = f"LEVEL UP! Difficulty {difficulty}"
                level_up_time = time.time()
                env.close()
                env, agent = make_env_agent()
                obs, _ = env.reset()
                ep_reward = episode = deadlocks = 0
                all_rewards.clear()
                all_solved.clear()

    while running:
        now = time.time()
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
                        ep_reward = episode = deadlocks = 0
                        all_rewards.clear()
                        all_solved.clear()
                        level_up_msg = ""

        if not paused and episode < n_episodes:
            if mode == "fast":
                for _ in range(5):
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
                    if do_one_step():
                        finish_episode()
                        check_level_up()

        # ── Render ─────────────────────────────────────────────────────
        screen.fill(COL_BG)
        draw_grid(screen, getattr(env, "_world", None), GRID_AREA)
        pygame.draw.rect(screen, COL_GRID_LINE, GRID_AREA, 2, border_radius=4)

        title = font_big.render("ThinkerAgent — Learning Live", True, COL_WHITE)
        screen.blit(title, (GRID_AREA.centerx - title.get_width() // 2, GRID_AREA.bottom + 8))

        # Stats
        x, y = STATS_X, 15
        def text(txt, f=font, color=COL_WHITE):
            nonlocal y
            s = f.render(txt, True, color)
            screen.blit(s, (x, y))
            y += s.get_height() + 3

        text(f"Episode: {episode:,} / {n_episodes:,}", font_big)
        text(f"Difficulty: {difficulty}")

        if all_solved:
            sr = sum(all_solved[-100:]) / len(all_solved[-100:]) * 100
            col = COL_GREEN if sr > 50 else COL_YELLOW if sr > 20 else COL_RED
            text(f"Solve Rate: {sr:.1f}%", color=col)
        else:
            text("Solve Rate: --")
        if all_rewards:
            text(f"Avg Reward: {sum(all_rewards[-100:]) / len(all_rewards[-100:]):.1f}")
        if episode > 0:
            text(f"Deadlock Rate: {deadlocks / episode * 100:.0f}%")

        y += 6
        wm_acc = agent.world_model_accuracy
        wm_col = COL_GREEN if wm_acc > 0.8 else COL_YELLOW if wm_acc > 0.5 else COL_RED
        text(f"World Model: {wm_acc:.0%}", color=wm_col)
        plan_str = "ON" if agent.planning_active else "OFF"
        plan_col = COL_GREEN if agent.planning_active else COL_DIM
        text(f"Planning: {plan_str}", color=plan_col)
        text(f"Dreamed: {agent.total_dreams} episodes", color=COL_DIM)
        text(f"Buffer: {len(agent.replay_buffer)} transitions", color=COL_DIM)

        y += 4
        mode_labels = {"slow": "SLOW", "visual": "VISUAL", "fast": "FAST"}
        text(f"Mode: {'PAUSED' if paused else mode_labels.get(mode, mode)}", color=COL_DIM)
        if level_up_msg:
            text(level_up_msg, font_big, COL_GREEN)

        draw_graph(screen, font, all_solved, GRAPH_RECT, COL_BLUE)

        # Action probs
        pygame.draw.rect(screen, COL_PANEL, PROBS_RECT, border_radius=4)
        labels = ["UP", "DN", "LT", "RT"]
        bar_max = PROBS_RECT.width - 65
        py = PROBS_RECT.y + 5
        for lbl, p in zip(labels, action_probs):
            t = font.render(f"{lbl} {p:.2f}", True, COL_DIM)
            screen.blit(t, (PROBS_RECT.x + 4, py))
            bw = int(p * bar_max)
            pygame.draw.rect(screen, COL_BLUE, (PROBS_RECT.x + 50, py + 2, bw, 9), border_radius=2)
            py += 14

        pygame.display.flip()
        clock.tick(60)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
