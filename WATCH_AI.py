"""WATCH_AI - Watch AI solve classic Sokoban puzzles.

Modes:
  --solver   BFS optimal solver (default) - finds shortest solution
  --rl       Trained PPO agent - watch the neural network play

Controls:
  Space  - pause / resume
  S      - slow down
  F      - speed up
  N      - next level
  P      - previous level
  ESC    - quit
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import pygame

from env.sokoban import SokobanState, solve, DIR_NAMES
from env.level_loader import LevelLoader
from env.sokoban_renderer import SokobanRenderer


SPEED_PRESETS = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0]


def get_danger_zones(state: SokobanState) -> List[Tuple[int, int]]:
    """Find cells where a box would be deadlocked (corner trap)."""
    zones = []
    for y in range(state.height):
        for x in range(state.width):
            if (x, y) in state.walls or (x, y) in state.targets:
                continue
            up = (x, y - 1) in state.walls
            dn = (x, y + 1) in state.walls
            lt = (x - 1, y) in state.walls
            rt = (x + 1, y) in state.walls
            if (up or dn) and (lt or rt):
                zones.append((x, y))
    return zones


def compute_path(state: SokobanState, actions: List[int]) -> List[Tuple[int, int]]:
    """Simulate actions to get path of player positions."""
    path = [state.player]
    s = state
    for a in actions:
        ns = s.move(a)
        if ns:
            s = ns
            path.append(s.player)
    return path


def run_solver_mode(levels: List[SokobanState], renderer: SokobanRenderer) -> None:
    """Watch BFS solver find and execute optimal solutions."""
    total = len(levels)
    level_idx = 0
    speed_idx = 5  # 1.0s default

    def load_level(idx: int):
        nonlocal level_idx, state, solution, plan_path, step, solved, danger
        level_idx = idx % total
        state = levels[level_idx].clone()
        boxes = len(state.boxes)
        print(f"\n  Level {level_idx + 1}/{total}  "
              f"({state.width}x{state.height}, {boxes} boxes)")

        print("  Solving...", end=" ", flush=True)
        solution = solve(state, max_states=1_000_000)
        if solution:
            print(f"found {len(solution)}-move solution")
            plan_path = compute_path(state, solution)
        else:
            print("no solution found!")
            solution = []
            plan_path = []

        step = 0
        solved = False
        danger = get_danger_zones(state)

    state = levels[0].clone()
    solution: List[int] = []
    plan_path: List[Tuple[int, int]] = []
    step = 0
    solved = False
    danger: List[Tuple[int, int]] = []

    load_level(0)

    paused = False
    last_step = time.time()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_s:
                    speed_idx = min(speed_idx + 1, len(SPEED_PRESETS) - 1)
                    print(f"  Delay: {SPEED_PRESETS[speed_idx]:.2f}s")
                elif event.key == pygame.K_f:
                    speed_idx = max(speed_idx - 1, 0)
                    print(f"  Delay: {SPEED_PRESETS[speed_idx]:.2f}s")
                elif event.key == pygame.K_n:
                    load_level(level_idx + 1)
                    last_step = time.time()
                elif event.key == pygame.K_p:
                    load_level(level_idx - 1)
                    last_step = time.time()

        now = time.time()
        delay = SPEED_PRESETS[speed_idx]

        if not paused and not solved and solution and (now - last_step) >= delay:
            last_step = now
            if step < len(solution):
                action = solution[step]
                ns = state.move(action)
                if ns:
                    state = ns
                    step += 1
                    # Update remaining path
                    if step < len(plan_path):
                        plan_path = plan_path[step:]
                    else:
                        plan_path = []

                    if state.solved:
                        solved = True
                        print(f"  SOLVED in {step} moves!")
                else:
                    step += 1

        # Auto-advance after solve
        if solved and (now - last_step) > 2.5:
            load_level(level_idx + 1)
            last_step = time.time()

        renderer.render(
            state,
            level_num=level_idx + 1,
            total_levels=total,
            moves=step,
            solved=solved,
            ai_mode=True,
            step_num=step,
            max_steps=len(solution) if solution else 0,
            planned_path=plan_path if not solved else None,
            danger_zones=danger,
            info_text="BFS Solver | Space: Pause  S/F: Speed  N/P: Level  ESC: Quit",
        )


def run_rl_mode(levels: List[SokobanState], renderer: SokobanRenderer,
                model_path: str) -> None:
    """Watch trained PPO agent play Sokoban."""
    from TRAIN_AND_WATCH import SokobanNet, SokobanRLEnv, MAX_GRID
    import torch
    from torch.distributions import Categorical

    # Load trained network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SokobanNet().to(device)

    if os.path.exists(model_path):
        data = torch.load(model_path, map_location=device, weights_only=True)
        net.load_state_dict(data["net"])
        net.eval()
        print(f"  Loaded model from {model_path}")
    else:
        print(f"  WARNING: No model at {model_path}, using untrained agent")
        print(f"  Run 'python TRAIN_AND_WATCH.py' first to train!")

    env = SokobanRLEnv(levels, max_steps=300)

    total = len(levels)
    level_idx = 0
    speed_idx = 4  # 0.7s

    def load_level(idx: int):
        nonlocal level_idx, state, step, solved, danger, failed, obs
        level_idx = idx % total
        obs = env.reset(level_idx=level_idx)
        state = env.get_render_state()
        print(f"\n  Level {level_idx + 1}/{total}  "
              f"({state.width}x{state.height}, {len(state.boxes)} boxes)")
        step = 0
        solved = False
        failed = False
        danger = get_danger_zones(state)

    state = levels[0].clone()
    obs = np.zeros((5, MAX_GRID, MAX_GRID), dtype=np.float32)
    step = 0
    solved = False
    failed = False
    danger: List[Tuple[int, int]] = []

    load_level(0)

    paused = False
    last_step = time.time()
    max_steps = 300
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_s:
                    speed_idx = min(speed_idx + 1, len(SPEED_PRESETS) - 1)
                elif event.key == pygame.K_f:
                    speed_idx = max(speed_idx - 1, 0)
                elif event.key == pygame.K_n:
                    load_level(level_idx + 1)
                    last_step = time.time()
                elif event.key == pygame.K_p:
                    load_level(level_idx - 1)
                    last_step = time.time()

        now = time.time()
        delay = SPEED_PRESETS[speed_idx]

        if not paused and not solved and not failed and step < max_steps and (now - last_step) >= delay:
            last_step = now

            state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                probs, value = net(state_t)
            action = probs.argmax(dim=-1).item()  # greedy at inference

            obs, reward, done, info = env.step(action)
            state = env.get_render_state()
            step += 1
            danger = get_danger_zones(state)

            if info.get("solved"):
                solved = True
                print(f"  SOLVED in {step} moves!")
            elif done:
                reason = "deadlock" if info.get("deadlock") else "timeout"
                print(f"  Failed ({reason}) after {step} moves")
                failed = True

        if (solved or failed) and (now - last_step) > 2.5:
            load_level(level_idx + 1)
            last_step = time.time()

        renderer.render(
            state,
            level_num=level_idx + 1,
            total_levels=total,
            moves=step,
            solved=solved,
            ai_mode=True,
            step_num=step,
            max_steps=max_steps,
            danger_zones=danger,
            info_text="RL Agent (PPO) | Space: Pause  S/F: Speed  N/P: Level  ESC: Quit",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch AI solve Sokoban")
    parser.add_argument("--rl", action="store_true",
                        help="Use trained RL agent instead of BFS solver")
    parser.add_argument("--model", type=str, default="checkpoints/ppo_sokoban_60.pt",
                        help="Path to trained model (for --rl mode)")
    args = parser.parse_args()

    pygame.init()

    # Load levels
    levels_file = os.path.join(os.path.dirname(__file__), "levels", "classic_60.txt")
    loader = LevelLoader(levels_file)
    total = loader.get_total_levels()

    if total == 0:
        print("ERROR: No levels found!")
        sys.exit(1)

    levels = [loader.get_level(i) for i in range(total)]

    print("=" * 55)
    print("  WATCH AI - Classic Sokoban")
    print("=" * 55)
    mode_str = "RL Agent" if args.rl else "BFS Solver"
    print(f"  Mode: {mode_str}")
    print(f"  Levels: {total}")
    print("=" * 55)

    renderer = SokobanRenderer(cell_size=48)

    if args.rl:
        run_rl_mode(levels, renderer, args.model)
    else:
        run_solver_mode(levels, renderer)

    renderer.close()
    pygame.quit()
    print("\n  Goodbye!")


if __name__ == "__main__":
    main()
