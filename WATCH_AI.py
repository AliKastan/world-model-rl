"""WATCH_AI — Watch the ThinkerAgent solve puzzles step by step.

Launches a PyGame window showing the AI's thinking process visually:
  - Planned path drawn as green line
  - Danger zones (deadlock cells) as red overlay
  - Console prints the full thought process
  - Smooth animated movement between steps

Controls:
  Space  — pause / resume AI
  S      — slow down (increase delay)
  F      — speed up (decrease delay)
  N      — skip to next level
  ESC    — quit
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pygame

from env.puzzle_world import PuzzleWorld, ACTION_NAMES, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT
from env.level_generator import LevelGenerator
from env.renderer import PuzzleRenderer
from agents.model_based.thinker_agent import ThinkerAgent
from agents.model_based.mental_sim import MentalSimulator, ThoughtStep


# ── Speed presets ──────────────────────────────────────────────────────
SPEED_SLOW = 2.0
SPEED_NORMAL = 1.0
SPEED_FAST = 0.3
SPEED_MIN = 0.1
SPEED_MAX = 4.0

ACTION_DELTAS = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}


def _compute_planned_path(world: PuzzleWorld, actions: List[int]) -> List[Tuple[int, int]]:
    """Simulate actions on a clone to get the sequence of agent positions."""
    path = [tuple(world.agent_pos)]
    sim = world.clone()
    for a in actions:
        sim.step(a)
        path.append(tuple(sim.agent_pos))
    return path


def _get_danger_zones(world: PuzzleWorld) -> List[Tuple[int, int]]:
    """Find all cells where pushing a box would cause a deadlock."""
    zones: List[Tuple[int, int]] = []
    for y in range(world.height):
        for x in range(world.width):
            if world._box_is_deadlocked(x, y):
                # Only mark floor cells as danger (not walls)
                from env.objects import Wall, Target
                cell = world.get_cell(x, y)
                is_wall = any(isinstance(o, Wall) for o in cell)
                is_target = any(isinstance(o, Target) for o in cell)
                if not is_wall and not is_target:
                    zones.append((x, y))
    return zones


def _print_thought(thought: ThoughtStep, step_num: int) -> None:
    """Print the AI's thought process to the console."""
    print()
    print(f"{'─' * 60}")
    print(f"  Step {step_num}")
    print(f"{'─' * 60}")
    print(f"  🧠 Thinking... {thought.perception}")
    print()

    # Show strategies with results
    for i, sim_result in enumerate(thought.simulations):
        s = sim_result.strategy
        reward_str = f"reward: {sim_result.final_reward:.0f}"
        if sim_result.success:
            mark = "✅"
        elif sim_result.failure_reason and "deadlock" in sim_result.failure_reason:
            mark = "❌ DEADLOCK!"
        elif sim_result.final_reward < 0:
            mark = "❌"
        else:
            mark = "⚠️"
        print(f"  🤔 Strategy {i + 1}: {s.name} — {s.description}")
        print(f"     [{reward_str}, {sim_result.steps_simulated} steps] {mark}")
        if sim_result.failure_reason:
            print(f"     └─ {sim_result.failure_reason}")

    print()
    if thought.chosen_strategy:
        print(f"  ✨ Decision: {thought.chosen_strategy.name}, "
              f"confidence {thought.confidence:.2f}")
    else:
        print(f"  ✨ Decision: explore, confidence {thought.confidence:.2f}")
    print(f"  💡 {thought.reasoning}")


def main() -> None:
    pygame.init()

    gen = LevelGenerator()
    renderer = PuzzleRenderer(cell_size=64)
    difficulty = 3

    # ── Generate first puzzle ──────────────────────────────────────────
    print("=" * 60)
    print("  🎮 WATCH AI — ThinkerAgent Demo")
    print("=" * 60)
    print()
    print("  Controls:")
    print("    Space  — pause / resume")
    print("    S      — slow down")
    print("    F      — speed up")
    print("    N      — next level")
    print("    ESC    — quit")
    print()

    seed_counter = 0

    def new_level() -> Tuple[PuzzleWorld, ThinkerAgent]:
        nonlocal seed_counter, difficulty
        print(f"\n{'=' * 60}")
        print(f"  Generating Level (difficulty {difficulty}, seed {seed_counter})...")
        print(f"{'=' * 60}")
        world = gen.generate(difficulty=difficulty, seed=seed_counter)
        seed_counter += 1
        optimal = getattr(world, "_optimal_steps", "?")
        print(f"  Grid: {world.width}x{world.height}, optimal solution: {optimal} steps")
        print()
        agent = ThinkerAgent(use_perfect_model=True, planner_type="astar", mode="perfect")
        agent.set_world(world)
        return world, agent

    world, agent = new_level()

    # ── State ──────────────────────────────────────────────────────────
    paused = False
    step_delay = SPEED_NORMAL
    step_num = 0
    last_step_time = time.time()
    running = True
    solved = False
    solved_time = 0.0

    # Overlay data
    planned_path: List[Tuple[int, int]] = []
    danger_zones: List[Tuple[int, int]] = _get_danger_zones(world)

    # Do initial think to get the plan
    obs = world.get_observation()
    first_action, first_thought = agent._mental_sim.think(obs, world)
    _print_thought(first_thought, 0)

    # Extract planned path from agent's chosen strategy
    if first_thought.chosen_strategy and first_thought.chosen_strategy.action_sequence:
        planned_path = _compute_planned_path(world, first_thought.chosen_strategy.action_sequence)

    # ── Main loop ──────────────────────────────────────────────────────
    while running:
        # ── Events ─────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    state_str = "PAUSED" if paused else "RESUMED"
                    print(f"\n  ⏸️  {state_str}")
                elif event.key == pygame.K_s:
                    step_delay = min(step_delay + 0.3, SPEED_MAX)
                    print(f"  🐢 Delay: {step_delay:.1f}s")
                elif event.key == pygame.K_f:
                    step_delay = max(step_delay - 0.3, SPEED_MIN)
                    print(f"  🐇 Delay: {step_delay:.1f}s")
                elif event.key == pygame.K_n:
                    difficulty = min(difficulty + 1, 10)
                    world, agent = new_level()
                    step_num = 0
                    solved = False
                    danger_zones = _get_danger_zones(world)
                    planned_path = []
                    last_step_time = time.time()
                    # Initial think for new level
                    obs = world.get_observation()
                    first_action, first_thought = agent._mental_sim.think(obs, world)
                    _print_thought(first_thought, 0)
                    if first_thought.chosen_strategy and first_thought.chosen_strategy.action_sequence:
                        planned_path = _compute_planned_path(
                            world, first_thought.chosen_strategy.action_sequence
                        )

        # ── AI step ────────────────────────────────────────────────────
        now = time.time()
        if not paused and not solved and (now - last_step_time) >= step_delay:
            last_step_time = now
            step_num += 1

            obs = world.get_observation()
            agent.set_world(world)

            # Use the mental simulator directly for visible thinking
            action, thought = agent._mental_sim.think(obs, world)
            _print_thought(thought, step_num)

            # Update planned path from current strategy
            if thought.chosen_strategy and thought.chosen_strategy.action_sequence:
                planned_path = _compute_planned_path(
                    world, thought.chosen_strategy.action_sequence
                )
            else:
                planned_path = []

            # Execute the action in the real world
            _obs, reward, terminated, truncated, info = world.step(action)
            action_name = ACTION_NAMES.get(action, "?")
            print(f"  ➡️  Action: {action_name}, reward: {reward:.1f}")

            # Update danger zones (boxes may have moved)
            danger_zones = _get_danger_zones(world)

            # Check if solved
            if world.solved or terminated:
                solved = True
                solved_time = now
                optimal = getattr(world, "_optimal_steps", "?")
                print(f"\n  🎉 SOLVED in {step_num} steps! (optimal: {optimal})")
                print(f"  Efficiency: {optimal}/{step_num} = "
                      f"{(int(optimal) / step_num * 100) if isinstance(optimal, int) and step_num > 0 else '?'}%")

            # Safety: truncated (too many steps)
            if truncated:
                print(f"\n  ⏰ Truncated after {step_num} steps — moving to next level")
                solved = True
                solved_time = now

        # ── Auto-advance after solve ──────────────────────────────────
        if solved and (now - solved_time) > 3.0:
            difficulty = min(difficulty + 1, 10)
            world, agent = new_level()
            step_num = 0
            solved = False
            danger_zones = _get_danger_zones(world)
            planned_path = []
            last_step_time = time.time()
            # Initial think
            obs = world.get_observation()
            first_action, first_thought = agent._mental_sim.think(obs, world)
            _print_thought(first_thought, 0)
            if first_thought.chosen_strategy and first_thought.chosen_strategy.action_sequence:
                planned_path = _compute_planned_path(
                    world, first_thought.chosen_strategy.action_sequence
                )

        # ── Render ─────────────────────────────────────────────────────
        overlay = {
            "danger_zones": danger_zones,
            "selected_path": planned_path,
        }
        renderer.render(world, extra_overlay=overlay)
        pygame.display.flip()
        pygame.time.Clock().tick(60)

    renderer.close()
    pygame.quit()
    print("\n  👋 Goodbye!")


if __name__ == "__main__":
    main()
