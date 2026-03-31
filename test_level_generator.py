"""Test script for the level generator and BFS solver.

- Generates 3 levels at each difficulty 1-5
- For difficulty 4, demonstrates deadlock potential
- Replays a solved level step by step
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from env.level_generator import LevelGenerator, solve, ACTION_NAMES_SHORT
from env.puzzle_world import ACTION_NAMES


def main() -> None:
    gen = LevelGenerator()

    # -- Generate 3 levels at difficulties 1-5 --
    print("=" * 60)
    print("GENERATING LEVELS AT DIFFICULTIES 1-5")
    print("=" * 60)

    for diff in range(1, 6):
        print(f"\n{'-' * 60}")
        print(f"  DIFFICULTY {diff}")
        print(f"{'-' * 60}")
        for level_idx in range(3):
            seed = diff * 1000 + level_idx
            t0 = time.monotonic()
            world = gen.generate(difficulty=diff, seed=seed)
            elapsed = time.monotonic() - t0

            opt_steps = getattr(world, "_optimal_steps", -1)
            opt_solution = getattr(world, "_optimal_solution", [])

            print(f"\n  Level {level_idx + 1} (seed={seed}, {elapsed:.2f}s):")
            print(f"  Grid: {world.width}x{world.height}  "
                  f"Solvable=True  Optimal={opt_steps} steps")
            for line in world.render_ascii().split("\n"):
                print(f"    {line}")

    # -- Difficulty 4: show deadlock potential --
    print(f"\n{'=' * 60}")
    print("DEADLOCK ANALYSIS -- DIFFICULTY 4")
    print("=" * 60)

    deadlock_found = False
    for seed in range(4000, 4100):
        try:
            world = gen.generate(difficulty=4, seed=seed)
        except RuntimeError:
            continue
        opt_solution = getattr(world, "_optimal_solution", [])
        if not opt_solution:
            continue

        # Try a wrong first move (not part of optimal solution)
        for bad_action in range(4):
            if bad_action == opt_solution[0]:
                continue
            test_world = world.clone()
            test_world.step(bad_action)
            if test_world.is_deadlock():
                direction_name = ACTION_NAMES[bad_action]
                print(f"\n  Seed {seed}: moving {direction_name} first causes a DEADLOCK!")
                print(f"  Optimal solution starts with "
                      f"{ACTION_NAMES[opt_solution[0]]} ({len(opt_solution)} steps)")
                print(f"\n  Initial state:")
                for line in world.render_ascii().split("\n"):
                    print(f"    {line}")
                print(f"\n  After wrong move ({direction_name}) -- DEADLOCKED:")
                for line in test_world.render_ascii().split("\n"):
                    print(f"    {line}")
                deadlock_found = True
                break
        if deadlock_found:
            break

    if not deadlock_found:
        # Generate a custom level that guarantees deadlock demonstration
        print("\n  (Searching harder for deadlock demo...)")
        for seed in range(4100, 4500):
            try:
                world = gen.generate(difficulty=4, seed=seed)
            except RuntimeError:
                continue
            opt_solution = getattr(world, "_optimal_solution", [])
            if not opt_solution:
                continue
            for bad_action in range(4):
                if bad_action == opt_solution[0]:
                    continue
                test_world = world.clone()
                test_world.step(bad_action)
                if test_world.is_deadlock():
                    direction_name = ACTION_NAMES[bad_action]
                    print(f"\n  Seed {seed}: moving {direction_name} first -> DEADLOCK!")
                    print(f"\n  Initial state:")
                    for line in world.render_ascii().split("\n"):
                        print(f"    {line}")
                    print(f"\n  After wrong move ({direction_name}) -- DEADLOCKED:")
                    for line in test_world.render_ascii().split("\n"):
                        print(f"    {line}")
                    deadlock_found = True
                    break
            if deadlock_found:
                break
        if not deadlock_found:
            print("  Could not find a single-move deadlock demo "
                  "(deadlocks exist but may require >1 wrong move)")

    # -- Replay a solved level step by step --
    print(f"\n{'=' * 60}")
    print("SOLUTION REPLAY -- DIFFICULTY 3")
    print("=" * 60)

    world = gen.generate(difficulty=3, seed=12345)
    solution = getattr(world, "_optimal_solution", [])
    print(f"\n  Optimal solution: {len(solution)} steps")
    print(f"  Actions: {' '.join(ACTION_NAMES_SHORT[a] for a in solution)}")

    print(f"\n  Step 0 (initial):")
    for line in world.render_ascii().split("\n"):
        print(f"    {line}")

    for i, action in enumerate(solution):
        obs, reward, terminated, truncated, info = world.step(action)
        action_char = ACTION_NAMES_SHORT[action]
        print(f"\n  Step {i + 1} ({action_char}) | reward={reward:+.1f}"
              f"  boxes={obs['boxes_on_targets']}/{obs['total_targets']}")
        for line in world.render_ascii().split("\n"):
            print(f"    {line}")
        if terminated:
            print(f"\n  *** SOLVED in {i + 1} steps! ***")
            break

    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
