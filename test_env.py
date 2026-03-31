"""Test script for the puzzle environment.

Creates a 7x7 grid with walls, 1 box, 1 target, and an agent.
Pushes the box onto the target and verifies the puzzle is solved.
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from env.objects import Wall, Floor, Box, Target
from env.puzzle_world import PuzzleWorld, ACTION_RIGHT


def build_test_level() -> PuzzleWorld:
    """Create a 7x7 room: walls on border, box at (3,3), target at (5,3), agent at (2,3)."""
    world = PuzzleWorld(width=7, height=7)

    for y in range(7):
        for x in range(7):
            if x == 0 or x == 6 or y == 0 or y == 6:
                world.place_object(Wall(pos=(x, y)), x, y)
            else:
                world.place_object(Floor(pos=(x, y)), x, y)

    # Place target first (it goes under the box if they overlap)
    world.place_object(Target(pos=(5, 3)), 5, 3)
    world.place_object(Box(pos=(3, 3)), 3, 3)
    world.agent_pos = (2, 3)

    return world


def main() -> None:
    world = build_test_level()

    print("=== Initial State ===")
    print(world.render_ascii())
    print()

    # Move right 3 times: agent pushes box from (3,3) -> (4,3) -> (5,3)
    for i in range(3):
        obs, reward, terminated, truncated, info = world.step(ACTION_RIGHT)
        print(f"=== After step {i + 1} (RIGHT) | reward={reward:.1f} ===")
        print(world.render_ascii())
        print(f"  agent_pos={world.agent_pos}  boxes_on_targets={obs['boxes_on_targets']}"
              f"  solved={world.solved}")
        print()

        if terminated:
            print("PUZZLE SOLVED!")
            break

    # Verify
    assert world.solved, "Expected puzzle to be solved after 3 right moves!"
    print("All assertions passed.")


if __name__ == "__main__":
    main()
