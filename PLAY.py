"""Play Sokoban levels interactively.

Controls:
    Arrow Keys  - Move player
    R           - Reset current level
    N           - Next level
    P           - Previous level
    U           - Undo last move
    ESC         - Quit
"""

from __future__ import annotations

import os
import sys
from typing import List

import pygame

from env.sokoban import SokobanState
from env.level_loader import LevelLoader
from env.sokoban_renderer import SokobanRenderer


def main() -> None:
    pygame.init()

    levels_file = os.path.join(os.path.dirname(__file__), "levels", "classic_60.txt")
    loader = LevelLoader(levels_file)
    total = loader.get_total_levels()

    if total == 0:
        print("ERROR: No levels found!")
        sys.exit(1)

    print("=" * 45)
    print("  SOKOBAN - Classic 60 Levels")
    print("=" * 45)
    print(f"  Loaded {total} levels")
    print("=" * 45)

    renderer = SokobanRenderer(cell_size=48)

    level_idx = 0
    state: SokobanState = loader.get_level(0)
    history: List[SokobanState] = []
    moves = 0
    solved = False

    def load_level(idx: int) -> None:
        nonlocal state, history, moves, solved, level_idx
        level_idx = idx % total
        state = loader.get_level(level_idx)
        history = []
        moves = 0
        solved = False
        print(f"  Level {level_idx + 1}: {state.width}x{state.height}, "
              f"{len(state.boxes)} boxes")

    load_level(0)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_n:
                    load_level(level_idx + 1)
                elif event.key == pygame.K_p:
                    load_level(level_idx - 1)
                elif event.key == pygame.K_r:
                    load_level(level_idx)
                elif event.key == pygame.K_u and history:
                    state = history.pop()
                    moves -= 1
                    solved = False
                elif not solved:
                    action = None
                    if event.key == pygame.K_UP:
                        action = 0
                    elif event.key == pygame.K_DOWN:
                        action = 1
                    elif event.key == pygame.K_LEFT:
                        action = 2
                    elif event.key == pygame.K_RIGHT:
                        action = 3

                    if action is not None:
                        new_state = state.move(action)
                        if new_state is not None:
                            history.append(state)
                            state = new_state
                            moves += 1
                            if state.solved:
                                solved = True
                                print(f"  SOLVED Level {level_idx + 1} "
                                      f"in {moves} moves!")

        renderer.render(
            state,
            level_num=level_idx + 1,
            total_levels=total,
            moves=moves,
            solved=solved,
        )

    renderer.close()
    pygame.quit()


if __name__ == "__main__":
    main()
