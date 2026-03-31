"""Level loader and procedural generator for Sokoban.

- Loads levels from XSB-format text files
- Generates simple training levels (1-2 boxes, small grids) for RL
- All generated levels are verified solvable with BFS
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import List, Optional, Tuple

from env.sokoban import SokobanState, solve


# ═══════════════════════════════════════════════════════════════════════
# XSB File Loader
# ═══════════════════════════════════════════════════════════════════════


class LevelLoader:
    """Load Sokoban levels from an XSB-format file."""

    def __init__(self, filepath: str = "levels/classic_60.txt") -> None:
        self.filepath = filepath
        self.levels: List[SokobanState] = []
        if os.path.isfile(filepath):
            self.levels = self._parse_file(filepath)

    def get_level(self, index: int) -> SokobanState:
        return self.levels[index].clone()

    def get_total_levels(self) -> int:
        return len(self.levels)

    def _parse_file(self, filepath: str) -> List[SokobanState]:
        with open(filepath, "r") as f:
            content = f.read()

        levels = []
        current_lines: List[str] = []

        for line in content.split("\n"):
            stripped = line.rstrip()
            # Level separator: blank line or line starting with ;
            if not stripped or stripped.startswith(";"):
                if current_lines and any("#" in l for l in current_lines):
                    xsb = "\n".join(current_lines)
                    try:
                        state = SokobanState.from_xsb(xsb)
                        if state.boxes and state.targets:
                            levels.append(state)
                    except Exception:
                        pass
                    current_lines = []
            else:
                current_lines.append(stripped)

        # Last level
        if current_lines and any("#" in l for l in current_lines):
            xsb = "\n".join(current_lines)
            try:
                state = SokobanState.from_xsb(xsb)
                if state.boxes and state.targets:
                    levels.append(state)
            except Exception:
                pass

        return levels


# ═══════════════════════════════════════════════════════════════════════
# Procedural Level Generator (for RL training)
# ═══════════════════════════════════════════════════════════════════════


def _is_connected(cells, walls) -> bool:
    """Check all cells form a connected region (BFS)."""
    if not cells:
        return True
    start = cells[0]
    visited = {start}
    queue = deque([start])
    cell_set = set(cells)
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nb = (x + dx, y + dy)
            if nb in cell_set and nb not in visited and nb not in walls:
                visited.add(nb)
                queue.append(nb)
    return len(visited) >= len(cell_set)


def generate_training_level(
    width: int = 6,
    height: int = 6,
    n_boxes: int = 1,
    seed: Optional[int] = None,
    max_retries: int = 500,
    min_solution: int = 3,
    max_solution: int = 80,
) -> Optional[Tuple[SokobanState, List[int]]]:
    """Generate a random solvable Sokoban level.

    Returns (state, solution) or None if generation fails.
    """
    rng = random.Random(seed)

    for _ in range(max_retries):
        # Border walls
        walls = set()
        for x in range(width):
            walls.add((x, 0))
            walls.add((x, height - 1))
        for y in range(height):
            walls.add((0, y))
            walls.add((width - 1, y))

        # Interior cells
        interior = [
            (x, y) for x in range(1, width - 1) for y in range(1, height - 1)
        ]

        # Random interior walls (up to 20% of interior)
        rng.shuffle(interior)
        n_int_walls = rng.randint(0, max(1, len(interior) // 5))
        candidate_walls = interior[:n_int_walls]

        # Add interior walls one by one, checking connectivity
        for w in candidate_walls:
            walls.add(w)

        free = [p for p in interior if p not in walls]
        if len(free) < n_boxes * 2 + 1:
            continue

        if not _is_connected(free, walls):
            continue

        # Place targets (not in corners)
        rng.shuffle(free)
        targets = set()
        for p in free:
            if len(targets) >= n_boxes:
                break
            targets.add(p)
        if len(targets) < n_boxes:
            continue

        # Place boxes (different from targets, not in corners)
        remaining = [p for p in free if p not in targets]
        rng.shuffle(remaining)
        boxes = set()
        for p in remaining:
            if len(boxes) >= n_boxes:
                break
            x, y = p
            # Skip corner positions
            h_blocked = (x - 1, y) in walls and (x + 1, y) in walls
            v_blocked = (x, y - 1) in walls and (x, y + 1) in walls
            adj_walls = sum(
                1 for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                if (x + dx, y + dy) in walls
            )
            if adj_walls >= 2:
                up = (x, y - 1) in walls
                dn = (x, y + 1) in walls
                lt = (x - 1, y) in walls
                rt = (x + 1, y) in walls
                if (up or dn) and (lt or rt):
                    continue
            boxes.add(p)

        if len(boxes) < n_boxes:
            continue

        # Place player
        player_spots = [p for p in free if p not in targets and p not in boxes]
        if not player_spots:
            continue
        player = rng.choice(player_spots)

        state = SokobanState(
            frozenset(walls), frozenset(boxes), frozenset(targets),
            player, width, height,
        )

        # Solve
        solution = solve(state, max_states=200_000)
        if solution is not None and min_solution <= len(solution) <= max_solution:
            return state, solution

    return None


def generate_level_set(
    n_levels: int = 60,
    base_seed: int = 42,
) -> List[Tuple[SokobanState, List[int]]]:
    """Generate a set of levels with increasing difficulty."""
    levels = []
    specs = [
        # (count, width, height, n_boxes, min_sol, max_sol)
        (10, 5, 5, 1, 3, 20),      # 1-10: tiny, 1 box
        (10, 6, 6, 1, 5, 30),      # 11-20: small, 1 box
        (10, 7, 7, 2, 6, 50),      # 21-30: medium, 2 boxes
        (10, 7, 7, 2, 10, 70),     # 31-40: medium-hard, 2 boxes
        (10, 8, 8, 2, 12, 90),     # 41-50: hard, 2 boxes
        (10, 9, 9, 3, 10, 120),    # 51-60: very hard, 3 boxes
    ]

    seed = base_seed
    for count, w, h, nb, min_s, max_s in specs:
        generated = 0
        attempts = 0
        while generated < count and attempts < count * 200:
            result = generate_training_level(
                width=w, height=h, n_boxes=nb,
                seed=seed, min_solution=min_s, max_solution=max_s,
            )
            seed += 1
            attempts += 1
            if result is not None:
                levels.append(result)
                generated += 1

    return levels


def save_levels(levels: List[Tuple[SokobanState, List[int]]], filepath: str) -> None:
    """Save levels to XSB file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    lines = []
    for i, (state, solution) in enumerate(levels):
        lines.append(f"; Level {i + 1} (solution: {len(solution)} moves)")
        lines.append(state.to_xsb())
        lines.append("")
    with open(filepath, "w") as f:
        f.write("\n".join(lines))


def generate_rl_training_levels(
    n_levels: int = 500,
    seed: int = 1000,
) -> List[SokobanState]:
    """Generate simple 1-box levels for RL training."""
    levels = []
    s = seed
    while len(levels) < n_levels:
        result = generate_training_level(
            width=random.choice([5, 6, 7]),
            height=random.choice([5, 6, 7]),
            n_boxes=1, seed=s, min_solution=3, max_solution=30,
        )
        s += 1
        if result is not None:
            levels.append(result[0])
    return levels


# ═══════════════════════════════════════════════════════════════════════
# CLI: Generate levels
# ═══════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    print("Generating 60 Sokoban levels...")
    levels = generate_level_set(n_levels=60, base_seed=42)
    filepath = os.path.join(os.path.dirname(__file__), "..", "levels", "classic_60.txt")
    save_levels(levels, filepath)
    print(f"Saved {len(levels)} levels to {filepath}")
    for i, (state, sol) in enumerate(levels):
        print(f"  Level {i + 1}: {state.width}x{state.height}, "
              f"{len(state.boxes)} boxes, {len(sol)} moves")
