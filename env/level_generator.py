"""Procedural level generator with BFS solver for solvability validation.

Generates Sokoban-style puzzles across 10 difficulty tiers for curriculum
learning. Every generated level is validated as solvable before being
returned.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from env.objects import (
    Box,
    Door,
    Floor,
    IceTile,
    Key,
    PressureSwitch,
    SwitchWall,
    Target,
    Wall,
)
from env.puzzle_world import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    DIRECTION_DELTAS,
    PuzzleWorld,
)

# ---------------------------------------------------------------------------
# BFS Solver
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SolverState:
    """Hashable snapshot of the game state for BFS search."""

    agent_pos: Tuple[int, int]
    box_positions: FrozenSet[Tuple[int, int]]
    collected_keys: FrozenSet[int]
    open_doors: FrozenSet[int]
    active_switches: FrozenSet[int]


def _extract_state(world: PuzzleWorld) -> SolverState:
    """Extract a hashable state from a :class:`PuzzleWorld`."""
    boxes: List[Tuple[int, int]] = []
    keys_collected = frozenset(k.key_id for k in world.inventory)
    open_doors: List[int] = []
    active_switches: List[int] = []

    for y in range(world.height):
        for x in range(world.width):
            for obj in world.get_cell(x, y):
                if isinstance(obj, Box):
                    boxes.append((x, y))
                elif isinstance(obj, Door) and not obj.locked:
                    open_doors.append(obj.door_id)
                elif isinstance(obj, PressureSwitch) and obj.activated:
                    active_switches.append(obj.switch_id)

    return SolverState(
        agent_pos=world.agent_pos,
        box_positions=frozenset(boxes),
        collected_keys=keys_collected,
        open_doors=frozenset(open_doors),
        active_switches=frozenset(active_switches),
    )


def _is_goal(world: PuzzleWorld) -> bool:
    """Check if all targets have a box on them."""
    total_targets = 0
    boxes_on = 0
    for y in range(world.height):
        for x in range(world.width):
            for obj in world.get_cell(x, y):
                if isinstance(obj, Target):
                    total_targets += 1
                if isinstance(obj, Box) and obj.on_target:
                    boxes_on += 1
    return total_targets > 0 and boxes_on >= total_targets


ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
ACTION_NAMES_SHORT = {ACTION_UP: "U", ACTION_DOWN: "D", ACTION_LEFT: "L", ACTION_RIGHT: "R"}


def solve(
    world: PuzzleWorld,
    max_states: int = 500_000,
    timeout_seconds: float = 5.0,
) -> Optional[List[int]]:
    """BFS solver — find the shortest action sequence that solves *world*.

    Args:
        world: The puzzle to solve (not mutated).
        max_states: Maximum number of states to explore before giving up.
        timeout_seconds: Wall-clock time limit.

    Returns:
        Optimal action sequence, or ``None`` if unsolvable / budget exceeded.
    """
    start_world = world.clone()
    start_state = _extract_state(start_world)

    if _is_goal(start_world):
        return []

    visited: Set[SolverState] = {start_state}
    # Queue entries: (world_snapshot, action_history)
    queue: deque[Tuple[PuzzleWorld, List[int]]] = deque()
    queue.append((start_world, []))

    t0 = time.monotonic()
    states_explored = 0

    while queue:
        if states_explored >= max_states:
            return None
        if time.monotonic() - t0 > timeout_seconds:
            return None

        current_world, history = queue.popleft()
        states_explored += 1

        for action in ACTIONS:
            child = current_world.clone()
            child.step(action)

            # Deadlock pruning
            if child.is_deadlock():
                continue

            child_state = _extract_state(child)
            if child_state in visited:
                continue
            visited.add(child_state)

            new_history = history + [action]

            if _is_goal(child):
                return new_history

            queue.append((child, new_history))

    return None  # exhausted search space — unsolvable


# ---------------------------------------------------------------------------
# Difficulty tier specifications
# ---------------------------------------------------------------------------

@dataclass
class _TierSpec:
    """Parameters for a single difficulty tier."""

    grid_w: int
    grid_h: int
    num_boxes: int
    num_targets: int  # always == num_boxes
    interior_walls_range: Tuple[int, int]  # (min, max) random interior walls
    num_keys: int = 0
    num_doors: int = 0
    ice_tile_range: Tuple[int, int] = (0, 0)
    num_switches: int = 0
    num_switch_walls: int = 0
    # solver budget (bigger levels need more)
    solver_max_states: int = 500_000
    solver_timeout: float = 5.0
    max_steps: int = 200


_TIERS: Dict[int, _TierSpec] = {
    1: _TierSpec(
        grid_w=6, grid_h=6, num_boxes=2, num_targets=2,
        interior_walls_range=(0, 2), max_steps=50,
        solver_max_states=100_000, solver_timeout=3.0,
    ),
    2: _TierSpec(
        grid_w=7, grid_h=7, num_boxes=2, num_targets=2,
        interior_walls_range=(2, 5), max_steps=80,
        solver_max_states=150_000, solver_timeout=3.0,
    ),
    3: _TierSpec(
        grid_w=7, grid_h=7, num_boxes=3, num_targets=3,
        interior_walls_range=(3, 7), max_steps=100,
        solver_max_states=200_000, solver_timeout=4.0,
    ),
    4: _TierSpec(
        grid_w=8, grid_h=8, num_boxes=3, num_targets=3,
        interior_walls_range=(5, 10), max_steps=120,
        solver_max_states=300_000, solver_timeout=5.0,
    ),
    5: _TierSpec(
        grid_w=8, grid_h=8, num_boxes=2, num_targets=2,
        interior_walls_range=(4, 8), num_keys=1, num_doors=1,
        max_steps=120,
        solver_max_states=300_000, solver_timeout=5.0,
    ),
    6: _TierSpec(
        grid_w=8, grid_h=8, num_boxes=3, num_targets=3,
        interior_walls_range=(3, 6), ice_tile_range=(3, 7),
        max_steps=150,
        solver_max_states=300_000, solver_timeout=5.0,
    ),
    7: _TierSpec(
        grid_w=9, grid_h=9, num_boxes=3, num_targets=3,
        interior_walls_range=(4, 8),
        num_switches=1, num_switch_walls=1,
        max_steps=150,
        solver_max_states=400_000, solver_timeout=5.0,
    ),
    8: _TierSpec(
        grid_w=10, grid_h=10, num_boxes=3, num_targets=3,
        interior_walls_range=(5, 10),
        num_keys=1, num_doors=1,
        ice_tile_range=(2, 5),
        num_switches=1, num_switch_walls=1,
        max_steps=200,
        solver_max_states=500_000, solver_timeout=5.0,
    ),
    9: _TierSpec(
        grid_w=10, grid_h=10, num_boxes=4, num_targets=4,
        interior_walls_range=(6, 12), max_steps=200,
        solver_max_states=500_000, solver_timeout=5.0,
    ),
    10: _TierSpec(
        grid_w=12, grid_h=12, num_boxes=4, num_targets=4,
        interior_walls_range=(8, 16),
        num_keys=1, num_doors=1,
        ice_tile_range=(2, 5),
        num_switches=1, num_switch_walls=1,
        max_steps=300,
        solver_max_states=500_000, solver_timeout=5.0,
    ),
}

# ---------------------------------------------------------------------------
# Level generator
# ---------------------------------------------------------------------------


class LevelGenerator:
    """Procedural level generator producing solvable puzzles at 10 difficulty tiers."""

    def __init__(self, max_retries: int = 100) -> None:
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        difficulty: int,
        seed: Optional[int] = None,
    ) -> PuzzleWorld:
        """Generate a random solvable level at the given *difficulty* (1-10).

        Args:
            difficulty: Difficulty tier (1-10).
            seed: Optional RNG seed for reproducibility.

        Returns:
            A :class:`PuzzleWorld` whose solution has been validated.

        Raises:
            RuntimeError: If no solvable level could be produced within the
                retry budget.
        """
        if difficulty < 1 or difficulty > 10:
            raise ValueError(f"difficulty must be 1-10, got {difficulty}")
        spec = _TIERS[difficulty]
        rng = random.Random(seed)

        for attempt in range(self.max_retries):
            # Derive a per-attempt seed so retries explore new layouts
            attempt_seed = rng.randint(0, 2**31)
            world = self._build_level(spec, random.Random(attempt_seed))
            if world is None:
                continue

            solution = solve(
                world,
                max_states=spec.solver_max_states,
                timeout_seconds=spec.solver_timeout,
            )
            if solution is not None and len(solution) > 0:
                # Stash metadata on the world object for benchmarking
                world._optimal_solution = solution  # type: ignore[attr-defined]
                world._optimal_steps = len(solution)  # type: ignore[attr-defined]
                world._difficulty = difficulty  # type: ignore[attr-defined]
                return world

        raise RuntimeError(
            f"Failed to generate a solvable level at difficulty {difficulty} "
            f"after {self.max_retries} attempts"
        )

    def generate_batch(
        self,
        difficulty: int,
        count: int,
        seed: Optional[int] = None,
    ) -> List[PuzzleWorld]:
        """Generate *count* solvable levels at *difficulty*."""
        rng = random.Random(seed)
        levels: List[PuzzleWorld] = []
        for _ in range(count):
            level_seed = rng.randint(0, 2**31)
            levels.append(self.generate(difficulty, seed=level_seed))
        return levels

    def validate_level(self, world: PuzzleWorld) -> Dict[str, Any]:
        """Analyse *world* and return validation metadata."""
        solution = solve(world)

        num_boxes = 0
        num_keys = 0
        has_ice = False
        has_switches = False
        for y in range(world.height):
            for x in range(world.width):
                for obj in world.get_cell(x, y):
                    if isinstance(obj, Box):
                        num_boxes += 1
                    elif isinstance(obj, Key):
                        num_keys += 1
                    elif isinstance(obj, IceTile):
                        has_ice = True
                    elif isinstance(obj, PressureSwitch):
                        has_switches = True

        solvable = solution is not None
        optimal_steps = len(solution) if solution else -1
        difficulty_score = self._estimate_difficulty(
            world, optimal_steps, num_boxes, num_keys, has_ice, has_switches,
        )

        return {
            "solvable": solvable,
            "optimal_steps": optimal_steps,
            "optimal_solution": solution if solution else [],
            "num_boxes": num_boxes,
            "num_keys": num_keys,
            "has_ice": has_ice,
            "has_switches": has_switches,
            "difficulty_score": difficulty_score,
        }

    # ------------------------------------------------------------------
    # Internal: level construction
    # ------------------------------------------------------------------

    def _build_level(
        self,
        spec: _TierSpec,
        rng: random.Random,
    ) -> Optional[PuzzleWorld]:
        """Attempt to build a single level from *spec*. Returns ``None`` on failure."""
        w, h = spec.grid_w, spec.grid_h
        world = PuzzleWorld(width=w, height=h)
        world.max_steps = spec.max_steps

        # 1. Border walls + interior floor
        for y in range(h):
            for x in range(w):
                if x == 0 or x == w - 1 or y == 0 or y == h - 1:
                    world.place_object(Wall(pos=(x, y)), x, y)
                else:
                    world.place_object(Floor(pos=(x, y)), x, y)

        # Interior cells available for placement
        interior: List[Tuple[int, int]] = [
            (x, y) for x in range(1, w - 1) for y in range(1, h - 1)
        ]

        # 2. Random interior walls (ensure connectivity)
        num_walls = rng.randint(*spec.interior_walls_range)
        wall_positions = self._place_interior_walls(world, interior, num_walls, rng)
        free_cells = [c for c in interior if c not in wall_positions]

        if len(free_cells) < (
            spec.num_boxes + spec.num_targets + spec.num_keys + spec.num_doors
            + spec.num_switches + spec.num_switch_walls + 1  # +1 for agent
        ):
            return None  # not enough room

        rng.shuffle(free_cells)
        idx = 0

        # 3a. Place targets
        target_positions: List[Tuple[int, int]] = []
        for _ in range(spec.num_targets):
            tx, ty = free_cells[idx]; idx += 1
            world.place_object(Target(pos=(tx, ty)), tx, ty)
            target_positions.append((tx, ty))

        # 3b. Place boxes — make sure they don't start on a target and aren't
        #     in an immediate deadlock corner.
        box_positions: List[Tuple[int, int]] = []
        for _ in range(spec.num_boxes):
            placed = False
            for tries in range(50):
                if idx >= len(free_cells):
                    return None
                bx, by = free_cells[idx]; idx += 1
                if (bx, by) in target_positions:
                    continue
                # Reject immediate corner deadlocks
                if self._is_corner(world, bx, by):
                    continue
                world.place_object(Box(pos=(bx, by)), bx, by)
                box_positions.append((bx, by))
                placed = True
                break
            if not placed:
                return None

        # 3c. Keys and doors
        for kid in range(spec.num_keys):
            if idx >= len(free_cells):
                return None
            kx, ky = free_cells[idx]; idx += 1
            world.place_object(Key(pos=(kx, ky), key_id=kid), kx, ky)

        for did in range(spec.num_doors):
            if idx >= len(free_cells):
                return None
            dx, dy = free_cells[idx]; idx += 1
            world.place_object(Door(pos=(dx, dy), door_id=did), dx, dy)

        # 3d. Ice tiles
        num_ice = rng.randint(*spec.ice_tile_range)
        for _ in range(num_ice):
            if idx >= len(free_cells):
                break
            ix, iy = free_cells[idx]; idx += 1
            # Don't place ice on a cell that already has box/target/key/door
            if (ix, iy) in box_positions or (ix, iy) in target_positions:
                continue
            world.place_object(IceTile(pos=(ix, iy)), ix, iy)

        # 3e. Switches + switch walls
        for sid in range(spec.num_switches):
            if idx >= len(free_cells):
                break
            sx, sy = free_cells[idx]; idx += 1
            world.place_object(
                PressureSwitch(pos=(sx, sy), switch_id=sid), sx, sy,
            )

        for sid in range(spec.num_switch_walls):
            if idx >= len(free_cells):
                break
            swx, swy = free_cells[idx]; idx += 1
            world.place_object(
                SwitchWall(pos=(swx, swy), linked_switch_id=sid), swx, swy,
            )

        # 4. Agent at a valid remaining position
        if idx >= len(free_cells):
            return None
        ax, ay = free_cells[idx]
        # Make sure agent isn't on a box or target
        while (ax, ay) in box_positions or (ax, ay) in target_positions:
            idx += 1
            if idx >= len(free_cells):
                return None
            ax, ay = free_cells[idx]
        world.agent_pos = (ax, ay)

        # Quick connectivity check
        if not self._check_connectivity(world):
            return None

        return world

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _place_interior_walls(
        self,
        world: PuzzleWorld,
        interior: List[Tuple[int, int]],
        count: int,
        rng: random.Random,
    ) -> Set[Tuple[int, int]]:
        """Place up to *count* interior walls while maintaining connectivity."""
        candidates = list(interior)
        rng.shuffle(candidates)
        placed: Set[Tuple[int, int]] = set()

        for x, y in candidates:
            if len(placed) >= count:
                break
            # Temporarily place wall
            world.place_object(Wall(pos=(x, y)), x, y)
            placed.add((x, y))

            # Check connectivity of remaining free cells
            free = [c for c in interior if c not in placed]
            if not self._cells_connected(world, free):
                # Revert
                self._replace_cell(world, x, y, Floor(pos=(x, y)))
                placed.discard((x, y))

        return placed

    @staticmethod
    def _cells_connected(world: PuzzleWorld, cells: List[Tuple[int, int]]) -> bool:
        """Return True if all *cells* form a single connected component via BFS."""
        if not cells:
            return True
        cell_set = set(cells)
        start = cells[0]
        visited: Set[Tuple[int, int]] = {start}
        queue: deque[Tuple[int, int]] = deque([start])

        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in cell_set and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return len(visited) == len(cell_set)

    @staticmethod
    def _replace_cell(world: PuzzleWorld, x: int, y: int, obj: GameObject) -> None:
        """Replace all objects in a cell with a single *obj*."""
        world.grid[y][x] = [obj]
        obj.pos = (x, y)

    @staticmethod
    def _is_corner(world: PuzzleWorld, x: int, y: int) -> bool:
        """Return True if (*x*, *y*) is a corner (two perpendicular walls)."""
        def wall_at(wx: int, wy: int) -> bool:
            if not world._in_bounds(wx, wy):
                return True
            return any(isinstance(o, Wall) for o in world.get_cell(wx, wy))

        up = wall_at(x, y - 1)
        down = wall_at(x, y + 1)
        left = wall_at(x - 1, y)
        right = wall_at(x + 1, y)
        return (up and left) or (up and right) or (down and left) or (down and right)

    @staticmethod
    def _check_connectivity(world: PuzzleWorld) -> bool:
        """Verify that all non-wall cells are reachable from the agent position."""
        w, h = world.width, world.height
        non_wall: Set[Tuple[int, int]] = set()
        for y in range(h):
            for x in range(w):
                if not any(isinstance(o, Wall) for o in world.get_cell(x, y)):
                    # Also skip closed switch-walls for connectivity purposes
                    if any(isinstance(o, SwitchWall) and not o.open for o in world.get_cell(x, y)):
                        continue
                    non_wall.add((x, y))

        if not non_wall:
            return False

        # BFS from agent
        start = world.agent_pos
        if start not in non_wall:
            return False
        visited: Set[Tuple[int, int]] = {start}
        queue: deque[Tuple[int, int]] = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in non_wall and (nx, ny) not in visited:
                    # Doors block passage but are still "reachable" cells
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        # All non-wall, non-switch-wall cells should be reachable
        # (doors count as reachable even if locked — the agent might have keys)
        return len(visited) == len(non_wall)

    @staticmethod
    def _estimate_difficulty(
        world: PuzzleWorld,
        optimal_steps: int,
        num_boxes: int,
        num_keys: int,
        has_ice: bool,
        has_switches: bool,
    ) -> float:
        """Heuristic difficulty score (0-100)."""
        score = 0.0
        # Grid size contribution (0-20)
        area = world.width * world.height
        score += min(20.0, area / 7.0)
        # Optimal solution length (0-30)
        if optimal_steps > 0:
            score += min(30.0, optimal_steps * 0.8)
        # Boxes (0-15)
        score += num_boxes * 5.0
        # Keys (0-10)
        score += num_keys * 5.0
        # Mechanics (0-10 each)
        if has_ice:
            score += 10.0
        if has_switches:
            score += 10.0
        return min(100.0, score)
