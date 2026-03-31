"""Planners — search over world model predictions to find action sequences.

Three implementations:
  - BFSPlanner: breadth-first search (optimal, slow)
  - AStarPlanner: A* with manhattan heuristic (optimal, faster)
  - BeamSearchPlanner: beam search (not optimal, fast — good for learned models)
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from env.objects import Box, Door, Key, PressureSwitch, Target
from env.puzzle_world import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    PuzzleWorld,
)
from agents.model_based.world_model import PerfectWorldModel

ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]


# ═══════════════════════════════════════════════════════════════════════
# State helpers (shared across planners)
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PlannerState:
    """Hashable snapshot for visited-set tracking."""

    agent_pos: Tuple[int, int]
    box_positions: FrozenSet[Tuple[int, int]]
    collected_keys: FrozenSet[int]
    open_doors: FrozenSet[int]
    active_switches: FrozenSet[int]


def _extract_state(world: PuzzleWorld) -> PlannerState:
    boxes: List[Tuple[int, int]] = []
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

    return PlannerState(
        agent_pos=world.agent_pos,
        box_positions=frozenset(boxes),
        collected_keys=frozenset(k.key_id for k in world.inventory),
        open_doors=frozenset(open_doors),
        active_switches=frozenset(active_switches),
    )


def _is_goal(world: PuzzleWorld) -> bool:
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


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _heuristic(world: PuzzleWorld) -> float:
    """Sum of manhattan distances from each unplaced box to its nearest unoccupied target.

    Admissible: every box must move at least this far, so never overestimates.
    """
    boxes: List[Tuple[int, int]] = []
    targets: List[Tuple[int, int]] = []
    occupied_targets: Set[Tuple[int, int]] = set()

    for y in range(world.height):
        for x in range(world.width):
            for obj in world.get_cell(x, y):
                if isinstance(obj, Box):
                    if obj.on_target:
                        occupied_targets.add((x, y))
                    else:
                        boxes.append((x, y))
                if isinstance(obj, Target):
                    targets.append((x, y))

    free_targets = [t for t in targets if t not in occupied_targets]

    if not boxes:
        return 0.0

    # Greedy assignment: each box → nearest free target
    total = 0.0
    used: Set[int] = set()
    for bx, by in boxes:
        best_dist = float("inf")
        best_idx = -1
        for i, (tx, ty) in enumerate(free_targets):
            if i in used:
                continue
            d = _manhattan((bx, by), (tx, ty))
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx >= 0:
            total += best_dist
            used.add(best_idx)
        else:
            # More boxes than targets — shouldn't happen, but be safe
            total += 10.0

    return total


# ═══════════════════════════════════════════════════════════════════════
# BFSPlanner
# ═══════════════════════════════════════════════════════════════════════


class BFSPlanner:
    """Breadth-first search over world model predictions.

    Optimal (shortest action sequence) but slow for larger puzzles.
    """

    def plan(
        self,
        world_model: PerfectWorldModel,
        current_world: PuzzleWorld,
        max_depth: int = 50,
        max_states: int = 200_000,
    ) -> Optional[List[int]]:
        """Find optimal action sequence to solve the puzzle.

        Returns:
            Action list, or None if no solution found within budget.
        """
        if _is_goal(current_world):
            return []

        start_state = _extract_state(current_world)
        visited: Set[PlannerState] = {start_state}
        queue: deque[Tuple[PuzzleWorld, List[int]]] = deque()
        queue.append((current_world.clone(), []))

        states_explored = 0

        while queue:
            if states_explored >= max_states:
                return None

            world, actions = queue.popleft()
            if len(actions) >= max_depth:
                continue

            states_explored += 1

            for action in ACTIONS:
                next_world, reward, done = world_model.predict(world, action)

                # Deadlock pruning
                if next_world.is_deadlock():
                    continue

                state = _extract_state(next_world)
                if state in visited:
                    continue
                visited.add(state)

                new_actions = actions + [action]

                if _is_goal(next_world):
                    return new_actions

                if not done:
                    queue.append((next_world, new_actions))

        return None


# ═══════════════════════════════════════════════════════════════════════
# AStarPlanner
# ═══════════════════════════════════════════════════════════════════════


class AStarPlanner:
    """A* search with admissible manhattan heuristic.

    Guaranteed optimal and typically much faster than BFS for larger puzzles.
    """

    def plan(
        self,
        world_model: PerfectWorldModel,
        current_world: PuzzleWorld,
        max_depth: int = 50,
        max_states: int = 200_000,
    ) -> Optional[List[int]]:
        if _is_goal(current_world):
            return []

        start_state = _extract_state(current_world)
        # Priority queue: (f_score, tiebreak_counter, world, actions)
        counter = 0
        h = _heuristic(current_world)
        open_set: List[Tuple[float, int, PuzzleWorld, List[int]]] = []
        heapq.heappush(open_set, (h, counter, current_world.clone(), []))

        g_scores: Dict[PlannerState, float] = {start_state: 0.0}
        states_explored = 0

        while open_set:
            if states_explored >= max_states:
                return None

            f, _, world, actions = heapq.heappop(open_set)
            if len(actions) >= max_depth:
                continue

            current_state = _extract_state(world)
            current_g = len(actions)

            # Skip if we've found a better path to this state
            if current_state in g_scores and g_scores[current_state] < current_g:
                continue

            states_explored += 1

            for action in ACTIONS:
                next_world, reward, done = world_model.predict(world, action)

                if next_world.is_deadlock():
                    continue

                next_state = _extract_state(next_world)
                new_g = current_g + 1

                if next_state in g_scores and g_scores[next_state] <= new_g:
                    continue
                g_scores[next_state] = new_g

                new_actions = actions + [action]

                if _is_goal(next_world):
                    return new_actions

                if not done:
                    h = _heuristic(next_world)
                    f = new_g + h
                    counter += 1
                    heapq.heappush(open_set, (f, counter, next_world, new_actions))

        return None


# ═══════════════════════════════════════════════════════════════════════
# BeamSearchPlanner
# ═══════════════════════════════════════════════════════════════════════


class BeamSearchPlanner:
    """Beam search — keep top-k paths at each depth.

    Not optimal, but fast for complex puzzles. Best suited for learned
    world models where BFS/A* might explore too many bad predictions.
    """

    def plan(
        self,
        world_model: PerfectWorldModel,
        current_world: PuzzleWorld,
        beam_width: int = 5,
        max_depth: int = 30,
    ) -> Optional[List[int]]:
        if _is_goal(current_world):
            return []

        # Each beam entry: (score, world, actions)
        beam: List[Tuple[float, PuzzleWorld, List[int]]] = [
            (0.0, current_world.clone(), [])
        ]
        visited: Set[PlannerState] = {_extract_state(current_world)}

        for depth in range(max_depth):
            candidates: List[Tuple[float, PuzzleWorld, List[int]]] = []

            for score, world, actions in beam:
                for action in ACTIONS:
                    next_world, reward, done = world_model.predict(world, action)

                    if next_world.is_deadlock():
                        continue

                    state = _extract_state(next_world)
                    if state in visited:
                        continue
                    visited.add(state)

                    new_actions = actions + [action]

                    if _is_goal(next_world):
                        return new_actions

                    if not done:
                        # Score: cumulative reward + heuristic bonus
                        h = _heuristic(next_world)
                        new_score = score + reward - h
                        candidates.append((new_score, next_world, new_actions))

            if not candidates:
                return None

            # Keep top beam_width candidates
            candidates.sort(key=lambda c: c[0], reverse=True)
            beam = candidates[:beam_width]

        # Return best partial plan if no solution found
        return None


# ═══════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════


def create_planner(planner_type: str = "astar"):
    """Create a planner by name."""
    planners = {
        "bfs": BFSPlanner,
        "astar": AStarPlanner,
        "beam": BeamSearchPlanner,
    }
    if planner_type not in planners:
        raise ValueError(f"Unknown planner: {planner_type!r}. Choose from {list(planners)}")
    return planners[planner_type]()


# ═══════════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    import time
    from env.level_generator import LevelGenerator

    gen = LevelGenerator()
    wm = PerfectWorldModel()

    print("=" * 70)
    print("Planner Comparison")
    print("=" * 70)

    for difficulty in [1, 2, 3, 4, 5]:
        world = gen.generate(difficulty=difficulty, seed=42)
        print(f"\nDifficulty {difficulty} ({world.width}x{world.height}):")

        for name, planner in [
            ("BFS    ", BFSPlanner()),
            ("A*     ", AStarPlanner()),
            ("Beam(5)", BeamSearchPlanner()),
        ]:
            t0 = time.perf_counter()
            if isinstance(planner, BeamSearchPlanner):
                result = planner.plan(wm, world, beam_width=5, max_depth=50)
            else:
                result = planner.plan(wm, world, max_depth=50)
            dt = (time.perf_counter() - t0) * 1000

            if result is not None:
                # Verify solution
                verify = world.clone()
                for a in result:
                    verify.step(a)
                solved = _is_goal(verify)
                print(f"  {name}: {len(result):3d} steps, {dt:7.1f}ms  {'SOLVED' if solved else 'FAILED'}")
            else:
                print(f"  {name}: no solution, {dt:7.1f}ms")

    print("\nDone.")


if __name__ == "__main__":
    main()
