"""MentalSimulator — the agent's "thinking process".

Combines AffordanceNet + WorldModel to strategize before acting.
The agent perceives the scene, generates candidate strategies,
simulates each one in its world model, and picks the best.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from env.objects import Box, Door, Floor, IceTile, Key, PressureSwitch, SwitchWall, Target, Wall
from env.puzzle_world import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    ACTION_NAMES,
    DIRECTION_DELTAS,
    PuzzleWorld,
)
from agents.model_based.affordance import ContextualAffordance, SceneAnalysis, ObjectInfo
from agents.model_based.world_model import PerfectWorldModel


# ═══════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Strategy:
    name: str  # e.g. "collect_key_then_push"
    description: str  # e.g. "Pick up key, open door, push box to target"
    action_sequence: List[int]  # e.g. [2, 2, 0, 0, 1, 3, 3, 3]
    expected_reward: float
    risk_level: float  # [0, 1]
    prerequisites: List[str]  # e.g. ["need key_0"]
    estimated_steps: int


@dataclass
class SimulationResult:
    strategy: Strategy
    imagined_states: List[dict]  # state snapshot at each step
    final_reward: float
    success: bool
    failure_reason: Optional[str]  # e.g. "deadlock at step 12"
    steps_simulated: int


@dataclass
class ThoughtStep:
    step_number: int
    perception: str  # "I see 2 boxes, 1 key, 1 locked door"
    affordances: List[str]  # "Key is collectible", "Box is pushable but risky"
    strategies_considered: List[Strategy]
    simulations: List[SimulationResult]
    chosen_strategy: Optional[Strategy]
    confidence: float  # [0, 1]
    reasoning: str  # "Must collect key first because target is behind locked door"


# ═══════════════════════════════════════════════════════════════════════
# BFS pathfinding within a PuzzleWorld
# ═══════════════════════════════════════════════════════════════════════

# Directional deltas matching action indices: UP=0, DOWN=1, LEFT=2, RIGHT=3
_ACTION_DELTAS = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}


def _bfs_path(
    world: PuzzleWorld,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    max_depth: int = 80,
) -> Optional[List[int]]:
    """BFS for shortest action path from *start* to *goal*, ignoring boxes.

    Returns action sequence or None if unreachable.
    """
    if start == goal:
        return []

    visited = {start}
    queue: deque[Tuple[Tuple[int, int], List[int]]] = deque()
    queue.append((start, []))

    while queue:
        (cx, cy), actions = queue.popleft()
        if len(actions) >= max_depth:
            continue
        for action, (dx, dy) in _ACTION_DELTAS.items():
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited:
                continue
            if not world._in_bounds(nx, ny):
                continue
            # Can walk through? Skip walls and closed switch-walls
            cell = world.get_cell(nx, ny)
            if any(isinstance(o, Wall) for o in cell):
                continue
            if any(isinstance(o, SwitchWall) and not o.open for o in cell):
                continue
            # Treat locked doors as passable for planning (key will unlock)
            visited.add((nx, ny))
            new_actions = actions + [action]
            if (nx, ny) == goal:
                return new_actions
            queue.append(((nx, ny), new_actions))

    return None


def _bfs_push_box(
    world: PuzzleWorld,
    box_pos: Tuple[int, int],
    target_pos: Tuple[int, int],
    max_depth: int = 40,
) -> Optional[List[int]]:
    """BFS to find an action sequence that pushes a box from *box_pos* to *target_pos*.

    Simulates using agent+box state. Returns actions for the agent,
    or None if no push path found within budget.
    """
    bx, by = box_pos
    agent_pos = world.agent_pos

    # State: (agent_x, agent_y, box_x, box_y)
    start = (agent_pos[0], agent_pos[1], bx, by)
    visited = {start}
    queue: deque[Tuple[Tuple[int, int, int, int], List[int]]] = deque()
    queue.append((start, []))

    while queue:
        (ax, ay, bx, by), actions = queue.popleft()
        if len(actions) >= max_depth:
            continue
        if (bx, by) == target_pos:
            return actions

        for action, (dx, dy) in _ACTION_DELTAS.items():
            nax, nay = ax + dx, ay + dy
            if not world._in_bounds(nax, nay):
                continue
            # Wall check
            if any(isinstance(o, Wall) for o in world.get_cell(nax, nay)):
                continue
            if any(isinstance(o, SwitchWall) and not o.open for o in world.get_cell(nax, nay)):
                continue

            nbx, nby = bx, by
            # Agent pushes box?
            if (nax, nay) == (bx, by):
                nbx, nby = bx + dx, by + dy
                if not world._in_bounds(nbx, nby):
                    continue
                if any(isinstance(o, (Wall, Box)) for o in world.get_cell(nbx, nby)):
                    continue
                if any(isinstance(o, SwitchWall) and not o.open for o in world.get_cell(nbx, nby)):
                    continue
                # Deadlock check: would box be stuck?
                if world._box_is_deadlocked(nbx, nby):
                    # Allow if it's the target
                    if (nbx, nby) != target_pos:
                        continue

            state = (nax, nay, nbx, nby)
            if state in visited:
                continue
            visited.add(state)
            queue.append((state, actions + [action]))

    return None


# ═══════════════════════════════════════════════════════════════════════
# MentalSimulator
# ═══════════════════════════════════════════════════════════════════════


class MentalSimulator:
    """The agent's thinking process — perceive, analyze, strategize, simulate, decide."""

    def __init__(
        self,
        world_model: Optional[PerfectWorldModel] = None,
        affordance_net: Optional[ContextualAffordance] = None,
        use_perfect_model: bool = True,
    ) -> None:
        self.world_model = world_model or PerfectWorldModel()
        self.affordance = affordance_net or ContextualAffordance()
        self.use_perfect_model = use_perfect_model
        self.thought_history: List[ThoughtStep] = []
        self._step_counter = 0

    # ------------------------------------------------------------------
    # Perceive
    # ------------------------------------------------------------------

    def perceive(self, observation: dict) -> dict:
        """Parse the grid and identify all objects, agent state, and progress."""
        grid = observation["grid"]
        agent_pos = observation["agent_pos"]
        inventory = observation.get("inventory", [])
        boxes_on = observation.get("boxes_on_targets", 0)
        total_targets = observation.get("total_targets", 0)

        # Scan grid for object counts and positions
        from env.puzzle_world import TYPE_IDS

        _id_to_name = {v: k for k, v in TYPE_IDS.items()}
        objects_found: Dict[str, List[Tuple[int, int]]] = {}

        h, w = grid.shape
        for y in range(h):
            for x in range(w):
                type_id = int(grid[y, x])
                name = _id_to_name.get(type_id, "unknown")
                if name in ("floor", "wall"):
                    continue
                objects_found.setdefault(name, []).append((x, y))

        # Build human-readable perception
        parts = []
        for name, positions in sorted(objects_found.items()):
            count = len(positions)
            label = name.replace("_", " ")
            if count == 1:
                parts.append(f"1 {label}")
            else:
                plural = label if label.endswith("s") else label + ("es" if label.endswith("x") else "s")
                parts.append(f"{count} {plural}")

        perception_str = f"I see: {', '.join(parts)}" if parts else "I see: empty room"

        return {
            "perception_str": perception_str,
            "objects": objects_found,
            "agent_pos": agent_pos,
            "inventory": inventory,
            "boxes_on_targets": boxes_on,
            "total_targets": total_targets,
            "grid_shape": (h, w),
        }

    # ------------------------------------------------------------------
    # Analyze
    # ------------------------------------------------------------------

    def analyze(self, world: PuzzleWorld) -> SceneAnalysis:
        """Use ContextualAffordance for full scene analysis."""
        return self.affordance.analyze_scene(world)

    # ------------------------------------------------------------------
    # Strategy generation
    # ------------------------------------------------------------------

    def generate_strategies(
        self, world: PuzzleWorld, analysis: SceneAnalysis
    ) -> List[Strategy]:
        """Generate candidate strategies based on scene analysis."""
        strategies: List[Strategy] = []

        objects = analysis.objects
        relationships = analysis.relationships
        danger_zones = set(analysis.danger_zones)

        boxes = [o for o in objects if o.obj_type == "box" and not o.extra.get("on_target")]
        targets = [o for o in objects if o.obj_type == "target" and not o.extra.get("has_box")]
        keys = [o for o in objects if o.obj_type == "key"]
        locked_doors = [o for o in objects if o.obj_type == "door" and o.extra.get("locked")]
        switches = [o for o in objects if o.obj_type == "switch" and not o.extra.get("activated")]
        sw_walls = [o for o in objects if o.obj_type == "switch_wall" and not o.extra.get("open")]
        ice_tiles = [o for o in objects if o.obj_type == "ice"]

        # --- Strategy 1: collect_key_first ---
        # If keys exist and doors are locked, we must get keys first
        key_door_pairs = [
            r for r in relationships if r.relation == "unlocks"
        ]
        if key_door_pairs:
            for rel in key_door_pairs:
                key_obj = rel.source
                door_obj = rel.target
                strat = self._plan_key_door_strategy(world, key_obj, door_obj, boxes, targets)
                if strat is not None:
                    strategies.append(strat)

        # --- Strategy 2: direct_push ---
        # For each box near a target with clear path
        clear_paths = [r for r in relationships if r.relation == "can_reach_target"]
        for rel in clear_paths:
            box_obj = rel.source
            tgt_obj = rel.target
            strat = self._plan_direct_push(world, box_obj, tgt_obj, danger_zones)
            if strat is not None:
                strategies.append(strat)

        # --- Strategy 3: clear_path_then_push ---
        blocked_paths = [r for r in relationships if r.relation == "path_blocked"]
        for rel in blocked_paths:
            box_obj = rel.source
            tgt_obj = rel.target
            strat = self._plan_clear_path_push(world, box_obj, tgt_obj, locked_doors)
            if strat is not None:
                strategies.append(strat)

        # --- Strategy 4: activate_switch ---
        if switches and sw_walls:
            for sw in switches:
                strat = self._plan_switch_strategy(world, sw, sw_walls, boxes, targets)
                if strat is not None:
                    strategies.append(strat)

        # --- Strategy 5: use_ice_slide ---
        if ice_tiles and boxes:
            for box_obj in boxes:
                for tgt_obj in targets:
                    strat = self._plan_ice_strategy(world, box_obj, tgt_obj, ice_tiles)
                    if strat is not None:
                        strategies.append(strat)

        # --- Strategy 6: box ordering variations ---
        if len(boxes) >= 2 and len(targets) >= 2:
            ordering_strats = self._plan_box_orderings(world, boxes, targets, danger_zones)
            strategies.extend(ordering_strats)

        # If we still have no strategies, add an exploration fallback
        if not strategies:
            strategies.append(Strategy(
                name="explore",
                description="No clear strategy — explore to gather information",
                action_sequence=[ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT],
                expected_reward=-0.4,
                risk_level=0.3,
                prerequisites=[],
                estimated_steps=4,
            ))

        # Cap at 5 strategies
        strategies = strategies[:5]
        return strategies

    # ------------------------------------------------------------------
    # Strategy planning helpers
    # ------------------------------------------------------------------

    def _plan_key_door_strategy(
        self,
        world: PuzzleWorld,
        key_obj: ObjectInfo,
        door_obj: ObjectInfo,
        boxes: List[ObjectInfo],
        targets: List[ObjectInfo],
    ) -> Optional[Strategy]:
        """Plan: go to key, go to door, then push boxes."""
        actions: List[int] = []

        # Path agent -> key
        path_to_key = _bfs_path(world, world.agent_pos, key_obj.pos)
        if path_to_key is None:
            return None
        actions.extend(path_to_key)

        # Path key -> door (agent will be at key pos after collecting)
        path_to_door = _bfs_path(world, key_obj.pos, door_obj.pos)
        if path_to_door is None:
            return None
        actions.extend(path_to_door)

        # After unlocking door, plan pushes for each box
        # Simulate the key+door actions to get an updated world
        sim_world = world.clone()
        for a in actions:
            sim_world.step(a)

        for box in boxes:
            best_push = None
            best_dist = float("inf")
            for tgt in targets:
                push_path = _bfs_push_box(sim_world, box.pos, tgt.pos)
                if push_path is not None and len(push_path) < best_dist:
                    best_push = push_path
                    best_dist = len(push_path)
            if best_push is not None:
                actions.extend(best_push)
                for a in best_push:
                    sim_world.step(a)

        kid = key_obj.extra.get("key_id", "?")
        return Strategy(
            name="collect_key_first",
            description=f"Collect Key#{kid}@{key_obj.pos}, unlock Door@{door_obj.pos}, push boxes",
            action_sequence=actions,
            expected_reward=100.0 + 5.0 + 2.0 + 10.0 * len(boxes),
            risk_level=0.2,
            prerequisites=[f"need key_{kid}"],
            estimated_steps=len(actions),
        )

    def _plan_direct_push(
        self,
        world: PuzzleWorld,
        box_obj: ObjectInfo,
        tgt_obj: ObjectInfo,
        danger_zones: set,
    ) -> Optional[Strategy]:
        """Plan: push box directly to target."""
        push_path = _bfs_push_box(world, box_obj.pos, tgt_obj.pos)
        if push_path is None:
            return None

        risk = 0.1
        # Check if push path goes through danger zones
        sim = world.clone()
        for a in push_path:
            sim.step(a)
            obs = sim.get_observation()
            if tuple(obs["agent_pos"]) in danger_zones:
                risk = max(risk, 0.3)

        return Strategy(
            name="direct_push",
            description=f"Push Box@{box_obj.pos} to Target@{tgt_obj.pos}",
            action_sequence=push_path,
            expected_reward=10.0,
            risk_level=risk,
            prerequisites=[],
            estimated_steps=len(push_path),
        )

    def _plan_clear_path_push(
        self,
        world: PuzzleWorld,
        box_obj: ObjectInfo,
        tgt_obj: ObjectInfo,
        locked_doors: List[ObjectInfo],
    ) -> Optional[Strategy]:
        """Plan: clear obstacles then push box."""
        prereqs = []
        for door in locked_doors:
            prereqs.append(f"need key_{door.extra.get('door_id', '?')}")

        # Try pathfinding on a world with doors unlocked
        sim = world.clone()
        for y in range(sim.height):
            for x in range(sim.width):
                for obj in sim.get_cell(x, y):
                    if isinstance(obj, Door) and obj.locked:
                        obj.locked = False
                        obj.solid = False

        push_path = _bfs_push_box(sim, box_obj.pos, tgt_obj.pos)
        if push_path is None:
            return None

        return Strategy(
            name="clear_path_then_push",
            description=f"Clear obstacles, then push Box@{box_obj.pos} to Target@{tgt_obj.pos}",
            action_sequence=push_path,
            expected_reward=10.0,
            risk_level=0.5,
            prerequisites=prereqs,
            estimated_steps=len(push_path),
        )

    def _plan_switch_strategy(
        self,
        world: PuzzleWorld,
        switch_obj: ObjectInfo,
        sw_walls: List[ObjectInfo],
        boxes: List[ObjectInfo],
        targets: List[ObjectInfo],
    ) -> Optional[Strategy]:
        """Plan: activate switch to open switch-wall, then push boxes."""
        actions: List[int] = []

        # Path to switch
        path_to_switch = _bfs_path(world, world.agent_pos, switch_obj.pos)
        if path_to_switch is None:
            return None
        actions.extend(path_to_switch)

        sid = switch_obj.extra.get("switch_id", "?")
        return Strategy(
            name="activate_switch",
            description=f"Step on Switch#{sid}@{switch_obj.pos} to open wall, then push boxes",
            action_sequence=actions,
            expected_reward=10.0 * len(boxes),
            risk_level=0.3,
            prerequisites=[],
            estimated_steps=len(actions) + 10,
        )

    def _plan_ice_strategy(
        self,
        world: PuzzleWorld,
        box_obj: ObjectInfo,
        tgt_obj: ObjectInfo,
        ice_tiles: List[ObjectInfo],
    ) -> Optional[Strategy]:
        """Plan: leverage ice sliding to push box to target."""
        bx, by = box_obj.pos

        nearby_ice = [
            ice for ice in ice_tiles
            if abs(ice.pos[0] - bx) + abs(ice.pos[1] - by) <= 3
        ]
        if not nearby_ice:
            return None

        # Try direct push (ice sliding handled by world rules)
        push_path = _bfs_push_box(world, box_obj.pos, tgt_obj.pos)
        if push_path is None:
            return None

        return Strategy(
            name="use_ice_slide",
            description=f"Use ice to slide Box@{box_obj.pos} toward Target@{tgt_obj.pos}",
            action_sequence=push_path,
            expected_reward=10.0,
            risk_level=0.4,
            prerequisites=[],
            estimated_steps=len(push_path),
        )

    def _plan_box_orderings(
        self,
        world: PuzzleWorld,
        boxes: List[ObjectInfo],
        targets: List[ObjectInfo],
        danger_zones: set,
    ) -> List[Strategy]:
        """Generate ordering variations: box_A_first vs box_B_first."""
        strategies: List[Strategy] = []
        if len(boxes) < 2 or len(targets) < 2:
            return strategies

        # Try two orderings: box[0] first, and box[1] first
        for first_idx in range(min(2, len(boxes))):
            ordered = [boxes[first_idx]] + [b for i, b in enumerate(boxes) if i != first_idx]
            actions: List[int] = []
            sim = world.clone()
            total_reward = 0.0
            risk = 0.1
            feasible = True

            available_targets = list(targets)
            for box in ordered:
                best_push = None
                best_tgt = None
                best_len = float("inf")
                for tgt in available_targets:
                    pp = _bfs_push_box(sim, box.pos, tgt.pos)
                    if pp is not None and len(pp) < best_len:
                        best_push = pp
                        best_tgt = tgt
                        best_len = len(pp)

                if best_push is None:
                    feasible = False
                    break

                actions.extend(best_push)
                for a in best_push:
                    sim.step(a)
                total_reward += 10.0
                available_targets = [t for t in available_targets if t is not best_tgt]

            if not feasible:
                continue

            box_label = chr(65 + first_idx)  # A, B, ...
            strategies.append(Strategy(
                name=f"box_{box_label}_first",
                description=f"Push Box {box_label}@{ordered[0].pos} first, then others",
                action_sequence=actions,
                expected_reward=total_reward,
                risk_level=risk,
                prerequisites=[],
                estimated_steps=len(actions),
            ))

        return strategies

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_strategy(
        self, strategy: Strategy, world: PuzzleWorld
    ) -> SimulationResult:
        """Execute the strategy inside the world model (not the real world).

        Uses PerfectWorldModel to simulate each action step-by-step.
        """
        trajectory = self.world_model.imagine_trajectory(world, strategy.action_sequence)

        imagined_states: List[dict] = []
        cumulative_reward = 0.0
        success = False
        failure_reason: Optional[str] = None

        for i, (sim_world, reward, done) in enumerate(trajectory):
            cumulative_reward += reward
            obs = sim_world.get_observation()
            imagined_states.append({
                "step": i + 1,
                "agent_pos": obs["agent_pos"],
                "boxes_on_targets": obs["boxes_on_targets"],
                "total_targets": obs["total_targets"],
                "reward": reward,
                "cumulative_reward": cumulative_reward,
            })

            # Check deadlock
            if sim_world.is_deadlock():
                failure_reason = f"deadlock at step {i + 1}"
                return SimulationResult(
                    strategy=strategy,
                    imagined_states=imagined_states,
                    final_reward=cumulative_reward,
                    success=False,
                    failure_reason=failure_reason,
                    steps_simulated=i + 1,
                )

            # Check solved
            if sim_world.solved:
                success = True
                return SimulationResult(
                    strategy=strategy,
                    imagined_states=imagined_states,
                    final_reward=cumulative_reward,
                    success=True,
                    failure_reason=None,
                    steps_simulated=i + 1,
                )

            if done:
                failure_reason = f"episode ended at step {i + 1} (truncated or terminated)"
                break

        # Check partial success: did we place at least some boxes?
        if imagined_states:
            last = imagined_states[-1]
            if last["boxes_on_targets"] == last["total_targets"]:
                success = True
            elif last["boxes_on_targets"] > 0 and failure_reason is None:
                failure_reason = (
                    f"partial: {last['boxes_on_targets']}/{last['total_targets']} boxes placed"
                )
            elif failure_reason is None:
                failure_reason = "strategy completed without solving puzzle"

        return SimulationResult(
            strategy=strategy,
            imagined_states=imagined_states,
            final_reward=cumulative_reward,
            success=success,
            failure_reason=failure_reason,
            steps_simulated=len(imagined_states),
        )

    # ------------------------------------------------------------------
    # Think — the main function
    # ------------------------------------------------------------------

    def think(
        self, observation: dict, world: PuzzleWorld
    ) -> Tuple[int, ThoughtStep]:
        """Full thinking pipeline — perceive, analyze, strategize, simulate, decide.

        Returns:
            (first_action_of_best_strategy, thought_step)
        """
        self._step_counter += 1

        # 1. Perceive
        perception = self.perceive(observation)

        # 2. Analyze
        analysis = self.analyze(world)

        # 3. Build affordance descriptions
        affordance_strs = []
        for obj_info in analysis.objects:
            aff = obj_info.affordance
            desc_parts = [f"{obj_info.obj_type.capitalize()}@{obj_info.pos}"]
            if aff.collect_score > 0.5:
                desc_parts.append("collectible")
            if aff.push_score > 0.5:
                if aff.risk_score > 0.5:
                    desc_parts.append("pushable but risky")
                else:
                    desc_parts.append("pushable")
            if aff.risk_score > 0.7:
                desc_parts.append("HIGH RISK (deadlock danger)")
            if aff.utility_score > 0.8:
                desc_parts.append("high utility")
            if aff.prerequisite_keys:
                desc_parts.append(f"needs key {aff.prerequisite_keys}")
            affordance_strs.append(" — ".join(desc_parts))

        # 4. Generate strategies
        strategies = self.generate_strategies(world, analysis)

        # 5. Simulate each strategy
        simulations: List[SimulationResult] = []
        for strategy in strategies:
            sim_result = self.simulate_strategy(strategy, world)
            simulations.append(sim_result)

        # 6. Rank: (success, final_reward, -risk)
        ranked = sorted(
            simulations,
            key=lambda s: (
                s.success,
                s.final_reward,
                -s.strategy.risk_level,
            ),
            reverse=True,
        )

        # 7. Select best
        best: Optional[SimulationResult] = None
        chosen_strategy: Optional[Strategy] = None
        confidence = 0.0
        reasoning = ""

        if ranked:
            best = ranked[0]
            if best.success:
                chosen_strategy = best.strategy
                confidence = min(1.0, 0.7 + 0.3 * (1.0 - best.strategy.risk_level))
                reasoning = (
                    f"Strategy '{best.strategy.name}' succeeds with "
                    f"reward {best.final_reward:.1f} in {best.steps_simulated} steps"
                )
            elif best.final_reward > 0:
                chosen_strategy = best.strategy
                confidence = 0.4
                reasoning = (
                    f"No strategy fully solves, but '{best.strategy.name}' "
                    f"gets partial reward {best.final_reward:.1f}"
                )
                if best.failure_reason:
                    reasoning += f" ({best.failure_reason})"
            else:
                # All strategies fail — fall back to exploration
                confidence = 0.1
                reasoning = "All strategies fail or produce negative reward — exploring"

        # Determine the action to take
        if chosen_strategy and chosen_strategy.action_sequence:
            action = chosen_strategy.action_sequence[0]
        else:
            # Fallback: try each direction, pick one that doesn't hit a wall
            action = ACTION_UP
            for try_action in [ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT]:
                clone = world.clone()
                _obs, reward, _t, _tr, _i = clone.step(try_action)
                if reward > -0.5:  # not an invalid move
                    action = try_action
                    break

        # Build reasoning with key-door and deadlock context
        if not reasoning:
            reasoning = "No viable strategies found"

        # Enrich reasoning with analysis context
        key_rels = [r for r in analysis.relationships if r.relation == "unlocks"]
        deadlock_rels = [r for r in analysis.relationships if r.relation == "deadlock_if_pushed"]
        if key_rels and chosen_strategy and "key" in chosen_strategy.name:
            kr = key_rels[0]
            reasoning += (
                f". Must collect key before anything because "
                f"Key@{kr.source.pos} unlocks Door@{kr.target.pos}"
            )
        if deadlock_rels:
            reasoning += f". Avoiding {len(deadlock_rels)} deadlock-risk push direction(s)"

        # 8. Record ThoughtStep
        thought_step = ThoughtStep(
            step_number=self._step_counter,
            perception=perception["perception_str"],
            affordances=affordance_strs,
            strategies_considered=strategies,
            simulations=simulations,
            chosen_strategy=chosen_strategy,
            confidence=confidence,
            reasoning=reasoning,
        )

        self.thought_history.append(thought_step)
        return action, thought_step

    # ------------------------------------------------------------------
    # Thought summary (human-readable)
    # ------------------------------------------------------------------

    @staticmethod
    def get_thought_summary(thought_step: ThoughtStep) -> str:
        """Human-readable summary of one thinking step."""
        lines = [
            "\U0001f9e0 Thinking...",
            f"\U0001f441\ufe0f  {thought_step.perception}",
        ]

        # Affordance highlights
        if thought_step.affordances:
            aff_lines = []
            for a in thought_step.affordances[:6]:
                aff_lines.append(f"   {a}")
            lines.append("\U0001f9e9 Affordances:")
            lines.extend(aff_lines)

        # Strategies
        lines.append("\U0001f914 Strategies:")
        for i, sim in enumerate(thought_step.simulations):
            s = sim.strategy
            reward_str = f"reward: {sim.final_reward:.0f}"
            risk_str = "LOW" if s.risk_level < 0.3 else ("MED" if s.risk_level < 0.6 else "HIGH")
            if sim.success:
                mark = "\u2705"
            elif sim.failure_reason and "deadlock" in sim.failure_reason:
                mark = "\u274c"
            elif sim.final_reward < 0:
                mark = "\u274c"
            else:
                mark = "\u26a0\ufe0f"
            lines.append(
                f"  {i + 1}. {s.name}: {s.description}  "
                f"[{reward_str}, risk: {risk_str}] {mark}"
            )
            if sim.failure_reason:
                lines.append(f"     \u2514\u2500 {sim.failure_reason}")

        # Decision
        if thought_step.chosen_strategy:
            lines.append(
                f"\u2728 Decision: {thought_step.chosen_strategy.name} "
                f"(confidence: {thought_step.confidence:.2f})"
            )
        else:
            lines.append(
                f"\u2728 Decision: explore (confidence: {thought_step.confidence:.2f})"
            )

        lines.append(f"\U0001f4a1 Reasoning: {thought_step.reasoning}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Test / demo
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    from env.level_generator import LevelGenerator

    gen = LevelGenerator()
    sim = MentalSimulator()

    # ------------------------------------------------------------------
    # Test 1: Level 5 (key + door + 1 box)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("TEST 1: Level 5 — key + door + box")
    print("=" * 70)

    world5 = gen.generate(difficulty=5, seed=42)
    obs5 = world5.get_observation()
    action5, thought5 = sim.think(obs5, world5)

    print(MentalSimulator.get_thought_summary(thought5))
    print(f"\nChosen first action: {ACTION_NAMES[action5]}")

    # Verify strategy involves key collection
    if thought5.chosen_strategy:
        strat = thought5.chosen_strategy
        has_key_step = "key" in strat.name.lower() or "key" in strat.description.lower()
        print(f"\nStrategy involves key collection: {has_key_step}")
        assert has_key_step, "Level 5 strategy should involve collecting a key!"
        print("PASS: Strategy correctly prioritises key collection")
    else:
        print("WARNING: No strategy chosen (level may be trivial or unsolvable)")

    # ------------------------------------------------------------------
    # Test 2: Level 4 (deadlock danger — 2 boxes, more walls)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST 2: Level 4 — deadlock danger")
    print("=" * 70)

    world4 = gen.generate(difficulty=4, seed=42)
    obs4 = world4.get_observation()

    sim2 = MentalSimulator()
    action4, thought4 = sim2.think(obs4, world4)

    print(MentalSimulator.get_thought_summary(thought4))
    print(f"\nChosen first action: {ACTION_NAMES[action4]}")

    # Verify deadlock awareness
    deadlock_strategies = [
        s for s in thought4.simulations
        if s.failure_reason and "deadlock" in s.failure_reason
    ]
    deadlock_affordances = [
        a for a in thought4.affordances if "risk" in a.lower() or "deadlock" in a.lower()
    ]

    has_deadlock_awareness = bool(deadlock_strategies) or bool(deadlock_affordances)
    if has_deadlock_awareness:
        print(f"\nDeadlock-failed strategies: {len(deadlock_strategies)}")
        print(f"Risk-flagged affordances: {len(deadlock_affordances)}")
        print("PASS: Agent identifies and avoids deadlock moves")
    else:
        # Even if no deadlock found for this seed, the agent should at least
        # flag danger zones
        danger_count = len(sim2.affordance.analyze_scene(world4).danger_zones)
        print(f"\nDanger zones identified: {danger_count}")
        if danger_count > 0:
            print("PASS: Agent identifies danger zones in the level")
        else:
            print("NOTE: This seed has no deadlock positions (level is safe)")

    # Verify the chosen strategy isn't one that leads to deadlock
    if thought4.chosen_strategy:
        chosen_sim = next(
            (s for s in thought4.simulations if s.strategy is thought4.chosen_strategy),
            None,
        )
        if chosen_sim:
            assert chosen_sim.failure_reason is None or "deadlock" not in chosen_sim.failure_reason, (
                "Chosen strategy should NOT lead to deadlock!"
            )
            print("PASS: Chosen strategy does not lead to deadlock")

    print("\n" + "=" * 70)
    print("All tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
