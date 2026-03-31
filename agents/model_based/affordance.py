"""Affordance analysis — answering "What can I do with this object?"

Provides:
- :class:`AffordanceNet` — neural network that predicts per-object affordance
  vectors from local context.
- :class:`ContextualAffordance` — rule-based + learned scene analyser that
  computes pairwise relationships, priority rankings, and danger zones.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.objects import (
    Box,
    Door,
    Floor,
    IceTile,
    Key,
    OneWayTile,
    PressureSwitch,
    SwitchWall,
    Target,
    Wall,
)
from env.puzzle_world import PuzzleWorld, TYPE_IDS

# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

NUM_TYPE_IDS = 14  # 0-13 as defined in TYPE_IDS


class AffordanceType(Enum):
    PUSHABLE = auto()
    BLOCKING = auto()
    COLLECTIBLE = auto()
    UNLOCKS = auto()
    SLIPPERY = auto()
    ACTIVATABLE = auto()
    CONDITIONAL_BLOCK = auto()
    GOAL_POSITION = auto()
    DIRECTIONAL = auto()
    DEADLOCK_RISK = auto()


@dataclass
class AffordanceVector:
    """Per-object affordance prediction."""

    can_interact: float = 0.0
    push_score: float = 0.0
    collect_score: float = 0.0
    risk_score: float = 0.0
    utility_score: float = 0.0
    prerequisite_keys: List[int] = field(default_factory=list)


@dataclass
class ObjectInfo:
    """One object in the scene with its affordance."""

    obj_type: str
    pos: Tuple[int, int]
    affordance: AffordanceVector
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """Pairwise relationship between two objects."""

    source: ObjectInfo
    target: ObjectInfo
    relation: str  # e.g. "can_reach_target", "deadlock_if_pushed", "unlocks"
    score: float = 0.0
    description: str = ""


@dataclass
class SceneAnalysis:
    """Full analysis of the current puzzle state."""

    objects: List[ObjectInfo]
    relationships: List[Relationship]
    suggested_actions: List[str]
    danger_zones: List[Tuple[int, int]]
    required_sequence: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# AffordanceNet
# ---------------------------------------------------------------------------

class AffordanceNet(nn.Module):
    """Neural network predicting :class:`AffordanceVector` from local context.

    Inputs
    ------
    object_type : int
        Type-id (0-13) -> Embedding(14, 16).
    position : (x, y)
        Normalised to [0, 1] -> Linear(2, 8).
    neighbors : (n, s, e, w)
        Four adjacent type-ids -> shared Embedding -> Linear(4*16, 32).
    agent_distance : float
        Manhattan distance (normalised) -> Linear(1, 4).
    context : (n_keys_held, boxes_remaining)
        -> Linear(2, 8).

    Output: 5 sigmoid heads (can_interact, push, collect, risk, utility).
    """

    def __init__(self) -> None:
        super().__init__()
        self.type_emb = nn.Embedding(NUM_TYPE_IDS + 1, 16, padding_idx=NUM_TYPE_IDS)
        # +1 for out-of-bounds sentinel

        self.pos_mlp = nn.Linear(2, 8)
        self.neigh_mlp = nn.Linear(4 * 16, 32)
        self.dist_mlp = nn.Linear(1, 4)
        self.ctx_mlp = nn.Linear(2, 8)

        combined = 16 + 8 + 32 + 4 + 8  # 68
        self.trunk = nn.Sequential(
            nn.Linear(combined, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.head_interact = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.head_push = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.head_collect = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.head_risk = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.head_utility = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())

    def forward(
        self,
        obj_type: torch.Tensor,       # (B,) int
        position: torch.Tensor,        # (B, 2) float
        neighbors: torch.Tensor,       # (B, 4) int
        agent_dist: torch.Tensor,      # (B, 1) float
        context: torch.Tensor,         # (B, 2) float
    ) -> Tuple[torch.Tensor, ...]:
        """Return five (B, 1) sigmoid outputs."""
        te = self.type_emb(obj_type)                      # (B, 16)
        pe = torch.relu(self.pos_mlp(position))           # (B, 8)
        ne = self.type_emb(neighbors)                     # (B, 4, 16)
        ne = torch.relu(self.neigh_mlp(ne.view(ne.size(0), -1)))  # (B, 32)
        de = torch.relu(self.dist_mlp(agent_dist))        # (B, 4)
        ce = torch.relu(self.ctx_mlp(context))            # (B, 8)

        x = torch.cat([te, pe, ne, de, ce], dim=-1)       # (B, 68)
        h = self.trunk(x)                                  # (B, 32)

        return (
            self.head_interact(h),
            self.head_push(h),
            self.head_collect(h),
            self.head_risk(h),
            self.head_utility(h),
        )

    def predict_single(
        self,
        obj_type: int,
        position: Tuple[float, float],
        neighbors: List[int],
        agent_dist: float,
        context: Tuple[float, float],
    ) -> AffordanceVector:
        """Convenience: predict for one object, return an :class:`AffordanceVector`."""
        self.eval()
        with torch.no_grad():
            ot = torch.tensor([obj_type], dtype=torch.long)
            pos = torch.tensor([list(position)], dtype=torch.float32)
            nei = torch.tensor([neighbors], dtype=torch.long)
            ad = torch.tensor([[agent_dist]], dtype=torch.float32)
            ctx = torch.tensor([list(context)], dtype=torch.float32)
            ci, ps, cs, rs, us = self(ot, pos, nei, ad, ctx)
        return AffordanceVector(
            can_interact=ci.item(),
            push_score=ps.item(),
            collect_score=cs.item(),
            risk_score=rs.item(),
            utility_score=us.item(),
        )


# ---------------------------------------------------------------------------
# Scene-level helpers
# ---------------------------------------------------------------------------

def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _is_corner(world: PuzzleWorld, x: int, y: int) -> bool:
    """True if (x, y) has two perpendicular walls/OOB neighbours."""
    def blocked(cx: int, cy: int) -> bool:
        if not world._in_bounds(cx, cy):
            return True
        return world._has_wall(cx, cy)

    u, d = blocked(x, y - 1), blocked(x, y + 1)
    l, r = blocked(x - 1, y), blocked(x + 1, y)
    return (u and l) or (u and r) or (d and l) or (d and r)


def _cell_type_id(world: PuzzleWorld, x: int, y: int) -> int:
    """Return primary type-id for the cell, or NUM_TYPE_IDS for OOB."""
    if not world._in_bounds(x, y):
        return NUM_TYPE_IDS  # sentinel
    return world._cell_type_id(x, y)


def _bfs_reachable(world: PuzzleWorld, start: Tuple[int, int]) -> set:
    """Return the set of cells reachable from *start* (ignoring boxes)."""
    visited = {start}
    q: deque = deque([start])
    while q:
        cx, cy = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited:
                continue
            if not world._in_bounds(nx, ny):
                continue
            # Skip walls and closed switch-walls
            if world._has_wall(nx, ny):
                continue
            if any(isinstance(o, SwitchWall) and not o.open for o in world.get_cell(nx, ny)):
                continue
            visited.add((nx, ny))
            q.append((nx, ny))
    return visited


def _find_all_danger_zones(world: PuzzleWorld) -> List[Tuple[int, int]]:
    """Return every empty cell where a box would be deadlocked."""
    zones: List[Tuple[int, int]] = []
    for y in range(world.height):
        for x in range(world.width):
            cell = world.get_cell(x, y)
            # Skip cells that already have a solid object
            if any(isinstance(o, (Wall, Box)) for o in cell):
                continue
            # Skip if a target is here (box on target is never a deadlock)
            if any(isinstance(o, Target) for o in cell):
                continue
            # Would a box here be deadlocked?
            if world._box_is_deadlocked(x, y):
                zones.append((x, y))
    return zones


# ---------------------------------------------------------------------------
# ContextualAffordance
# ---------------------------------------------------------------------------

class ContextualAffordance:
    """Scene-level affordance analyser combining rule-based reasoning with
    an optional learned :class:`AffordanceNet`.

    Parameters
    ----------
    net : AffordanceNet or None
        If provided, neural predictions are blended with rule-based scores.
        Otherwise, pure rule-based analysis is used.
    """

    def __init__(self, net: Optional[AffordanceNet] = None) -> None:
        self.net = net

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_scene(self, world: PuzzleWorld) -> SceneAnalysis:
        """Analyse the full puzzle state and return a :class:`SceneAnalysis`."""
        # 1. Gather object info + individual affordances
        objects = self._gather_objects(world)

        # 2. Compute pairwise relationships
        relationships = self._compute_relationships(world, objects)

        # 3. Identify danger zones
        danger_zones = _find_all_danger_zones(world)

        # 4. Required sequencing (keys before doors, etc.)
        required_sequence = self._compute_sequence(world, objects, relationships)

        # 5. Priority-ranked suggestions
        suggested_actions = self._rank_actions(objects, relationships, required_sequence)

        return SceneAnalysis(
            objects=objects,
            relationships=relationships,
            suggested_actions=suggested_actions,
            danger_zones=danger_zones,
            required_sequence=required_sequence,
        )

    # ------------------------------------------------------------------
    # Object gathering
    # ------------------------------------------------------------------

    def _gather_objects(self, world: PuzzleWorld) -> List[ObjectInfo]:
        objs: List[ObjectInfo] = []
        ax, ay = world.agent_pos
        inv_keys = {k.key_id for k in world.inventory}
        n_keys = len(world.inventory)
        # Count remaining boxes not on target
        boxes_remaining = sum(
            1
            for y in range(world.height)
            for x in range(world.width)
            for o in world.get_cell(x, y)
            if isinstance(o, Box) and not o.on_target
        )

        for y in range(world.height):
            for x in range(world.width):
                for obj in world.get_cell(x, y):
                    info = self._analyze_object(
                        world, obj, x, y, ax, ay, n_keys, boxes_remaining, inv_keys,
                    )
                    if info is not None:
                        objs.append(info)
        return objs

    def _analyze_object(
        self,
        world: PuzzleWorld,
        obj: Any,
        x: int, y: int,
        ax: int, ay: int,
        n_keys: int,
        boxes_remaining: int,
        inv_keys: set,
    ) -> Optional[ObjectInfo]:
        """Compute affordance for a single object."""
        dist = _manhattan((ax, ay), (x, y))
        max_dim = max(world.width, world.height)
        norm_dist = dist / max(1, max_dim * 2)
        norm_x = x / max(1, world.width - 1)
        norm_y = y / max(1, world.height - 1)

        neighbors = [
            _cell_type_id(world, x, y - 1),
            _cell_type_id(world, x, y + 1),
            _cell_type_id(world, x - 1, y),
            _cell_type_id(world, x + 1, y),
        ]

        # --- Rule-based affordance ---
        aff = AffordanceVector()
        extra: Dict[str, Any] = {}

        if isinstance(obj, Box):
            aff.can_interact = 1.0 if dist == 1 else max(0.0, 1.0 - norm_dist)
            aff.push_score = 0.8
            # Risk: is any adjacent cell a deadlock zone?
            danger_adj = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if world._in_bounds(nx, ny) and not any(
                    isinstance(o, Target) for o in world.get_cell(nx, ny)
                ):
                    if world._box_is_deadlocked(nx, ny):
                        danger_adj += 1
            aff.risk_score = min(1.0, danger_adj * 0.35)
            if _is_corner(world, x, y) and not obj.on_target:
                aff.risk_score = 1.0
            # Utility: distance to nearest target
            min_target_dist = self._nearest_target_dist(world, x, y)
            aff.utility_score = max(0.0, 1.0 - min_target_dist / max(1, max_dim))
            if obj.on_target:
                aff.utility_score = 1.0
                aff.risk_score = 0.0
            extra["on_target"] = obj.on_target
            obj_type_str = "box"

        elif isinstance(obj, Key):
            aff.can_interact = 1.0
            aff.collect_score = 1.0
            # Utility: does a matching locked door exist?
            has_door = self._has_matching_door(world, obj.key_id)
            aff.utility_score = 1.0 if has_door else 0.3
            extra["key_id"] = obj.key_id
            obj_type_str = "key"

        elif isinstance(obj, Door):
            if obj.locked:
                has_key = obj.door_id in inv_keys
                aff.can_interact = 1.0 if has_key else 0.0
                aff.risk_score = 0.0
                aff.utility_score = 0.9
                aff.prerequisite_keys = [] if has_key else [obj.door_id]
                extra["door_id"] = obj.door_id
                extra["locked"] = True
            else:
                aff.can_interact = 1.0
                extra["door_id"] = obj.door_id
                extra["locked"] = False
            obj_type_str = "door"

        elif isinstance(obj, Target):
            aff.utility_score = 1.0
            # Check if a box is already here
            has_box = any(isinstance(o, Box) for o in world.get_cell(x, y))
            extra["has_box"] = has_box
            obj_type_str = "target"

        elif isinstance(obj, IceTile):
            aff.risk_score = 0.3  # inherently risky
            aff.utility_score = 0.2
            obj_type_str = "ice"

        elif isinstance(obj, PressureSwitch):
            aff.can_interact = 1.0
            aff.utility_score = 0.7
            extra["switch_id"] = obj.switch_id
            extra["activated"] = obj.activated
            obj_type_str = "switch"

        elif isinstance(obj, SwitchWall):
            extra["linked_switch_id"] = obj.linked_switch_id
            extra["open"] = obj.open
            aff.can_interact = 0.0
            obj_type_str = "switch_wall"

        elif isinstance(obj, Wall):
            return None  # skip walls

        elif isinstance(obj, Floor):
            return None  # skip plain floors

        else:
            return None

        # --- Neural prediction (blend if available) ---
        if self.net is not None:
            type_id = _cell_type_id(world, x, y)
            n_aff = self.net.predict_single(
                obj_type=type_id,
                position=(norm_x, norm_y),
                neighbors=neighbors,
                agent_dist=norm_dist,
                context=(float(n_keys), float(boxes_remaining)),
            )
            # Blend: 60% rule-based, 40% learned
            aff.can_interact = 0.6 * aff.can_interact + 0.4 * n_aff.can_interact
            aff.push_score = 0.6 * aff.push_score + 0.4 * n_aff.push_score
            aff.collect_score = 0.6 * aff.collect_score + 0.4 * n_aff.collect_score
            aff.risk_score = 0.6 * aff.risk_score + 0.4 * n_aff.risk_score
            aff.utility_score = 0.6 * aff.utility_score + 0.4 * n_aff.utility_score

        return ObjectInfo(obj_type=obj_type_str, pos=(x, y), affordance=aff, extra=extra)

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    def _compute_relationships(
        self, world: PuzzleWorld, objects: List[ObjectInfo],
    ) -> List[Relationship]:
        rels: List[Relationship] = []
        boxes = [o for o in objects if o.obj_type == "box"]
        targets = [o for o in objects if o.obj_type == "target"]
        keys = [o for o in objects if o.obj_type == "key"]
        doors = [o for o in objects if o.obj_type == "door" and o.extra.get("locked")]
        switches = [o for o in objects if o.obj_type == "switch"]
        sw_walls = [o for o in objects if o.obj_type == "switch_wall"]
        ice_tiles = [o for o in objects if o.obj_type == "ice"]

        reachable = _bfs_reachable(world, world.agent_pos)

        # Box <-> Target
        for box in boxes:
            if box.extra.get("on_target"):
                continue
            for tgt in targets:
                if tgt.extra.get("has_box"):
                    continue
                dist = _manhattan(box.pos, tgt.pos)
                path_clear = tgt.pos in reachable and box.pos in reachable
                score = max(0.0, 1.0 - dist / max(1, world.width + world.height))
                desc = (
                    f"Box@{box.pos} -> Target@{tgt.pos}: "
                    f"dist={dist}, path={'clear' if path_clear else 'blocked'}"
                )
                rels.append(Relationship(
                    source=box, target=tgt,
                    relation="can_reach_target" if path_clear else "path_blocked",
                    score=score if path_clear else score * 0.3,
                    description=desc,
                ))

        # Box <-> Wall corners (deadlock danger)
        for box in boxes:
            if box.extra.get("on_target"):
                continue
            bx, by = box.pos
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = bx + dx, by + dy
                if not world._in_bounds(nx, ny):
                    continue
                if not any(isinstance(o, Target) for o in world.get_cell(nx, ny)):
                    if world._box_is_deadlocked(nx, ny):
                        rels.append(Relationship(
                            source=box,
                            target=box,  # self-referential
                            relation="deadlock_if_pushed",
                            score=1.0,
                            description=(
                                f"DEADLOCK: pushing Box@{box.pos} "
                                f"toward ({nx},{ny}) traps it"
                            ),
                        ))

        # Key <-> Door
        for key in keys:
            kid = key.extra["key_id"]
            for door in doors:
                if door.extra["door_id"] == kid:
                    rels.append(Relationship(
                        source=key, target=door,
                        relation="unlocks",
                        score=1.0,
                        description=f"Key#{kid}@{key.pos} unlocks Door#{kid}@{door.pos}",
                    ))

        # Box <-> IceTile
        for box in boxes:
            for ice in ice_tiles:
                if _manhattan(box.pos, ice.pos) <= 2:
                    rels.append(Relationship(
                        source=box, target=ice,
                        relation="ice_slide",
                        score=0.5,
                        description=f"Box@{box.pos} near ice@{ice.pos} — will slide",
                    ))

        # Box/Agent <-> Switch -> SwitchWall
        for sw in switches:
            sid = sw.extra["switch_id"]
            for sww in sw_walls:
                if sww.extra["linked_switch_id"] == sid:
                    rels.append(Relationship(
                        source=sw, target=sww,
                        relation="toggles",
                        score=0.8,
                        description=(
                            f"Switch#{sid}@{sw.pos} toggles "
                            f"SwitchWall@{sww.pos}"
                        ),
                    ))

        return rels

    # ------------------------------------------------------------------
    # Sequencing
    # ------------------------------------------------------------------

    def _compute_sequence(
        self,
        world: PuzzleWorld,
        objects: List[ObjectInfo],
        relationships: List[Relationship],
    ) -> Optional[List[str]]:
        """Determine required ordering of actions (keys before doors, etc.)."""
        steps: List[str] = []

        # Keys before their doors
        key_door_pairs = [
            r for r in relationships if r.relation == "unlocks"
        ]
        for r in key_door_pairs:
            steps.append(f"Collect Key#{r.source.extra['key_id']}@{r.source.pos}")
            steps.append(f"Unlock Door#{r.target.extra['door_id']}@{r.target.pos}")

        # Then push boxes to targets
        box_target = [
            r for r in relationships
            if r.relation == "can_reach_target" and r.score > 0.2
        ]
        # Sort by score descending (easiest/best first)
        box_target.sort(key=lambda r: -r.score)
        seen_boxes = set()
        for r in box_target:
            bpos = r.source.pos
            if bpos in seen_boxes:
                continue
            seen_boxes.add(bpos)
            steps.append(
                f"Push Box@{bpos} -> Target@{r.target.pos}"
            )

        return steps if steps else None

    # ------------------------------------------------------------------
    # Action ranking
    # ------------------------------------------------------------------

    def _rank_actions(
        self,
        objects: List[ObjectInfo],
        relationships: List[Relationship],
        sequence: Optional[List[str]],
    ) -> List[str]:
        """Return prioritised action suggestions."""
        if sequence:
            return [f"{i + 1}) {s}" for i, s in enumerate(sequence)]

        # Fallback: rank objects by (utility - risk)
        scored: List[Tuple[float, str]] = []
        for obj in objects:
            a = obj.affordance
            net_score = a.utility_score - a.risk_score
            if obj.obj_type == "key":
                scored.append((net_score + 1.0, f"Collect Key@{obj.pos}"))
            elif obj.obj_type == "door" and obj.extra.get("locked"):
                scored.append((net_score + 0.5, f"Unlock Door@{obj.pos}"))
            elif obj.obj_type == "box" and not obj.extra.get("on_target"):
                scored.append((net_score, f"Push Box@{obj.pos} toward target"))
        scored.sort(key=lambda t: -t[0])
        return [f"{i + 1}) {s}" for i, (_, s) in enumerate(scored)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nearest_target_dist(world: PuzzleWorld, x: int, y: int) -> int:
        best = world.width + world.height
        for ty in range(world.height):
            for tx in range(world.width):
                if any(isinstance(o, Target) for o in world.get_cell(tx, ty)):
                    d = _manhattan((x, y), (tx, ty))
                    if d < best:
                        best = d
        return best

    @staticmethod
    def _has_matching_door(world: PuzzleWorld, key_id: int) -> bool:
        for y in range(world.height):
            for x in range(world.width):
                for obj in world.get_cell(x, y):
                    if isinstance(obj, Door) and obj.locked and obj.door_id == key_id:
                        return True
        return False


# ---------------------------------------------------------------------------
# Training data generation & pre-training
# ---------------------------------------------------------------------------

@dataclass
class _Sample:
    obj_type: int
    position: Tuple[float, float]
    neighbors: List[int]
    agent_dist: float
    context: Tuple[float, float]
    # Labels
    can_interact: float
    push_score: float
    collect_score: float
    risk_score: float
    utility_score: float


def generate_training_data(
    n_samples: int = 5000,
    seed: int = 42,
) -> List[_Sample]:
    """Generate supervised affordance labels from procedurally generated levels.

    For each object the ground-truth affordance is computed from game rules
    and BFS solver results.
    """
    from env.level_generator import LevelGenerator, solve

    gen = LevelGenerator()
    rng = random.Random(seed)
    samples: List[_Sample] = []

    attempts = 0
    while len(samples) < n_samples and attempts < n_samples * 5:
        attempts += 1
        difficulty = rng.randint(1, 7)
        level_seed = rng.randint(0, 2**31)
        try:
            world = gen.generate(difficulty=difficulty, seed=level_seed)
        except RuntimeError:
            continue

        solution = getattr(world, "_optimal_solution", None)
        max_dim = max(world.width, world.height)
        ax, ay = world.agent_pos
        n_keys = len(world.inventory)
        boxes_remaining = sum(
            1
            for y in range(world.height)
            for x in range(world.width)
            for o in world.get_cell(x, y)
            if isinstance(o, Box) and not o.on_target
        )

        for y in range(world.height):
            for x in range(world.width):
                for obj in world.get_cell(x, y):
                    s = _label_object(
                        world, obj, x, y, ax, ay, max_dim,
                        n_keys, boxes_remaining, solution,
                    )
                    if s is not None:
                        samples.append(s)
                        if len(samples) >= n_samples:
                            break
                if len(samples) >= n_samples:
                    break
            if len(samples) >= n_samples:
                break

    return samples[:n_samples]


def _label_object(
    world: PuzzleWorld,
    obj: Any,
    x: int, y: int,
    ax: int, ay: int,
    max_dim: int,
    n_keys: int,
    boxes_remaining: int,
    solution: Optional[List[int]],
) -> Optional[_Sample]:
    dist = _manhattan((ax, ay), (x, y))
    norm_dist = dist / max(1, max_dim * 2)
    norm_x = x / max(1, world.width - 1)
    norm_y = y / max(1, world.height - 1)
    type_id = _cell_type_id(world, x, y)

    neighbors = [
        _cell_type_id(world, x, y - 1),
        _cell_type_id(world, x, y + 1),
        _cell_type_id(world, x - 1, y),
        _cell_type_id(world, x + 1, y),
    ]

    ci = ps = cs = rs = us = 0.0

    if isinstance(obj, Box):
        ci = 1.0 if dist == 1 else max(0.0, 1.0 - norm_dist * 2)
        ps = 0.8
        # Risk from deadlock
        if _is_corner(world, x, y) and not obj.on_target:
            rs = 1.0
        else:
            danger_count = sum(
                1
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if world._in_bounds(x + dx, y + dy)
                and world._box_is_deadlocked(x + dx, y + dy)
                and not any(isinstance(o, Target) for o in world.get_cell(x + dx, y + dy))
            )
            rs = min(1.0, danger_count * 0.35)
        # Utility from distance to nearest target
        best_td = max_dim * 2
        for ty in range(world.height):
            for tx in range(world.width):
                if any(isinstance(o, Target) for o in world.get_cell(tx, ty)):
                    best_td = min(best_td, _manhattan((x, y), (tx, ty)))
        us = max(0.0, 1.0 - best_td / max(1, max_dim))
        if obj.on_target:
            us = 1.0
            rs = 0.0

    elif isinstance(obj, Key):
        ci = 1.0
        cs = 1.0
        has_door = False
        for dy in range(world.height):
            for dx in range(world.width):
                for o in world.get_cell(dx, dy):
                    if isinstance(o, Door) and o.locked and o.door_id == obj.key_id:
                        has_door = True
        us = 1.0 if has_door else 0.3

    elif isinstance(obj, Door) and obj.locked:
        inv_keys = {k.key_id for k in world.inventory}
        ci = 1.0 if obj.door_id in inv_keys else 0.0
        us = 0.9

    elif isinstance(obj, Target):
        us = 1.0

    elif isinstance(obj, IceTile):
        rs = 0.3
        us = 0.2

    elif isinstance(obj, PressureSwitch):
        ci = 1.0
        us = 0.7

    elif isinstance(obj, SwitchWall):
        ci = 0.0

    else:
        return None

    return _Sample(
        obj_type=type_id,
        position=(norm_x, norm_y),
        neighbors=neighbors,
        agent_dist=norm_dist,
        context=(float(n_keys), float(boxes_remaining)),
        can_interact=ci,
        push_score=ps,
        collect_score=cs,
        risk_score=rs,
        utility_score=us,
    )


def pretrain(
    net: AffordanceNet,
    dataset: List[_Sample],
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 64,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Pre-train *net* on supervised affordance data.

    Returns per-epoch loss and per-head accuracy history.
    """
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Tensorise dataset
    n = len(dataset)
    obj_types = torch.tensor([s.obj_type for s in dataset], dtype=torch.long)
    positions = torch.tensor([list(s.position) for s in dataset], dtype=torch.float32)
    neighbors = torch.tensor([s.neighbors for s in dataset], dtype=torch.long)
    agent_dists = torch.tensor([[s.agent_dist] for s in dataset], dtype=torch.float32)
    contexts = torch.tensor([list(s.context) for s in dataset], dtype=torch.float32)
    labels = torch.tensor(
        [[s.can_interact, s.push_score, s.collect_score, s.risk_score, s.utility_score]
         for s in dataset],
        dtype=torch.float32,
    )

    history: Dict[str, List[float]] = {"loss": [], "accuracy": []}
    head_names = ["interact", "push", "collect", "risk", "utility"]

    net.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        correct = np.zeros(5)
        total = 0

        for i in range(0, n, batch_size):
            idx = perm[i: i + batch_size]
            ot = obj_types[idx]
            pos = positions[idx]
            nei = neighbors[idx]
            ad = agent_dists[idx]
            ctx = contexts[idx]
            lab = labels[idx]

            preds = net(ot, pos, nei, ad, ctx)  # tuple of 5 (B, 1)
            pred_cat = torch.cat(preds, dim=-1)  # (B, 5)

            loss = criterion(pred_cat, lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
            pred_binary = (pred_cat > 0.5).float()
            correct += (pred_binary == (lab > 0.5).float()).sum(dim=0).cpu().numpy()
            total += len(idx)

        avg_loss = epoch_loss / n
        avg_acc = correct / total
        history["loss"].append(avg_loss)
        history["accuracy"].append(float(avg_acc.mean()))

        if verbose and (epoch + 1) % 10 == 0:
            acc_str = "  ".join(f"{head_names[i]}={avg_acc[i]:.2f}" for i in range(5))
            print(f"  Epoch {epoch + 1:>3d}/{epochs}: loss={avg_loss:.4f}  {acc_str}")

    return history
