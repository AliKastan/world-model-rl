"""PuzzleWorld -- grid-based 2D puzzle environment.

Implements Sokoban-style mechanics with extensions: keys/doors, ice tiles,
pressure switches, one-way tiles, and deadlock detection.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from env.objects import (
    Box,
    Door,
    Floor,
    GameObject,
    IceTile,
    Key,
    OneWayTile,
    PressureSwitch,
    SwitchWall,
    Target,
    Wall,
)

# Action constants
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

ACTION_NAMES = {ACTION_UP: "up", ACTION_DOWN: "down", ACTION_LEFT: "left", ACTION_RIGHT: "right"}

DIRECTION_DELTAS: Dict[str, Tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

DIR_TO_INT = {"up": 0, "down": 1, "left": 2, "right": 3}

# Type IDs for the observation grid
TYPE_IDS: Dict[str, int] = {
    "floor": 0,
    "wall": 1,
    "box": 2,
    "target": 3,
    "box_on_target": 4,
    "key": 5,
    "door_locked": 6,
    "door_unlocked": 7,
    "ice": 8,
    "switch_inactive": 9,
    "switch_active": 10,
    "switch_wall_closed": 11,
    "switch_wall_open": 12,
    "oneway": 13,
}


class PuzzleWorld:
    """Grid-based 2D puzzle environment with Sokoban-style mechanics."""

    def __init__(self, width: int = 10, height: int = 10) -> None:
        self.width = width
        self.height = height

        # Each cell is a list of GameObjects (supports stacking, e.g. Box on Target)
        self.grid: List[List[List[GameObject]]] = [
            [[] for _ in range(width)] for _ in range(height)
        ]

        self.agent_pos: Tuple[int, int] = (1, 1)
        self.agent_dir: str = "down"
        self.inventory: List[Key] = []
        self.steps: int = 0
        self.max_steps: int = 200
        self.solved: bool = False

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def place_object(self, obj: GameObject, x: int, y: int) -> None:
        """Place *obj* at grid position (*x*, *y*)."""
        obj.pos = (x, y)
        self.grid[y][x].append(obj)

    def get_cell(self, x: int, y: int) -> List[GameObject]:
        """Return the list of objects at (*x*, *y*)."""
        if not self._in_bounds(x, y):
            return []
        return self.grid[y][x]

    def _find_in_cell(self, x: int, y: int, cls: type) -> Optional[GameObject]:
        """Return the first object of *cls* in the cell, or ``None``."""
        for obj in self.get_cell(x, y):
            if isinstance(obj, cls):
                return obj
        return None

    def _remove_from_cell(self, x: int, y: int, obj: GameObject) -> None:
        cell = self.get_cell(x, y)
        if obj in cell:
            cell.remove(obj)

    def _cell_is_solid(self, x: int, y: int) -> bool:
        """Return True if any object in the cell blocks movement."""
        if not self._in_bounds(x, y):
            return True
        return any(o.solid for o in self.get_cell(x, y))

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _cell_type_id(self, x: int, y: int) -> int:
        """Return the primary type_id for the cell (highest-priority object)."""
        cell = self.get_cell(x, y)
        # Priority: box_on_target > box > key > door > switch_wall > switch > ice > oneway > target > wall > floor
        has_box = False
        has_target = False
        for obj in cell:
            if isinstance(obj, Box):
                has_box = True
                if obj.on_target:
                    return TYPE_IDS["box_on_target"]
            elif isinstance(obj, Target):
                has_target = True
            elif isinstance(obj, Key):
                return TYPE_IDS["key"]
            elif isinstance(obj, Door):
                return TYPE_IDS["door_unlocked"] if not obj.locked else TYPE_IDS["door_locked"]
            elif isinstance(obj, SwitchWall):
                return TYPE_IDS["switch_wall_open"] if obj.open else TYPE_IDS["switch_wall_closed"]
            elif isinstance(obj, PressureSwitch):
                return TYPE_IDS["switch_active"] if obj.activated else TYPE_IDS["switch_inactive"]
            elif isinstance(obj, IceTile):
                return TYPE_IDS["ice"]
            elif isinstance(obj, OneWayTile):
                return TYPE_IDS["oneway"]
        if has_box:
            return TYPE_IDS["box"]
        if has_target:
            return TYPE_IDS["target"]
        # Check for wall
        for obj in cell:
            if isinstance(obj, Wall):
                return TYPE_IDS["wall"]
        return TYPE_IDS["floor"]

    def get_observation(self) -> Dict[str, Any]:
        """Return the agent's full observation of the world state."""
        grid = np.zeros((self.height, self.width), dtype=np.int32)
        for y in range(self.height):
            for x in range(self.width):
                grid[y, x] = self._cell_type_id(x, y)

        boxes_on_targets = sum(
            1
            for y in range(self.height)
            for x in range(self.width)
            for obj in self.get_cell(x, y)
            if isinstance(obj, Box) and obj.on_target
        )
        total_targets = sum(
            1
            for y in range(self.height)
            for x in range(self.width)
            for obj in self.get_cell(x, y)
            if isinstance(obj, Target)
        )

        return {
            "grid": grid,
            "agent_pos": self.agent_pos,
            "agent_dir": DIR_TO_INT[self.agent_dir],
            "inventory": [k.key_id for k in self.inventory],
            "boxes_on_targets": boxes_on_targets,
            "total_targets": total_targets,
            "steps": self.steps,
        }

    # ------------------------------------------------------------------
    # Switch helpers
    # ------------------------------------------------------------------

    def _toggle_switches_at(self, x: int, y: int, activate: bool) -> None:
        """Activate or deactivate any pressure switch at (*x*, *y*) and toggle walls."""
        for obj in self.get_cell(x, y):
            if isinstance(obj, PressureSwitch):
                if activate and not obj.activated:
                    obj.activate()
                    self._toggle_linked_walls(obj.switch_id)
                elif not activate and obj.activated:
                    obj.deactivate()
                    self._toggle_linked_walls(obj.switch_id)

    def _toggle_linked_walls(self, switch_id: int) -> None:
        """Toggle all SwitchWalls linked to *switch_id*."""
        for y in range(self.height):
            for x in range(self.width):
                for obj in self.get_cell(x, y):
                    if isinstance(obj, SwitchWall) and obj.linked_switch_id == switch_id:
                        obj.toggle()

    def _anything_on_switch(self, x: int, y: int) -> bool:
        """Return True if the agent or a box is on the switch at (*x*, *y*)."""
        if self.agent_pos == (x, y):
            return True
        return any(isinstance(o, Box) for o in self.get_cell(x, y))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one action and return (obs, reward, terminated, truncated, info)."""
        reward = 0.0
        direction = ACTION_NAMES[action]
        dx, dy = DIRECTION_DELTAS[direction]
        ax, ay = self.agent_pos
        tx, ty = ax + dx, ay + dy  # target cell

        # Update facing direction regardless of move outcome
        self.agent_dir = direction

        moved = False

        if not self._in_bounds(tx, ty):
            reward += -0.5
        else:
            cell_objs = self.get_cell(tx, ty)

            # --- OneWayTile check: if target cell has a one-way that doesn't
            # allow our direction, block the move. ---
            for obj in cell_objs:
                if isinstance(obj, OneWayTile) and not obj.allows(direction):
                    reward += -0.5
                    self.steps += 1
                    terminated = self.solved
                    truncated = self.steps >= self.max_steps
                    return self.get_observation(), reward - 0.1, terminated, truncated, self._info()

            # --- Wall ---
            if any(isinstance(o, Wall) for o in cell_objs):
                reward += -0.5

            # --- SwitchWall (closed) ---
            elif any(isinstance(o, SwitchWall) and not o.open for o in cell_objs):
                reward += -0.5

            # --- Box ---
            elif any(isinstance(o, Box) for o in cell_objs):
                box: Box = next(o for o in cell_objs if isinstance(o, Box))
                bx, by = tx + dx, ty + dy  # cell behind the box

                if not self._in_bounds(bx, by) or self._cell_is_solid(bx, by):
                    reward += -0.5  # can't push
                else:
                    # Check if behind cell has a one-way that forbids this direction
                    behind_blocked = False
                    for o in self.get_cell(bx, by):
                        if isinstance(o, OneWayTile) and not o.allows(direction):
                            behind_blocked = True
                            break
                    if behind_blocked:
                        reward += -0.5
                    else:
                        # Was the box on a target before?
                        was_on_target = box.on_target

                        # Deactivate switch at box's old position if applicable
                        old_bx, old_by = tx, ty
                        self._remove_from_cell(tx, ty, box)

                        # Place box at new position
                        box.pos = (bx, by)
                        self.grid[by][bx].append(box)

                        # Check if new position has a target
                        has_target = any(isinstance(o, Target) for o in self.get_cell(bx, by))
                        box.on_target = has_target

                        if has_target and not was_on_target:
                            reward += 10.0
                        elif not has_target and was_on_target:
                            reward += -10.0

                        # Handle switches for box movement
                        self._toggle_switches_at(bx, by, True)
                        if not self._anything_on_switch(old_bx, old_by):
                            self._toggle_switches_at(old_bx, old_by, False)

                        moved = True

            # --- Door (locked) ---
            elif any(isinstance(o, Door) and o.locked for o in cell_objs):
                door: Door = next(o for o in cell_objs if isinstance(o, Door) and o.locked)
                matching_key = next(
                    (k for k in self.inventory if k.key_id == door.door_id), None
                )
                if matching_key is not None:
                    self.inventory.remove(matching_key)
                    door.unlock()
                    reward += 5.0
                    moved = True
                else:
                    reward += -0.5

            # --- Door (unlocked) ---
            elif any(isinstance(o, Door) and not o.locked for o in cell_objs):
                moved = True

            # --- Key ---
            elif any(isinstance(o, Key) for o in cell_objs):
                key: Key = next(o for o in cell_objs if isinstance(o, Key))
                self._remove_from_cell(tx, ty, key)
                self.inventory.append(key)
                reward += 2.0
                moved = True

            # --- Everything else (floor, target, ice, switch, open switch_wall) ---
            else:
                moved = True

        if moved:
            old_ax, old_ay = ax, ay
            self.agent_pos = (tx, ty)

            # Deactivate switch the agent just left
            if not self._anything_on_switch(old_ax, old_ay):
                self._toggle_switches_at(old_ax, old_ay, False)

            # Activate switch the agent just stepped on
            self._toggle_switches_at(tx, ty, True)

            # --- Ice sliding ---
            self._handle_ice_slide(direction)

        # Time penalty
        reward += -0.1

        # Check solved
        self._check_solved()

        if self.solved:
            reward += 100.0

        self.steps += 1
        terminated = self.solved
        truncated = self.steps >= self.max_steps

        return self.get_observation(), reward, terminated, truncated, self._info()

    def _handle_ice_slide(self, direction: str) -> None:
        """Slide the agent along ice tiles until they hit a non-ice tile or solid."""
        dx, dy = DIRECTION_DELTAS[direction]
        while True:
            x, y = self.agent_pos
            # Are we on ice?
            on_ice = any(isinstance(o, IceTile) for o in self.get_cell(x, y))
            if not on_ice:
                break
            nx, ny = x + dx, y + dy
            if not self._in_bounds(nx, ny) or self._cell_is_solid(nx, ny):
                break
            # Check one-way tile at next position
            blocked = False
            for o in self.get_cell(nx, ny):
                if isinstance(o, OneWayTile) and not o.allows(direction):
                    blocked = True
                    break
            if blocked:
                break

            old_x, old_y = x, y
            self.agent_pos = (nx, ny)

            # Handle switches
            if not self._anything_on_switch(old_x, old_y):
                self._toggle_switches_at(old_x, old_y, False)
            self._toggle_switches_at(nx, ny, True)

    def _check_solved(self) -> None:
        """Set self.solved = True if all targets have a box on them."""
        total_targets = 0
        boxes_on_targets = 0
        for y in range(self.height):
            for x in range(self.width):
                for obj in self.get_cell(x, y):
                    if isinstance(obj, Target):
                        total_targets += 1
                    if isinstance(obj, Box) and obj.on_target:
                        boxes_on_targets += 1
        if total_targets > 0 and boxes_on_targets >= total_targets:
            self.solved = True

    def _info(self) -> Dict[str, Any]:
        obs = self.get_observation()
        return {
            "is_deadlock": self.is_deadlock(),
            "steps": self.steps,
            "boxes_placed": obs["boxes_on_targets"],
            "total_targets": obs["total_targets"],
            "inventory": [k.key_id for k in self.inventory],
        }

    # ------------------------------------------------------------------
    # Deadlock detection
    # ------------------------------------------------------------------

    def is_deadlock(self) -> bool:
        """Simple deadlock detection.

        A deadlock exists when any box that is NOT on a target is stuck:
        - Corner deadlock: two adjacent walls at 90 degrees.
        - Wall-edge deadlock: box is against a wall with no reachable target
          along that wall line.
        """
        for y in range(self.height):
            for x in range(self.width):
                for obj in self.get_cell(x, y):
                    if isinstance(obj, Box) and not obj.on_target:
                        if self._box_is_deadlocked(x, y):
                            return True
        return False

    def _box_is_deadlocked(self, x: int, y: int) -> bool:
        """Check if a box at (*x*, *y*) is in an unrecoverable position."""
        # Check corners: wall or out-of-bounds on two adjacent sides
        up_blocked = not self._in_bounds(x, y - 1) or self._has_wall(x, y - 1)
        down_blocked = not self._in_bounds(x, y + 1) or self._has_wall(x, y + 1)
        left_blocked = not self._in_bounds(x - 1, y) or self._has_wall(x - 1, y)
        right_blocked = not self._in_bounds(x + 1, y) or self._has_wall(x + 1, y)

        # Corner deadlock
        if (up_blocked and left_blocked) or (up_blocked and right_blocked) or \
           (down_blocked and left_blocked) or (down_blocked and right_blocked):
            return True

        # Wall-edge deadlock: box against a wall edge with no target along that wall
        if up_blocked or down_blocked:
            # Box is against a horizontal wall — check if there's a target in this row
            # that the box could reach (i.e. no wall segment blocks it)
            if not self._target_reachable_along_wall(x, y, horizontal=True):
                return True

        if left_blocked or right_blocked:
            if not self._target_reachable_along_wall(x, y, horizontal=False):
                return True

        return False

    def _has_wall(self, x: int, y: int) -> bool:
        return any(isinstance(o, Wall) for o in self.get_cell(x, y))

    def _target_reachable_along_wall(self, bx: int, by: int, horizontal: bool) -> bool:
        """Check if there's a target along the wall that the box can slide to."""
        if horizontal:
            # Scan left and right in the same row
            for x in range(self.width):
                if any(isinstance(o, Target) for o in self.get_cell(x, by)):
                    return True
        else:
            for y in range(self.height):
                if any(isinstance(o, Target) for o in self.get_cell(bx, y)):
                    return True
        return False

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------

    def clone(self) -> PuzzleWorld:
        """Deep copy of the entire world state.

        Critical for mental simulation -- the agent clones the world
        to "think ahead" without affecting the real environment.
        """
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # ASCII rendering
    # ------------------------------------------------------------------

    def render_ascii(self) -> str:
        """Return a pretty ASCII representation for debugging."""
        lines: List[str] = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if self.agent_pos == (x, y):
                    row.append("@")
                    continue
                cell = self.get_cell(x, y)
                char = " "
                # Priority rendering
                for obj in cell:
                    if isinstance(obj, Box):
                        char = "*" if obj.on_target else "$"
                        break
                    if isinstance(obj, Key):
                        char = "k"
                        break
                    if isinstance(obj, Door):
                        char = "D" if obj.locked else "d"
                        break
                    if isinstance(obj, SwitchWall):
                        char = " " if obj.open else "%"
                        break
                    if isinstance(obj, PressureSwitch):
                        char = "^"
                        break
                    if isinstance(obj, IceTile):
                        char = "~"
                        break
                    if isinstance(obj, OneWayTile):
                        char = obj.symbol
                        break
                    if isinstance(obj, Wall):
                        char = "#"
                    elif isinstance(obj, Target):
                        char = "."
                row.append(char)
            lines.append("".join(row))
        return "\n".join(lines)
