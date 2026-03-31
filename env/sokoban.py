"""Core Sokoban engine — clean, fast, standalone.

No keys, doors, ice, or switches. Pure Sokoban: walls, boxes, targets, player.
"""

from __future__ import annotations

from collections import deque
from typing import FrozenSet, List, Optional, Set, Tuple

# Actions: 0=up, 1=down, 2=left, 3=right
DIR_DELTAS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
DIR_NAMES = ["up", "down", "left", "right"]


class SokobanState:
    """Immutable Sokoban game state."""

    __slots__ = ("walls", "boxes", "targets", "player", "width", "height")

    def __init__(
        self,
        walls: FrozenSet[Tuple[int, int]],
        boxes: FrozenSet[Tuple[int, int]],
        targets: FrozenSet[Tuple[int, int]],
        player: Tuple[int, int],
        width: int,
        height: int,
    ) -> None:
        self.walls = walls
        self.boxes = boxes
        self.targets = targets
        self.player = player
        self.width = width
        self.height = height

    def move(self, action: int) -> Optional["SokobanState"]:
        """Try to execute action. Returns new state or None if invalid."""
        dx, dy = DIR_DELTAS[action]
        px, py = self.player
        nx, ny = px + dx, py + dy

        if (nx, ny) in self.walls:
            return None

        if (nx, ny) in self.boxes:
            bx, by = nx + dx, ny + dy
            if (bx, by) in self.walls or (bx, by) in self.boxes:
                return None
            new_boxes = (self.boxes - frozenset({(nx, ny)})) | frozenset({(bx, by)})
            return SokobanState(
                self.walls, new_boxes, self.targets, (nx, ny), self.width, self.height
            )

        return SokobanState(
            self.walls, self.boxes, self.targets, (nx, ny), self.width, self.height
        )

    @property
    def solved(self) -> bool:
        return self.boxes == self.targets

    @property
    def n_boxes_on_target(self) -> int:
        return len(self.boxes & self.targets)

    def is_deadlocked(self) -> bool:
        """Check for simple corner deadlock."""
        for bx, by in self.boxes:
            if (bx, by) in self.targets:
                continue
            wall_up = (bx, by - 1) in self.walls
            wall_dn = (bx, by + 1) in self.walls
            wall_lt = (bx - 1, by) in self.walls
            wall_rt = (bx + 1, by) in self.walls
            if (wall_up or wall_dn) and (wall_lt or wall_rt):
                return True
        return False

    def box_distances(self) -> float:
        """Sum of Manhattan distances from each box to nearest target."""
        total = 0.0
        for bx, by in self.boxes:
            if (bx, by) in self.targets:
                continue
            dists = [abs(bx - tx) + abs(by - ty) for tx, ty in self.targets
                     if (tx, ty) not in (self.boxes - {(bx, by)})]
            total += min(dists) if dists else 100.0
        return total

    def key(self) -> Tuple:
        return (self.player, self.boxes)

    def clone(self) -> "SokobanState":
        return SokobanState(
            self.walls, self.boxes, self.targets,
            self.player, self.width, self.height,
        )

    # -- Serialization (XSB format) -----------------------------------------

    def to_xsb(self) -> str:
        lines = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                p = (x, y)
                if p in self.walls:
                    row.append("#")
                elif p == self.player and p in self.targets:
                    row.append("+")
                elif p == self.player:
                    row.append("@")
                elif p in self.boxes and p in self.targets:
                    row.append("*")
                elif p in self.boxes:
                    row.append("$")
                elif p in self.targets:
                    row.append(".")
                else:
                    row.append(" ")
            lines.append("".join(row).rstrip())
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)

    @staticmethod
    def from_xsb(text: str) -> "SokobanState":
        walls: Set[Tuple[int, int]] = set()
        boxes: Set[Tuple[int, int]] = set()
        targets: Set[Tuple[int, int]] = set()
        player = (1, 1)
        lines = text.split("\n")
        height = len(lines)
        width = max((len(l) for l in lines), default=0)

        for y, line in enumerate(lines):
            for x, ch in enumerate(line):
                if ch == "#":
                    walls.add((x, y))
                elif ch == "$":
                    boxes.add((x, y))
                elif ch == ".":
                    targets.add((x, y))
                elif ch == "@":
                    player = (x, y)
                elif ch == "*":
                    boxes.add((x, y))
                    targets.add((x, y))
                elif ch == "+":
                    player = (x, y)
                    targets.add((x, y))

        return SokobanState(
            frozenset(walls), frozenset(boxes), frozenset(targets),
            player, width, height,
        )


# ═══════════════════════════════════════════════════════════════════════
# BFS Solver
# ═══════════════════════════════════════════════════════════════════════


def solve(state: SokobanState, max_states: int = 300_000) -> Optional[List[int]]:
    """BFS solve. Returns action list or None."""
    if state.solved:
        return []

    queue: deque = deque()
    queue.append((state, []))
    visited = {state.key()}

    while queue and len(visited) < max_states:
        current, moves = queue.popleft()
        for action in range(4):
            new = current.move(action)
            if new is None:
                continue
            k = new.key()
            if k in visited:
                continue
            if new.is_deadlocked():
                continue
            visited.add(k)
            new_moves = moves + [action]
            if new.solved:
                return new_moves
            queue.append((new, new_moves))

    return None
