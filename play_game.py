#!/usr/bin/env python3
"""Standalone playable Sokoban puzzle game with beautiful rendering.

Controls:
  Arrow keys / WASD  — Move agent
  U                  — Undo last move
  R                  — Restart level
  N                  — Next level
  H                  — Show hint (optimal next move)
  1-5                — Jump to level
  ESC / Q            — Quit

No project imports needed — fully self-contained (only pygame + numpy).
"""

from __future__ import annotations

import copy
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pygame

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

CELL = 64
FPS = 60

# Colours
BG = (22, 22, 35)
GRID_LINE = (35, 35, 50)
C_AGENT = (99, 102, 241)
C_AGENT_EYE = (255, 255, 255)
C_BOX = (251, 191, 36)
C_BOX_DONE = (52, 211, 153)
C_TARGET = (52, 211, 153)
C_WALL = (75, 85, 99)
C_WALL_HI = (90, 100, 114)
C_WALL_LO = (55, 65, 79)
C_KEY = (251, 146, 60)
C_DOOR_LOCKED = (239, 68, 68)
C_DOOR_OPEN = (74, 222, 128)
C_FLOOR = (30, 30, 46)
C_HUD_BG = (15, 15, 25)
C_HUD_TEXT = (220, 220, 235)
C_HINT = (99, 102, 241)
C_DEADLOCK = (239, 68, 68)
C_GOLD = (251, 191, 36)
C_DIM = (60, 60, 80)

# Type IDs
FLOOR = 0
WALL = 1
BOX = 2
TARGET = 3
BOX_ON_TARGET = 4
KEY = 5
DOOR_LOCKED = 6
DOOR_OPEN = 7
AGENT = 8

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DELTAS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
DIR_NAMES = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}

# ═══════════════════════════════════════════════════════════════════════
# Levels — defined as string grids
# ═══════════════════════════════════════════════════════════════════════

# Legend:  # wall, . target, $ box, * box-on-target, @ agent, k key, D locked door
#          + agent-on-target, (space) floor

LEVELS = [
    {
        "name": "First Push",
        "grid": [
            "#######",
            "#     #",
            "# .$  #",
            "#  @  #",
            "#     #",
            "#     #",
            "#######",
        ],
        "optimal": 3,
    },
    {
        "name": "Two Boxes",
        "grid": [
            "########",
            "#      #",
            "# .$ . #",
            "#  $   #",
            "#   @  #",
            "#      #",
            "#      #",
            "########",
        ],
        "optimal": 11,
    },
    {
        "name": "Corner Trap",
        "grid": [
            "#########",
            "#       #",
            "# ## .  #",
            "# #  $  #",
            "#    $  #",
            "#  @  . #",
            "#       #",
            "#       #",
            "#########",
        ],
        "optimal": 6,
    },
    {
        "name": "Key & Door",
        "grid": [
            "########",
            "#   D  #",
            "#   #  #",
            "# k #$.#",
            "#   #  #",
            "#  @   #",
            "#      #",
            "########",
        ],
        "optimal": 20,
    },
    {
        "name": "The Gauntlet",
        "grid": [
            "########",
            "#      #",
            "# .  $ #",
            "#   #  #",
            "# $  . #",
            "#   #  #",
            "#  @   #",
            "########",
        ],
        "optimal": 11,
    },
]


# ═══════════════════════════════════════════════════════════════════════
# Game state
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class GameState:
    width: int
    height: int
    grid: np.ndarray          # (H, W) int — cell types
    agent: Tuple[int, int]
    agent_dir: int
    boxes: Set[Tuple[int, int]]
    targets: Set[Tuple[int, int]]
    keys: Set[Tuple[int, int]]
    doors: Dict[Tuple[int, int], bool]  # pos -> locked
    inventory: List[int]
    steps: int
    solved: bool

    def clone(self) -> "GameState":
        return GameState(
            width=self.width,
            height=self.height,
            grid=self.grid.copy(),
            agent=self.agent,
            agent_dir=self.agent_dir,
            boxes=set(self.boxes),
            targets=set(self.targets),
            keys=set(self.keys),
            doors=dict(self.doors),
            inventory=list(self.inventory),
            steps=self.steps,
            solved=self.solved,
        )


def _parse_level(level_data: dict) -> GameState:
    """Parse a string-grid level into a GameState."""
    lines = level_data["grid"]
    h = len(lines)
    w = max(len(line) for line in lines)
    grid = np.full((h, w), FLOOR, dtype=np.int32)
    agent = (1, 1)
    boxes: Set[Tuple[int, int]] = set()
    targets: Set[Tuple[int, int]] = set()
    keys: Set[Tuple[int, int]] = set()
    doors: Dict[Tuple[int, int], bool] = {}

    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch == "#":
                grid[y, x] = WALL
            elif ch == ".":
                grid[y, x] = TARGET
                targets.add((x, y))
            elif ch == "$":
                grid[y, x] = BOX
                boxes.add((x, y))
            elif ch == "*":
                grid[y, x] = BOX_ON_TARGET
                boxes.add((x, y))
                targets.add((x, y))
            elif ch == "@":
                agent = (x, y)
            elif ch == "+":
                agent = (x, y)
                targets.add((x, y))
            elif ch == "k":
                grid[y, x] = KEY
                keys.add((x, y))
            elif ch == "D":
                grid[y, x] = DOOR_LOCKED
                doors[(x, y)] = True

    return GameState(
        width=w, height=h, grid=grid, agent=agent, agent_dir=DOWN,
        boxes=boxes, targets=targets, keys=keys, doors=doors,
        inventory=[], steps=0, solved=False,
    )


def _is_wall(state: GameState, x: int, y: int) -> bool:
    if x < 0 or x >= state.width or y < 0 or y >= state.height:
        return True
    return state.grid[y, x] == WALL


def _is_solid(state: GameState, x: int, y: int) -> bool:
    if _is_wall(state, x, y):
        return True
    if (x, y) in state.boxes:
        return True
    if (x, y) in state.doors and state.doors[(x, y)]:
        return True
    return False


def _box_is_deadlocked(state: GameState, bx: int, by: int) -> bool:
    """Check if a box at (bx, by) is stuck in a corner (not on target)."""
    if (bx, by) in state.targets:
        return False
    u = _is_wall(state, bx, by - 1)
    d = _is_wall(state, bx, by + 1)
    l = _is_wall(state, bx - 1, by)
    r = _is_wall(state, bx + 1, by)
    return (u and l) or (u and r) or (d and l) or (d and r)


def _any_deadlock(state: GameState) -> bool:
    for bx, by in state.boxes:
        if (bx, by) not in state.targets and _box_is_deadlocked(state, bx, by):
            return True
    return False


def _step(state: GameState, action: int) -> Tuple[float, bool]:
    """Apply action to state in-place. Returns (reward, solved)."""
    dx, dy = DELTAS[action]
    state.agent_dir = action
    ax, ay = state.agent
    nx, ny = ax + dx, ay + dy

    # Out of bounds
    if nx < 0 or nx >= state.width or ny < 0 or ny >= state.height:
        return -0.5, False

    # Wall
    if state.grid[ny, nx] == WALL:
        return -0.5, False

    reward = -0.1  # time penalty

    # Locked door
    if (nx, ny) in state.doors and state.doors[(nx, ny)]:
        if state.inventory:
            state.doors[(nx, ny)] = False
            state.grid[ny, nx] = DOOR_OPEN
            state.inventory.pop()
            state.agent = (nx, ny)
            reward += 5.0
        else:
            return -0.5, False

    # Key
    elif (nx, ny) in state.keys:
        state.keys.discard((nx, ny))
        state.grid[ny, nx] = FLOOR
        state.inventory.append(1)
        state.agent = (nx, ny)
        reward += 2.0

    # Box
    elif (nx, ny) in state.boxes:
        bx, by = nx + dx, ny + dy
        if bx < 0 or bx >= state.width or by < 0 or by >= state.height:
            return -0.5, False
        if _is_solid(state, bx, by):
            return -0.5, False

        # Push box
        state.boxes.discard((nx, ny))
        state.boxes.add((bx, by))

        # Update grid
        if (nx, ny) in state.targets:
            state.grid[ny, nx] = TARGET
            reward -= 10.0  # left target
        else:
            state.grid[ny, nx] = FLOOR

        if (bx, by) in state.targets:
            state.grid[by, bx] = BOX_ON_TARGET
            reward += 10.0
        else:
            state.grid[by, bx] = BOX

        state.agent = (nx, ny)

    # Normal floor / target / open door
    else:
        state.agent = (nx, ny)

    state.steps += 1

    # Check solved
    if state.targets and all((bx, by) in state.boxes for bx, by in state.targets):
        state.solved = True
        reward += 100.0

    return reward, state.solved


# ═══════════════════════════════════════════════════════════════════════
# BFS solver (for hints and optimal count)
# ═══════════════════════════════════════════════════════════════════════

def _solver_key(state: GameState) -> Tuple:
    return (state.agent, frozenset(state.boxes), frozenset(state.keys), tuple(sorted(state.doors.items())))


def _solve(state: GameState, max_states: int = 200_000) -> Optional[List[int]]:
    """BFS solver. Returns optimal action list or None."""
    start = state.clone()
    if start.solved:
        return []

    visited: Set = {_solver_key(start)}
    queue: deque = deque()
    queue.append((start, []))

    explored = 0
    while queue and explored < max_states:
        cur, actions = queue.popleft()
        explored += 1

        for action in [UP, DOWN, LEFT, RIGHT]:
            child = cur.clone()
            _reward, solved = _step(child, action)

            if _any_deadlock(child):
                continue

            key = _solver_key(child)
            if key in visited:
                continue
            visited.add(key)

            new_actions = actions + [action]
            if solved:
                return new_actions

            queue.append((child, new_actions))

    return None


# ═══════════════════════════════════════════════════════════════════════
# Animation state
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AnimState:
    agent_from: Optional[Tuple[int, int]] = None
    agent_to: Optional[Tuple[int, int]] = None
    agent_t: float = 1.0
    box_from: Optional[Tuple[int, int]] = None
    box_to: Optional[Tuple[int, int]] = None
    box_t: float = 1.0
    solve_t: float = 0.0
    show_solved: bool = False
    deadlock_flash: float = 0.0
    hint_action: Optional[int] = None


# ═══════════════════════════════════════════════════════════════════════
# Renderer
# ═══════════════════════════════════════════════════════════════════════

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * min(max(t, 0.0), 1.0)


def _ease_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1.0 - (1.0 - t) ** 2


class Renderer:
    def __init__(self, state: GameState) -> None:
        self.cs = CELL
        self.hud_h = 48
        self.inv_h = 36
        w = state.width * self.cs
        h = state.height * self.cs + self.hud_h + self.inv_h
        pygame.init()
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Think Before You Act — Sokoban")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeuisemibold", 16)
        self.font_big = pygame.font.SysFont("segoeuibold", 28)
        self.font_hud = pygame.font.SysFont("segoeuisemibold", 14)
        self.anim = AnimState()
        self.game_time = 0.0

    def resize(self, state: GameState) -> None:
        w = state.width * self.cs
        h = state.height * self.cs + self.hud_h + self.inv_h
        self.screen = pygame.display.set_mode((w, h))

    def draw(self, state: GameState, level_name: str, optimal: int, level_idx: int) -> None:
        cs = self.cs
        dt = self.clock.get_time() / 1000.0
        self.game_time += dt

        # Advance animations
        if self.anim.agent_t < 1.0:
            self.anim.agent_t = min(1.0, self.anim.agent_t + dt / 0.12)
        if self.anim.box_t < 1.0:
            self.anim.box_t = min(1.0, self.anim.box_t + dt / 0.14)
        if self.anim.show_solved:
            self.anim.solve_t = min(1.0, self.anim.solve_t + dt / 0.5)
        if self.anim.deadlock_flash > 0:
            self.anim.deadlock_flash = max(0.0, self.anim.deadlock_flash - dt / 0.8)

        self.screen.fill(BG)

        oy = self.hud_h  # grid vertical offset

        # Draw grid
        for y in range(state.height):
            for x in range(state.width):
                rx, ry = x * cs, y * cs + oy
                tid = state.grid[y, x]

                # Floor
                pygame.draw.rect(self.screen, C_FLOOR, (rx, ry, cs, cs))
                # Grid lines
                pygame.draw.rect(self.screen, GRID_LINE, (rx, ry, cs, cs), 1)

                # Walls with 3D effect
                if tid == WALL:
                    pygame.draw.rect(self.screen, C_WALL, (rx + 2, ry + 2, cs - 4, cs - 4), border_radius=4)
                    pygame.draw.rect(self.screen, C_WALL_HI, (rx + 2, ry + 2, cs - 4, 3))
                    pygame.draw.rect(self.screen, C_WALL_LO, (rx + 2, ry + cs - 5, cs - 4, 3))

                # Target — pulsing circles
                if (x, y) in state.targets and (x, y) not in state.boxes:
                    pulse = 0.8 + 0.2 * math.sin(self.game_time * 3.0)
                    r = int(cs * 0.25 * pulse)
                    cx, cy = rx + cs // 2, ry + cs // 2
                    pygame.draw.circle(self.screen, C_TARGET, (cx, cy), r + 4, 2)
                    pygame.draw.circle(self.screen, C_TARGET, (cx, cy), r, 2)

                # Key — bobbing
                if (x, y) in state.keys:
                    bob = int(3 * math.sin(self.game_time * 2.5))
                    kx, ky = rx + cs // 2, ry + cs // 2 + bob
                    # Key shape: circle head + rectangle shaft
                    pygame.draw.circle(self.screen, C_KEY, (kx, ky - 6), 8)
                    pygame.draw.circle(self.screen, C_FLOOR, (kx, ky - 6), 4)
                    pygame.draw.rect(self.screen, C_KEY, (kx - 2, ky - 2, 4, 14))
                    pygame.draw.rect(self.screen, C_KEY, (kx, ky + 4, 6, 3))
                    pygame.draw.rect(self.screen, C_KEY, (kx, ky + 9, 4, 3))

                # Door
                if (x, y) in state.doors:
                    if state.doors[(x, y)]:
                        pygame.draw.rect(self.screen, C_DOOR_LOCKED, (rx + 4, ry + 4, cs - 8, cs - 8), border_radius=4)
                        # Padlock
                        pygame.draw.circle(self.screen, C_HUD_TEXT, (rx + cs // 2, ry + cs // 2 - 4), 8, 2)
                        pygame.draw.rect(self.screen, C_HUD_TEXT, (rx + cs // 2 - 6, ry + cs // 2, 12, 10), border_radius=2)
                    else:
                        s = pygame.Surface((cs - 8, cs - 8), pygame.SRCALPHA)
                        s.fill((*C_DOOR_OPEN, 100))
                        self.screen.blit(s, (rx + 4, ry + 4))

        # Draw boxes (may be animating)
        for bx, by in state.boxes:
            if self.anim.box_to == (bx, by) and self.anim.box_t < 1.0:
                fx, fy = self.anim.box_from
                t = _ease_out(self.anim.box_t)
                dx = _lerp(fx * cs, bx * cs, t)
                dy = _lerp(fy * cs + oy, by * cs + oy, t)
                # Squash effect
                sx = 1.0 + 0.15 * (1.0 - t)
                sy = 1.0 - 0.1 * (1.0 - t)
            else:
                dx, dy = bx * cs, by * cs + oy
                sx, sy = 1.0, 1.0

            on_target = (bx, by) in state.targets
            color = C_BOX_DONE if on_target else C_BOX

            bw = int(cs * 0.8 * sx)
            bh = int(cs * 0.8 * sy)
            brx = int(dx + (cs - bw) / 2)
            bry = int(dy + (cs - bh) / 2)
            pygame.draw.rect(self.screen, color, (brx, bry, bw, bh), border_radius=6)

            # Symbol
            sym = "\u2713" if on_target else "\u00d7"
            txt = self.font.render(sym, True, BG)
            self.screen.blit(txt, (int(dx) + cs // 2 - txt.get_width() // 2,
                                    int(dy) + cs // 2 - txt.get_height() // 2))

            # Glow outline for on-target
            if on_target:
                glow_alpha = int(60 + 40 * math.sin(self.game_time * 4))
                s = pygame.Surface((bw + 8, bh + 8), pygame.SRCALPHA)
                pygame.draw.rect(s, (*C_BOX_DONE, glow_alpha), (0, 0, bw + 8, bh + 8), 3, border_radius=8)
                self.screen.blit(s, (brx - 4, bry - 4))

            # Deadlock flash
            if self.anim.deadlock_flash > 0 and _box_is_deadlocked(state, bx, by):
                alpha = int(180 * self.anim.deadlock_flash)
                s = pygame.Surface((bw, bh), pygame.SRCALPHA)
                s.fill((*C_DEADLOCK, alpha))
                self.screen.blit(s, (brx, bry))

        # Draw agent (may be animating)
        if self.anim.agent_t < 1.0 and self.anim.agent_from and self.anim.agent_to:
            fx, fy = self.anim.agent_from
            tx, ty = self.anim.agent_to
            t = _ease_out(self.anim.agent_t)
            ax_px = _lerp(fx * cs, tx * cs, t) + cs // 2
            ay_px = _lerp(fy * cs + oy, ty * cs + oy, t) + cs // 2
            # Bounce
            bounce = 3 * math.sin(t * math.pi)
            ay_px -= bounce
        else:
            ax_px = state.agent[0] * cs + cs // 2
            ay_px = state.agent[1] * cs + oy + cs // 2

        ar = int(cs * 0.38)
        pygame.draw.circle(self.screen, C_AGENT, (int(ax_px), int(ay_px)), ar)

        # Eyes
        dx_e, dy_e = DELTAS.get(state.agent_dir, (0, 1))
        eye_off = ar // 3
        for side in (-1, 1):
            if dx_e != 0:
                ex = int(ax_px + dx_e * eye_off)
                ey = int(ay_px + side * eye_off * 0.6)
            else:
                ex = int(ax_px + side * eye_off * 0.6)
                ey = int(ay_px + dy_e * eye_off)
            pygame.draw.circle(self.screen, C_AGENT_EYE, (ex, ey), 4)
            pygame.draw.circle(self.screen, BG, (ex + dx_e * 2, ey + dy_e * 2), 2)

        # Hint arrow
        if self.anim.hint_action is not None:
            hdx, hdy = DELTAS[self.anim.hint_action]
            hx = state.agent[0] + hdx
            hy = state.agent[1] + hdy
            if 0 <= hx < state.width and 0 <= hy < state.height:
                s = pygame.Surface((cs, cs), pygame.SRCALPHA)
                s.fill((*C_HINT, 80))
                self.screen.blit(s, (hx * cs, hy * cs + oy))
                # Arrow
                arrow_txt = {UP: "\u2191", DOWN: "\u2193", LEFT: "\u2190", RIGHT: "\u2192"}
                atxt = self.font_big.render(arrow_txt[self.anim.hint_action], True, (*C_HINT, 200))
                self.screen.blit(atxt, (hx * cs + cs // 2 - atxt.get_width() // 2,
                                         hy * cs + oy + cs // 2 - atxt.get_height() // 2))

        # ── HUD ──
        pygame.draw.rect(self.screen, C_HUD_BG, (0, 0, self.screen.get_width(), self.hud_h))

        # Level name
        ltxt = self.font_hud.render(f"Level {level_idx + 1}: {level_name}", True, C_HUD_TEXT)
        self.screen.blit(ltxt, (10, 6))

        # Steps
        stxt = self.font_hud.render(f"Steps: {state.steps}", True, C_HUD_TEXT)
        self.screen.blit(stxt, (10, 26))

        # Optimal
        otxt = self.font_hud.render(f"Optimal: {optimal}", True, C_DIM)
        self.screen.blit(otxt, (120, 26))

        # Boxes progress
        boxes_on = sum(1 for b in state.boxes if b in state.targets)
        total_t = len(state.targets)
        ptxt = self.font_hud.render(f"Boxes: {boxes_on}/{total_t}", True, C_HUD_TEXT)
        self.screen.blit(ptxt, (self.screen.get_width() - 120, 6))

        # Difficulty stars
        for i in range(5):
            sx = self.screen.get_width() - 120 + i * 18
            color = C_GOLD if i <= level_idx else C_DIM
            pygame.draw.polygon(self.screen, color, _star_points(sx + 8, 34, 7, 3))

        # ── Inventory bar ──
        inv_y = self.screen.get_height() - self.inv_h
        pygame.draw.rect(self.screen, C_HUD_BG, (0, inv_y, self.screen.get_width(), self.inv_h))

        if state.inventory:
            ktxt = self.font_hud.render(f"Keys: {len(state.inventory)}", True, C_KEY)
            self.screen.blit(ktxt, (10, inv_y + 10))

        # Deadlock warning
        if _any_deadlock(state):
            wtxt = self.font_hud.render("DEADLOCK!", True, C_DEADLOCK)
            self.screen.blit(wtxt, (self.screen.get_width() - 100, inv_y + 10))

        # ── Solved overlay ──
        if self.anim.show_solved:
            t = _ease_out(self.anim.solve_t)
            s = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            s.fill((0, 0, 0, int(140 * t)))
            self.screen.blit(s, (0, 0))

            stxt = self.font_big.render("SOLVED!", True, C_BOX_DONE)
            cx = self.screen.get_width() // 2 - stxt.get_width() // 2
            cy = self.screen.get_height() // 2 - 40
            self.screen.blit(stxt, (cx, int(cy - 20 * (1 - t))))

            eff = (optimal / max(state.steps, 1)) * 100
            dtxt = self.font.render(
                f"Steps: {state.steps}  |  Optimal: {optimal}  |  Efficiency: {eff:.0f}%",
                True, C_HUD_TEXT,
            )
            self.screen.blit(dtxt, (self.screen.get_width() // 2 - dtxt.get_width() // 2,
                                     int(cy + 40 * t)))

            ntxt = self.font_hud.render("Press N for next level, R to retry", True, C_DIM)
            self.screen.blit(ntxt, (self.screen.get_width() // 2 - ntxt.get_width() // 2,
                                     int(cy + 70 * t)))

        pygame.display.flip()
        self.clock.tick(FPS)


def _star_points(cx: int, cy: int, r_out: int, r_in: int) -> List[Tuple[int, int]]:
    points = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5
        r = r_out if i % 2 == 0 else r_in
        points.append((int(cx + r * math.cos(angle)), int(cy - r * math.sin(angle))))
    return points


# ═══════════════════════════════════════════════════════════════════════
# Main game loop
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    level_idx = 0
    state = _parse_level(LEVELS[level_idx])
    renderer = Renderer(state)
    undo_stack: List[GameState] = []
    solution_cache: Dict[int, Optional[List[int]]] = {}

    running = True
    while running:
        level = LEVELS[level_idx]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.KEYDOWN:
                key = event.key

                # Quit
                if key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                    break

                # Level select
                if key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    new_idx = key - pygame.K_1
                    if new_idx < len(LEVELS):
                        level_idx = new_idx
                        state = _parse_level(LEVELS[level_idx])
                        renderer.resize(state)
                        renderer.anim = AnimState()
                        undo_stack.clear()
                    continue

                # Restart
                if key == pygame.K_r:
                    state = _parse_level(LEVELS[level_idx])
                    renderer.resize(state)
                    renderer.anim = AnimState()
                    undo_stack.clear()
                    continue

                # Next level
                if key == pygame.K_n:
                    level_idx = (level_idx + 1) % len(LEVELS)
                    state = _parse_level(LEVELS[level_idx])
                    renderer.resize(state)
                    renderer.anim = AnimState()
                    undo_stack.clear()
                    continue

                # Undo
                if key == pygame.K_u and undo_stack:
                    state = undo_stack.pop()
                    renderer.anim = AnimState()
                    continue

                # Hint
                if key == pygame.K_h:
                    if level_idx not in solution_cache:
                        solution_cache[level_idx] = _solve(state)
                    sol = solution_cache.get(level_idx)
                    # Re-solve from current state
                    sol = _solve(state, max_states=100_000)
                    if sol:
                        renderer.anim.hint_action = sol[0]
                    else:
                        renderer.anim.hint_action = None
                    continue

                # Movement
                action = None
                if key in (pygame.K_UP, pygame.K_w):
                    action = UP
                elif key in (pygame.K_DOWN, pygame.K_s):
                    action = DOWN
                elif key in (pygame.K_LEFT, pygame.K_a):
                    action = LEFT
                elif key in (pygame.K_RIGHT, pygame.K_d):
                    action = RIGHT

                if action is not None and not state.solved:
                    renderer.anim.hint_action = None
                    old_agent = state.agent
                    old_boxes = set(state.boxes)
                    undo_stack.append(state.clone())

                    reward, solved = _step(state, action)

                    # Animate agent
                    if state.agent != old_agent:
                        renderer.anim.agent_from = old_agent
                        renderer.anim.agent_to = state.agent
                        renderer.anim.agent_t = 0.0

                    # Animate box push
                    moved_boxes = state.boxes - old_boxes
                    if moved_boxes:
                        new_pos = moved_boxes.pop()
                        # Find which old box moved
                        removed = old_boxes - state.boxes
                        if removed:
                            old_pos = removed.pop()
                            renderer.anim.box_from = old_pos
                            renderer.anim.box_to = new_pos
                            renderer.anim.box_t = 0.0

                    # Deadlock flash
                    if _any_deadlock(state):
                        renderer.anim.deadlock_flash = 1.0

                    # Solved
                    if solved:
                        renderer.anim.show_solved = True
                        renderer.anim.solve_t = 0.0

                    # Invalidate hint cache
                    solution_cache.pop(level_idx, None)

        renderer.draw(state, level["name"], level["optimal"], level_idx)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
