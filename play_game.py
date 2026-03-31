#!/usr/bin/env python3
"""Sokoban -- 60 Levels: a complete Pygame-based Sokoban game.

Uses the project's own env.sokoban and env.level_loader modules.

Controls:
    Arrow keys  Move player
    U           Undo last move
    R           Restart level
    N           Next level
    P           Previous level
    H           Hint (show best move via BFS solver)
    ESC         Back to level select
    Q           Quit
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import pygame

from env.sokoban import SokobanState, solve, DIR_DELTAS
from env.level_loader import LevelLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEVELS_FILE = os.path.join(BASE_DIR, "levels", "classic_60.txt")
PROGRESS_FILE = os.path.join(BASE_DIR, "levels", "progress.json")

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
BG = (22, 22, 35)
WALL_COLOR = (75, 85, 99)
WALL_LIGHT = (95, 105, 119)
WALL_DARK = (55, 65, 79)
FLOOR_LINE = (40, 40, 55)
BOX_COLOR = (251, 191, 36)
BOX_ON_TARGET = (52, 211, 153)
BOX_DEADLOCK = (220, 50, 50)
TARGET_COLOR = (52, 211, 153)
PLAYER_COLOR = (99, 102, 241)
TEXT_COLOR = (230, 230, 240)
DIM_TEXT = (140, 140, 160)
GREEN = (52, 211, 153)
WHITE = (230, 230, 240)
GRAY = (80, 80, 100)
HINT_COLOR = (52, 211, 153)
SOLVED_BANNER_COLOR = (52, 211, 153)
DEADLOCK_COLOR = (220, 50, 50)
MENU_BTN_SOLVED_BG = (30, 70, 50)
MENU_BTN_AVAIL_BG = (35, 35, 55)
MENU_BTN_LOCKED_BG = (30, 30, 42)
MENU_BTN_BORDER = (50, 50, 65)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 960, 720
FPS = 60
ANIM_DURATION = 0.15          # seconds for movement lerp
SOLVED_BANNER_TIME = 2.5      # seconds before auto-advance
DEADLOCK_FLASH_TIME = 1.0
HINT_DISPLAY_TIME = 2.0

# Menu grid
MENU_COLS = 10
MENU_BTN_W, MENU_BTN_H = 70, 55
MENU_GAP_X, MENU_GAP_Y = 10, 10

# Arrow key -> action index (matches DIR_DELTAS order in env.sokoban)
KEY_TO_ACTION = {
    pygame.K_UP: 0,     # (0, -1)
    pygame.K_DOWN: 1,   # (0,  1)
    pygame.K_LEFT: 2,   # (-1, 0)
    pygame.K_RIGHT: 3,  # ( 1, 0)
}

ACTION_ARROWS: Dict[int, str] = {0: "\u2191", 1: "\u2193", 2: "\u2190", 3: "\u2192"}

# ---------------------------------------------------------------------------
# Progress persistence
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    """Return {"solved": set[int], "best": {level_idx: steps}}."""
    default: dict = {"solved": set(), "best": {}}
    if not os.path.isfile(PROGRESS_FILE):
        return default
    try:
        with open(PROGRESS_FILE, "r") as fh:
            raw = json.load(fh)
        default["solved"] = set(raw.get("solved", []))
        default["best"] = {int(k): v for k, v in raw.get("best", {}).items()}
    except Exception:
        pass
    return default


def save_progress(solved: set, best: dict) -> None:
    os.makedirs(os.path.dirname(PROGRESS_FILE) or ".", exist_ok=True)
    with open(PROGRESS_FILE, "w") as fh:
        json.dump({"solved": sorted(solved),
                    "best": {str(k): v for k, v in best.items()}}, fh)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

# ---------------------------------------------------------------------------
# Main game class
# ---------------------------------------------------------------------------

class SokobanGame:
    """Encapsulates all game state, rendering, and input handling."""

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Sokoban \u2014 60 Levels")
        self.clock = pygame.time.Clock()

        # Fonts -- use consolas with fallback
        family = "consolas"
        self.font_title = pygame.font.SysFont(family, 36, bold=True)
        self.font_med = pygame.font.SysFont(family, 22)
        self.font_sm = pygame.font.SysFont(family, 16)
        self.font_grid = pygame.font.SysFont(family, 20, bold=True)
        self.font_hint = pygame.font.SysFont(family, 48, bold=True)
        self.font_banner = pygame.font.SysFont(family, 52, bold=True)

        # Levels
        self.loader = LevelLoader(LEVELS_FILE)
        self.total_levels = self.loader.get_total_levels()

        # Persistent progress
        prog = load_progress()
        self.solved_set: set = prog["solved"]
        self.best_steps: Dict[int, int] = prog["best"]

        # Menu state
        self.selected_index = 0

        # Game-play state (set properly by load_level)
        self.level_index = 0
        self.state: Optional[SokobanState] = None
        self.undo_stack: List[SokobanState] = []
        self.steps = 0
        self.pushes = 0

        # Animation
        self.anim_start = 0.0
        self.anim_from: Tuple[float, float] = (0.0, 0.0)
        self.anim_to: Tuple[float, float] = (0.0, 0.0)
        self.animating = False

        # Deadlock flash
        self.deadlock_time = 0.0
        self.deadlock_active = False

        # Hint
        self.hint_action: Optional[int] = None
        self.hint_time = 0.0
        self.hint_solving = False

        # Solved overlay
        self.solved_time = 0.0
        self.solved_active = False
        self.solved_optimal: Optional[int] = None

        # Scene control
        self.scene = "menu"  # "menu" | "game"
        self.running = True

    # ------------------------------------------------------------------
    # Level management
    # ------------------------------------------------------------------

    def load_level(self, index: int) -> None:
        index = clamp(index, 0, self.total_levels - 1)
        self.level_index = index
        self.state = self.loader.get_level(index)
        self.undo_stack.clear()
        self.steps = 0
        self.pushes = 0
        self.deadlock_active = False
        self.hint_action = None
        self.hint_solving = False
        self.solved_active = False
        self.solved_optimal = None
        self.animating = False
        self.scene = "game"

    def restart_level(self) -> None:
        self.load_level(self.level_index)

    # ------------------------------------------------------------------
    # Movement + undo
    # ------------------------------------------------------------------

    def try_move(self, action: int) -> None:
        if self.state is None or self.animating or self.solved_active:
            return
        old_player = self.state.player
        new_state = self.state.move(action)
        if new_state is None:
            return

        self.undo_stack.append(self.state)
        pushed = (new_state.boxes != self.state.boxes)
        self.state = new_state
        self.steps += 1
        if pushed:
            self.pushes += 1

        self.hint_action = None

        # Start lerp animation
        self.anim_from = (float(old_player[0]), float(old_player[1]))
        self.anim_to = (float(new_state.player[0]), float(new_state.player[1]))
        self.anim_start = time.time()
        self.animating = True

        # Check win
        if new_state.solved:
            self.solved_active = True
            self.solved_time = time.time()
            self.solved_set.add(self.level_index)
            prev = self.best_steps.get(self.level_index)
            if prev is None or self.steps < prev:
                self.best_steps[self.level_index] = self.steps
            save_progress(self.solved_set, self.best_steps)
            self._solve_for_optimal()
        elif new_state.is_deadlocked():
            self.deadlock_active = True
            self.deadlock_time = time.time()

    def undo(self) -> None:
        if not self.undo_stack or self.animating or self.solved_active:
            return
        prev = self.undo_stack.pop()
        # Adjust counters -- we stored the state *before* the move
        pushed = (self.state is not None and prev.boxes != self.state.boxes)
        self.state = prev
        self.steps = max(0, self.steps - 1)
        if pushed:
            self.pushes = max(0, self.pushes - 1)
        self.deadlock_active = False
        self.hint_action = None

    # ------------------------------------------------------------------
    # Hint / optimal (threaded solver calls)
    # ------------------------------------------------------------------

    def request_hint(self) -> None:
        if self.state is None or self.animating or self.solved_active or self.hint_solving:
            return
        self.hint_solving = True
        snapshot = self.state.clone()

        def _worker() -> None:
            sol = solve(snapshot, max_states=300_000)
            if sol and len(sol) > 0:
                self.hint_action = sol[0]
                self.hint_time = time.time()
            self.hint_solving = False

        threading.Thread(target=_worker, daemon=True).start()

    def _solve_for_optimal(self) -> None:
        initial = self.loader.get_level(self.level_index)

        def _worker() -> None:
            sol = solve(initial, max_states=300_000)
            if sol is not None:
                self.solved_optimal = len(sol)

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _cell_size(self, state: SokobanState) -> int:
        margin_top, margin_bot = 50, 40
        avail_w = SCREEN_W - 40
        avail_h = SCREEN_H - margin_top - margin_bot - 10
        cw = avail_w // max(state.width, 1)
        ch = avail_h // max(state.height, 1)
        return min(cw, ch, 64)

    def _grid_origin(self, state: SokobanState, cell: int) -> Tuple[int, int]:
        gw = state.width * cell
        gh = state.height * cell
        ox = (SCREEN_W - gw) // 2
        oy = 50 + (SCREEN_H - 50 - 40 - gh) // 2
        return ox, oy

    # ------------------------------------------------------------------
    # Draw: game screen
    # ------------------------------------------------------------------

    def draw_game(self) -> None:
        state = self.state
        if state is None:
            return

        now = time.time()
        self.screen.fill(BG)

        cell = self._cell_size(state)
        ox, oy = self._grid_origin(state, cell)

        # Finish animation if elapsed
        if self.animating and (now - self.anim_start) >= ANIM_DURATION:
            self.animating = False

        # Expire transient overlays
        if self.deadlock_active and now - self.deadlock_time > DEADLOCK_FLASH_TIME:
            self.deadlock_active = False
        if self.hint_action is not None and now - self.hint_time > HINT_DISPLAY_TIME:
            self.hint_action = None

        # -- Floor grid lines --
        for x in range(state.width + 1):
            px = ox + x * cell
            pygame.draw.line(self.screen, FLOOR_LINE,
                             (px, oy), (px, oy + state.height * cell))
        for y in range(state.height + 1):
            py = oy + y * cell
            pygame.draw.line(self.screen, FLOOR_LINE,
                             (ox, py), (ox + state.width * cell, py))

        # -- Walls with 3-D bevel --
        for wx, wy in state.walls:
            rect = pygame.Rect(ox + wx * cell, oy + wy * cell, cell, cell)
            pygame.draw.rect(self.screen, WALL_COLOR, rect)
            pygame.draw.line(self.screen, WALL_LIGHT,
                             rect.topleft, rect.topright, 2)
            pygame.draw.line(self.screen, WALL_LIGHT,
                             rect.topleft, rect.bottomleft, 2)
            pygame.draw.line(self.screen, WALL_DARK,
                             rect.bottomleft, rect.bottomright, 2)
            pygame.draw.line(self.screen, WALL_DARK,
                             rect.topright, rect.bottomright, 2)

        # -- Targets (pulsing circle) --
        pulse = 0.6 + 0.4 * math.sin(now * 3.0)
        target_r = int(cell * 0.18 * pulse + cell * 0.08)
        for tx, ty in state.targets:
            if (tx, ty) in state.boxes:
                continue  # box covers target visually
            cx = ox + tx * cell + cell // 2
            cy = oy + ty * cell + cell // 2
            pygame.draw.circle(self.screen, TARGET_COLOR, (cx, cy), target_r)
            pygame.draw.circle(self.screen, TARGET_COLOR, (cx, cy),
                               target_r + 4, width=2)

        # -- Boxes --
        for bx, by in state.boxes:
            rect = pygame.Rect(ox + bx * cell + 3, oy + by * cell + 3,
                               cell - 6, cell - 6)
            on_target = (bx, by) in state.targets

            # Pick colour
            if self.deadlock_active and not on_target:
                w_up = (bx, by - 1) in state.walls
                w_dn = (bx, by + 1) in state.walls
                w_lt = (bx - 1, by) in state.walls
                w_rt = (bx + 1, by) in state.walls
                if (w_up or w_dn) and (w_lt or w_rt):
                    color = BOX_DEADLOCK
                else:
                    color = BOX_COLOR
            elif on_target:
                color = BOX_ON_TARGET
            else:
                color = BOX_COLOR

            pygame.draw.rect(self.screen, color, rect, border_radius=6)
            # inner highlight ring
            inner = pygame.Rect(rect.x + 3, rect.y + 3,
                                rect.width - 6, rect.height - 6)
            lighter = tuple(min(255, c + 30) for c in color)
            pygame.draw.rect(self.screen, lighter, inner, width=1,
                             border_radius=4)

        # -- Player (circle with highlight) --
        if self.animating:
            t = (now - self.anim_start) / ANIM_DURATION
            t = max(0.0, min(1.0, t))
            px = lerp(self.anim_from[0], self.anim_to[0], t)
            py = lerp(self.anim_from[1], self.anim_to[1], t)
        else:
            px, py = float(state.player[0]), float(state.player[1])

        pcx = ox + int(px * cell) + cell // 2
        pcy = oy + int(py * cell) + cell // 2
        player_r = cell // 2 - 4
        pygame.draw.circle(self.screen, PLAYER_COLOR, (pcx, pcy), player_r)
        highlight = tuple(min(255, c + 50) for c in PLAYER_COLOR)
        pygame.draw.circle(self.screen, highlight, (pcx, pcy),
                           player_r // 2, width=2)

        # -- Hint arrow overlay --
        if self.hint_action is not None:
            dx, dy = DIR_DELTAS[self.hint_action]
            hx = state.player[0] + dx
            hy = state.player[1] + dy
            arrow_cx = ox + hx * cell + cell // 2
            arrow_cy = oy + hy * cell + cell // 2
            alpha = 0.5 + 0.5 * math.sin(now * 6.0)
            a_color = tuple(int(c * alpha) for c in HINT_COLOR)
            arrow_surf = self.font_hint.render(
                ACTION_ARROWS[self.hint_action], True, a_color)
            ar = arrow_surf.get_rect(center=(arrow_cx, arrow_cy))
            self.screen.blit(arrow_surf, ar)

        # -- HUD top bar --
        best_str = ""
        b = self.best_steps.get(self.level_index)
        if b is not None:
            best_str = f"  |  Best: {b}"
        hud = (f"Level {self.level_index + 1}/{self.total_levels}  |  "
               f"Steps: {self.steps}  |  Pushes: {self.pushes}{best_str}")
        hud_surf = self.font_med.render(hud, True, TEXT_COLOR)
        self.screen.blit(hud_surf, (20, 12))

        # Hint-solving indicator
        if self.hint_solving:
            s_surf = self.font_sm.render("Solving...", True, HINT_COLOR)
            self.screen.blit(s_surf, (SCREEN_W - 130, 14))

        # -- Controls bar bottom --
        controls = ("Arrow=Move  U=Undo  R=Restart  "
                    "N=Next  P=Prev  H=Hint  ESC=Menu")
        ctrl_surf = self.font_sm.render(controls, True, DIM_TEXT)
        self.screen.blit(ctrl_surf,
                         ((SCREEN_W - ctrl_surf.get_width()) // 2,
                          SCREEN_H - 28))

        # -- Deadlock text --
        if self.deadlock_active:
            dl_surf = self.font_banner.render("DEADLOCK!", True, DEADLOCK_COLOR)
            dr = dl_surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2))
            backdrop = pygame.Surface((dr.width + 40, dr.height + 20),
                                      pygame.SRCALPHA)
            backdrop.fill((0, 0, 0, 160))
            self.screen.blit(backdrop, (dr.x - 20, dr.y - 10))
            self.screen.blit(dl_surf, dr)

        # -- Solved overlay --
        if self.solved_active:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            self.screen.blit(overlay, (0, 0))

            sol_surf = self.font_banner.render("SOLVED!", True,
                                               SOLVED_BANNER_COLOR)
            sr = sol_surf.get_rect(center=(SCREEN_W // 2,
                                           SCREEN_H // 2 - 40))
            self.screen.blit(sol_surf, sr)

            info = f"Steps: {self.steps}  |  Pushes: {self.pushes}"
            if self.solved_optimal is not None:
                info += f"  |  Optimal: {self.solved_optimal}"
            info_surf = self.font_med.render(info, True, TEXT_COLOR)
            ir = info_surf.get_rect(center=(SCREEN_W // 2,
                                            SCREEN_H // 2 + 20))
            self.screen.blit(info_surf, ir)

            remaining = SOLVED_BANNER_TIME - (now - self.solved_time)
            if remaining > 0:
                cd = self.font_sm.render(
                    f"Next level in {remaining:.1f}s ...", True, DIM_TEXT)
                cr = cd.get_rect(center=(SCREEN_W // 2,
                                         SCREEN_H // 2 + 60))
                self.screen.blit(cd, cr)
            else:
                nxt = self.level_index + 1
                if nxt < self.total_levels:
                    self.load_level(nxt)
                else:
                    self.scene = "menu"

    # ------------------------------------------------------------------
    # Draw: level-select menu
    # ------------------------------------------------------------------

    def _menu_grid_origin(self) -> Tuple[int, int]:
        rows = (self.total_levels + MENU_COLS - 1) // MENU_COLS
        grid_w = (MENU_COLS * MENU_BTN_W
                  + (MENU_COLS - 1) * MENU_GAP_X)
        grid_ox = (SCREEN_W - grid_w) // 2
        grid_oy = 130
        return grid_ox, grid_oy

    def _max_unlocked(self) -> int:
        if self.solved_set:
            return max(self.solved_set) + 1
        return 0

    def draw_menu(self) -> None:
        self.screen.fill(BG)

        # Title
        title = self.font_title.render(
            "SOKOBAN  \u2014  60 Levels", True, TEXT_COLOR)
        tr = title.get_rect(center=(SCREEN_W // 2, 50))
        self.screen.blit(title, tr)

        # Solved counter
        counter = self.font_med.render(
            f"Solved: {len(self.solved_set)}/{self.total_levels}",
            True, GREEN)
        cr = counter.get_rect(center=(SCREEN_W // 2, 90))
        self.screen.blit(counter, cr)

        grid_ox, grid_oy = self._menu_grid_origin()
        max_unlock = self._max_unlocked()
        mouse_pos = pygame.mouse.get_pos()

        for i in range(self.total_levels):
            col = i % MENU_COLS
            row = i // MENU_COLS
            x = grid_ox + col * (MENU_BTN_W + MENU_GAP_X)
            y = grid_oy + row * (MENU_BTN_H + MENU_GAP_Y)
            rect = pygame.Rect(x, y, MENU_BTN_W, MENU_BTN_H)

            is_solved = i in self.solved_set
            is_available = i <= max_unlock or is_solved
            is_selected = i == self.selected_index
            is_hovered = rect.collidepoint(mouse_pos) and is_available

            if is_solved:
                bg_color = MENU_BTN_SOLVED_BG
                border = GREEN
                txt_color = GREEN
            elif is_available:
                bg_color = MENU_BTN_AVAIL_BG
                border = WHITE
                txt_color = WHITE
            else:
                bg_color = MENU_BTN_LOCKED_BG
                border = GRAY
                txt_color = GRAY

            pygame.draw.rect(self.screen, bg_color, rect, border_radius=6)

            if is_selected or is_hovered:
                pygame.draw.rect(self.screen, border, rect,
                                 width=2, border_radius=6)
            else:
                pygame.draw.rect(self.screen, MENU_BTN_BORDER, rect,
                                 width=1, border_radius=6)

            # Best steps indicator for solved levels
            if is_solved and i in self.best_steps:
                tiny = self.font_sm.render(
                    str(self.best_steps[i]), True,
                    tuple(max(0, c - 40) for c in GREEN))
                self.screen.blit(tiny, (rect.right - tiny.get_width() - 4,
                                        rect.bottom - tiny.get_height() - 2))

            num_surf = self.font_grid.render(str(i + 1), True, txt_color)
            nr = num_surf.get_rect(center=(rect.centerx, rect.centery - 2))
            self.screen.blit(num_surf, nr)

        # Instructions
        instr = ("Arrow keys / Mouse to select  |  "
                 "Enter / Click to play  |  Q to quit")
        instr_surf = self.font_sm.render(instr, True, DIM_TEXT)
        self.screen.blit(instr_surf,
                         ((SCREEN_W - instr_surf.get_width()) // 2,
                          SCREEN_H - 28))

    # ------------------------------------------------------------------
    # Menu hit-test
    # ------------------------------------------------------------------

    def _menu_hit_test(self, pos: Tuple[int, int]) -> Optional[int]:
        grid_ox, grid_oy = self._menu_grid_origin()
        mx, my = pos
        for i in range(self.total_levels):
            col = i % MENU_COLS
            row = i // MENU_COLS
            x = grid_ox + col * (MENU_BTN_W + MENU_GAP_X)
            y = grid_oy + row * (MENU_BTN_H + MENU_GAP_Y)
            if x <= mx <= x + MENU_BTN_W and y <= my <= y + MENU_BTN_H:
                return i
        return None

    def _try_enter_level(self, idx: int) -> None:
        max_unlock = self._max_unlocked()
        if idx <= max_unlock or idx in self.solved_set:
            self.load_level(idx)

    # ------------------------------------------------------------------
    # Event handling: menu
    # ------------------------------------------------------------------

    def handle_menu_events(self, events: list) -> None:
        for ev in events:
            if ev.type == pygame.QUIT:
                self.running = False
                return

            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_q:
                    self.running = False
                    return
                elif ev.key == pygame.K_RIGHT:
                    self.selected_index = min(self.selected_index + 1,
                                              self.total_levels - 1)
                elif ev.key == pygame.K_LEFT:
                    self.selected_index = max(self.selected_index - 1, 0)
                elif ev.key == pygame.K_DOWN:
                    nxt = self.selected_index + MENU_COLS
                    if nxt < self.total_levels:
                        self.selected_index = nxt
                elif ev.key == pygame.K_UP:
                    nxt = self.selected_index - MENU_COLS
                    if nxt >= 0:
                        self.selected_index = nxt
                elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    self._try_enter_level(self.selected_index)

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                idx = self._menu_hit_test(ev.pos)
                if idx is not None:
                    self.selected_index = idx
                    self._try_enter_level(idx)

    # ------------------------------------------------------------------
    # Event handling: game
    # ------------------------------------------------------------------

    def handle_game_events(self, events: list) -> None:
        for ev in events:
            if ev.type == pygame.QUIT:
                self.running = False
                return

            if ev.type != pygame.KEYDOWN:
                continue

            if ev.key == pygame.K_q:
                self.running = False
                return
            elif ev.key == pygame.K_ESCAPE:
                self.scene = "menu"
                return
            elif ev.key in KEY_TO_ACTION:
                self.try_move(KEY_TO_ACTION[ev.key])
            elif ev.key == pygame.K_u:
                self.undo()
            elif ev.key == pygame.K_r:
                self.restart_level()
            elif ev.key == pygame.K_n:
                nxt = self.level_index + 1
                if nxt < self.total_levels:
                    self.load_level(nxt)
            elif ev.key == pygame.K_p:
                prv = self.level_index - 1
                if prv >= 0:
                    self.load_level(prv)
            elif ev.key == pygame.K_h:
                self.request_hint()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        while self.running:
            events = pygame.event.get()

            if self.scene == "menu":
                self.handle_menu_events(events)
                if self.running:
                    self.draw_menu()
            elif self.scene == "game":
                self.handle_game_events(events)
                if self.running:
                    self.draw_game()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    game = SokobanGame()
    game.run()


if __name__ == "__main__":
    main()
