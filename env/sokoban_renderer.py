"""Classic Sokoban renderer using pygame.

Clean, retro-style rendering directly from SokobanState.
No PuzzleWorld dependency.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pygame

from env.sokoban import SokobanState


# Classic Sokoban color palette
COLORS = {
    "bg": (35, 35, 50),
    "wall": (90, 75, 60),
    "wall_top": (120, 100, 80),
    "wall_shadow": (60, 50, 40),
    "floor": (55, 55, 70),
    "floor_line": (45, 45, 60),
    "box": (200, 150, 50),
    "box_dark": (160, 120, 40),
    "box_x": (180, 135, 45),
    "box_done": (80, 180, 80),
    "box_done_dark": (60, 140, 60),
    "target": (200, 70, 70),
    "target_glow": (220, 90, 90),
    "player": (60, 120, 220),
    "player_dark": (40, 90, 180),
    "player_eye": (255, 255, 255),
    "player_pupil": (20, 20, 40),
    "text": (220, 220, 230),
    "text_dim": (130, 130, 150),
    "header_bg": (25, 25, 38),
    "solved": (80, 220, 120),
    "path": (80, 200, 140),
    "danger": (200, 50, 50),
}

CELL = 48
HEADER_H = 60
FOOTER_H = 36


class SokobanRenderer:
    """Renders SokobanState with classic Sokoban aesthetics."""

    def __init__(self, cell_size: int = CELL) -> None:
        self.cell = cell_size
        self._screen: Optional[pygame.Surface] = None
        self._font: Optional[pygame.font.Font] = None
        self._font_big: Optional[pygame.font.Font] = None
        self._font_small: Optional[pygame.font.Font] = None
        self._clock = pygame.time.Clock()

    def _ensure_screen(self, state: SokobanState) -> pygame.Surface:
        """Create or resize screen to fit the level."""
        w = state.width * self.cell + 40
        h = state.height * self.cell + HEADER_H + FOOTER_H + 20
        w = max(w, 460)

        if self._screen is None or self._screen.get_size() != (w, h):
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Sokoban")

        if self._font is None:
            self._font = pygame.font.SysFont("consolas", 18)
            self._font_big = pygame.font.SysFont("consolas", 24, bold=True)
            self._font_small = pygame.font.SysFont("consolas", 13)

        return self._screen

    def render(
        self,
        state: SokobanState,
        level_num: int = 0,
        total_levels: int = 0,
        moves: int = 0,
        solved: bool = False,
        ai_mode: bool = False,
        step_num: int = 0,
        max_steps: int = 0,
        planned_path: Optional[List[Tuple[int, int]]] = None,
        danger_zones: Optional[List[Tuple[int, int]]] = None,
        info_text: str = "",
    ) -> None:
        """Draw everything."""
        screen = self._ensure_screen(state)
        screen.fill(COLORS["bg"])

        sw, sh = screen.get_size()
        grid_w = state.width * self.cell
        grid_h = state.height * self.cell
        ox = (sw - grid_w) // 2
        oy = HEADER_H + 5

        # Header
        pygame.draw.rect(screen, COLORS["header_bg"], (0, 0, sw, HEADER_H))

        if total_levels > 0:
            title = self._font_big.render(
                f"Level {level_num}/{total_levels}", True, COLORS["text"])
        else:
            title = self._font_big.render(f"Level {level_num}", True, COLORS["text"])
        screen.blit(title, (15, 8))

        boxes_on = state.n_boxes_on_target
        total_boxes = len(state.boxes)

        if ai_mode and max_steps > 0:
            info = self._font.render(
                f"Step {step_num}/{max_steps}  {boxes_on}/{total_boxes} boxes",
                True, COLORS["text"])
        else:
            info = self._font.render(
                f"Moves: {moves}  {boxes_on}/{total_boxes} boxes",
                True, COLORS["text"])
        screen.blit(info, (15, 35))

        if solved:
            msg = self._font_big.render("SOLVED!", True, COLORS["solved"])
            screen.blit(msg, (sw - msg.get_width() - 15, 15))

        # Danger zones
        if danger_zones:
            for (x, y) in danger_zones:
                rx, ry = ox + x * self.cell, oy + y * self.cell
                s = pygame.Surface((self.cell, self.cell), pygame.SRCALPHA)
                s.fill((200, 50, 50, 40))
                screen.blit(s, (rx, ry))

        # Grid
        self._draw_grid(screen, state, ox, oy)

        # Planned path overlay
        if planned_path and len(planned_path) > 1:
            points = [(ox + x * self.cell + self.cell // 2,
                        oy + y * self.cell + self.cell // 2)
                       for x, y in planned_path]
            pygame.draw.lines(screen, COLORS["path"], False, points, 3)

        # Footer
        fy = sh - FOOTER_H + 8
        if info_text:
            ft = self._font_small.render(info_text, True, COLORS["text_dim"])
        elif ai_mode:
            ft = self._font_small.render(
                "Space: Pause  S/F: Speed  N/P: Level  ESC: Quit",
                True, COLORS["text_dim"])
        else:
            ft = self._font_small.render(
                "Arrows: Move  R: Reset  U: Undo  N/P: Level  ESC: Quit",
                True, COLORS["text_dim"])
        screen.blit(ft, (15, fy))

        pygame.display.flip()
        self._clock.tick(60)

    def _draw_grid(self, screen: pygame.Surface, state: SokobanState,
                   ox: int, oy: int) -> None:
        """Draw the Sokoban grid with classic styling."""
        c = self.cell

        for y in range(state.height):
            for x in range(state.width):
                rx, ry = ox + x * c, oy + y * c
                pos = (x, y)

                if pos in state.walls:
                    # Wall: 3D brick look
                    pygame.draw.rect(screen, COLORS["wall_shadow"],
                                     (rx, ry, c, c))
                    pygame.draw.rect(screen, COLORS["wall"],
                                     (rx, ry, c - 2, c - 2))
                    pygame.draw.rect(screen, COLORS["wall_top"],
                                     (rx + 2, ry + 2, c - 6, c - 6))
                else:
                    # Floor
                    pygame.draw.rect(screen, COLORS["floor"],
                                     (rx, ry, c, c))
                    pygame.draw.rect(screen, COLORS["floor_line"],
                                     (rx, ry, c, c), 1)

                # Target diamond
                if pos in state.targets and pos not in state.boxes:
                    cx, cy = rx + c // 2, ry + c // 2
                    r = c // 5
                    pts = [(cx, cy - r), (cx + r, cy),
                           (cx, cy + r), (cx - r, cy)]
                    pygame.draw.polygon(screen, COLORS["target"], pts)
                    pygame.draw.polygon(screen, COLORS["target_glow"], pts, 2)

                # Box
                if pos in state.boxes:
                    m = 5
                    on_target = pos in state.targets
                    base = COLORS["box_done"] if on_target else COLORS["box"]
                    dark = COLORS["box_done_dark"] if on_target else COLORS["box_dark"]

                    # Shadow
                    pygame.draw.rect(screen, dark,
                                     (rx + m + 2, ry + m + 2, c - 2 * m, c - 2 * m),
                                     border_radius=3)
                    # Box body
                    pygame.draw.rect(screen, base,
                                     (rx + m, ry + m, c - 2 * m, c - 2 * m),
                                     border_radius=3)
                    # Cross mark on box
                    cx, cy = rx + c // 2, ry + c // 2
                    cr = c // 5
                    pygame.draw.line(screen, dark,
                                     (cx - cr, cy - cr), (cx + cr, cy + cr), 2)
                    pygame.draw.line(screen, dark,
                                     (cx - cr, cy + cr), (cx + cr, cy - cr), 2)

                # Player
                if pos == state.player:
                    cx, cy = rx + c // 2, ry + c // 2
                    r = c // 3
                    # Body
                    pygame.draw.circle(screen, COLORS["player_dark"],
                                       (cx + 1, cy + 1), r)
                    pygame.draw.circle(screen, COLORS["player"],
                                       (cx, cy), r)
                    # Eyes
                    pygame.draw.circle(screen, COLORS["player_eye"],
                                       (cx - r // 3, cy - r // 4), r // 4)
                    pygame.draw.circle(screen, COLORS["player_eye"],
                                       (cx + r // 3, cy - r // 4), r // 4)
                    pygame.draw.circle(screen, COLORS["player_pupil"],
                                       (cx - r // 3, cy - r // 4), r // 7)
                    pygame.draw.circle(screen, COLORS["player_pupil"],
                                       (cx + r // 3, cy - r // 4), r // 7)

    def close(self) -> None:
        if self._screen:
            pygame.display.quit()
            self._screen = None
