"""Premium visual renderer for the puzzle environment.

Renders :class:`PuzzleWorld` state using PyGame with smooth animations,
rich colours, drop shadows, particle effects, and a polished HUD.

Run standalone for interactive human play::

    python -m env.renderer
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pygame

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
from env.puzzle_world import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    PuzzleWorld,
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

COL_BG = (22, 22, 35)
COL_GRID = (40, 40, 55)
COL_AGENT = (99, 102, 241)
COL_BOX = (251, 191, 36)
COL_BOX_DONE = (52, 211, 153)
COL_TARGET = (52, 211, 153)
COL_WALL = (75, 85, 99)
COL_WALL_HI = (90, 100, 114)
COL_WALL_LO = (55, 65, 79)
COL_KEY = (251, 146, 60)
COL_DOOR_LOCKED = (239, 68, 68)
COL_DOOR_UNLOCKED = (74, 222, 128)
COL_ICE = (147, 197, 253)
COL_SW_INACTIVE = (168, 85, 247)
COL_SW_ACTIVE = (236, 72, 153)
COL_SWITCH_WALL = (168, 85, 247)
COL_ONEWAY = (56, 189, 248)
COL_WHITE = (255, 255, 255)
COL_HUD_BG = (15, 15, 25, 200)
COL_HUD_TEXT = (220, 220, 235)
COL_SHADOW = (0, 0, 0, 40)
COL_GOLD_STAR = (251, 191, 36)
COL_EMPTY_STAR = (60, 60, 80)
COL_CONFETTI = [
    (251, 191, 36), (52, 211, 153), (99, 102, 241),
    (236, 72, 153), (251, 146, 60), (56, 189, 248),
]

HUD_HEIGHT = 44
INV_BAR_HEIGHT = 36

# Direction helpers
DIR_OFFSETS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * min(max(t, 0.0), 1.0)


def _ease_out_quad(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1.0 - (1.0 - t) * (1.0 - t)


def _ease_out_back(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * pow(t - 1.0, 3) + c1 * pow(t - 1.0, 2)


def _surf_with_alpha(w: int, h: int) -> pygame.Surface:
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    return s


# ---------------------------------------------------------------------------
# Particle system
# ---------------------------------------------------------------------------

@dataclass
class _Particle:
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    life: float  # seconds remaining
    max_life: float
    size: float = 4.0
    gravity: float = 0.0


# ---------------------------------------------------------------------------
# Animation state helpers
# ---------------------------------------------------------------------------

@dataclass
class _AnimState:
    """Tracks per-frame animation state for smooth transitions."""
    # Agent lerp
    agent_visual_x: float = 0.0
    agent_visual_y: float = 0.0
    agent_prev_x: float = 0.0
    agent_prev_y: float = 0.0
    agent_move_t: float = 1.0  # 0..1, 1 = done
    agent_bounce_t: float = 1.0

    # Agent trail (fading afterimage)
    trail_x: float = -1.0
    trail_y: float = -1.0
    trail_alpha: float = 0.0

    # Box animations: key = (gx, gy) of box, value = anim progress
    box_push_anims: Dict[Tuple[int, int], float] = field(default_factory=dict)
    box_push_dirs: Dict[Tuple[int, int], Tuple[int, int]] = field(default_factory=dict)
    box_prev_pos: Dict[Tuple[int, int], Tuple[float, float]] = field(default_factory=dict)
    box_move_t: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # Target pulse
    target_pulse_phase: float = 0.0

    # Key float
    key_bob_phase: float = 0.0

    # Ice sparkle
    ice_sparkle_timer: float = 0.0
    ice_sparkles: List[Tuple[int, int, float]] = field(default_factory=list)

    # Switch wall alpha: key = (x, y)
    sw_wall_alpha: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # Solved overlay
    solved_t: float = 0.0
    solved_shown: bool = False

    # Particles
    particles: List[_Particle] = field(default_factory=list)

    # Confetti
    confetti: List[_Particle] = field(default_factory=list)

    # Door flash
    door_flash: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # Arrow pulse
    arrow_pulse_phase: float = 0.0

    # Switch depress
    switch_depress: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # Key collection fly animation: list of (start_x, start_y, progress 0..1)
    key_fly_anims: List[Tuple[float, float, float, int]] = field(default_factory=list)
    # (world_x, world_y, progress, key_id)


# ---------------------------------------------------------------------------
# PuzzleRenderer
# ---------------------------------------------------------------------------

class PuzzleRenderer:
    """PyGame renderer with smooth animations and rich visuals."""

    def __init__(self, cell_size: int = 64) -> None:
        self.cell_size = cell_size
        self._cs = cell_size
        self._initialised = False
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._anim = _AnimState()
        self._fonts: Dict[str, pygame.font.Font] = {}
        self._zoom = 1.0
        self._cam_x = 0.0
        self._cam_y = 0.0
        self._prev_agent_pos: Optional[Tuple[int, int]] = None
        self._prev_box_positions: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._prev_solved = False
        self._prev_door_locked: Dict[Tuple[int, int], bool] = {}
        self._prev_switch_active: Dict[Tuple[int, int], bool] = {}
        self._prev_key_positions: set = set()
        self._prev_sw_wall_open: Dict[Tuple[int, int], bool] = {}
        self._grid_w = 0
        self._grid_h = 0
        self._speed_mult = 1.0
        self._paused = False
        self._show_thought = False
        self._last_time = 0.0

    # ------------------------------------------------------------------
    # Init / resize
    # ------------------------------------------------------------------

    def _ensure_init(self, world: PuzzleWorld) -> None:
        gw, gh = world.width, world.height
        if self._initialised and gw == self._grid_w and gh == self._grid_h:
            return

        if not pygame.get_init():
            pygame.init()

        self._grid_w = gw
        self._grid_h = gh
        win_w = gw * self._cs
        win_h = gh * self._cs + HUD_HEIGHT + INV_BAR_HEIGHT
        self._screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
        pygame.display.set_caption("World-Model RL  |  Puzzle Renderer")
        self._clock = pygame.time.Clock()

        self._fonts["hud"] = pygame.font.SysFont("consolas,dejavusansmono,monospace", 16)
        self._fonts["hud_big"] = pygame.font.SysFont("consolas,dejavusansmono,monospace", 20, bold=True)
        self._fonts["solved"] = pygame.font.SysFont("consolas,dejavusansmono,monospace", 40, bold=True)
        self._fonts["solved_sub"] = pygame.font.SysFont("consolas,dejavusansmono,monospace", 20)
        self._fonts["icon"] = pygame.font.SysFont("consolas,dejavusansmono,monospace", 24, bold=True)
        self._fonts["small"] = pygame.font.SysFont("consolas,dejavusansmono,monospace", 12)

        self._initialised = True
        self._last_time = time.monotonic()

        # Seed anim state from world
        ax, ay = world.agent_pos
        self._anim.agent_visual_x = float(ax)
        self._anim.agent_visual_y = float(ay)
        self._anim.agent_prev_x = float(ax)
        self._anim.agent_prev_y = float(ay)
        self._prev_agent_pos = (ax, ay)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def render(
        self,
        world: PuzzleWorld,
        extra_overlay: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Render one frame of *world* to the PyGame window."""
        self._ensure_init(world)
        assert self._screen is not None and self._clock is not None

        now = time.monotonic()
        raw_dt = now - self._last_time
        self._last_time = now
        dt = raw_dt * self._speed_mult
        if self._paused:
            dt = 0.0

        self._detect_changes(world, dt)
        self._tick_animations(dt)
        self._tick_particles(raw_dt)  # particles always tick

        scr = self._screen
        scr.fill(COL_BG)

        cs = self._cs
        ox, oy = self._camera_offset(world)

        # Grid lines
        self._draw_grid_lines(scr, world, ox, oy)

        # Tiles layer (floors, ice, targets, switches, oneway, switch walls)
        self._draw_tiles(scr, world, ox, oy, dt)

        # Walls
        self._draw_walls(scr, world, ox, oy)

        # Objects layer (keys, doors, boxes)
        self._draw_objects(scr, world, ox, oy, dt)

        # Agent
        self._draw_agent(scr, world, ox, oy, dt)

        # Overlays (thought paths, danger zones)
        if extra_overlay:
            self._draw_overlays(scr, world, ox, oy, extra_overlay)

        # Particles
        self._draw_particles(scr, ox, oy)

        # Key collection fly animations
        self._draw_key_fly(scr, ox, oy)

        # HUD
        self._draw_hud(scr, world)
        self._draw_inventory_bar(scr, world)

        # Mini-map for large grids
        if world.width > 10 or world.height > 10:
            self._draw_minimap(scr, world)

        # Solved overlay
        if world.solved:
            self._draw_solved_overlay(scr, world, raw_dt)

        pygame.display.flip()
        self._clock.tick(60)

    def close(self) -> None:
        if self._initialised:
            pygame.quit()
            self._initialised = False

    # ------------------------------------------------------------------
    # Change detection (triggers animations)
    # ------------------------------------------------------------------

    def _detect_changes(self, world: PuzzleWorld, dt: float) -> None:
        ax, ay = world.agent_pos

        # Agent moved?
        if self._prev_agent_pos is not None and (ax, ay) != self._prev_agent_pos:
            px, py = self._prev_agent_pos
            self._anim.agent_prev_x = self._anim.agent_visual_x
            self._anim.agent_prev_y = self._anim.agent_visual_y
            self._anim.agent_move_t = 0.0
            self._anim.agent_bounce_t = 0.0
            self._anim.trail_x = float(px)
            self._anim.trail_y = float(py)
            self._anim.trail_alpha = 1.0

        self._prev_agent_pos = (ax, ay)

        # Box positions changed?
        current_boxes: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for y in range(world.height):
            for x in range(world.width):
                for obj in world.get_cell(x, y):
                    if isinstance(obj, Box):
                        current_boxes[(x, y)] = (x, y)

        # Detect box that appeared at new location
        for pos in current_boxes:
            if pos not in self._prev_box_positions:
                # Find where it came from
                for old_pos in self._prev_box_positions:
                    if old_pos not in current_boxes:
                        self._anim.box_prev_pos[pos] = (float(old_pos[0]), float(old_pos[1]))
                        self._anim.box_move_t[pos] = 0.0
                        dx = pos[0] - old_pos[0]
                        dy = pos[1] - old_pos[1]
                        self._anim.box_push_dirs[pos] = (dx, dy)
                        self._anim.box_push_anims[pos] = 0.0

                        # Check if box landed on target — spawn particles
                        for obj in world.get_cell(pos[0], pos[1]):
                            if isinstance(obj, Box) and obj.on_target:
                                self._spawn_target_burst(pos[0], pos[1])
                        break

        self._prev_box_positions = current_boxes

        # Door unlock detection
        current_doors: Dict[Tuple[int, int], bool] = {}
        for y in range(world.height):
            for x in range(world.width):
                for obj in world.get_cell(x, y):
                    if isinstance(obj, Door):
                        current_doors[(x, y)] = obj.locked
        for pos, locked in current_doors.items():
            prev_locked = self._prev_door_locked.get(pos)
            if prev_locked is True and not locked:
                # Door was just unlocked — trigger flash
                self._anim.door_flash[pos] = 0.0
        self._prev_door_locked = current_doors

        # Switch activation detection
        current_switches: Dict[Tuple[int, int], bool] = {}
        for y in range(world.height):
            for x in range(world.width):
                for obj in world.get_cell(x, y):
                    if isinstance(obj, PressureSwitch):
                        current_switches[(x, y)] = obj.activated
        for pos, active in current_switches.items():
            prev_active = self._prev_switch_active.get(pos)
            if prev_active is not None and active != prev_active:
                # Switch state changed — trigger depress animation
                self._anim.switch_depress[pos] = 0.0
        self._prev_switch_active = current_switches

        # Switch wall open/close fade detection
        current_sw_walls: Dict[Tuple[int, int], bool] = {}
        for y in range(world.height):
            for x in range(world.width):
                for obj in world.get_cell(x, y):
                    if isinstance(obj, SwitchWall):
                        current_sw_walls[(x, y)] = obj.open
        for pos, is_open in current_sw_walls.items():
            prev_open = self._prev_sw_wall_open.get(pos)
            if prev_open is not None and is_open != prev_open:
                # Start fade animation (0.0 = just changed, ticks up to 1.0)
                self._anim.sw_wall_alpha[pos] = 0.0
        self._prev_sw_wall_open = current_sw_walls

        # Key collection detection (fly to inventory)
        current_key_pos: set = set()
        for y in range(world.height):
            for x in range(world.width):
                for obj in world.get_cell(x, y):
                    if isinstance(obj, Key):
                        current_key_pos.add((x, y))
        for pos in self._prev_key_positions:
            if pos not in current_key_pos:
                # Key was collected — spawn fly animation
                # Try to find the key_id from inventory (most recently added)
                kid = world.inventory[-1].key_id if world.inventory else 0
                self._anim.key_fly_anims.append((float(pos[0]), float(pos[1]), 0.0, kid))
        self._prev_key_positions = current_key_pos

        # Solved?
        if world.solved and not self._prev_solved:
            self._anim.solved_t = 0.0
            self._anim.solved_shown = True
            self._spawn_confetti()
        self._prev_solved = world.solved

    # ------------------------------------------------------------------
    # Tick animations
    # ------------------------------------------------------------------

    def _tick_animations(self, dt: float) -> None:
        a = self._anim
        speed = 8.0  # lerp speed

        # Agent movement lerp
        if a.agent_move_t < 1.0:
            a.agent_move_t = min(1.0, a.agent_move_t + dt * speed)
        t = _ease_out_quad(a.agent_move_t)
        if self._prev_agent_pos:
            ax, ay = self._prev_agent_pos
            a.agent_visual_x = _lerp(a.agent_prev_x, float(ax), t)
            a.agent_visual_y = _lerp(a.agent_prev_y, float(ay), t)

        # Agent bounce
        if a.agent_bounce_t < 1.0:
            a.agent_bounce_t = min(1.0, a.agent_bounce_t + dt * 10.0)

        # Trail fade
        if a.trail_alpha > 0:
            a.trail_alpha = max(0.0, a.trail_alpha - dt * 5.0)

        # Box lerps
        done_boxes = []
        for pos, mt in list(a.box_move_t.items()):
            if mt < 1.0:
                a.box_move_t[pos] = min(1.0, mt + dt * speed * 0.85)
            else:
                done_boxes.append(pos)
        for pos in done_boxes:
            a.box_move_t.pop(pos, None)
            a.box_prev_pos.pop(pos, None)

        # Box push squash/stretch
        done_push = []
        for pos, pt in list(a.box_push_anims.items()):
            if pt < 1.0:
                a.box_push_anims[pos] = min(1.0, pt + dt * 7.0)
            else:
                done_push.append(pos)
        for pos in done_push:
            a.box_push_anims.pop(pos, None)
            a.box_push_dirs.pop(pos, None)

        # Target pulse
        a.target_pulse_phase += dt * 0.5
        if a.target_pulse_phase > 1.0:
            a.target_pulse_phase -= 1.0

        # Key bob
        a.key_bob_phase += dt * 0.67

        # Ice sparkles
        a.ice_sparkle_timer += dt
        if a.ice_sparkle_timer > 0.15:
            a.ice_sparkle_timer = 0.0
            # refresh sparkles
            a.ice_sparkles = [
                (random.randint(0, self._cs - 1),
                 random.randint(0, self._cs - 1),
                 random.random())
                for _ in range(6)
            ]

        # Arrow pulse
        a.arrow_pulse_phase += dt * 0.8
        if a.arrow_pulse_phase > 1.0:
            a.arrow_pulse_phase -= 1.0

        # Door flash
        done_flash = []
        for pos, ft in list(a.door_flash.items()):
            if ft < 1.0:
                a.door_flash[pos] = min(1.0, ft + dt * 3.3)
            else:
                done_flash.append(pos)
        for pos in done_flash:
            a.door_flash.pop(pos, None)

        # Switch depress
        done_sw = []
        for pos, st in list(a.switch_depress.items()):
            if st < 1.0:
                a.switch_depress[pos] = min(1.0, st + dt * 10.0)
            else:
                done_sw.append(pos)
        for pos in done_sw:
            a.switch_depress.pop(pos, None)

        # Switch wall alpha fade
        done_swa = []
        for pos, at in list(a.sw_wall_alpha.items()):
            if at < 1.0:
                a.sw_wall_alpha[pos] = min(1.0, at + dt * 3.3)
            else:
                done_swa.append(pos)
        for pos in done_swa:
            a.sw_wall_alpha.pop(pos, None)

        # Key collection fly
        updated_flies: List[Tuple[float, float, float, int]] = []
        for wx, wy, prog, kid in a.key_fly_anims:
            prog += dt * 3.0
            if prog < 1.0:
                updated_flies.append((wx, wy, prog, kid))
        a.key_fly_anims = updated_flies

        # Solved timer
        if a.solved_shown:
            a.solved_t = min(5.0, a.solved_t + dt)

    # ------------------------------------------------------------------
    # Particles
    # ------------------------------------------------------------------

    def _tick_particles(self, dt: float) -> None:
        alive: List[_Particle] = []
        for p in self._anim.particles:
            p.life -= dt
            if p.life > 0:
                p.x += p.vx * dt
                p.y += p.vy * dt
                p.vy += p.gravity * dt
                alive.append(p)
        self._anim.particles = alive

        alive_c: List[_Particle] = []
        for p in self._anim.confetti:
            p.life -= dt
            if p.life > 0:
                p.x += p.vx * dt
                p.y += p.vy * dt
                p.vy += p.gravity * dt
                alive_c.append(p)
        self._anim.confetti = alive_c

    def _spawn_target_burst(self, gx: int, gy: int) -> None:
        cs = self._cs
        cx = (gx + 0.5) * cs
        cy = (gy + 0.5) * cs
        for i in range(8):
            angle = (i / 8.0) * math.pi * 2
            speed = random.uniform(80, 160)
            self._anim.particles.append(_Particle(
                x=cx, y=cy,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                color=COL_TARGET,
                life=0.6, max_life=0.6,
                size=random.uniform(3, 6),
                gravity=120.0,
            ))

    def _spawn_confetti(self) -> None:
        assert self._screen is not None
        w = self._screen.get_width()
        for _ in range(30):
            self._anim.confetti.append(_Particle(
                x=random.uniform(0, w),
                y=random.uniform(-40, -10),
                vx=random.uniform(-30, 30),
                vy=random.uniform(60, 180),
                color=random.choice(COL_CONFETTI),
                life=3.0, max_life=3.0,
                size=random.uniform(4, 8),
                gravity=50.0,
            ))

    def _draw_particles(self, scr: pygame.Surface, ox: float, oy: float) -> None:
        yo = HUD_HEIGHT
        for p in self._anim.particles:
            alpha = int(255 * (p.life / p.max_life))
            s = _surf_with_alpha(int(p.size * 2), int(p.size * 2))
            col = (*p.color, alpha)
            pygame.draw.circle(s, col, (int(p.size), int(p.size)), int(p.size))
            scr.blit(s, (int(p.x + ox - p.size), int(p.y + oy + yo - p.size)))

        for p in self._anim.confetti:
            alpha = int(255 * min(1.0, p.life / 0.5))
            sz = int(p.size)
            s = _surf_with_alpha(sz, sz)
            col = (*p.color, alpha)
            s.fill(col)
            scr.blit(s, (int(p.x - sz // 2), int(p.y - sz // 2)))

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _camera_offset(self, world: PuzzleWorld) -> Tuple[float, float]:
        assert self._screen is not None
        sw = self._screen.get_width()
        sh = self._screen.get_height() - HUD_HEIGHT - INV_BAR_HEIGHT
        gw = world.width * self._cs * self._zoom
        gh = world.height * self._cs * self._zoom

        if gw <= sw and gh <= sh:
            ox = (sw - gw) / 2
            oy = (sh - gh) / 2
        else:
            # Follow agent
            ax = (self._anim.agent_visual_x + 0.5) * self._cs * self._zoom
            ay = (self._anim.agent_visual_y + 0.5) * self._cs * self._zoom
            ox = sw / 2 - ax
            oy = sh / 2 - ay
            ox = min(0, max(sw - gw, ox))
            oy = min(0, max(sh - gh, oy))

        return ox, oy

    # ------------------------------------------------------------------
    # Draw helpers
    # ------------------------------------------------------------------

    def _cell_rect(self, gx: int, gy: int, ox: float, oy: float) -> pygame.Rect:
        cs = self._cs * self._zoom
        x = ox + gx * cs
        y = oy + gy * cs + HUD_HEIGHT
        return pygame.Rect(int(x), int(y), int(cs), int(cs))

    def _cell_rectf(self, fx: float, fy: float, ox: float, oy: float) -> pygame.Rect:
        cs = self._cs * self._zoom
        x = ox + fx * cs
        y = oy + fy * cs + HUD_HEIGHT
        return pygame.Rect(int(x), int(y), int(cs), int(cs))

    @staticmethod
    def _shadow_rect(r: pygame.Rect) -> pygame.Rect:
        return pygame.Rect(r.x + 2, r.y + 2, r.w, r.h)

    def _draw_rounded_shadow(self, scr: pygame.Surface, rect: pygame.Rect, radius: int = 8) -> None:
        sr = self._shadow_rect(rect)
        shadow = _surf_with_alpha(sr.w + 4, sr.h + 4)
        pygame.draw.rect(shadow, COL_SHADOW, (0, 0, sr.w + 4, sr.h + 4), border_radius=radius)
        scr.blit(shadow, (sr.x - 2, sr.y - 2))

    # ------------------------------------------------------------------
    # Grid lines
    # ------------------------------------------------------------------

    def _draw_grid_lines(self, scr: pygame.Surface, world: PuzzleWorld, ox: float, oy: float) -> None:
        cs = self._cs * self._zoom
        y_off = HUD_HEIGHT
        for x in range(world.width + 1):
            px = int(ox + x * cs)
            pygame.draw.line(scr, COL_GRID, (px, int(oy + y_off)), (px, int(oy + world.height * cs + y_off)))
        for y in range(world.height + 1):
            py = int(oy + y * cs + y_off)
            pygame.draw.line(scr, COL_GRID, (int(ox), py), (int(ox + world.width * cs), py))

    # ------------------------------------------------------------------
    # Tiles (floor-level objects)
    # ------------------------------------------------------------------

    def _draw_tiles(self, scr: pygame.Surface, world: PuzzleWorld, ox: float, oy: float, dt: float) -> None:
        a = self._anim
        cs = self._cs * self._zoom
        br = max(1, int(8 * self._zoom))

        for gy in range(world.height):
            for gx in range(world.width):
                cell = world.get_cell(gx, gy)
                rect = self._cell_rect(gx, gy, ox, oy)
                inset = rect.inflate(-2, -2)

                for obj in cell:
                    if isinstance(obj, Target):
                        # Pulsing bullseye
                        pulse = 0.4 + 0.3 * math.sin(a.target_pulse_phase * math.pi * 2)
                        alpha = int(255 * pulse)
                        s = _surf_with_alpha(rect.w, rect.h)
                        cx, cy = rect.w // 2, rect.h // 2
                        r_outer = int(cs * 0.35)
                        r_mid = int(cs * 0.22)
                        r_inner = int(cs * 0.1)
                        col = (*COL_TARGET, alpha)
                        pygame.draw.circle(s, col, (cx, cy), r_outer, 2)
                        pygame.draw.circle(s, col, (cx, cy), r_mid, 2)
                        pygame.draw.circle(s, (*COL_TARGET, min(255, alpha + 40)), (cx, cy), r_inner)
                        scr.blit(s, rect.topleft)

                    elif isinstance(obj, IceTile):
                        # Frost tile with sparkles
                        s = _surf_with_alpha(rect.w, rect.h)
                        pygame.draw.rect(s, (*COL_ICE, 60), (0, 0, rect.w, rect.h), border_radius=br)
                        # Frost pattern: diagonal lines
                        for i in range(0, rect.w + rect.h, max(1, int(12 * self._zoom))):
                            x1 = min(i, rect.w - 1)
                            y1 = max(0, i - rect.w + 1)
                            x2 = max(0, i - rect.h + 1)
                            y2 = min(i, rect.h - 1)
                            pygame.draw.line(s, (*COL_ICE, 30), (x1, y1), (x2, y2))
                        # Sparkles
                        for sx, sy, brightness in a.ice_sparkles:
                            sx_s = int(sx * self._zoom)
                            sy_s = int(sy * self._zoom)
                            if 0 <= sx_s < rect.w and 0 <= sy_s < rect.h:
                                alpha_s = int(200 * brightness)
                                pygame.draw.circle(s, (*COL_WHITE, alpha_s), (sx_s, sy_s), max(1, int(2 * self._zoom)))
                        scr.blit(s, rect.topleft)

                    elif isinstance(obj, PressureSwitch):
                        # Raised button or pressed flat
                        is_active = obj.activated
                        col = COL_SW_ACTIVE if is_active else COL_SW_INACTIVE
                        pad = int(cs * 0.2)
                        btn_rect = rect.inflate(-pad * 2, -pad * 2)
                        if is_active:
                            # Flat and glowing
                            s = _surf_with_alpha(btn_rect.w + 8, btn_rect.h + 8)
                            pygame.draw.rect(s, (*COL_SW_ACTIVE, 100), (0, 0, btn_rect.w + 8, btn_rect.h + 8), border_radius=br)
                            scr.blit(s, (btn_rect.x - 4, btn_rect.y - 4))
                            pygame.draw.rect(scr, col, btn_rect, border_radius=br)
                        else:
                            # Raised look
                            self._draw_rounded_shadow(scr, btn_rect, br)
                            pygame.draw.rect(scr, col, btn_rect, border_radius=br)
                            hi = btn_rect.inflate(-4, -4)
                            hi.h = 3
                            pygame.draw.rect(scr, tuple(min(255, c + 30) for c in col), hi, border_radius=2)

                    elif isinstance(obj, SwitchWall):
                        # Fade animation: if recently toggled, animate alpha
                        fade_t = a.sw_wall_alpha.get((gx, gy), 1.0)
                        if obj.open:
                            # Fading out: alpha goes 180 -> 0
                            alpha_sw = int(180 * max(0.0, 1.0 - _ease_out_quad(fade_t)))
                        else:
                            # Fading in: alpha goes 0 -> 180
                            if fade_t < 1.0:
                                alpha_sw = int(180 * _ease_out_quad(fade_t))
                            else:
                                alpha_sw = 180
                        if alpha_sw > 2:
                            s = _surf_with_alpha(rect.w, rect.h)
                            pygame.draw.rect(s, (*COL_SWITCH_WALL, alpha_sw), (0, 0, rect.w, rect.h), border_radius=br)
                            # Diagonal hash lines
                            line_alpha = int(alpha_sw * 0.55)
                            for i in range(0, rect.w + rect.h, max(1, int(10 * self._zoom))):
                                x1 = min(i, rect.w - 1)
                                y1 = max(0, i - rect.w + 1)
                                x2 = max(0, i - rect.h + 1)
                                y2 = min(i, rect.h - 1)
                                pygame.draw.line(s, (*COL_SWITCH_WALL, line_alpha), (x1, y1), (x2, y2))
                            scr.blit(s, rect.topleft)

                    elif isinstance(obj, OneWayTile):
                        # Arrow pointing in direction
                        pulse = 0.6 + 0.4 * math.sin(a.arrow_pulse_phase * math.pi * 2)
                        alpha_a = int(200 * pulse)
                        s = _surf_with_alpha(rect.w, rect.h)
                        cx, cy = rect.w // 2, rect.h // 2
                        sz = int(cs * 0.25)
                        direction = obj.direction
                        # Draw arrow triangle
                        if direction == "up":
                            pts = [(cx, cy - sz), (cx - sz, cy + sz // 2), (cx + sz, cy + sz // 2)]
                        elif direction == "down":
                            pts = [(cx, cy + sz), (cx - sz, cy - sz // 2), (cx + sz, cy - sz // 2)]
                        elif direction == "left":
                            pts = [(cx - sz, cy), (cx + sz // 2, cy - sz), (cx + sz // 2, cy + sz)]
                        else:  # right
                            pts = [(cx + sz, cy), (cx - sz // 2, cy - sz), (cx - sz // 2, cy + sz)]
                        pygame.draw.polygon(s, (*COL_ONEWAY, alpha_a), pts)
                        scr.blit(s, rect.topleft)

    # ------------------------------------------------------------------
    # Walls
    # ------------------------------------------------------------------

    def _draw_walls(self, scr: pygame.Surface, world: PuzzleWorld, ox: float, oy: float) -> None:
        br = max(1, int(8 * self._zoom))
        for gy in range(world.height):
            for gx in range(world.width):
                for obj in world.get_cell(gx, gy):
                    if isinstance(obj, Wall):
                        rect = self._cell_rect(gx, gy, ox, oy)
                        # 3D brick effect
                        pygame.draw.rect(scr, COL_WALL, rect, border_radius=br)
                        # Top highlight
                        hi = pygame.Rect(rect.x + 2, rect.y + 1, rect.w - 4, 3)
                        pygame.draw.rect(scr, COL_WALL_HI, hi, border_radius=1)
                        # Bottom shadow
                        lo = pygame.Rect(rect.x + 2, rect.y + rect.h - 4, rect.w - 4, 3)
                        pygame.draw.rect(scr, COL_WALL_LO, lo, border_radius=1)
                        break

    # ------------------------------------------------------------------
    # Objects (key, door, box)
    # ------------------------------------------------------------------

    def _draw_objects(self, scr: pygame.Surface, world: PuzzleWorld, ox: float, oy: float, dt: float) -> None:
        a = self._anim
        cs_z = self._cs * self._zoom
        br = max(1, int(8 * self._zoom))

        for gy in range(world.height):
            for gx in range(world.width):
                cell = world.get_cell(gx, gy)
                for obj in cell:
                    if isinstance(obj, Key):
                        self._draw_key(scr, gx, gy, ox, oy, obj)
                    elif isinstance(obj, Door):
                        self._draw_door(scr, gx, gy, ox, oy, obj)
                    elif isinstance(obj, Box):
                        self._draw_box(scr, gx, gy, ox, oy, obj)

    def _draw_key(self, scr: pygame.Surface, gx: int, gy: int, ox: float, oy: float, obj: Key) -> None:
        a = self._anim
        cs_z = self._cs * self._zoom
        rect = self._cell_rect(gx, gy, ox, oy)
        # Bob animation
        bob = math.sin(a.key_bob_phase * math.pi * 2) * 3 * self._zoom
        cx = rect.centerx
        cy = int(rect.centery + bob)
        # Key shape: circle head + teeth
        head_r = int(cs_z * 0.16)
        pygame.draw.circle(scr, COL_KEY, (cx, cy - int(cs_z * 0.08)), head_r)
        pygame.draw.circle(scr, COL_BG, (cx, cy - int(cs_z * 0.08)), head_r - max(1, int(3 * self._zoom)))
        # Shaft
        shaft_w = max(1, int(3 * self._zoom))
        shaft_len = int(cs_z * 0.25)
        pygame.draw.line(scr, COL_KEY, (cx, cy), (cx, cy + shaft_len), shaft_w)
        # Teeth
        tooth = max(1, int(4 * self._zoom))
        pygame.draw.line(scr, COL_KEY, (cx, cy + shaft_len - tooth * 2), (cx + tooth, cy + shaft_len - tooth * 2), shaft_w)
        pygame.draw.line(scr, COL_KEY, (cx, cy + shaft_len), (cx + tooth, cy + shaft_len), shaft_w)

    def _draw_door(self, scr: pygame.Surface, gx: int, gy: int, ox: float, oy: float, obj: Door) -> None:
        a = self._anim
        cs_z = self._cs * self._zoom
        br = max(1, int(8 * self._zoom))
        rect = self._cell_rect(gx, gy, ox, oy)
        inset = rect.inflate(-4, -4)

        # Flash animation?
        flash_t = a.door_flash.get((gx, gy), 1.0)

        if obj.locked:
            col = COL_DOOR_LOCKED
            self._draw_rounded_shadow(scr, inset, br)
            pygame.draw.rect(scr, col, inset, border_radius=br)
            # Padlock icon
            cx, cy = inset.centerx, inset.centery
            lock_w = int(cs_z * 0.22)
            lock_h = int(cs_z * 0.18)
            lock_rect = pygame.Rect(cx - lock_w // 2, cy, lock_w, lock_h)
            pygame.draw.rect(scr, (200, 50, 50), lock_rect, border_radius=3)
            # Shackle
            shackle_r = int(cs_z * 0.12)
            pygame.draw.arc(scr, (200, 50, 50),
                            (cx - shackle_r, cy - shackle_r, shackle_r * 2, shackle_r * 2),
                            0, math.pi, max(1, int(3 * self._zoom)))
        else:
            # Unlocked: semi-transparent green
            s = _surf_with_alpha(inset.w, inset.h)
            alpha = 120
            if flash_t < 1.0:
                # Flash white then fade to green
                wb = int(255 * max(0, 1.0 - flash_t * 3))
                col_f = (min(255, COL_DOOR_UNLOCKED[0] + wb),
                         min(255, COL_DOOR_UNLOCKED[1] + wb),
                         min(255, COL_DOOR_UNLOCKED[2] + wb),
                         alpha + int(80 * (1 - flash_t)))
            else:
                col_f = (*COL_DOOR_UNLOCKED, alpha)
            pygame.draw.rect(s, col_f, (0, 0, inset.w, inset.h), border_radius=br)
            scr.blit(s, inset.topleft)

    def _draw_box(self, scr: pygame.Surface, gx: int, gy: int, ox: float, oy: float, obj: Box) -> None:
        a = self._anim
        cs_z = self._cs * self._zoom
        br = max(1, int(8 * self._zoom))
        pos_key = (gx, gy)

        # Determine visual position (lerp if animating)
        if pos_key in a.box_prev_pos and pos_key in a.box_move_t:
            t = _ease_out_quad(a.box_move_t[pos_key])
            prev = a.box_prev_pos[pos_key]
            vx = _lerp(prev[0], float(gx), t)
            vy = _lerp(prev[1], float(gy), t)
        else:
            vx, vy = float(gx), float(gy)

        rect = self._cell_rectf(vx, vy, ox, oy)

        # Squash & stretch
        sx_scale = 1.0
        sy_scale = 1.0
        if pos_key in a.box_push_anims:
            pt = a.box_push_anims[pos_key]
            d = a.box_push_dirs.get(pos_key, (1, 0))
            # Squash in push direction at start, stretch at end
            squash = math.sin(pt * math.pi) * 0.15
            if d[0] != 0:  # horizontal push
                sx_scale = 1.0 - squash
                sy_scale = 1.0 + squash * 0.5
            else:
                sx_scale = 1.0 + squash * 0.5
                sy_scale = 1.0 - squash

        inset = rect.inflate(-6, -6)
        # Apply squash/stretch
        new_w = int(inset.w * sx_scale)
        new_h = int(inset.h * sy_scale)
        draw_rect = pygame.Rect(
            inset.centerx - new_w // 2,
            inset.centery - new_h // 2,
            new_w, new_h,
        )

        # Shadow
        self._draw_rounded_shadow(scr, draw_rect, br)

        if obj.on_target:
            # Glowing green outline pulse
            glow_alpha = int(80 + 60 * math.sin(time.monotonic() * 4))
            glow_s = _surf_with_alpha(draw_rect.w + 8, draw_rect.h + 8)
            pygame.draw.rect(glow_s, (*COL_BOX_DONE, glow_alpha), (0, 0, draw_rect.w + 8, draw_rect.h + 8), border_radius=br + 2)
            scr.blit(glow_s, (draw_rect.x - 4, draw_rect.y - 4))
            pygame.draw.rect(scr, COL_BOX_DONE, draw_rect, border_radius=br)
            # Checkmark
            self._draw_checkmark(scr, draw_rect)
        else:
            pygame.draw.rect(scr, COL_BOX, draw_rect, border_radius=br)
            # Inner shadow (top-left lighter, bottom-right darker)
            inner = draw_rect.inflate(-4, -4)
            hi = pygame.Rect(inner.x, inner.y, inner.w, 2)
            pygame.draw.rect(scr, (255, 210, 80), hi, border_radius=1)
            # X pattern
            self._draw_x_pattern(scr, draw_rect)

    def _draw_checkmark(self, scr: pygame.Surface, rect: pygame.Rect) -> None:
        cx, cy = rect.centerx, rect.centery
        sz = int(rect.w * 0.25)
        pts = [
            (cx - sz, cy),
            (cx - sz // 3, cy + sz * 2 // 3),
            (cx + sz, cy - sz // 2),
        ]
        pygame.draw.lines(scr, COL_WHITE, False, pts, max(2, int(3 * self._zoom)))

    def _draw_x_pattern(self, scr: pygame.Surface, rect: pygame.Rect) -> None:
        cx, cy = rect.centerx, rect.centery
        sz = int(rect.w * 0.15)
        col = (220, 165, 20)
        lw = max(1, int(2 * self._zoom))
        pygame.draw.line(scr, col, (cx - sz, cy - sz), (cx + sz, cy + sz), lw)
        pygame.draw.line(scr, col, (cx + sz, cy - sz), (cx - sz, cy + sz), lw)

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------

    def _draw_agent(self, scr: pygame.Surface, world: PuzzleWorld, ox: float, oy: float, dt: float) -> None:
        a = self._anim
        cs_z = self._cs * self._zoom

        # Trail afterimage
        if a.trail_alpha > 0.02:
            tr = self._cell_rectf(a.trail_x, a.trail_y, ox, oy)
            alpha = int(150 * a.trail_alpha)
            s = _surf_with_alpha(tr.w, tr.h)
            cx, cy = tr.w // 2, tr.h // 2
            r = int(cs_z * 0.35)
            pygame.draw.circle(s, (*COL_AGENT, alpha), (cx, cy), r)
            scr.blit(s, tr.topleft)

        # Agent body
        rect = self._cell_rectf(a.agent_visual_x, a.agent_visual_y, ox, oy)

        # Bounce scale
        bounce = 1.0
        if a.agent_bounce_t < 1.0:
            bounce = 1.0 + 0.1 * math.sin(a.agent_bounce_t * math.pi)

        cx = rect.centerx
        cy = rect.centery
        r = int(cs_z * 0.35 * bounce)

        # Shadow
        shadow_s = _surf_with_alpha(r * 2 + 8, r * 2 + 8)
        pygame.draw.circle(shadow_s, COL_SHADOW, (r + 4, r + 6), r)
        scr.blit(shadow_s, (cx - r - 4, cy - r - 2))

        # Body
        pygame.draw.circle(scr, COL_AGENT, (cx, cy), r)
        # Highlight
        pygame.draw.circle(scr, (130, 133, 255), (cx - r // 4, cy - r // 4), r // 3)

        # Eyes that follow movement direction
        direction = world.agent_dir
        edx, edy = DIR_OFFSETS.get(direction, (0, -1))
        eye_off = int(r * 0.3)
        eye_r = max(2, int(r * 0.18))
        pupil_r = max(1, int(r * 0.09))

        if direction in ("left", "right"):
            e1 = (cx + edx * eye_off, cy - eye_off // 2)
            e2 = (cx + edx * eye_off, cy + eye_off // 2)
        else:
            e1 = (cx - eye_off // 2, cy + edy * eye_off)
            e2 = (cx + eye_off // 2, cy + edy * eye_off)

        for ex, ey in (e1, e2):
            pygame.draw.circle(scr, COL_WHITE, (ex, ey), eye_r)
            pygame.draw.circle(scr, (30, 30, 50), (ex + edx * pupil_r, ey + edy * pupil_r), pupil_r)

    # ------------------------------------------------------------------
    # Overlays (thought paths, danger zones)
    # ------------------------------------------------------------------

    def _draw_overlays(self, scr: pygame.Surface, world: PuzzleWorld, ox: float, oy: float,
                       overlay: Dict[str, Any]) -> None:
        cs_z = self._cs * self._zoom

        # Danger zones
        danger = overlay.get("danger_zones", [])
        for (dx, dy) in danger:
            rect = self._cell_rect(dx, dy, ox, oy)
            s = _surf_with_alpha(rect.w, rect.h)
            pygame.draw.rect(s, (239, 68, 68, 60), (0, 0, rect.w, rect.h), border_radius=4)
            scr.blit(s, rect.topleft)

        # Thought paths (dotted lines)
        thought_paths = overlay.get("thought_paths", [])
        colors = [(200, 200, 255), (255, 200, 200), (200, 255, 200), (255, 255, 200)]
        for i, path in enumerate(thought_paths):
            col = colors[i % len(colors)]
            for j in range(len(path) - 1):
                ax_, ay_ = path[j]
                bx_, by_ = path[j + 1]
                p1 = (int(ox + (ax_ + 0.5) * cs_z), int(oy + (ay_ + 0.5) * cs_z + HUD_HEIGHT))
                p2 = (int(ox + (bx_ + 0.5) * cs_z), int(oy + (by_ + 0.5) * cs_z + HUD_HEIGHT))
                self._draw_dotted_line(scr, col, p1, p2)

        # Selected path (solid green)
        sel = overlay.get("selected_path", [])
        if len(sel) >= 2:
            pts = [(int(ox + (px + 0.5) * cs_z), int(oy + (py + 0.5) * cs_z + HUD_HEIGHT))
                   for (px, py) in sel]
            pygame.draw.lines(scr, COL_TARGET, False, pts, max(2, int(3 * self._zoom)))

        # Affordance labels
        labels = overlay.get("affordance_labels", {})
        font = self._fonts.get("small")
        if font:
            for (lx, ly), text in labels.items():
                rect = self._cell_rect(lx, ly, ox, oy)
                surf = font.render(text, True, COL_WHITE)
                scr.blit(surf, (rect.x + 2, rect.y - 14))

    @staticmethod
    def _draw_dotted_line(scr: pygame.Surface, color: Tuple[int, int, int],
                          p1: Tuple[int, int], p2: Tuple[int, int], dash_len: int = 6) -> None:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = max(1, int(math.hypot(dx, dy)))
        for i in range(0, dist, dash_len * 2):
            t1 = i / dist
            t2 = min(1.0, (i + dash_len) / dist)
            x1 = int(p1[0] + dx * t1)
            y1 = int(p1[1] + dy * t1)
            x2 = int(p1[0] + dx * t2)
            y2 = int(p1[1] + dy * t2)
            pygame.draw.line(scr, color, (x1, y1), (x2, y2), 2)

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _draw_hud(self, scr: pygame.Surface, world: PuzzleWorld) -> None:
        w = scr.get_width()
        # Background bar
        hud_s = _surf_with_alpha(w, HUD_HEIGHT)
        hud_s.fill(COL_HUD_BG)
        scr.blit(hud_s, (0, 0))

        font = self._fonts["hud"]
        font_b = self._fonts["hud_big"]
        y_mid = HUD_HEIGHT // 2

        # Left: difficulty
        diff = getattr(world, "_difficulty", 0)
        stars = ""
        for i in range(1, 6):
            stars += "*" if i <= diff else "."
        left_text = f"Level [{stars}]"
        surf = font.render(left_text, True, COL_HUD_TEXT)
        scr.blit(surf, (10, y_mid - surf.get_height() // 2))

        # Centre: step counter
        step_text = f"Step {world.steps}/{world.max_steps}"
        surf_c = font_b.render(step_text, True, COL_HUD_TEXT)
        scr.blit(surf_c, (w // 2 - surf_c.get_width() // 2, y_mid - surf_c.get_height() // 2 - 4))
        # Progress bar
        bar_w = 120
        bar_h = 6
        bar_x = w // 2 - bar_w // 2
        bar_y = y_mid + 10
        pygame.draw.rect(scr, (50, 50, 70), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        frac = min(1.0, world.steps / max(1, world.max_steps))
        fill_col = (52, 211, 153) if frac < 0.7 else (251, 191, 36) if frac < 0.9 else (239, 68, 68)
        pygame.draw.rect(scr, fill_col, (bar_x, bar_y, int(bar_w * frac), bar_h), border_radius=3)

        # Right: box progress
        obs = world.get_observation()
        box_text = f"{obs['boxes_on_targets']}/{obs['total_targets']} boxes"
        surf_r = font.render(box_text, True, COL_HUD_TEXT)
        rx = w - surf_r.get_width() - 10
        scr.blit(surf_r, (rx, y_mid - surf_r.get_height() // 2 - 4))
        # Mini progress bar
        bp_w = 60
        bp_x = w - bp_w - 10
        bp_y = y_mid + 10
        pygame.draw.rect(scr, (50, 50, 70), (bp_x, bp_y, bp_w, bar_h), border_radius=3)
        bfrac = obs["boxes_on_targets"] / max(1, obs["total_targets"])
        pygame.draw.rect(scr, COL_TARGET, (bp_x, bp_y, int(bp_w * bfrac), bar_h), border_radius=3)

    def _draw_inventory_bar(self, scr: pygame.Surface, world: PuzzleWorld) -> None:
        w = scr.get_width()
        h = scr.get_height()
        bar_y = h - INV_BAR_HEIGHT

        inv_s = _surf_with_alpha(w, INV_BAR_HEIGHT)
        inv_s.fill(COL_HUD_BG)
        scr.blit(inv_s, (0, bar_y))

        font = self._fonts["hud"]
        if world.inventory:
            txt = "Keys: "
            surf = font.render(txt, True, COL_HUD_TEXT)
            scr.blit(surf, (10, bar_y + INV_BAR_HEIGHT // 2 - surf.get_height() // 2))
            kx = 10 + surf.get_width() + 4
            for key in world.inventory:
                pygame.draw.circle(scr, COL_KEY, (kx + 8, bar_y + INV_BAR_HEIGHT // 2), 8)
                kid_surf = self._fonts["small"].render(str(key.key_id), True, COL_BG)
                scr.blit(kid_surf, (kx + 8 - kid_surf.get_width() // 2, bar_y + INV_BAR_HEIGHT // 2 - kid_surf.get_height() // 2))
                kx += 24
        else:
            surf = font.render("Inventory: empty", True, (80, 80, 100))
            scr.blit(surf, (10, bar_y + INV_BAR_HEIGHT // 2 - surf.get_height() // 2))

        # Speed/pause indicators on the right
        indicators = []
        if self._paused:
            indicators.append("PAUSED")
        if self._speed_mult != 1.0:
            indicators.append(f"{self._speed_mult:.1f}x")
        if self._show_thought:
            indicators.append("THINK")
        if indicators:
            txt = "  ".join(indicators)
            surf = font.render(txt, True, COL_GOLD_STAR)
            scr.blit(surf, (w - surf.get_width() - 10, bar_y + INV_BAR_HEIGHT // 2 - surf.get_height() // 2))

    # ------------------------------------------------------------------
    # Solved overlay
    # ------------------------------------------------------------------

    def _draw_solved_overlay(self, scr: pygame.Surface, world: PuzzleWorld, dt: float) -> None:
        a = self._anim
        w = scr.get_width()
        h = scr.get_height()

        # Dim background
        alpha_dim = min(120, int(a.solved_t * 200))
        dim = _surf_with_alpha(w, h)
        dim.fill((0, 0, 0, alpha_dim))
        scr.blit(dim, (0, 0))

        if a.solved_t < 0.3:
            return

        # "SOLVED!" text with glow
        font = self._fonts["solved"]
        text = "SOLVED!"
        # Glow
        glow_alpha = int(80 + 40 * math.sin(time.monotonic() * 3))
        glow_surf = font.render(text, True, (*COL_TARGET, glow_alpha))
        gw = glow_surf.get_width()
        gh = glow_surf.get_height()
        scr.blit(glow_surf, (w // 2 - gw // 2 + 2, h // 2 - gh // 2 - 40 + 2))
        # Main text
        text_surf = font.render(text, True, COL_WHITE)
        scr.blit(text_surf, (w // 2 - text_surf.get_width() // 2, h // 2 - text_surf.get_height() // 2 - 40))

        # Stats
        if a.solved_t > 0.6:
            sub_font = self._fonts["solved_sub"]
            opt = getattr(world, "_optimal_steps", None)
            lines = [f"Steps taken: {world.steps}"]
            if opt is not None and opt > 0:
                eff = opt / max(1, world.steps) * 100
                lines.append(f"Optimal: {opt}  Efficiency: {eff:.0f}%")

            for i, line in enumerate(lines):
                surf = sub_font.render(line, True, COL_HUD_TEXT)
                scr.blit(surf, (w // 2 - surf.get_width() // 2, h // 2 + 10 + i * 28))

    # ------------------------------------------------------------------
    # Key collection fly
    # ------------------------------------------------------------------

    def _draw_key_fly(self, scr: pygame.Surface, ox: float, oy: float) -> None:
        """Draw keys spiralling from their grid position up to the inventory bar."""
        a = self._anim
        cs_z = self._cs * self._zoom
        h = scr.get_height()
        inv_target_y = float(h - INV_BAR_HEIGHT // 2)
        inv_target_x = 80.0  # roughly where inventory icons sit

        for wx, wy, prog, kid in a.key_fly_anims:
            t = _ease_out_quad(prog)
            # Start position (grid cell centre)
            start_x = ox + (wx + 0.5) * cs_z
            start_y = oy + (wy + 0.5) * cs_z + HUD_HEIGHT
            # Spiral path: lerp with a sine wobble
            cur_x = _lerp(start_x, inv_target_x, t) + math.sin(prog * math.pi * 4) * 15 * (1 - t)
            cur_y = _lerp(start_y, inv_target_y, t)

            alpha = int(255 * (1 - t * 0.6))
            size = max(2, int(10 * (1 - t * 0.5) * self._zoom))
            s = _surf_with_alpha(size * 2, size * 2)
            pygame.draw.circle(s, (*COL_KEY, alpha), (size, size), size)
            scr.blit(s, (int(cur_x - size), int(cur_y - size)))

    # ------------------------------------------------------------------
    # Mini-map
    # ------------------------------------------------------------------

    def _draw_minimap(self, scr: pygame.Surface, world: PuzzleWorld) -> None:
        """Draw a small overview map in the bottom-right corner for large grids."""
        w = scr.get_width()
        h = scr.get_height()
        margin = 8
        max_map_size = 100
        cell_px = min(max_map_size // world.width, max_map_size // world.height, 6)
        cell_px = max(2, cell_px)
        map_w = world.width * cell_px
        map_h = world.height * cell_px
        mx = w - map_w - margin
        my = h - INV_BAR_HEIGHT - map_h - margin

        # Background
        bg = _surf_with_alpha(map_w + 4, map_h + 4)
        bg.fill((15, 15, 25, 180))
        scr.blit(bg, (mx - 2, my - 2))

        for gy in range(world.height):
            for gx in range(world.width):
                px = mx + gx * cell_px
                py = my + gy * cell_px
                r = pygame.Rect(px, py, cell_px, cell_px)

                # Determine colour
                if (gx, gy) == world.agent_pos:
                    pygame.draw.rect(scr, COL_AGENT, r)
                    continue
                cell = world.get_cell(gx, gy)
                col = COL_BG
                for obj in cell:
                    if isinstance(obj, Box):
                        col = COL_BOX_DONE if obj.on_target else COL_BOX
                        break
                    elif isinstance(obj, Wall):
                        col = COL_WALL
                        break
                    elif isinstance(obj, Target):
                        col = COL_TARGET
                    elif isinstance(obj, Key):
                        col = COL_KEY
                        break
                    elif isinstance(obj, Door):
                        col = COL_DOOR_LOCKED if obj.locked else COL_DOOR_UNLOCKED
                        break
                    elif isinstance(obj, IceTile):
                        col = (100, 140, 200)
                    elif isinstance(obj, PressureSwitch):
                        col = COL_SW_ACTIVE if obj.activated else COL_SW_INACTIVE
                    elif isinstance(obj, SwitchWall) and not obj.open:
                        col = (120, 60, 180)
                pygame.draw.rect(scr, col, r)

        # Border
        pygame.draw.rect(scr, COL_GRID, (mx - 2, my - 2, map_w + 4, map_h + 4), 1)

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def handle_scroll(self, dy: int) -> None:
        if dy > 0:
            self._zoom = min(2.0, self._zoom * 1.1)
        elif dy < 0:
            self._zoom = max(0.5, self._zoom / 1.1)


# ---------------------------------------------------------------------------
# Standalone interactive mode
# ---------------------------------------------------------------------------

def _run_standalone() -> None:
    """Launch interactive human-playable puzzle with rendered visuals."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from env.level_generator import LevelGenerator
    from env.puzzle_world import ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT

    gen = LevelGenerator()
    difficulty = 3
    renderer = PuzzleRenderer(cell_size=64)

    def new_level() -> PuzzleWorld:
        seed = random.randint(0, 2**31)
        w = gen.generate(difficulty=difficulty, seed=seed)
        print(f"\n--- New level (difficulty {difficulty}, seed {seed}) ---")
        print(w.render_ascii())
        opt = getattr(w, "_optimal_steps", "?")
        print(f"Optimal: {opt} steps\n")
        return w

    world = new_level()

    KEY_MAP = {
        pygame.K_UP: ACTION_UP,
        pygame.K_w: ACTION_UP,
        pygame.K_DOWN: ACTION_DOWN,
        pygame.K_s: ACTION_DOWN,
        pygame.K_LEFT: ACTION_LEFT,
        pygame.K_a: ACTION_LEFT,
        pygame.K_RIGHT: ACTION_RIGHT,
        pygame.K_d: ACTION_RIGHT,
    }

    running = True
    auto_next_timer = 0.0

    while running:
        dt = renderer._clock.tick(60) / 1000.0 if renderer._clock else 0.016

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                renderer.handle_scroll(event.y)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    world = new_level()
                    auto_next_timer = 0.0
                elif event.key == pygame.K_n:
                    world = new_level()
                    auto_next_timer = 0.0
                elif event.key == pygame.K_SPACE:
                    renderer._paused = not renderer._paused
                elif event.key == pygame.K_f:
                    renderer._speed_mult = 4.0 if renderer._speed_mult != 4.0 else 1.0
                elif event.key == pygame.K_t:
                    renderer._show_thought = not renderer._show_thought
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                                   pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8,
                                   pygame.K_9):
                    difficulty = event.key - pygame.K_0
                    print(f"Switched to difficulty {difficulty}")
                    world = new_level()
                    auto_next_timer = 0.0
                elif event.key in KEY_MAP and not world.solved:
                    action = KEY_MAP[event.key]
                    obs, reward, term, trunc, info = world.step(action)
                    print(world.render_ascii())
                    print(f"  step={world.steps} reward={reward:+.1f} "
                          f"boxes={obs['boxes_on_targets']}/{obs['total_targets']}")
                    if term:
                        print("  >>> SOLVED! <<<")

                # Slow motion: toggle 0.25x
                if event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    renderer._speed_mult = 0.25 if renderer._speed_mult != 0.25 else 1.0

        # Auto-next on solve
        if world.solved:
            auto_next_timer += dt
            if auto_next_timer > 3.0:
                world = new_level()
                auto_next_timer = 0.0

        renderer.render(world)

        # FPS in title
        fps = renderer._clock.get_fps() if renderer._clock else 0
        pygame.display.set_caption(f"World-Model RL  |  Difficulty {difficulty}  |  {fps:.0f} FPS")

    renderer.close()


if __name__ == "__main__":
    _run_standalone()
