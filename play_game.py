#!/usr/bin/env python3
"""
SOKOBAN - A complete, self-contained puzzle game built with pygame.
60 levels: 17 handcrafted + 43 procedurally generated.

Controls:
    Arrow keys  Move player
    U           Undo last move
    R           Restart level
    N           Next level
    P           Previous level
    ESC         Back to level select
    Q           Quit
"""

import json
import math
import os
import random
import sys

import pygame

# ---------------------------------------------------------------------------
# LEVELS (XSB format) - 17 original handcrafted levels
# ---------------------------------------------------------------------------

HANDCRAFTED_LEVELS = [
    # Level 1 - 7x7, 2 boxes. Gentle intro.
    "\n".join([
        "#######",
        "#.  @ #",
        "#  $  #",
        "#  #  #",
        "#  $  #",
        "#.    #",
        "#######",
    ]),
    # Level 2 - 7x7, 2 boxes
    "\n".join([
        "#######",
        "#     #",
        "# $.$ #",
        "#  #  #",
        "#  .  #",
        "#  @  #",
        "#######",
    ]),
    # Level 3 - 8x7, 2 boxes
    "\n".join([
        "########",
        "#   .  #",
        "# $  # #",
        "#   #  #",
        "# #  $ #",
        "# @  . #",
        "########",
    ]),
    # Level 4 - 8x8, 3 boxes
    "\n".join([
        "########",
        "#      #",
        "# .$$  #",
        "# .#   #",
        "# . $  #",
        "#   #  #",
        "#   @  #",
        "########",
    ]),
    # Level 5 - 9x8, 3 boxes
    "\n".join([
        "#########",
        "#   #   #",
        "# $ . $ #",
        "#  .#.  #",
        "#   $   #",
        "#  ###  #",
        "#   @   #",
        "#########",
    ]),
    # Level 6 - 9x9, 3 boxes
    "\n".join([
        "#########",
        "#       #",
        "# # # # #",
        "# $ . $ #",
        "#  .#.  #",
        "#   $   #",
        "# #   # #",
        "#   @   #",
        "#########",
    ]),
    # Level 7 - 10x9, 3 boxes
    "\n".join([
        "##########",
        "#    #   #",
        "#  $   $ #",
        "# ##.##  #",
        "#   .  # #",
        "# # .  $ #",
        "#   ##   #",
        "#    @   #",
        "##########",
    ]),
    # Level 8 - 10x9, 4 boxes
    "\n".join([
        "##########",
        "#  ....  #",
        "#  #  #  #",
        "#  $  $  #",
        "## $  $ ##",
        "#   ##   #",
        "#   @    #",
        "#   ##   #",
        "##########",
    ]),
    # Level 9 - 10x10, 4 boxes
    "\n".join([
        "##########",
        "#   #    #",
        "# $   $  #",
        "#  .#.#  #",
        "## #   ###",
        "#  .#.   #",
        "#  $ $   #",
        "#   #    #",
        "#   @    #",
        "##########",
    ]),
    # Level 10 - 11x10, 4 boxes
    "\n".join([
        "###########",
        "#    #    #",
        "#  $   $  #",
        "# ##.#.## #",
        "#         #",
        "#  #. .#  #",
        "#  $ # $  #",
        "#  # # #  #",
        "#    @    #",
        "###########",
    ]),
    # Level 11 - 11x11, 5 boxes
    "\n".join([
        "###########",
        "#   #     #",
        "# $   # $ #",
        "#  ##.##  #",
        "#   . .   #",
        "## # # # ##",
        "#   . .   #",
        "#  ## ##  #",
        "# $ $ # $ #",
        "#    @    #",
        "###########",
    ]),
    # Level 12 - 12x11, 5 boxes
    "\n".join([
        "############",
        "#     #    #",
        "# $ $   $  #",
        "#  #.#.#.  #",
        "#     #    #",
        "## ##   ## #",
        "#    #     #",
        "#  .   .   #",
        "#  # # # $ #",
        "#  @   $ # #",
        "############",
    ]),
    # Level 13 - 12x11, 5 boxes
    "\n".join([
        "############",
        "#     ##   #",
        "# $.$ #  $ #",
        "#  #  #    #",
        "# .#.## # ##",
        "#     $    #",
        "## ####    #",
        "#    $ #   #",
        "#  ..#   # #",
        "#  @       #",
        "############",
    ]),
    # Level 14 - 12x12, 5 boxes
    "\n".join([
        "############",
        "#          #",
        "# # $.$ #  #",
        "#    #   # #",
        "# #.#.#    #",
        "#    #   # #",
        "# #  $ $ # #",
        "#    #     #",
        "# ## # ##  #",
        "#  . . $   #",
        "#      @   #",
        "############",
    ]),
    # Level 15 - 13x12, 5 boxes
    "\n".join([
        "#############",
        "#     #     #",
        "# $ $   $ # #",
        "#  #.#.#.   #",
        "#     #   # #",
        "## ##   ##  #",
        "#    # #    #",
        "#  .   .  # #",
        "#  # # # $  #",
        "#    $      #",
        "#      @    #",
        "#############",
    ]),
    # Level 16 - 13x13, 6 boxes
    "\n".join([
        "#############",
        "#     #     #",
        "# $ $   $ $ #",
        "#  #.#.#.#  #",
        "#     #     #",
        "## ##   # ###",
        "#    # #    #",
        "#  .   .  # #",
        "#  # # # $  #",
        "#    #   $  #",
        "#  . #      #",
        "#      @    #",
        "#############",
    ]),
    # Level 17 - 14x13, 6 boxes
    "\n".join([
        "##############",
        "#      #     #",
        "# $ $    $ $ #",
        "#  #.#..#.#  #",
        "#     ##     #",
        "## ##    # ###",
        "#    # ##    #",
        "#  .   .   # #",
        "#  # # # #   #",
        "#    #   $ # #",
        "#      $     #",
        "#        @   #",
        "##############",
    ]),
]


def _generate_level(level_index, rng):
    """Generate a simple procedural Sokoban level."""
    difficulty = level_index / 60.0
    w = rng.randint(8, 10) + int(difficulty * 4)
    h = rng.randint(8, 10) + int(difficulty * 4)
    w = min(w, 14)
    h = min(h, 14)
    num_boxes = rng.randint(2, 3) + int(difficulty * 3)
    num_boxes = min(num_boxes, 6)

    for _ in range(200):
        grid = [['#'] * w for _ in range(h)]
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                grid[y][x] = ' '

        num_walls = rng.randint(3, 6) + int(difficulty * 5)
        for _ in range(num_walls):
            wx = rng.randint(2, w - 3)
            wy = rng.randint(2, h - 3)
            grid[wy][wx] = '#'

        free = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if grid[y][x] == ' ':
                    free.append((x, y))

        if len(free) < num_boxes * 2 + 3:
            continue

        start = free[0]
        visited = set()
        stack = [start]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + ddx, cy + ddy
                if 0 <= nx < w and 0 <= ny < h and grid[ny][nx] == ' ' and (nx, ny) not in visited:
                    stack.append((nx, ny))

        reachable = [c for c in free if c in visited]
        if len(reachable) < num_boxes * 2 + 3:
            continue

        rng.shuffle(reachable)

        targets = []
        boxes = []
        for pos in reachable:
            if len(targets) < num_boxes:
                targets.append(pos)
            elif len(boxes) < num_boxes:
                x, y = pos
                adj_walls = 0
                for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if grid[y + ddy][x + ddx] == '#':
                        adj_walls += 1
                if adj_walls < 2 and pos not in targets:
                    boxes.append(pos)
            else:
                break

        if len(targets) != num_boxes or len(boxes) != num_boxes:
            continue

        remaining = [c for c in reachable if c not in targets and c not in boxes]
        if not remaining:
            continue

        player = remaining[0]

        for tx, ty in targets:
            grid[ty][tx] = '.'
        for bx, by in boxes:
            if grid[by][bx] == '.':
                grid[by][bx] = '*'
            else:
                grid[by][bx] = '$'
        px, py = player
        if grid[py][px] == '.':
            grid[py][px] = '+'
        else:
            grid[py][px] = '@'

        return "\n".join("".join(row) for row in grid)

    return "\n".join([
        "########",
        "#      #",
        "# $.   #",
        "#   .$ #",
        "#   @  #",
        "########",
    ])


def _build_levels():
    levels = list(HANDCRAFTED_LEVELS)
    rng = random.Random(42)
    while len(levels) < 60:
        lvl = _generate_level(len(levels), rng)
        levels.append(lvl)
    return levels


LEVELS = _build_levels()

# ---------------------------------------------------------------------------
# GAME ENGINE
# ---------------------------------------------------------------------------

def parse_level(xsb):
    """Parse XSB string into game state dict."""
    lines = xsb.split('\n')
    height = len(lines)
    width = max(len(l) for l in lines)
    walls = set()
    boxes = set()
    targets = set()
    player = None
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch == '#':
                walls.add((x, y))
            elif ch == '$':
                boxes.add((x, y))
            elif ch == '.':
                targets.add((x, y))
            elif ch == '@':
                player = (x, y)
            elif ch == '*':
                boxes.add((x, y))
                targets.add((x, y))
            elif ch == '+':
                player = (x, y)
                targets.add((x, y))
    if player is None:
        player = (1, 1)
    return {
        'walls': walls,
        'boxes': set(boxes),
        'targets': targets,
        'player': player,
        'width': width,
        'height': height,
    }


def try_move(state, dx, dy):
    """Attempt move. Returns (new_state, pushed) or (None, False)."""
    px, py = state['player']
    nx, ny = px + dx, py + dy
    if (nx, ny) in state['walls']:
        return None, False
    if (nx, ny) in state['boxes']:
        bx, by = nx + dx, ny + dy
        if (bx, by) in state['walls'] or (bx, by) in state['boxes']:
            return None, False
        new_boxes = set(state['boxes'])
        new_boxes.remove((nx, ny))
        new_boxes.add((bx, by))
        return {
            'walls': state['walls'], 'boxes': new_boxes,
            'targets': state['targets'], 'player': (nx, ny),
            'width': state['width'], 'height': state['height'],
        }, True
    return {
        'walls': state['walls'], 'boxes': set(state['boxes']),
        'targets': state['targets'], 'player': (nx, ny),
        'width': state['width'], 'height': state['height'],
    }, False


def is_solved(state):
    return len(state['boxes']) > 0 and state['boxes'] == state['targets']


def check_deadlocks(state):
    """Return set of deadlocked box positions (corner deadlocks not on target)."""
    deadlocked = set()
    for bx, by in state['boxes']:
        if (bx, by) in state['targets']:
            continue
        w = state['walls']
        left = (bx - 1, by) in w
        right = (bx + 1, by) in w
        up = (bx, by - 1) in w
        down = (bx, by + 1) in w
        if (left or right) and (up or down):
            deadlocked.add((bx, by))
    return deadlocked


# ---------------------------------------------------------------------------
# SAVE / LOAD
# ---------------------------------------------------------------------------

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves")
SAVE_FILE = os.path.join(SAVE_DIR, "progress.json")


def load_progress():
    try:
        with open(SAVE_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"solved": {}, "current": 0}


def save_progress(progress):
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(SAVE_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


# ---------------------------------------------------------------------------
# COLORS AND CONSTANTS
# ---------------------------------------------------------------------------

COL_BG = (18, 18, 30)
COL_FLOOR = (32, 32, 48)
COL_GRID = (42, 42, 58)
COL_WALL = (65, 70, 85)
COL_WALL_LIGHT = (80, 85, 100)
COL_WALL_DARK = (45, 50, 60)
COL_PLAYER = (99, 102, 241)
COL_BOX = (240, 180, 40)
COL_BOX_SHADOW = (210, 150, 30)
COL_BOX_HI = (255, 210, 80)
COL_BOX_DONE = (60, 200, 120)
COL_TARGET = (52, 211, 153)
COL_PANEL_BG = (25, 25, 40)
COL_INDIGO = (99, 102, 241)
COL_TEXT = (200, 210, 230)
COL_TEXT_DIM = (150, 160, 180)
COL_AMBER = (240, 180, 40)
COL_GREEN = (60, 200, 120)
COL_RED = (220, 60, 60)

WIN_W, WIN_H = 1000, 750
GAME_AREA_W = 700
PANEL_W = 300
ANIM_FRAMES = 6


def lerp(a, b, t):
    return a + (b - a) * max(0.0, min(1.0, t))


def draw_rounded_rect(surf, color, rect, radius=6):
    pygame.draw.rect(surf, color, rect, border_radius=radius)


# ---------------------------------------------------------------------------
# LEVEL SELECT SCREEN
# ---------------------------------------------------------------------------

def level_select_screen(screen, clock, progress, fonts):
    selected = progress.get("current", 0)

    while True:
        mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return -1
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return -1
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return selected
                if event.key == pygame.K_RIGHT:
                    selected = min(selected + 1, 59)
                if event.key == pygame.K_LEFT:
                    selected = max(selected - 1, 0)
                if event.key == pygame.K_DOWN:
                    selected = min(selected + 10, 59)
                if event.key == pygame.K_UP:
                    selected = max(selected - 10, 0)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i in range(60):
                    col = i % 10
                    row = i // 10
                    bx = 80 + col * 84
                    by = 160 + row * 78
                    r = pygame.Rect(bx, by, 64, 64)
                    if r.collidepoint(mx, my):
                        return i

        screen.fill(COL_BG)

        title_surf = fonts['title'].render("SOKOBAN", True, COL_INDIGO)
        screen.blit(title_surf, (WIN_W // 2 - title_surf.get_width() // 2, 30))

        sub_surf = fonts['medium'].render("60 Levels", True, COL_TEXT_DIM)
        screen.blit(sub_surf, (WIN_W // 2 - sub_surf.get_width() // 2, 85))

        solved_set = progress.get("solved", {})

        for i in range(60):
            col = i % 10
            row = i // 10
            bx = 80 + col * 84
            by = 160 + row * 78
            rect = pygame.Rect(bx, by, 64, 64)

            is_current = (i == selected)
            is_solved_lvl = str(i) in solved_set

            if is_solved_lvl:
                bg = (30, 80, 50)
                tc = COL_GREEN
            elif is_current:
                bg = (60, 60, 100)
                tc = COL_INDIGO
            else:
                bg = (40, 40, 55)
                tc = COL_TEXT_DIM

            draw_rounded_rect(screen, bg, rect, 8)
            if is_current:
                pygame.draw.rect(screen, COL_INDIGO, rect, 2, border_radius=8)

            num_surf = fonts['small'].render(str(i + 1), True, tc)
            screen.blit(num_surf, (bx + 32 - num_surf.get_width() // 2,
                                   by + 32 - num_surf.get_height() // 2))

        footer = fonts['medium'].render(f"Solved: {len(solved_set)}/60", True, COL_TEXT_DIM)
        screen.blit(footer, (WIN_W // 2 - footer.get_width() // 2, WIN_H - 55))

        hint = fonts['tiny'].render(
            "Arrow keys + Enter to select  |  Click a level  |  ESC to quit",
            True, (100, 110, 130))
        screen.blit(hint, (WIN_W // 2 - hint.get_width() // 2, WIN_H - 28))

        pygame.display.flip()
        clock.tick(30)


# ---------------------------------------------------------------------------
# GAME PLAY
# ---------------------------------------------------------------------------

DIR_MAP = {
    pygame.K_UP: (0, -1),
    pygame.K_DOWN: (0, 1),
    pygame.K_LEFT: (-1, 0),
    pygame.K_RIGHT: (1, 0),
}


def play_level(screen, clock, level_index, progress, fonts):
    """Play a single level. Returns: 'select', 'next', 'prev', 'quit', or 'solved'."""
    state = parse_level(LEVELS[level_index])
    initial_state = {
        'walls': state['walls'],
        'boxes': frozenset(state['boxes']),
        'targets': state['targets'],
        'player': state['player'],
        'width': state['width'],
        'height': state['height'],
    }

    moves = 0
    pushes = 0
    facing = (0, 1)
    undo_stack = []
    deadlocked_boxes = set()
    deadlock_flash = 0
    solved_timer = 0
    solved = False

    anim_progress = 1.0
    anim_from = state['player']
    anim_to = state['player']
    anim_box_from = None
    anim_box_to = None

    cell = min(650 // state['height'], 650 // state['width'])
    cell = max(28, min(64, cell))
    grid_w = state['width'] * cell
    grid_h = state['height'] * cell
    offset_x = (GAME_AREA_W - grid_w) // 2
    offset_y = (WIN_H - grid_h) // 2

    while True:
        dt = clock.tick(60)
        ticks = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            if event.type == pygame.KEYDOWN:
                if solved:
                    if event.key in (pygame.K_n, pygame.K_RETURN):
                        return 'next'
                    if event.key == pygame.K_ESCAPE:
                        return 'select'
                    continue

                if event.key == pygame.K_ESCAPE:
                    return 'select'
                if event.key == pygame.K_q:
                    return 'quit'
                if event.key == pygame.K_n:
                    return 'next'
                if event.key == pygame.K_p:
                    return 'prev'
                if event.key == pygame.K_r:
                    # Restart
                    state = parse_level(LEVELS[level_index])
                    moves = 0
                    pushes = 0
                    undo_stack.clear()
                    deadlocked_boxes = set()
                    deadlock_flash = 0
                    anim_progress = 1.0
                    anim_from = state['player']
                    anim_to = state['player']
                    anim_box_from = None
                    anim_box_to = None
                    continue
                if event.key == pygame.K_u and undo_stack:
                    if anim_progress < 1.0:
                        continue
                    prev = undo_stack.pop()
                    state['player'] = prev[0]
                    state['boxes'] = set(prev[1])
                    moves = prev[2]
                    pushes = prev[3]
                    facing = prev[4]
                    deadlocked_boxes = set()
                    deadlock_flash = 0
                    anim_progress = 1.0
                    anim_from = state['player']
                    anim_to = state['player']
                    anim_box_from = None
                    anim_box_to = None
                    continue

                if event.key in DIR_MAP and anim_progress >= 1.0:
                    dx, dy = DIR_MAP[event.key]
                    facing = (dx, dy)

                    undo_stack.append((
                        state['player'],
                        frozenset(state['boxes']),
                        moves,
                        pushes,
                        facing,
                    ))

                    old_player = state['player']
                    new_state, pushed = try_move(state, dx, dy)
                    if new_state is None:
                        undo_stack.pop()
                        continue

                    anim_from = old_player
                    anim_to = new_state['player']
                    anim_progress = 0.0
                    if pushed:
                        anim_box_from = anim_to
                        anim_box_to = (anim_to[0] + dx, anim_to[1] + dy)
                    else:
                        anim_box_from = None
                        anim_box_to = None

                    state = new_state
                    moves += 1
                    if pushed:
                        pushes += 1

                    dl = check_deadlocks(state)
                    if dl:
                        deadlocked_boxes = dl
                        deadlock_flash = 60
                    else:
                        deadlocked_boxes = set()
                        deadlock_flash = 0

                    if is_solved(state):
                        solved = True
                        solved_timer = 0
                        key = str(level_index)
                        solved_dict = progress.get("solved", {})
                        prev_best = solved_dict.get(key, {})
                        if not prev_best or moves < prev_best.get("moves", 9999):
                            solved_dict[key] = {"moves": moves, "pushes": pushes}
                        progress["solved"] = solved_dict
                        progress["current"] = level_index
                        save_progress(progress)

        # Update animation
        if anim_progress < 1.0:
            anim_progress += 1.0 / ANIM_FRAMES
            if anim_progress >= 1.0:
                anim_progress = 1.0

        if deadlock_flash > 0:
            deadlock_flash -= 1

        if solved:
            solved_timer += dt

        # ---- RENDER ----
        screen.fill(COL_BG)

        # Floor tiles
        for y in range(state['height']):
            for x in range(state['width']):
                sx = offset_x + x * cell
                sy = offset_y + y * cell
                if (x, y) in state['walls']:
                    continue
                pygame.draw.rect(screen, COL_FLOOR, (sx, sy, cell, cell))
                pygame.draw.rect(screen, COL_GRID, (sx, sy, cell, cell), 1)

        # Targets (pulsing)
        pulse = math.sin(ticks * 0.003) * 0.3 + 0.7
        for tx, ty in state['targets']:
            if (tx, ty) in state['boxes']:
                continue
            sx = offset_x + tx * cell + cell // 2
            sy = offset_y + ty * cell + cell // 2
            outer_r = int(cell * 0.3 * pulse)
            inner_r = max(3, cell // 8)
            ring_surf = pygame.Surface((outer_r * 2 + 4, outer_r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(ring_surf, (*COL_TARGET, 80),
                               (outer_r + 2, outer_r + 2), outer_r, 2)
            screen.blit(ring_surf, (sx - outer_r - 2, sy - outer_r - 2))
            pygame.draw.circle(screen, COL_TARGET, (sx, sy), inner_r)

        # Walls with 3D bevel
        for wx, wy in state['walls']:
            sx = offset_x + wx * cell
            sy = offset_y + wy * cell
            pygame.draw.rect(screen, COL_WALL, (sx, sy, cell, cell))
            pygame.draw.line(screen, COL_WALL_LIGHT, (sx, sy), (sx + cell - 1, sy), 2)
            pygame.draw.line(screen, COL_WALL_LIGHT, (sx, sy), (sx, sy + cell - 1), 2)
            pygame.draw.line(screen, COL_WALL_DARK,
                             (sx + cell - 1, sy), (sx + cell - 1, sy + cell - 1), 2)
            pygame.draw.line(screen, COL_WALL_DARK,
                             (sx, sy + cell - 1), (sx + cell - 1, sy + cell - 1), 2)

        # Boxes
        for bx, by in state['boxes']:
            if anim_progress < 1.0 and anim_box_from and (bx, by) == anim_box_to:
                continue

            on_target = (bx, by) in state['targets']
            is_dead = (bx, by) in deadlocked_boxes and deadlock_flash > 0

            sx = offset_x + bx * cell
            sy = offset_y + by * cell
            margin = max(2, cell // 10)

            if is_dead and (deadlock_flash // 6) % 2 == 0:
                box_col = COL_RED
                shadow_col = (180, 40, 40)
                hi_col = (255, 100, 100)
            elif on_target:
                box_col = COL_BOX_DONE
                shadow_col = (40, 160, 90)
                hi_col = (100, 230, 150)
            else:
                box_col = COL_BOX
                shadow_col = COL_BOX_SHADOW
                hi_col = COL_BOX_HI

            if on_target and not is_dead:
                glow_surf = pygame.Surface((cell + 8, cell + 8), pygame.SRCALPHA)
                pygame.draw.rect(glow_surf, (*COL_BOX_DONE, 40),
                                 (0, 0, cell + 8, cell + 8), border_radius=10)
                screen.blit(glow_surf, (sx - 4, sy - 4))

            rect = pygame.Rect(sx + margin, sy + margin,
                                cell - margin * 2, cell - margin * 2)
            draw_rounded_rect(screen, box_col, rect, 5)

            inner = pygame.Rect(sx + margin + 3, sy + margin + 3,
                                 cell - margin * 2 - 6, cell - margin * 2 - 6)
            draw_rounded_rect(screen, shadow_col, inner, 4)

            hi_rect = pygame.Rect(sx + margin + 2, sy + margin + 1,
                                   cell - margin * 2 - 4,
                                   max(3, (cell - margin * 2) // 4))
            draw_rounded_rect(screen, hi_col, hi_rect, 3)

            cx = sx + cell // 2
            cy = sy + cell // 2
            ms = max(3, cell // 6)
            if on_target and not is_dead:
                pygame.draw.line(screen, (255, 255, 255),
                                 (cx - ms, cy), (cx - ms // 3, cy + ms), 2)
                pygame.draw.line(screen, (255, 255, 255),
                                 (cx - ms // 3, cy + ms), (cx + ms, cy - ms), 2)
            else:
                pygame.draw.line(screen, (255, 255, 255),
                                 (cx - ms, cy - ms), (cx + ms, cy + ms), 2)
                pygame.draw.line(screen, (255, 255, 255),
                                 (cx - ms, cy + ms), (cx + ms, cy - ms), 2)

        # Animated box
        if anim_progress < 1.0 and anim_box_from and anim_box_to:
            abx = lerp(anim_box_from[0], anim_box_to[0], anim_progress)
            aby = lerp(anim_box_from[1], anim_box_to[1], anim_progress)
            sx = offset_x + abx * cell
            sy = offset_y + aby * cell
            margin = max(2, cell // 10)
            on_tgt = anim_box_to in state['targets']
            bc = COL_BOX_DONE if on_tgt else COL_BOX
            rect = pygame.Rect(int(sx) + margin, int(sy) + margin,
                                cell - margin * 2, cell - margin * 2)
            draw_rounded_rect(screen, bc, rect, 5)

        # Player
        if anim_progress < 1.0:
            ppx = lerp(anim_from[0], anim_to[0], anim_progress)
            ppy = lerp(anim_from[1], anim_to[1], anim_progress)
        else:
            ppx, ppy = state['player']

        pcx = int(offset_x + ppx * cell + cell // 2)
        pcy = int(offset_y + ppy * cell + cell // 2)
        pr = max(8, cell // 2 - 4)

        pygame.draw.circle(screen, COL_PLAYER, (pcx, pcy), pr)
        pygame.draw.circle(screen, (70, 72, 200), (pcx, pcy), pr, 2)

        # Eyes that face movement direction
        eye_off = max(2, pr // 3)
        eye_r = max(2, pr // 5)
        ex1 = pcx + facing[0] * eye_off - facing[1] * eye_off // 2
        ey1 = pcy + facing[1] * eye_off + facing[0] * eye_off // 2
        ex2 = pcx + facing[0] * eye_off + facing[1] * eye_off // 2
        ey2 = pcy + facing[1] * eye_off - facing[0] * eye_off // 2
        pygame.draw.circle(screen, (255, 255, 255), (ex1, ey1), eye_r)
        pygame.draw.circle(screen, (255, 255, 255), (ex2, ey2), eye_r)

        # ---- INFO PANEL ----
        panel_x = GAME_AREA_W
        pygame.draw.rect(screen, COL_PANEL_BG, (panel_x, 0, PANEL_W, WIN_H))
        pygame.draw.line(screen, COL_GRID, (panel_x, 0), (panel_x, WIN_H), 1)

        py_off = 30
        lbl = fonts['big'].render(f"Level {level_index + 1}", True, COL_INDIGO)
        screen.blit(lbl, (panel_x + PANEL_W // 2 - lbl.get_width() // 2, py_off))
        py_off += 55

        moves_surf = fonts['medium'].render(f"Moves: {moves}", True, COL_TEXT)
        screen.blit(moves_surf, (panel_x + 25, py_off))
        py_off += 35

        push_surf = fonts['medium'].render(f"Pushes: {pushes}", True, COL_TEXT)
        screen.blit(push_surf, (panel_x + 25, py_off))
        py_off += 45

        placed = sum(1 for b in state['boxes'] if b in state['targets'])
        total = len(state['targets'])
        box_col_txt = COL_GREEN if placed == total else COL_AMBER
        box_surf = fonts['medium'].render(f"Boxes: {placed}/{total}", True, box_col_txt)
        screen.blit(box_surf, (panel_x + 25, py_off))
        py_off += 35

        # Progress bar
        bar_x = panel_x + 25
        bar_w = PANEL_W - 50
        bar_h = 14
        pygame.draw.rect(screen, (40, 40, 55),
                         (bar_x, py_off, bar_w, bar_h), border_radius=7)
        fill_w = int(bar_w * placed / max(1, total))
        if fill_w > 0:
            pygame.draw.rect(screen, box_col_txt,
                             (bar_x, py_off, fill_w, bar_h), border_radius=7)
        py_off += 40

        # Deadlock warning
        if deadlock_flash > 0:
            dl_surf = fonts['medium'].render("Deadlock!", True, COL_RED)
            screen.blit(dl_surf, (panel_x + PANEL_W // 2 - dl_surf.get_width() // 2, py_off))
        py_off += 40

        # Best score
        solved_dict = progress.get("solved", {})
        best = solved_dict.get(str(level_index))
        if best:
            best_surf = fonts['small'].render(
                f"Best: {best['moves']} moves, {best['pushes']} pushes",
                True, COL_GREEN)
            screen.blit(best_surf, (panel_x + 25, py_off))
        py_off += 40

        # Controls help
        controls = [
            "Controls:",
            "Arrows  Move",
            "U       Undo",
            "R       Restart",
            "N       Next level",
            "P       Prev level",
            "ESC     Level select",
            "Q       Quit",
        ]
        cy_ctrl = WIN_H - 220
        for i, line in enumerate(controls):
            col = COL_TEXT if i == 0 else COL_TEXT_DIM
            s = fonts['tiny'].render(line, True, col)
            screen.blit(s, (panel_x + 25, cy_ctrl + i * 22))

        # ---- SOLVED OVERLAY ----
        if solved:
            overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            screen.blit(overlay, (0, 0))

            solved_text = fonts['title'].render("SOLVED!", True, COL_GREEN)
            screen.blit(solved_text,
                        (WIN_W // 2 - solved_text.get_width() // 2, WIN_H // 2 - 80))

            stats = fonts['medium'].render(
                f"Moves: {moves}   Pushes: {pushes}", True, COL_TEXT)
            screen.blit(stats,
                        (WIN_W // 2 - stats.get_width() // 2, WIN_H // 2 - 10))

            hint_text = fonts['small'].render(
                "Press N or Enter for next level  |  ESC for level select",
                True, COL_TEXT_DIM)
            screen.blit(hint_text,
                        (WIN_W // 2 - hint_text.get_width() // 2, WIN_H // 2 + 40))

            if solved_timer > 2500:
                return 'solved'

        pygame.display.flip()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Sokoban")
    clock = pygame.time.Clock()

    fonts = {
        'title': pygame.font.SysFont("segoeui,arial,helvetica", 48, bold=True),
        'big': pygame.font.SysFont("segoeui,arial,helvetica", 32, bold=True),
        'medium': pygame.font.SysFont("segoeui,arial,helvetica", 22),
        'small': pygame.font.SysFont("segoeui,arial,helvetica", 18),
        'tiny': pygame.font.SysFont("consolas,couriernew,monospace", 16),
    }

    progress = load_progress()

    while True:
        level_index = level_select_screen(screen, clock, progress, fonts)
        if level_index < 0:
            break

        while True:
            progress["current"] = level_index
            result = play_level(screen, clock, level_index, progress, fonts)

            if result == 'quit':
                save_progress(progress)
                pygame.quit()
                sys.exit()
            elif result == 'select':
                break
            elif result in ('next', 'solved'):
                level_index = min(level_index + 1, 59)
            elif result == 'prev':
                level_index = max(level_index - 1, 0)

    save_progress(progress)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
