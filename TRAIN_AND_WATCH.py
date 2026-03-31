"""
TRAIN_AND_WATCH.py - Watch a PPO agent learn Sokoban from scratch.
Completely self-contained. Dependencies: pygame, torch, numpy only.

Controls:
  V     - Visual mode (render every episode, ~2 eps/sec)
  F     - Fast mode (skip rendering, full speed)
  S     - Slow mode (render every step, 200ms delay)
  Space - Pause / Resume
  Q     - Quit
"""

import random, math, time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import pygame

# =====================================================================
# 1. SOKOBAN ENGINE
# =====================================================================

class SokobanState:
    __slots__ = ("width", "height", "walls", "boxes", "targets", "player")

    def __init__(self, w, h, walls, boxes, targets, player):
        self.width = w
        self.height = h
        self.walls = frozenset(walls)
        self.boxes = frozenset(boxes)
        self.targets = frozenset(targets)
        self.player = player

    def move(self, action):
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        nx, ny = self.player[0] + dx, self.player[1] + dy
        if (nx, ny) in self.walls:
            return None
        if (nx, ny) in self.boxes:
            bx, by = nx + dx, ny + dy
            if (bx, by) in self.walls or (bx, by) in self.boxes:
                return None
            new_boxes = (self.boxes - {(nx, ny)}) | {(bx, by)}
            return SokobanState(self.width, self.height, self.walls,
                                new_boxes, self.targets, (nx, ny))
        return SokobanState(self.width, self.height, self.walls,
                            self.boxes, self.targets, (nx, ny))

    def is_solved(self):
        return self.boxes == self.targets

    def is_deadlocked(self):
        for bx, by in self.boxes:
            if (bx, by) in self.targets:
                continue
            wl = (bx - 1, by) in self.walls
            wr = (bx + 1, by) in self.walls
            wu = (bx, by - 1) in self.walls
            wd = (bx, by + 1) in self.walls
            if (wl or wr) and (wu or wd):
                return True
        return False

    def box_distances(self):
        total = 0
        for bx, by in self.boxes:
            total += min(abs(bx - tx) + abs(by - ty) for tx, ty in self.targets)
        return total

    def key(self):
        return (self.player, self.boxes)


def bfs_solve(state, max_states=200000):
    if state.is_solved():
        return True
    visited = {state.key()}
    queue = deque([state])
    while queue and len(visited) < max_states:
        s = queue.popleft()
        for a in range(4):
            ns = s.move(a)
            if ns is None:
                continue
            k = ns.key()
            if k in visited:
                continue
            if ns.is_solved():
                return True
            if ns.is_deadlocked():
                continue
            visited.add(k)
            queue.append(ns)
    return False


# =====================================================================
# 2. LEVEL GENERATOR
# =====================================================================

PHASE_CONFIG = [
    {"min_w": 5, "max_w": 6, "min_h": 5, "max_h": 6, "boxes": 1,
     "max_steps": 150, "name": "Phase 1: One Box"},
    {"min_w": 6, "max_w": 8, "min_h": 6, "max_h": 8, "boxes": 2,
     "max_steps": 250, "name": "Phase 2: Two Boxes"},
    {"min_w": 7, "max_w": 9, "min_h": 7, "max_h": 9, "boxes": 3,
     "max_steps": 400, "name": "Phase 3: Three Boxes"},
]


def generate_level(width, height, n_boxes, seed=None):
    rng = random.Random(seed)
    walls = set()
    for x in range(width):
        walls.add((x, 0))
        walls.add((x, height - 1))
    for y in range(height):
        walls.add((0, y))
        walls.add((width - 1, y))
    interior = [(x, y) for x in range(1, width - 1)
                for y in range(1, height - 1)]
    n_iw = rng.randint(0, max(0, len(interior) // 5 - n_boxes * 2))
    rng.shuffle(interior)
    for i in range(min(n_iw, len(interior))):
        walls.add(interior[i])
    free = [p for p in interior if p not in walls]
    if len(free) < 2 * n_boxes + 1:
        return None
    # flood-fill connectivity check
    start = free[0]
    visited = {start}
    stack = [start]
    while stack:
        cx, cy = stack.pop()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (cx + dx, cy + dy)
            if nb not in visited and nb not in walls \
                    and 0 <= nb[0] < width and 0 <= nb[1] < height:
                visited.add(nb)
                stack.append(nb)
    if len(visited) != len(free):
        return None
    rng.shuffle(free)
    targets = free[:n_boxes]
    boxes_pos = free[n_boxes:2 * n_boxes]
    player = free[2 * n_boxes]
    if set(boxes_pos) & set(targets):
        return None
    state = SokobanState(width, height, walls, boxes_pos, targets, player)
    if state.is_deadlocked():
        return None
    if not bfs_solve(state):
        return None
    return state


def make_level_for_phase(phase_idx):
    cfg = PHASE_CONFIG[phase_idx]
    for _ in range(500):
        w = random.randint(cfg["min_w"], cfg["max_w"])
        h = random.randint(cfg["min_h"], cfg["max_h"])
        s = generate_level(w, h, cfg["boxes"], seed=random.randint(0, 999999))
        if s is not None:
            return s
    return None


# =====================================================================
# 3. GYM ENVIRONMENT
# =====================================================================

class SokobanEnv:
    OBS_W, OBS_H = 10, 10

    def __init__(self):
        self.phase = 0
        self.state = None
        self.steps = 0
        self.prev_dist = 0
        self.prev_on = 0

    def reset(self):
        self.state = make_level_for_phase(self.phase)
        if self.state is None:
            self.state = make_level_for_phase(0)
        self.steps = 0
        self.prev_dist = self.state.box_distances()
        self.prev_on = len(self.state.boxes & self.state.targets)
        return self._obs()

    def _obs(self):
        obs = np.zeros((5, self.OBS_H, self.OBS_W), dtype=np.float32)
        s = self.state
        for x, y in s.walls:
            if x < self.OBS_W and y < self.OBS_H:
                obs[0, y, x] = 1.0
        for x, y in s.boxes - s.targets:
            if x < self.OBS_W and y < self.OBS_H:
                obs[1, y, x] = 1.0
        for x, y in s.targets - s.boxes:
            if x < self.OBS_W and y < self.OBS_H:
                obs[2, y, x] = 1.0
        px, py = s.player
        if px < self.OBS_W and py < self.OBS_H:
            obs[3, py, px] = 1.0
        for x, y in s.boxes & s.targets:
            if x < self.OBS_W and y < self.OBS_H:
                obs[4, y, x] = 1.0
        return obs

    def step(self, action):
        self.steps += 1
        ns = self.state.move(action)
        max_steps = PHASE_CONFIG[self.phase]["max_steps"]
        if ns is None:
            done = self.steps >= max_steps
            return self._obs(), -1.0, done, {"invalid": True}
        self.state = ns
        if ns.is_solved():
            return self._obs(), 100.0, True, {"solved": True}
        if ns.is_deadlocked():
            return self._obs(), -50.0, True, {"deadlock": True}
        reward = -0.1
        new_dist = ns.box_distances()
        reward += float(self.prev_dist - new_dist)
        self.prev_dist = new_dist
        new_on = len(ns.boxes & ns.targets)
        if new_on > self.prev_on:
            reward += 10.0 * (new_on - self.prev_on)
        elif new_on < self.prev_on:
            reward -= 10.0 * (self.prev_on - new_on)
        self.prev_on = new_on
        done = self.steps >= max_steps
        return self._obs(), reward, done, {}


# =====================================================================
# 4. NEURAL NETWORK
# =====================================================================

class SokobanNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 10 * 10, 512)
        self.policy_head = nn.Linear(512, 4)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        policy = F.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x)
        return policy, value


# =====================================================================
# 5. PPO AGENT
# =====================================================================

class PPOAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = SokobanNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=2.5e-4,
                                          eps=1e-5)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_eps = 0.2
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.ppo_epochs = 4
        self.mini_batch_size = 128
        self.steps_per_update = 512
        # buffers
        self.buf_obs = []
        self.buf_act = []
        self.buf_rew = []
        self.buf_lp = []
        self.buf_val = []
        self.buf_done = []
        # tracking
        self.total_steps = 0
        self.updates = 0
        self.last_loss = 0.0

    def encode_state(self, obs):
        return torch.FloatTensor(obs).unsqueeze(0).to(self.device)

    def select_action(self, obs):
        with torch.no_grad():
            probs, value = self.net(self.encode_state(obs))
            dist = Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), value.item()

    def get_action_probs(self, obs):
        with torch.no_grad():
            probs, _ = self.net(self.encode_state(obs))
            return probs.cpu().numpy()[0]

    def store_transition(self, obs, action, reward, log_prob, value, done):
        self.buf_obs.append(obs.copy())
        self.buf_act.append(action)
        self.buf_rew.append(reward)
        self.buf_lp.append(log_prob)
        self.buf_val.append(value)
        self.buf_done.append(done)
        self.total_steps += 1

    def ready_to_update(self):
        return len(self.buf_obs) >= self.steps_per_update

    def compute_gae(self):
        n = len(self.buf_rew)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            if self.buf_done[t]:
                next_val = 0.0
                last_gae = 0.0
            else:
                next_val = self.buf_val[t + 1] if t + 1 < n else 0.0
            delta = self.buf_rew[t] + self.gamma * next_val - self.buf_val[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae
        returns = advantages + np.array(self.buf_val, dtype=np.float32)
        return advantages, returns

    def update(self):
        if not self.buf_obs:
            return {}
        advantages, returns = self.compute_gae()
        states_t = torch.FloatTensor(np.array(self.buf_obs)).to(self.device)
        actions_t = torch.LongTensor(self.buf_act).to(self.device)
        old_lp_t = torch.FloatTensor(self.buf_lp).to(self.device)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        n = len(self.buf_obs)
        total_loss = 0.0
        count = 0
        for _ in range(self.ppo_epochs):
            idx = np.arange(n)
            np.random.shuffle(idx)
            for start in range(0, n, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n)
                mb = idx[start:end]
                probs, values = self.net(states_t[mb])
                dist = Categorical(probs)
                new_lp = dist.log_prob(actions_t[mb])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_lp - old_lp_t[mb])
                s1 = ratio * adv_t[mb]
                s2 = torch.clamp(ratio, 1 - self.clip_eps,
                                 1 + self.clip_eps) * adv_t[mb]
                pi_loss = -torch.min(s1, s2).mean()
                v_loss = F.mse_loss(values.squeeze(-1), ret_t[mb])
                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()
                count += 1
        self.buf_obs.clear()
        self.buf_act.clear()
        self.buf_rew.clear()
        self.buf_lp.clear()
        self.buf_val.clear()
        self.buf_done.clear()
        self.updates += 1
        self.last_loss = total_loss / max(count, 1)
        return {"loss": self.last_loss}


# =====================================================================
# 6. PYGAME VISUALIZATION
# =====================================================================

WIN_W, WIN_H = 1000, 700
BG = (18, 18, 30)
PANEL_BG = (25, 25, 40)
WALL_COL = (65, 70, 85)
WALL_LIGHT = (90, 95, 110)
WALL_DARK = (45, 48, 60)
PLAYER_COL = (99, 102, 241)
BOX_COL = (240, 180, 40)
BOX_ON_COL = (60, 200, 120)
TARGET_COL = (52, 211, 153)
FLOOR_COL = (30, 32, 46)
TEXT_COL = (220, 220, 230)
DIM_TEXT = (130, 130, 150)
ACCENT = (99, 102, 241)
GRAPH_BG = (30, 30, 48)
ACTION_NAMES = ["UP", "DN", "LT", "RT"]


def draw_grid(surf, state, area_x, area_y, area_w, area_h, pulse_t):
    if state is None:
        return
    gw, gh = state.width, state.height
    cell = min(area_w // gw, area_h // gh)
    ox = area_x + (area_w - gw * cell) // 2
    oy = area_y + (area_h - gh * cell) // 2
    on_target = state.boxes & state.targets
    off_boxes = state.boxes - state.targets
    empty_targets = state.targets - state.boxes

    # floor and walls
    for y in range(gh):
        for x in range(gw):
            r = pygame.Rect(ox + x * cell, oy + y * cell, cell, cell)
            if (x, y) in state.walls:
                pygame.draw.rect(surf, WALL_COL, r)
                pygame.draw.line(surf, WALL_LIGHT, r.topleft, r.topright, 2)
                pygame.draw.line(surf, WALL_LIGHT, r.topleft, r.bottomleft, 2)
                pygame.draw.line(surf, WALL_DARK, r.bottomleft, r.bottomright, 2)
                pygame.draw.line(surf, WALL_DARK, r.topright, r.bottomright, 2)
            else:
                pygame.draw.rect(surf, FLOOR_COL, r)
                pygame.draw.rect(surf, (40, 42, 56), r, 1)

    # targets (pulsing concentric circles)
    for tx, ty in empty_targets:
        cx = ox + tx * cell + cell // 2
        cy = oy + ty * cell + cell // 2
        pulse = int(3 * math.sin(pulse_t * 3))
        for i, alpha in enumerate([40, 80, 160]):
            rad = cell // 4 + 2 - i + pulse
            if rad > 0:
                c = (TARGET_COL[0] * alpha // 255,
                     TARGET_COL[1] * alpha // 255,
                     TARGET_COL[2] * alpha // 255)
                pygame.draw.circle(surf, c, (cx, cy), rad, 2)

    # boxes not on target
    for bx, by in off_boxes:
        r = pygame.Rect(ox + bx * cell + 3, oy + by * cell + 3,
                        cell - 6, cell - 6)
        pygame.draw.rect(surf, BOX_COL, r, border_radius=4)
        sr = pygame.Rect(r.x + 3, r.y + 3, r.w - 6, r.h - 6)
        dark = (max(BOX_COL[0] - 50, 0), max(BOX_COL[1] - 50, 0),
                max(BOX_COL[2] - 50, 0))
        pygame.draw.rect(surf, dark, sr, border_radius=3)

    # boxes on target (with glow)
    for bx, by in on_target:
        r = pygame.Rect(ox + bx * cell + 3, oy + by * cell + 3,
                        cell - 6, cell - 6)
        glow_r = pygame.Rect(r.x - 2, r.y - 2, r.w + 4, r.h + 4)
        glow = (BOX_ON_COL[0] // 2, BOX_ON_COL[1] // 2, BOX_ON_COL[2] // 2)
        pygame.draw.rect(surf, glow, glow_r, border_radius=6)
        pygame.draw.rect(surf, BOX_ON_COL, r, border_radius=4)
        dark = (max(BOX_ON_COL[0] - 40, 0), max(BOX_ON_COL[1] - 40, 0),
                max(BOX_ON_COL[2] - 40, 0))
        sr = pygame.Rect(r.x + 3, r.y + 3, r.w - 6, r.h - 6)
        pygame.draw.rect(surf, dark, sr, border_radius=3)

    # player (circle with eyes)
    px, py = state.player
    cx = ox + px * cell + cell // 2
    cy = oy + py * cell + cell // 2
    rad = cell // 2 - 4
    pygame.draw.circle(surf, PLAYER_COL, (cx, cy), rad)
    er = max(rad // 5, 2)
    pygame.draw.circle(surf, (255, 255, 255),
                       (cx - rad // 3, cy - rad // 4), er)
    pygame.draw.circle(surf, (255, 255, 255),
                       (cx + rad // 3, cy - rad // 4), er)
    pygame.draw.circle(surf, (20, 20, 40),
                       (cx - rad // 3, cy - rad // 4), max(er // 2, 1))
    pygame.draw.circle(surf, (20, 20, 40),
                       (cx + rad // 3, cy - rad // 4), max(er // 2, 1))


def draw_panel(surf, stats, agent, obs, fonts, graph_data, pulse_t):
    px = 630
    pygame.draw.rect(surf, PANEL_BG, (620, 0, 380, WIN_H))
    y = 12
    f_big, f_med, f_sm = fonts

    # Episode
    t = f_med.render(f"Episode: {stats['episode']}", True, TEXT_COL)
    surf.blit(t, (px, y))
    y += 32

    # Phase
    t = f_sm.render(PHASE_CONFIG[stats['phase']]['name'], True, ACCENT)
    surf.blit(t, (px, y))
    y += 26

    # Solve rate with colour coding
    sr = stats['solve_rate']
    if sr < 10:
        sc = (230, 60, 60)
    elif sr < 30:
        sc = (230, 140, 40)
    elif sr < 60:
        sc = (220, 200, 50)
    elif sr < 80:
        sc = (80, 200, 120)
    else:
        sc = ACCENT
    t = f_sm.render("Solve Rate:", True, DIM_TEXT)
    surf.blit(t, (px, y))
    t = f_big.render(f"{sr:.1f}%", True, sc)
    surf.blit(t, (px + 140, y - 6))
    y += 42

    # Avg reward, avg steps, deadlock rate
    for label, key, fmt in [("Avg Reward", "avg_reward", ".1f"),
                             ("Avg Steps", "avg_steps", ".0f"),
                             ("Deadlock %", "deadlock_rate", ".1f")]:
        t = f_sm.render(f"{label}: {stats[key]:{fmt}}", True, DIM_TEXT)
        surf.blit(t, (px, y))
        y += 22
    y += 8

    # Learning curve graph
    gx, gy, gw, gh = px, y, 320, 210
    pygame.draw.rect(surf, GRAPH_BG, (gx, gy, gw, gh), border_radius=6)
    t = f_sm.render("Solve Rate History", True, DIM_TEXT)
    surf.blit(t, (gx + 4, gy + 2))
    # dashed 50% line
    mid_y = gy + 20 + (gh - 30) // 2
    for dx in range(0, gw - 40, 8):
        pygame.draw.line(surf, (60, 60, 80),
                         (gx + 30 + dx, mid_y), (gx + 30 + dx + 4, mid_y))
    t = f_sm.render("50%", True, (80, 80, 100))
    surf.blit(t, (gx + 2, mid_y - 7))
    # plot data
    if len(graph_data) > 1:
        max_ep = max(1, graph_data[-1][0])
        plot_x, plot_y2 = gx + 30, gy + 20
        plot_w, plot_h = gw - 40, gh - 30
        pts = []
        for ep, val in graph_data:
            ppx = plot_x + int(ep / max_ep * plot_w)
            ppy = plot_y2 + plot_h - int(val / 100.0 * plot_h)
            pts.append((ppx, ppy))
        if len(pts) >= 2:
            pygame.draw.lines(surf, ACCENT, False, pts, 2)
    y += gh + 10

    # Action probability bars
    if obs is not None:
        probs = agent.get_action_probs(obs)
        t = f_sm.render("Action Probs:", True, DIM_TEXT)
        surf.blit(t, (px, y))
        y += 20
        bar_w = 180
        for i in range(4):
            lbl = f_sm.render(ACTION_NAMES[i], True, TEXT_COL)
            surf.blit(lbl, (px, y + 1))
            bx = px + 35
            pygame.draw.rect(surf, (40, 40, 55),
                             (bx, y, bar_w, 16), border_radius=3)
            fw = int(probs[i] * bar_w)
            if fw > 0:
                pygame.draw.rect(surf, ACCENT,
                                 (bx, y, fw, 16), border_radius=3)
            pct = f_sm.render(f"{probs[i] * 100:.0f}%", True, TEXT_COL)
            surf.blit(pct, (bx + bar_w + 5, y + 1))
            y += 20
        y += 6

    # Training info
    t = f_sm.render(f"Total Steps: {agent.total_steps}", True, DIM_TEXT)
    surf.blit(t, (px, y))
    y += 20
    t = f_sm.render(f"Updates: {agent.updates}", True, DIM_TEXT)
    surf.blit(t, (px, y))
    y += 20
    t = f_sm.render(f"Loss: {agent.last_loss:.4f}", True, DIM_TEXT)
    surf.blit(t, (px, y))


def draw_controls(surf, font, mode):
    bar_y = WIN_H - 30
    pygame.draw.rect(surf, (20, 20, 35), (0, bar_y, WIN_W, 30))
    labels = ["[V]isual", "[F]ast", "[S]low", "[Space]Pause", "[Q]uit"]
    x = 10
    for lb in labels:
        active = ((lb.startswith("[V]") and mode == "visual")
                  or (lb.startswith("[F]") and mode == "fast")
                  or (lb.startswith("[S]") and mode == "slow")
                  or (lb.startswith("[Space]") and mode == "paused"))
        col = ACCENT if active else DIM_TEXT
        t = font.render(lb, True, col)
        surf.blit(t, (x, bar_y + 6))
        x += t.get_width() + 20


# =====================================================================
# 7. MAIN TRAINING LOOP
# =====================================================================

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Train & Watch - PPO learns Sokoban")
    clock = pygame.time.Clock()

    f_big = pygame.font.SysFont("consolas", 36, bold=True)
    f_med = pygame.font.SysFont("consolas", 26)
    f_sm = pygame.font.SysFont("consolas", 16)
    f_overlay = pygame.font.SysFont("consolas", 48, bold=True)
    fonts = (f_big, f_med, f_sm)

    agent = PPOAgent()
    env = SokobanEnv()

    episode = 0
    recent_solves = deque(maxlen=100)
    recent_rewards = deque(maxlen=100)
    recent_steps = deque(maxlen=100)
    recent_deadlocks = deque(maxlen=100)
    graph_data = []

    mode = "visual"
    running = True
    obs = env.reset()
    ep_reward = 0.0
    ep_steps = 0

    phase_flash_text = None
    phase_flash_end = 0.0

    def get_stats():
        sr = (sum(recent_solves) / max(len(recent_solves), 1)) * 100
        ar = sum(recent_rewards) / max(len(recent_rewards), 1)
        ast = sum(recent_steps) / max(len(recent_steps), 1)
        dr = (sum(recent_deadlocks) / max(len(recent_deadlocks), 1)) * 100
        return {"episode": episode, "phase": env.phase,
                "solve_rate": sr, "avg_reward": ar,
                "avg_steps": ast, "deadlock_rate": dr}

    def check_phase():
        nonlocal phase_flash_text, phase_flash_end
        if len(recent_solves) < 20:
            return
        sr = sum(recent_solves) / len(recent_solves)
        if env.phase == 0 and sr > 0.70:
            env.phase = 1
            recent_solves.clear()
            phase_flash_text = "PHASE 2: Two Boxes!"
            phase_flash_end = time.time() + 2.0
        elif env.phase == 1 and sr > 0.50:
            env.phase = 2
            recent_solves.clear()
            phase_flash_text = "PHASE 3: Three Boxes!"
            phase_flash_end = time.time() + 2.0

    def finish_episode(solved, deadlocked):
        nonlocal episode
        episode += 1
        recent_solves.append(1 if solved else 0)
        recent_rewards.append(ep_reward)
        recent_steps.append(ep_steps)
        recent_deadlocks.append(1 if deadlocked else 0)
        if episode % 50 == 0:
            sr = (sum(recent_solves) / max(len(recent_solves), 1)) * 100
            graph_data.append((episode, sr))
        check_phase()

    def run_one_step():
        nonlocal obs, ep_reward, ep_steps
        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        agent.store_transition(obs, action, reward, log_prob, value, done)
        ep_reward += reward
        ep_steps += 1
        if done:
            finish_episode(info.get("solved", False),
                           info.get("deadlock", False))
            obs = env.reset()
            ep_reward = 0.0
            ep_steps = 0
            if agent.ready_to_update():
                agent.update()
            return True
        else:
            obs = next_obs
            if agent.ready_to_update():
                agent.update()
            return False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_v:
                    mode = "visual"
                elif event.key == pygame.K_f:
                    mode = "fast"
                elif event.key == pygame.K_s:
                    mode = "slow"
                elif event.key == pygame.K_SPACE:
                    mode = "paused" if mode != "paused" else "visual"

        # advance training
        if mode == "paused":
            pass
        elif mode == "slow":
            run_one_step()
        elif mode == "visual":
            for _ in range(300):
                ended = run_one_step()
                if ended:
                    break
        elif mode == "fast":
            for _ in range(2000):
                run_one_step()

        # render
        t_now = time.time()
        pulse_t = t_now % (2 * math.pi / 3) * 3
        screen.fill(BG)

        if mode == "fast":
            ft = f_overlay.render("FAST MODE", True, ACCENT)
            r = ft.get_rect(center=(310, 300))
            screen.blit(ft, r)
            et = f_med.render(f"Episode {episode}", True, DIM_TEXT)
            screen.blit(et, (310 - et.get_width() // 2, 350))
        else:
            draw_grid(screen, env.state, 10, 10, 600, 600, pulse_t)
            pname = PHASE_CONFIG[env.phase]["name"]
            pt = f_sm.render(pname, True, DIM_TEXT)
            screen.blit(pt, (310 - pt.get_width() // 2, 618))

        draw_panel(screen, get_stats(), agent, obs, fonts, graph_data, pulse_t)
        draw_controls(screen, f_sm, mode)

        # phase flash overlay
        if phase_flash_text and t_now < phase_flash_end:
            overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            screen.blit(overlay, (0, 0))
            ft = f_overlay.render(phase_flash_text, True, (80, 255, 160))
            r = ft.get_rect(center=(WIN_W // 2, WIN_H // 2))
            screen.blit(ft, r)
        elif phase_flash_text and t_now >= phase_flash_end:
            phase_flash_text = None

        # pause overlay
        if mode == "paused":
            overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 100))
            screen.blit(overlay, (0, 0))
            pt = f_overlay.render("PAUSED", True, (255, 255, 255))
            r = pt.get_rect(center=(WIN_W // 2, WIN_H // 2))
            screen.blit(pt, r)

        pygame.display.flip()

        if mode == "slow":
            pygame.time.wait(200)
        else:
            clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
