"""COMPARE.py -- Side-by-side PPO vs ThinkerAgent learning Sokoban.

Completely self-contained. Only dependencies: pygame, torch, numpy.
Both agents train simultaneously on identical puzzles (same seed).

Controls:
  V      -- Visual mode (1 episode per frame each)
  F      -- Fast mode   (5 episodes per frame each)
  S      -- Slow mode   (1 episode per frame, 200ms delay)
  Space  -- Pause / Resume
  ESC    -- Quit
"""

from __future__ import annotations

import random
import time
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# =====================================================================
# 1. SOKOBAN ENGINE
# =====================================================================

DIR_DELTAS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


class SokobanState:
    """Immutable Sokoban game state with move/solved/deadlock/distances."""

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
            return SokobanState(self.walls, new_boxes, self.targets, (nx, ny), self.width, self.height)
        return SokobanState(self.walls, self.boxes, self.targets, (nx, ny), self.width, self.height)

    @property
    def solved(self) -> bool:
        return self.boxes == self.targets

    @property
    def n_boxes_on_target(self) -> int:
        return len(self.boxes & self.targets)

    def is_deadlocked(self) -> bool:
        for bx, by in self.boxes:
            if (bx, by) in self.targets:
                continue
            wu = (bx, by - 1) in self.walls
            wd = (bx, by + 1) in self.walls
            wl = (bx - 1, by) in self.walls
            wr = (bx + 1, by) in self.walls
            if (wu or wd) and (wl or wr):
                return True
        return False

    def box_distances(self) -> float:
        total = 0.0
        for bx, by in self.boxes:
            if (bx, by) in self.targets:
                continue
            dists = [abs(bx - tx) + abs(by - ty) for tx, ty in self.targets
                     if (tx, ty) not in (self.boxes - {(bx, by)})]
            total += min(dists) if dists else 100.0
        return total

    def clone(self) -> "SokobanState":
        return SokobanState(self.walls, self.boxes, self.targets, self.player, self.width, self.height)

    def key(self) -> Tuple:
        return (self.player, self.boxes)


def bfs_solve(state: SokobanState, max_states: int = 300_000) -> Optional[List[int]]:
    """BFS solver. Returns action list or None."""
    if state.solved:
        return []
    queue: deque = deque([(state, [])])
    visited = {state.key()}
    while queue and len(visited) < max_states:
        cur, moves = queue.popleft()
        for a in range(4):
            nxt = cur.move(a)
            if nxt is None:
                continue
            k = nxt.key()
            if k in visited:
                continue
            if nxt.is_deadlocked():
                continue
            visited.add(k)
            nm = moves + [a]
            if nxt.solved:
                return nm
            queue.append((nxt, nm))
    return None


# =====================================================================
# 2. LEVEL GENERATOR + ENV
# =====================================================================

def _is_connected(cells: List[Tuple[int, int]], walls: Set[Tuple[int, int]]) -> bool:
    if not cells:
        return True
    start = cells[0]
    visited = {start}
    q = deque([start])
    cs = set(cells)
    while q:
        x, y = q.popleft()
        for dx, dy in DIR_DELTAS:
            nb = (x + dx, y + dy)
            if nb in cs and nb not in visited and nb not in walls:
                visited.add(nb)
                q.append(nb)
    return len(visited) >= len(cs)


def generate_level(seed: int, width: int = 0, height: int = 0, n_boxes: int = 1,
                   max_retries: int = 500) -> Optional[Tuple[SokobanState, List[int]]]:
    """Generate a random solvable level with the given seed."""
    rng = random.Random(seed)
    if width == 0:
        width = rng.choice([5, 6, 7])
    if height == 0:
        height = rng.choice([5, 6, 7])

    for _ in range(max_retries):
        walls: Set[Tuple[int, int]] = set()
        for x in range(width):
            walls.add((x, 0)); walls.add((x, height - 1))
        for y in range(height):
            walls.add((0, y)); walls.add((width - 1, y))
        interior = [(x, y) for x in range(1, width - 1) for y in range(1, height - 1)]
        rng.shuffle(interior)
        n_iw = rng.randint(0, max(1, len(interior) // 5))
        for w in interior[:n_iw]:
            walls.add(w)
        free = [p for p in interior if p not in walls]
        if len(free) < n_boxes * 2 + 1:
            continue
        if not _is_connected(free, walls):
            continue
        rng.shuffle(free)
        targets = set(free[:n_boxes])
        if len(targets) < n_boxes:
            continue
        remaining = [p for p in free if p not in targets]
        rng.shuffle(remaining)
        boxes: Set[Tuple[int, int]] = set()
        for p in remaining:
            if len(boxes) >= n_boxes:
                break
            x, y = p
            wu = (x, y - 1) in walls; wd = (x, y + 1) in walls
            wl = (x - 1, y) in walls; wr = (x + 1, y) in walls
            if (wu or wd) and (wl or wr):
                continue
            boxes.add(p)
        if len(boxes) < n_boxes:
            continue
        player_spots = [p for p in free if p not in targets and p not in boxes]
        if not player_spots:
            continue
        player = rng.choice(player_spots)
        st = SokobanState(frozenset(walls), frozenset(boxes), frozenset(targets),
                          player, width, height)
        sol = bfs_solve(st, max_states=200_000)
        if sol is not None and 3 <= len(sol) <= 30:
            return st, sol
    return None


class SokobanEnv:
    """Lightweight Sokoban gym-like env with 5-channel obs and reward shaping."""

    def __init__(self, max_h: int = 10, max_w: int = 10, max_steps: int = 200) -> None:
        self.max_h = max_h
        self.max_w = max_w
        self.max_steps = max_steps
        self.state: Optional[SokobanState] = None
        self._steps = 0
        self._prev_dist = 0.0
        self._gen_seed = 2000

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._gen_seed = seed
        result = None
        while result is None:
            result = generate_level(self._gen_seed)
            self._gen_seed += 1
        self.state = result[0]
        self._steps = 0
        self._prev_dist = self.state.box_distances()
        return self._obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.state is not None
        self._steps += 1
        old_on = self.state.n_boxes_on_target
        ns = self.state.move(action)
        if ns is None:
            trunc = self._steps >= self.max_steps
            return self._obs(), -1.0, False, trunc, {"solved": False, "deadlock": False}
        self.state = ns
        new_on = self.state.n_boxes_on_target
        new_dist = self.state.box_distances()
        reward = -0.1
        if self.state.solved:
            return self._obs(), reward + 100.0, True, False, {"solved": True, "deadlock": False}
        if self.state.is_deadlocked():
            return self._obs(), reward - 50.0, True, False, {"solved": False, "deadlock": True}
        if new_on > old_on:
            reward += 10.0
        elif new_on < old_on:
            reward -= 10.0
        dd = self._prev_dist - new_dist
        if dd > 0.5:
            reward += 1.0
        elif dd < -0.5:
            reward -= 1.0
        self._prev_dist = new_dist
        trunc = self._steps >= self.max_steps
        return self._obs(), reward, False, trunc, {"solved": False, "deadlock": False}

    def _obs(self) -> np.ndarray:
        obs = np.zeros((5, self.max_h, self.max_w), dtype=np.float32)
        s = self.state
        if s is None:
            return obs
        for x, y in s.walls:
            if y < self.max_h and x < self.max_w:
                obs[0, y, x] = 1.0
        for x, y in s.boxes:
            if y < self.max_h and x < self.max_w:
                if (x, y) in s.targets:
                    obs[4, y, x] = 1.0
                else:
                    obs[1, y, x] = 1.0
        for x, y in s.targets:
            if y < self.max_h and x < self.max_w:
                if (x, y) not in s.boxes:
                    obs[2, y, x] = 1.0
        px, py = s.player
        if py < self.max_h and px < self.max_w:
            obs[3, py, px] = 1.0
        return obs


# =====================================================================
# 3. PPO AGENT (SokobanNet + full PPO)
# =====================================================================

class SokobanNet(nn.Module):
    def __init__(self, grid_h: int = 10, grid_w: int = 10) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        flat = 64 * grid_h * grid_w
        self.fc = nn.Sequential(nn.Linear(flat, 512), nn.ReLU())
        self.policy_head = nn.Linear(512, 4)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).flatten(1)
        h = self.fc(h)
        return F.softmax(self.policy_head(h), dim=-1), self.value_head(h)


class PPOAgent:
    def __init__(self, grid_h: int = 10, grid_w: int = 10, lr: float = 2.5e-4) -> None:
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = SokobanNet(grid_h, grid_w).to(self.dev)
        self.opt = Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.gamma = 0.99
        self.gae_lam = 0.95
        self.clip_eps = 0.2
        self.ent_coef = 0.01
        self.val_coef = 0.5
        self.epochs = 4
        self.mb_size = 256
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        self.net.eval()
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=self.dev).unsqueeze(0)
            probs, val = self.net(t)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(val.item())

    def store(self, s, a, r, lp, v, d):
        self.states.append(s); self.actions.append(a); self.rewards.append(r)
        self.log_probs.append(lp); self.values.append(v); self.dones.append(d)

    def update(self) -> None:
        if not self.states:
            return
        self.net.train()
        d = self.dev
        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=d)
        actions = torch.tensor(self.actions, dtype=torch.long, device=d)
        old_lp = torch.tensor(self.log_probs, dtype=torch.float32, device=d)
        rews = np.array(self.rewards, dtype=np.float32)
        vals = np.array(self.values, dtype=np.float32)
        dns = np.array(self.dones, dtype=np.float32)
        n = len(rews)
        adv = np.zeros(n, dtype=np.float32)
        lg = 0.0
        for t in reversed(range(n)):
            nv = 0.0 if t == n - 1 else vals[t + 1]
            nt = 1.0 - dns[t]
            delta = rews[t] + self.gamma * nv * nt - vals[t]
            lg = delta + self.gamma * self.gae_lam * nt * lg
            adv[t] = lg
        rets = adv + vals
        adv_t = torch.tensor(adv, dtype=torch.float32, device=d)
        ret_t = torch.tensor(rets, dtype=torch.float32, device=d)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ns = len(self.states)
        idx = np.arange(ns)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, ns, self.mb_size):
                bi = idx[s:s + self.mb_size]
                bt = torch.tensor(bi, dtype=torch.long, device=d)
                pr, vl = self.net(states[bt])
                dist = torch.distributions.Categorical(pr)
                nlp = dist.log_prob(actions[bt])
                ent = dist.entropy().mean()
                ratio = torch.exp(nlp - old_lp[bt])
                s1 = ratio * adv_t[bt]
                s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[bt]
                pl = -torch.min(s1, s2).mean()
                vl_loss = F.mse_loss(vl.squeeze(-1), ret_t[bt])
                loss = pl + self.val_coef * vl_loss - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
        self.states.clear(); self.actions.clear(); self.rewards.clear()
        self.log_probs.clear(); self.values.clear(); self.dones.clear()


# =====================================================================
# 4. THINKER AGENT (PPO + 1-step lookahead)
# =====================================================================

class ThinkerAgent:
    """PPO policy + 1-step lookahead using SokobanState.move().

    For each of 4 actions, simulates on a copy of SokobanState and
    estimates the reward. Picks the action with the best estimated reward.
    Falls back to PPO policy when all actions give the same reward.
    """

    def __init__(self, grid_h: int = 10, grid_w: int = 10) -> None:
        self.ppo = PPOAgent(grid_h, grid_w)
        self.world_model_accuracy = 0.0
        self.planning_active = False
        self._steps_collected = 0

    @staticmethod
    def _estimate_reward(state: SokobanState, action: int) -> float:
        old_on = state.n_boxes_on_target
        old_dist = state.box_distances()
        ns = state.move(action)
        if ns is None:
            return -1.0
        r = -0.1
        if ns.solved:
            return r + 100.0
        if ns.is_deadlocked():
            return r - 50.0
        new_on = ns.n_boxes_on_target
        if new_on > old_on:
            r += 10.0
        elif new_on < old_on:
            r -= 10.0
        dd = old_dist - ns.box_distances()
        if dd > 0.5:
            r += 1.0
        elif dd < -0.5:
            r -= 1.0
        return r

    def select_action(self, obs: np.ndarray, env_state: Optional[SokobanState]
                      ) -> Tuple[int, float, float]:
        if self.planning_active and env_state is not None:
            rewards = [self._estimate_reward(env_state, a) for a in range(4)]
            best = max(rewards)
            if rewards.count(best) < len(rewards):
                best_a = int(np.argmax(rewards))
                _, lp, v = self.ppo.select_action(obs)
                return best_a, lp, v
        return self.ppo.select_action(obs)

    def store_and_learn(self, obs, a, r, lp, v, done) -> None:
        self.ppo.store(obs, a, r, lp, v, done)
        self._steps_collected += 1
        if self._steps_collected >= 1024:
            self.ppo.update()
            self._steps_collected = 0
            self.planning_active = True
            self.world_model_accuracy = min(0.99, self.world_model_accuracy + 0.05)


# =====================================================================
# 5. AGENT RUNNER (env + agent + stats bookkeeping)
# =====================================================================

class AgentRunner:
    def __init__(self, name: str, color: Tuple[int, int, int], is_thinker: bool = False):
        self.name = name
        self.color = color
        self.is_thinker = is_thinker
        self.env = SokobanEnv()
        self.agent: ThinkerAgent | PPOAgent = ThinkerAgent() if is_thinker else PPOAgent()
        self.obs = np.zeros(1)
        self.ep_reward = 0.0
        self.episode = 0
        self._steps_batch = 0
        self.all_solved: List[bool] = []
        self.all_steps: List[int] = []
        self.deadlocks = 0
        self._ep_steps = 0
        self._reset()

    def _reset(self, seed: Optional[int] = None) -> None:
        self.obs, _ = self.env.reset(seed=seed)
        self.ep_reward = 0.0
        self._ep_steps = 0

    @property
    def state(self) -> Optional[SokobanState]:
        return self.env.state

    @property
    def solve_rate(self) -> float:
        if not self.all_solved:
            return 0.0
        r = self.all_solved[-100:]
        return sum(r) / len(r) * 100

    @property
    def deadlock_rate(self) -> float:
        return (self.deadlocks / self.episode * 100) if self.episode > 0 else 0.0

    @property
    def avg_steps(self) -> float:
        if not self.all_steps:
            return 0.0
        r = self.all_steps[-100:]
        return sum(r) / len(r)

    def run_episode(self, level_seed: Optional[int] = None) -> None:
        self._reset(seed=level_seed)
        while True:
            if self.is_thinker:
                ta: ThinkerAgent = self.agent  # type: ignore
                a, lp, v = ta.select_action(self.obs, self.state)
            else:
                pa: PPOAgent = self.agent  # type: ignore
                a, lp, v = pa.select_action(self.obs)
            nobs, rew, term, trunc, info = self.env.step(a)
            done = term or trunc
            self._ep_steps += 1
            if self.is_thinker:
                ta = self.agent  # type: ignore
                ta.store_and_learn(self.obs, a, rew, lp, v, done)
            else:
                pa = self.agent  # type: ignore
                pa.store(self.obs, a, rew, lp, v, done)
                self._steps_batch += 1
            self.obs = nobs
            self.ep_reward += rew
            if done:
                break
        # finish episode
        self.all_solved.append(info.get("solved", False))
        self.all_steps.append(self._ep_steps)
        if info.get("deadlock", False):
            self.deadlocks += 1
        self.episode += 1
        if not self.is_thinker and self._steps_batch >= 1024:
            pa = self.agent  # type: ignore
            pa.update()
            self._steps_batch = 0


# =====================================================================
# 6. VISUALIZATION (1300x750)
# =====================================================================

WIN_W, WIN_H = 1300, 750

# Colors
C_BG      = (22, 22, 35)
C_PANEL   = (25, 25, 40)
C_GRID_BG = (30, 30, 48)
C_GRIDLN  = (40, 40, 55)
C_WALL    = (75, 85, 99)
C_WALL_HI = (95, 105, 119)
C_BOX     = (251, 191, 36)
C_BOX_ON  = (52, 211, 153)
C_TARGET  = (52, 211, 153)
C_PLAYER  = (99, 102, 241)
C_PLIGHT  = (140, 143, 255)
C_WHITE   = (220, 220, 235)
C_DIM     = (120, 120, 140)
C_RED     = (239, 68, 68)
C_BLUE    = (99, 102, 241)
C_GREEN   = (52, 211, 153)
C_YELLOW  = (251, 191, 36)

# Layout rects
TITLE_RECT  = pygame.Rect(0, 0, WIN_W, 40)
LEFT_GRID   = pygame.Rect(20, 50, 460, 440)
RIGHT_GRID  = pygame.Rect(820, 50, 460, 440)
CENTER_PANEL = pygame.Rect(500, 50, 300, 440)
GRAPH_RECT  = pygame.Rect(20, 520, 1260, 200)


def draw_sokoban(surface: pygame.Surface, state: Optional[SokobanState], area: pygame.Rect) -> None:
    """Render Sokoban grid with 3D walls, player with eyes, amber boxes, green targets."""
    pygame.draw.rect(surface, C_GRID_BG, area, border_radius=8)
    if state is None:
        return
    w, h = state.width, state.height
    cs = min(area.width // w, area.height // h)
    ox = area.x + (area.width - w * cs) // 2
    oy = area.y + (area.height - h * cs) // 2

    for gy in range(h):
        for gx in range(w):
            rect = pygame.Rect(ox + gx * cs, oy + gy * cs, cs, cs)
            pos = (gx, gy)

            if pos in state.walls:
                # 3D wall effect
                pygame.draw.rect(surface, C_WALL, rect, border_radius=3)
                hi = pygame.Rect(rect.x + 2, rect.y + 2, rect.width - 4, rect.height // 2 - 2)
                pygame.draw.rect(surface, C_WALL_HI, hi, border_radius=2)
                continue

            pygame.draw.rect(surface, C_GRID_BG, rect)
            pygame.draw.rect(surface, C_GRIDLN, rect, 1)

            if pos in state.targets:
                pygame.draw.circle(surface, C_TARGET, rect.center, cs // 4, 2)

            if pos in state.boxes:
                on = pos in state.targets
                c = C_BOX_ON if on else C_BOX
                br = rect.inflate(-6, -6)
                pygame.draw.rect(surface, c, br, border_radius=4)
                # Highlight strip on box
                hs = pygame.Rect(br.x + 2, br.y + 2, br.width - 4, br.height // 3)
                hc = (min(c[0] + 30, 255), min(c[1] + 30, 255), min(c[2] + 30, 255))
                pygame.draw.rect(surface, hc, hs, border_radius=2)

            if pos == state.player:
                cx, cy = rect.center
                r = cs // 3
                pygame.draw.circle(surface, C_PLAYER, (cx, cy), r)
                # Eyes
                er = max(2, cs // 12)
                pygame.draw.circle(surface, C_WHITE, (cx - er - 1, cy - er), er)
                pygame.draw.circle(surface, C_WHITE, (cx + er + 1, cy - er), er)
                pygame.draw.circle(surface, (30, 30, 50), (cx - er - 1, cy - er), er // 2 + 1)
                pygame.draw.circle(surface, (30, 30, 50), (cx + er + 1, cy - er), er // 2 + 1)


def rolling_rate(data: List[bool], window: int = 100) -> List[float]:
    if not data:
        return []
    out: List[float] = []
    for i in range(len(data)):
        s = max(0, i - window + 1)
        chunk = data[s:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def draw_center_panel(surface: pygame.Surface, font: pygame.font.Font, font_lg: pygame.font.Font,
                      ppo: AgentRunner, thinker: AgentRunner) -> None:
    """Draw the center stats panel."""
    pygame.draw.rect(surface, C_PANEL, CENTER_PANEL, border_radius=8)
    x = CENTER_PANEL.x + 16
    y = CENTER_PANEL.y + 12

    def txt(t: str, f=font, c=C_WHITE, indent: int = 0) -> None:
        nonlocal y
        s = f.render(t, True, c)
        surface.blit(s, (x + indent, y))
        y += s.get_height() + 3

    def rate_color(r: float) -> Tuple[int, int, int]:
        if r > 50:
            return C_GREEN
        if r > 20:
            return C_YELLOW
        return C_RED

    # PPO section
    txt("PPO", font_lg, C_RED)
    txt(f"Episode: {ppo.episode:,}", indent=4)
    sr = ppo.solve_rate
    txt(f"Solve Rate: {sr:.1f}%", c=rate_color(sr), indent=4)
    txt(f"Deadlocks: {ppo.deadlock_rate:.0f}%", indent=4)
    txt(f"Avg Steps: {ppo.avg_steps:.0f}", indent=4)
    y += 8

    # Divider
    pygame.draw.line(surface, C_GRIDLN, (x, y), (x + CENTER_PANEL.width - 32, y), 1)
    y += 10

    # Thinker section
    txt("Thinker", font_lg, C_BLUE)
    txt(f"Episode: {thinker.episode:,}", indent=4)
    sr2 = thinker.solve_rate
    txt(f"Solve Rate: {sr2:.1f}%", c=rate_color(sr2), indent=4)
    txt(f"Deadlocks: {thinker.deadlock_rate:.0f}%", indent=4)
    txt(f"Avg Steps: {thinker.avg_steps:.0f}", indent=4)
    ta: ThinkerAgent = thinker.agent  # type: ignore
    plan_c = C_GREEN if ta.planning_active else C_RED
    txt(f"Planning: {'ON' if ta.planning_active else 'OFF'}", c=plan_c, indent=4)
    y += 8

    # Divider
    pygame.draw.line(surface, C_GRIDLN, (x, y), (x + CENTER_PANEL.width - 32, y), 1)
    y += 10

    # Comparison
    if ppo.solve_rate > 0:
        ratio = thinker.solve_rate / ppo.solve_rate
        txt(f"Thinker is {ratio:.1f}x better", font_lg, C_BLUE)
    elif thinker.solve_rate > 0:
        txt("Thinker is ahead!", font_lg, C_BLUE)
    else:
        txt("Both learning...", font_lg, C_DIM)


def draw_graph(surface: pygame.Surface, font: pygame.font.Font,
               ppo_solved: List[bool], thinker_solved: List[bool]) -> None:
    """Draw dual learning curve graph."""
    rect = GRAPH_RECT
    pygame.draw.rect(surface, C_GRID_BG, rect, border_radius=8)

    # Title
    lbl = font.render("Learning Curve (rolling 100)", True, C_DIM)
    surface.blit(lbl, (rect.x + 12, rect.y + 6))

    gx = rect.x + 12
    gw = rect.width - 24
    gy = rect.y + 28
    gh = rect.height - 40

    # Axes
    pygame.draw.line(surface, C_DIM, (gx, gy + gh), (gx + gw, gy + gh), 1)
    pygame.draw.line(surface, C_DIM, (gx, gy), (gx, gy + gh), 1)

    # 50% dashed gridline
    mid_y = gy + gh // 2
    for dx in range(0, gw, 8):
        pygame.draw.line(surface, (50, 50, 70), (gx + dx, mid_y), (gx + dx + 4, mid_y), 1)
    surface.blit(font.render("50%", True, C_DIM), (gx + gw - 34, mid_y - 12))
    surface.blit(font.render("100%", True, C_DIM), (gx + gw - 42, gy - 2))

    def plot(rates: List[float], color: Tuple[int, int, int]) -> None:
        if len(rates) < 2:
            return
        n = len(rates)
        xs = gw / max(n - 1, 1)
        pts = [(gx + int(i * xs), gy + gh - int(r * gh)) for i, r in enumerate(rates)]
        if len(pts) >= 2:
            pygame.draw.lines(surface, color, False, pts, 2)

    plot(rolling_rate(ppo_solved), C_RED)
    plot(rolling_rate(thinker_solved), C_BLUE)

    # Legend top-right
    lx = rect.x + rect.width - 220
    ly = rect.y + 6
    pygame.draw.line(surface, C_RED, (lx, ly + 6), (lx + 20, ly + 6), 2)
    surface.blit(font.render("PPO", True, C_RED), (lx + 24, ly))
    lx += 90
    pygame.draw.line(surface, C_BLUE, (lx, ly + 6), (lx + 20, ly + 6), 2)
    surface.blit(font.render("Thinker", True, C_BLUE), (lx + 24, ly))


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Think Before You Act -- PPO vs ThinkerAgent")
    clock = pygame.time.Clock()

    font    = pygame.font.SysFont("consolas,dejavusansmono,monospace", 14)
    font_lg = pygame.font.SysFont("consolas,dejavusansmono,monospace", 18, bold=True)
    font_tt = pygame.font.SysFont("consolas,dejavusansmono,monospace", 22, bold=True)

    ppo_runner     = AgentRunner("PPO -- Brute Force",       C_RED,  is_thinker=False)
    thinker_runner = AgentRunner("ThinkerAgent -- Think First", C_BLUE, is_thinker=True)

    mode = "visual"
    paused = False
    last_step = time.time()
    level_seed_counter = 5000
    running = True

    while running:
        now = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_v:
                    mode = "visual"
                elif event.key == pygame.K_f:
                    mode = "fast"
                elif event.key == pygame.K_s:
                    mode = "slow"

        # -- Training --
        if not paused:
            def train_batch(n: int) -> None:
                nonlocal level_seed_counter
                for _ in range(n):
                    seed = level_seed_counter
                    level_seed_counter += 1
                    ppo_runner.run_episode(level_seed=seed)
                    thinker_runner.run_episode(level_seed=seed)

            if mode == "fast":
                train_batch(5)
            elif mode == "visual":
                train_batch(1)
            elif mode == "slow":
                if now - last_step >= 0.2:
                    last_step = now
                    train_batch(1)

        # -- Render --
        screen.fill(C_BG)

        # Title bar
        title = font_tt.render("Think Before You Act  --  PPO vs ThinkerAgent", True, C_WHITE)
        screen.blit(title, (WIN_W // 2 - title.get_width() // 2, 9))

        # Left grid - PPO
        draw_sokoban(screen, ppo_runner.state, LEFT_GRID)
        lbl = font_lg.render("PPO -- Brute Force", True, C_RED)
        screen.blit(lbl, (LEFT_GRID.centerx - lbl.get_width() // 2, LEFT_GRID.bottom + 4))

        # Right grid - Thinker
        draw_sokoban(screen, thinker_runner.state, RIGHT_GRID)
        lbl2 = font_lg.render("ThinkerAgent -- Think First", True, C_BLUE)
        screen.blit(lbl2, (RIGHT_GRID.centerx - lbl2.get_width() // 2, RIGHT_GRID.bottom + 4))

        # Center panel
        draw_center_panel(screen, font, font_lg, ppo_runner, thinker_runner)

        # Graph
        draw_graph(screen, font, ppo_runner.all_solved, thinker_runner.all_solved)

        # Bottom controls bar
        mode_labels = {"visual": "VISUAL (1 ep/frame)", "fast": "FAST (5 eps/frame)",
                       "slow": "SLOW (1 ep/frame, 200ms delay)"}
        status = "PAUSED" if paused else mode_labels.get(mode, mode)
        ctrl = font.render(f"[{status}]   V=visual  F=fast  S=slow  Space=pause  ESC=quit", True, C_DIM)
        screen.blit(ctrl, (20, WIN_H - 22))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
