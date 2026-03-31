"""TRAIN_AND_WATCH_THINKER -- Watch a ThinkerAgent learn Sokoban in real-time.

A ThinkerAgent combines a PPO policy with a learned world model for planning.
It imagines future states before acting, and dreams up extra training data.

Opens a PyGame window (1000x700). Left side shows the Sokoban grid,
right side shows training stats, world model info, learning curve, action probs.

Controls:
  Space  -- Pause / Resume
  F      -- Fast mode   (10 episodes per frame, skip rendering)
  S      -- Slow mode   (1 step per frame, 150ms delay)
  V      -- Visual mode (1 episode per frame, render final state)
  ESC    -- Quit

Self-contained. Only dependencies: pygame, torch, numpy.
"""

from __future__ import annotations

import random
import sys
import time
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# ══════════════════════════════════════════════════════════════════════
# Sokoban Engine
# ══════════════════════════════════════════════════════════════════════

DIR_DELTAS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


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


# ══════════════════════════════════════════════════════════════════════
# BFS Solver
# ══════════════════════════════════════════════════════════════════════

def solve(state: SokobanState, max_states: int = 300_000) -> Optional[List[int]]:
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


# ══════════════════════════════════════════════════════════════════════
# Level Generator
# ══════════════════════════════════════════════════════════════════════

def _is_connected(cells, walls) -> bool:
    if not cells:
        return True
    start = cells[0]
    visited = {start}
    queue = deque([start])
    cell_set = set(cells)
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nb = (x + dx, y + dy)
            if nb in cell_set and nb not in visited and nb not in walls:
                visited.add(nb)
                queue.append(nb)
    return len(visited) >= len(cell_set)


def generate_training_level(
    width: int = 6, height: int = 6, n_boxes: int = 1,
    seed: Optional[int] = None, max_retries: int = 500,
    min_solution: int = 3, max_solution: int = 80,
) -> Optional[Tuple[SokobanState, List[int]]]:
    rng = random.Random(seed)
    for _ in range(max_retries):
        walls: Set[Tuple[int, int]] = set()
        for x in range(width):
            walls.add((x, 0))
            walls.add((x, height - 1))
        for y in range(height):
            walls.add((0, y))
            walls.add((width - 1, y))
        interior = [(x, y) for x in range(1, width - 1) for y in range(1, height - 1)]
        rng.shuffle(interior)
        n_int_walls = rng.randint(0, max(1, len(interior) // 5))
        for w in interior[:n_int_walls]:
            walls.add(w)
        free = [p for p in interior if p not in walls]
        if len(free) < n_boxes * 2 + 1:
            continue
        if not _is_connected(free, walls):
            continue
        rng.shuffle(free)
        targets: Set[Tuple[int, int]] = set()
        for p in free:
            if len(targets) >= n_boxes:
                break
            targets.add(p)
        if len(targets) < n_boxes:
            continue
        remaining = [p for p in free if p not in targets]
        rng.shuffle(remaining)
        boxes: Set[Tuple[int, int]] = set()
        for p in remaining:
            if len(boxes) >= n_boxes:
                break
            x, y = p
            up = (x, y - 1) in walls
            dn = (x, y + 1) in walls
            lt = (x - 1, y) in walls
            rt = (x + 1, y) in walls
            if (up or dn) and (lt or rt):
                continue
            boxes.add(p)
        if len(boxes) < n_boxes:
            continue
        player_spots = [p for p in free if p not in targets and p not in boxes]
        if not player_spots:
            continue
        player = rng.choice(player_spots)
        state = SokobanState(
            frozenset(walls), frozenset(boxes), frozenset(targets),
            player, width, height,
        )
        solution = solve(state, max_states=200_000)
        if solution is not None and min_solution <= len(solution) <= max_solution:
            return state, solution
    return None


# ══════════════════════════════════════════════════════════════════════
# Gym-style Environment with Phase Progression
# ══════════════════════════════════════════════════════════════════════

class SokobanEnv:
    """Sokoban environment with 5-channel obs, reward shaping, phase progression."""

    def __init__(self, max_h: int = 10, max_w: int = 10, max_steps: int = 200) -> None:
        self.max_h = max_h
        self.max_w = max_w
        self.max_steps = max_steps
        self._state: Optional[SokobanState] = None
        self._steps = 0
        self._prev_dist = 0.0
        self._gen_seed = 2000
        # Phase progression
        self.phase = 1
        self._episode_count = 0
        self._recent_solved: List[bool] = []

    @property
    def state(self) -> Optional[SokobanState]:
        return self._state

    def _phase_params(self) -> Tuple[List[int], int, int, int]:
        """Returns (sizes, n_boxes, min_sol, max_sol) for current phase."""
        if self.phase == 1:
            return [5], 1, 3, 15
        elif self.phase == 2:
            return [5, 6], 1, 3, 30
        else:
            return [5, 6, 7], 1, 5, 40

    def _check_phase_up(self) -> None:
        if len(self._recent_solved) < 100:
            return
        rate = sum(self._recent_solved[-100:]) / 100
        if self.phase == 1 and rate > 0.40:
            self.phase = 2
        elif self.phase == 2 and rate > 0.35:
            self.phase = 3

    def reset(self) -> Tuple[np.ndarray, dict]:
        self._episode_count += 1
        self._check_phase_up()
        sizes, n_boxes, min_sol, max_sol = self._phase_params()
        result = None
        while result is None:
            w = random.choice(sizes)
            h = random.choice(sizes)
            result = generate_training_level(
                width=w, height=h, n_boxes=n_boxes,
                seed=self._gen_seed, min_solution=min_sol, max_solution=max_sol,
            )
            self._gen_seed += 1
        self._state = result[0]
        self._steps = 0
        self._prev_dist = self._state.box_distances()
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self._state is not None
        self._steps += 1
        old_on = self._state.n_boxes_on_target
        new_state = self._state.move(action)
        if new_state is None:
            reward = -1.0
            truncated = self._steps >= self.max_steps
            return self._get_obs(), reward, False, truncated, self._info(invalid=True)
        self._state = new_state
        new_on = self._state.n_boxes_on_target
        new_dist = self._state.box_distances()
        reward = -0.1
        if self._state.solved:
            reward += 100.0
            self._recent_solved.append(True)
            return self._get_obs(), reward, True, False, self._info(solved=True)
        if self._state.is_deadlocked():
            reward += -50.0
            self._recent_solved.append(False)
            return self._get_obs(), reward, True, False, self._info(deadlock=True)
        if new_on > old_on:
            reward += 10.0
        elif new_on < old_on:
            reward += -10.0
        dist_diff = self._prev_dist - new_dist
        if dist_diff > 0.5:
            reward += 1.0
        elif dist_diff < -0.5:
            reward += -1.0
        self._prev_dist = new_dist
        truncated = self._steps >= self.max_steps
        if truncated:
            self._recent_solved.append(False)
        return self._get_obs(), reward, False, truncated, self._info()

    def _get_obs(self) -> np.ndarray:
        s = self._state
        obs = np.zeros((5, self.max_h, self.max_w), dtype=np.float32)
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

    def _info(self, solved=False, deadlock=False, invalid=False) -> dict:
        s = self._state
        return {
            "solved": solved, "deadlock": deadlock, "invalid_move": invalid,
            "steps": self._steps,
            "boxes_on_target": s.n_boxes_on_target if s else 0,
        }

    def close(self) -> None:
        pass


# ══════════════════════════════════════════════════════════════════════
# Neural Networks
# ══════════════════════════════════════════════════════════════════════

class SokobanNet(nn.Module):
    """CNN policy+value network."""

    def __init__(self, grid_h: int = 10, grid_w: int = 10) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        flat_dim = 64 * grid_h * grid_w
        self.fc = nn.Sequential(nn.Linear(flat_dim, 512), nn.ReLU())
        self.policy_head = nn.Linear(512, 4)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = h.flatten(1)
        h = self.fc(h)
        logits = self.policy_head(h)
        probs = F.softmax(logits, dim=-1)
        value = self.value_head(h)
        return probs, value


class WorldModel(nn.Module):
    """Learned dynamics model: predicts next_state, reward, done from (state, action)."""

    def __init__(self, grid_h: int = 10, grid_w: int = 10) -> None:
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        # State encoder
        self.state_enc = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        # Action embedding
        self.action_emb = nn.Embedding(4, 64)
        # Decoder: (128, H, W) -> (5, H, W)
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 5, 3, padding=1), nn.Sigmoid(),
        )
        # Reward + done heads (from state features after encoding)
        self.reward_head = nn.Linear(64, 1)
        self.done_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        state:  (B, 5, H, W)
        action: (B,) long
        Returns: pred_state (B,5,H,W), pred_reward (B,), pred_done (B,)
        """
        feat = self.state_enc(state)                        # (B, 64, H, W)
        act_emb = self.action_emb(action)                   # (B, 64)
        act_map = act_emb.unsqueeze(-1).unsqueeze(-1)       # (B, 64, 1, 1)
        act_map = act_map.expand(-1, -1, self.grid_h, self.grid_w)  # (B, 64, H, W)
        combined = torch.cat([feat, act_map], dim=1)        # (B, 128, H, W)
        pred_state = self.decoder(combined)                 # (B, 5, H, W)
        # Global average pool of feat for reward/done
        pooled = feat.mean(dim=(2, 3))                      # (B, 64)
        pred_reward = self.reward_head(pooled).squeeze(-1)   # (B,)
        pred_done = self.done_head(pooled).squeeze(-1)       # (B,)
        return pred_state, pred_reward, pred_done


# ══════════════════════════════════════════════════════════════════════
# Replay Buffer
# ══════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """Circular buffer storing (state, action, next_state, reward, done) transitions."""

    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, int, np.ndarray, float, float]] = []
        self.pos = 0

    def push(self, state: np.ndarray, action: int, next_state: np.ndarray,
             reward: float, done: float) -> None:
        entry = (state, action, next_state, reward, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.pos] = entry
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, next_states, rewards, dones = [], [], [], [], []
        for i in indices:
            s, a, ns, r, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            next_states.append(ns)
            rewards.append(r)
            dones.append(d)
        return (np.array(states), np.array(actions), np.array(next_states),
                np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════
# Thinker Agent
# ══════════════════════════════════════════════════════════════════════

class ThinkerAgent:
    """PPO policy + learned world model with planning and dreaming."""

    def __init__(self, grid_h: int = 10, grid_w: int = 10, lr: float = 2.5e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95, clip_eps: float = 0.2,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 max_grad_norm: float = 0.5, update_epochs: int = 4,
                 mini_batch_size: int = 256) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size

        # Policy network
        self.policy_net = SokobanNet(grid_h, grid_w).to(self._device)
        self.policy_opt = Adam(self.policy_net.parameters(), lr=lr, eps=1e-5)

        # World model
        self.world_model = WorldModel(grid_h, grid_w).to(self._device)
        self.wm_opt = Adam(self.world_model.parameters(), lr=1e-3)

        # Replay buffer for world model training
        self.replay = ReplayBuffer(50_000)

        # PPO rollout storage
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

        # World model tracking
        self.wm_accuracy = 0.0
        self.wm_train_steps = 0
        self.planning_mode = False
        self.dream_episodes = 0
        self._step_counter = 0

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)

    def get_action_probs(self, obs: np.ndarray) -> np.ndarray:
        self.policy_net.eval()
        with torch.no_grad():
            probs, _ = self.policy_net(self._to_tensor(obs))
        return probs.squeeze(0).cpu().numpy()

    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """Select action, using planning if world model is accurate enough."""
        self.policy_net.eval()
        self.world_model.eval()

        enough_data = len(self.replay) >= 500
        self.planning_mode = self.wm_accuracy > 0.7 and enough_data

        if self.planning_mode:
            action = self._plan(obs)
            # Get log_prob and value for the chosen action
            with torch.no_grad():
                t = self._to_tensor(obs)
                probs, value = self.policy_net(t)
                dist = torch.distributions.Categorical(probs)
                log_prob = dist.log_prob(torch.tensor([action], device=self._device))
            return action, float(log_prob.item()), float(value.item())
        else:
            with torch.no_grad():
                t = self._to_tensor(obs)
                probs, value = self.policy_net(t)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            return int(action.item()), float(log_prob.item()), float(value.item())

    def _plan(self, obs: np.ndarray, depth: int = 3) -> int:
        """Simulate all 4 first actions up to depth steps using world model."""
        best_action = 0
        best_value = float("-inf")

        with torch.no_grad():
            for first_action in range(4):
                sim_state = self._to_tensor(obs)
                total_reward = 0.0
                for d in range(depth):
                    if d == 0:
                        act = first_action
                    else:
                        probs, _ = self.policy_net(sim_state)
                        act = int(probs.argmax(dim=-1).item())
                    act_t = torch.tensor([act], dtype=torch.long, device=self._device)
                    pred_state, pred_reward, pred_done = self.world_model(sim_state, act_t)
                    total_reward += float(pred_reward.item()) * (self.gamma ** d)
                    sim_state = pred_state
                    if float(pred_done.item()) > 0.5:
                        break
                # Add terminal value estimate
                _, term_val = self.policy_net(sim_state)
                total_reward += float(term_val.item()) * (self.gamma ** depth)
                if total_reward > best_value:
                    best_value = total_reward
                    best_action = first_action

        return best_action

    def store_transition(self, state, action, reward, log_prob, value, done) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def store_replay(self, state: np.ndarray, action: int, next_state: np.ndarray,
                     reward: float, done: bool) -> None:
        self.replay.push(state, action, next_state, reward, float(done))
        self._step_counter += 1

    def learn(self) -> None:
        """Train world model and potentially dream."""
        # Train world model every 50 steps
        if self._step_counter % 50 == 0 and len(self.replay) >= 256:
            self._train_world_model(n_steps=10)

        # Dream every 200 steps
        if self._step_counter % 200 == 0 and self.wm_accuracy > 0.6 and len(self.replay) >= 500:
            self._dream(n_episodes=30)

    def _train_world_model(self, n_steps: int = 10, batch_size: int = 256) -> None:
        """Train world model on replay buffer."""
        self.world_model.train()
        total_loss = 0.0
        total_state_err = 0.0

        for _ in range(n_steps):
            bs = min(batch_size, len(self.replay))
            states, actions, next_states, rewards, dones = self.replay.sample(bs)

            s_t = torch.tensor(states, dtype=torch.float32, device=self._device)
            a_t = torch.tensor(actions, dtype=torch.long, device=self._device)
            ns_t = torch.tensor(next_states, dtype=torch.float32, device=self._device)
            r_t = torch.tensor(rewards, dtype=torch.float32, device=self._device)
            d_t = torch.tensor(dones, dtype=torch.float32, device=self._device)

            pred_s, pred_r, pred_d = self.world_model(s_t, a_t)

            state_loss = F.mse_loss(pred_s, ns_t)
            reward_loss = F.mse_loss(pred_r, r_t)
            done_loss = F.binary_cross_entropy(pred_d, d_t)
            loss = state_loss + reward_loss + done_loss

            self.wm_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
            self.wm_opt.step()

            total_loss += loss.item()
            total_state_err += state_loss.item()

        self.wm_train_steps += n_steps
        avg_state_err = total_state_err / n_steps
        # Accuracy heuristic: 1 - clamp(state_error, 0, 1)
        self.wm_accuracy = max(0.0, min(1.0, 1.0 - avg_state_err))

    def _dream(self, n_episodes: int = 30) -> None:
        """Generate imagined episodes using the world model for extra PPO data."""
        self.world_model.eval()
        self.policy_net.eval()

        dream_states: List[np.ndarray] = []
        dream_actions: List[int] = []
        dream_rewards: List[float] = []
        dream_log_probs: List[float] = []
        dream_values: List[float] = []
        dream_dones: List[bool] = []

        with torch.no_grad():
            for _ in range(n_episodes):
                # Start from a random real state
                if len(self.replay) == 0:
                    break
                idx = np.random.randint(len(self.replay))
                start_state = self.replay.buffer[idx][0]
                sim_state = self._to_tensor(start_state)

                for step in range(20):  # max dream length
                    obs_np = sim_state.squeeze(0).cpu().numpy()
                    probs, value = self.policy_net(sim_state)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                    act_t = action.unsqueeze(0) if action.dim() == 0 else action
                    pred_s, pred_r, pred_d = self.world_model(sim_state, act_t)

                    done = float(pred_d.item()) > 0.5

                    dream_states.append(obs_np)
                    dream_actions.append(int(action.item()))
                    dream_rewards.append(float(pred_r.item()))
                    dream_log_probs.append(float(log_prob.item()))
                    dream_values.append(float(value.item()))
                    dream_dones.append(done)

                    if done:
                        break
                    sim_state = pred_s

        self.dream_episodes += n_episodes

        # Do a PPO update on dream data if enough
        if len(dream_states) >= 64:
            self._ppo_update_on(
                dream_states, dream_actions, dream_rewards,
                dream_log_probs, dream_values, dream_dones,
            )

    def update(self) -> Dict[str, float]:
        """PPO update on real experience."""
        if not self.states:
            return {}
        return self._ppo_update_on(
            self.states, self.actions, self.rewards,
            self.log_probs, self.values, self.dones, clear_main=True,
        )

    def _ppo_update_on(
        self, states_list, actions_list, rewards_list,
        log_probs_list, values_list, dones_list, clear_main=False,
    ) -> Dict[str, float]:
        self.policy_net.train()
        dev = self._device

        states = torch.tensor(np.array(states_list), dtype=torch.float32, device=dev)
        actions = torch.tensor(actions_list, dtype=torch.long, device=dev)
        old_lp = torch.tensor(log_probs_list, dtype=torch.float32, device=dev)
        rewards = np.array(rewards_list, dtype=np.float32)
        values = np.array(values_list, dtype=np.float32)
        dones = np.array(dones_list, dtype=np.float32)

        # GAE
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            nv = 0.0 if t == n - 1 else values[t + 1]
            nt = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * nv * nt - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * nt * last_gae
            advantages[t] = last_gae
        returns = advantages + values

        adv_t = torch.tensor(advantages, dtype=torch.float32, device=dev)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=dev)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n_samples = len(states_list)
        indices = np.arange(n_samples)
        total_pl = total_vl = total_ent = 0.0
        n_updates = 0

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, self.mini_batch_size):
                idx = indices[start:start + self.mini_batch_size]
                idx_t = torch.tensor(idx, dtype=torch.long, device=dev)

                b_states = states[idx_t]
                b_actions = actions[idx_t]
                b_old_lp = old_lp[idx_t]
                b_adv = adv_t[idx_t]
                b_ret = ret_t[idx_t]

                probs, vals = self.policy_net(b_states)
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - b_old_lp)
                s1 = ratio * b_adv
                s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = F.mse_loss(vals.squeeze(-1), b_ret)

                loss = (policy_loss + self.value_coef * value_loss
                        - self.entropy_coef * entropy)

                self.policy_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.policy_opt.step()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_ent += entropy.item()
                n_updates += 1

        if clear_main:
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.log_probs.clear()
            self.values.clear()
            self.dones.clear()

        n_updates = max(n_updates, 1)
        return {
            "policy_loss": total_pl / n_updates,
            "value_loss": total_vl / n_updates,
            "entropy": total_ent / n_updates,
        }


# ══════════════════════════════════════════════════════════════════════
# Colours & Layout
# ══════════════════════════════════════════════════════════════════════

COL_BG         = (22, 22, 35)
COL_PANEL      = (30, 30, 50)
COL_GRID_LINE  = (40, 40, 55)
COL_WALL       = (75, 85, 99)
COL_BOX        = (251, 191, 36)
COL_BOX_DONE   = (52, 211, 153)
COL_TARGET     = (52, 211, 153)
COL_PLAYER     = (99, 102, 241)
COL_WHITE      = (220, 220, 235)
COL_DIM        = (120, 120, 140)
COL_GREEN      = (52, 211, 153)
COL_YELLOW     = (251, 191, 36)
COL_RED        = (239, 68, 68)
COL_AMBER      = (245, 158, 11)
COL_GRAPH_BG   = (30, 30, 50)
COL_GRAPH_LINE = (52, 211, 153)
COL_DASH       = (50, 50, 70)

WIN_W, WIN_H = 1000, 700
GRID_AREA    = pygame.Rect(10, 10, 560, 560)
STATS_X      = 590
STATS_W      = 400
GRAPH_RECT   = pygame.Rect(STATS_X, 430, STATS_W, 160)
PROBS_RECT   = pygame.Rect(STATS_X, 600, STATS_W, 60)

N_EPISODES       = 10_000
STEPS_PER_UPDATE = 1024
SLOW_DELAY       = 0.15


# ══════════════════════════════════════════════════════════════════════
# Grid Renderer
# ══════════════════════════════════════════════════════════════════════

def draw_grid(surface: pygame.Surface, state, area: pygame.Rect) -> None:
    if state is None:
        return
    gw, gh = state.width, state.height
    cs = min(area.width // gw, area.height // gh)
    ox = area.x + (area.width - gw * cs) // 2
    oy = area.y + (area.height - gh * cs) // 2
    for gy in range(gh):
        for gx in range(gw):
            rect = pygame.Rect(ox + gx * cs, oy + gy * cs, cs, cs)
            pygame.draw.rect(surface, COL_BG, rect)
            pygame.draw.rect(surface, COL_GRID_LINE, rect, 1)
    for (wx, wy) in state.walls:
        rect = pygame.Rect(ox + wx * cs, oy + wy * cs, cs, cs)
        pygame.draw.rect(surface, COL_WALL, rect, border_radius=3)
    for (tx, ty) in state.targets:
        cx = ox + tx * cs + cs // 2
        cy = oy + ty * cs + cs // 2
        pygame.draw.circle(surface, COL_TARGET, (cx, cy), cs // 4, 2)
    boxes_on_target = state.boxes & state.targets
    for (bx, by) in state.boxes:
        rect = pygame.Rect(ox + bx * cs, oy + by * cs, cs, cs)
        colour = COL_BOX_DONE if (bx, by) in boxes_on_target else COL_BOX
        pygame.draw.rect(surface, colour, rect.inflate(-6, -6), border_radius=4)
    px, py = state.player
    pcx = ox + px * cs + cs // 2
    pcy = oy + py * cs + cs // 2
    pygame.draw.circle(surface, COL_PLAYER, (pcx, pcy), cs // 3)


# ══════════════════════════════════════════════════════════════════════
# Stats Panel
# ══════════════════════════════════════════════════════════════════════

def draw_stats(
    surface: pygame.Surface,
    font: pygame.font.Font,
    font_big: pygame.font.Font,
    episode: int,
    all_rewards: List[float],
    all_solved: List[bool],
    deadlocks: int,
    avg_steps: float,
    mode: str,
    paused: bool,
    agent: ThinkerAgent,
    phase: int,
) -> None:
    x = STATS_X
    y = 15

    def text(txt, f=font, color=COL_WHITE):
        nonlocal y
        s = f.render(txt, True, color)
        surface.blit(s, (x, y))
        y += s.get_height() + 4

    text(f"Episode: {episode} / {N_EPISODES}", font_big)

    # Phase
    phase_col = COL_GREEN if phase >= 3 else COL_YELLOW if phase >= 2 else COL_DIM
    text(f"Phase: {phase}/3", color=phase_col)

    # Solve rate
    if all_solved:
        recent = all_solved[-100:]
        sr = sum(recent) / len(recent) * 100
        col = COL_GREEN if sr > 50 else COL_YELLOW if sr > 20 else COL_RED
        text(f"Solve Rate: {sr:.1f}%", color=col)
    else:
        text("Solve Rate: --%")

    # Average reward
    if all_rewards:
        ar = sum(all_rewards[-100:]) / len(all_rewards[-100:])
        text(f"Avg Reward: {ar:.1f}")
    else:
        text("Avg Reward: --")

    # Deadlock rate
    if episode > 0:
        dr = deadlocks / episode * 100
        text(f"Deadlock Rate: {dr:.0f}%")
    else:
        text("Deadlock Rate: --%")

    text(f"Avg Steps: {avg_steps:.0f}" if avg_steps > 0 else "Avg Steps: --")

    y += 6

    # World model accuracy
    wm_pct = agent.wm_accuracy * 100
    if wm_pct >= 75:
        wm_col = COL_GREEN
    elif wm_pct >= 50:
        wm_col = COL_AMBER
    else:
        wm_col = COL_RED
    text(f"World Model: {wm_pct:.0f}%", color=wm_col)

    # Mode: Planning vs Exploring
    if agent.planning_mode:
        mode_label = "Mode: Planning [brain]"
        mode_col = COL_GREEN
    else:
        mode_label = "Mode: Exploring [search]"
        mode_col = COL_YELLOW
    text(mode_label, color=mode_col)

    # Replay buffer size
    text(f"Replay Buffer: {len(agent.replay):,}")

    # Dream episodes
    text(f"Dreams: {agent.dream_episodes:,} episodes")

    y += 6

    # Run mode indicator
    mode_map = {"visual": "VISUAL", "fast": "FAST", "slow": "SLOW"}
    label = "PAUSED" if paused else mode_map.get(mode, mode.upper())
    text(f"[{label}]  V/F/S/Space/Esc", color=COL_DIM)


# ══════════════════════════════════════════════════════════════════════
# Learning Curve Graph
# ══════════════════════════════════════════════════════════════════════

def draw_graph(
    surface: pygame.Surface, font: pygame.font.Font,
    all_solved: List[bool], rect: pygame.Rect,
) -> None:
    pygame.draw.rect(surface, COL_GRAPH_BG, rect, border_radius=4)
    lbl = font.render("Solve Rate (rolling 100)", True, COL_DIM)
    surface.blit(lbl, (rect.x + 4, rect.y + 2))

    gx = rect.x + 4
    gw = rect.width - 8
    gy = rect.y + 20
    gh = rect.height - 28

    pygame.draw.line(surface, COL_DIM, (gx, gy + gh), (gx + gw, gy + gh), 1)
    pygame.draw.line(surface, COL_DIM, (gx, gy), (gx, gy + gh), 1)

    mid_y = gy + gh // 2
    for dx in range(0, gw, 6):
        pygame.draw.line(surface, COL_DASH, (gx + dx, mid_y), (gx + dx + 3, mid_y), 1)

    if len(all_solved) < 2:
        return

    window = 100
    rates: List[float] = []
    for i in range(len(all_solved)):
        start = max(0, i - window + 1)
        chunk = all_solved[start : i + 1]
        rates.append(sum(chunk) / len(chunk))

    n = len(rates)
    x_scale = gw / max(n - 1, 1)
    points = []
    for i, r in enumerate(rates):
        px = gx + int(i * x_scale)
        py = gy + gh - int(r * gh)
        points.append((px, py))

    if len(points) >= 2:
        pygame.draw.lines(surface, COL_GRAPH_LINE, False, points, 2)


# ══════════════════════════════════════════════════════════════════════
# Action Probability Bars
# ══════════════════════════════════════════════════════════════════════

def draw_action_probs(
    surface: pygame.Surface, font: pygame.font.Font,
    probs: np.ndarray, rect: pygame.Rect,
) -> None:
    pygame.draw.rect(surface, COL_PANEL, rect, border_radius=4)
    labels = ["UP", "DN", "LT", "RT"]
    bar_h = 10
    bar_max_w = rect.width - 70
    y = rect.y + 6
    for lbl, p in zip(labels, probs):
        txt = font.render(f"{lbl} {p:.2f}", True, COL_DIM)
        surface.blit(txt, (rect.x + 4, y))
        bw = int(p * bar_max_w)
        bar_rect = pygame.Rect(rect.x + 50, y + 2, max(bw, 0), bar_h)
        pygame.draw.rect(surface, COL_PLAYER, bar_rect, border_radius=2)
        y += bar_h + 4


# ══════════════════════════════════════════════════════════════════════
# Trainer
# ══════════════════════════════════════════════════════════════════════

class Trainer:
    """Encapsulates the ThinkerAgent training state."""

    def __init__(self) -> None:
        self.env = SokobanEnv(max_h=10, max_w=10, max_steps=200)
        self.agent = ThinkerAgent(grid_h=10, grid_w=10)

        self.obs, _ = self.env.reset()
        self.ep_reward = 0.0
        self.ep_steps = 0

        self.episode = 0
        self.total_steps = 0
        self.deadlocks = 0

        self.all_rewards: List[float] = []
        self.all_solved: List[bool] = []
        self.all_ep_steps: List[int] = []
        self.action_probs = np.array([0.25, 0.25, 0.25, 0.25])
        self._prev_obs: Optional[np.ndarray] = None

    @property
    def avg_steps(self) -> float:
        if not self.all_ep_steps:
            return 0.0
        recent = self.all_ep_steps[-100:]
        return sum(recent) / len(recent)

    def do_one_step(self) -> bool:
        """Execute one environment step. Returns True if episode ended."""
        self._prev_obs = self.obs.copy()
        self.action_probs[:] = self.agent.get_action_probs(self.obs)
        action, log_prob, value = self.agent.select_action(self.obs)
        next_obs, reward, term, trunc, info = self.env.step(action)
        done = term or trunc

        # Store in PPO rollout buffer
        self.agent.store_transition(self.obs, action, reward, log_prob, value, done)

        # Store in replay buffer for world model
        self.agent.store_replay(self._prev_obs, action, next_obs, reward, done)

        self.obs = next_obs
        self.ep_reward += reward
        self.ep_steps += 1
        self.total_steps += 1

        # Incremental world model training
        self.agent.learn()

        if done:
            self._finish_episode(info)

        # PPO update
        if self.total_steps % STEPS_PER_UPDATE == 0 and self.total_steps > 0:
            self.agent.update()

        return done

    def _finish_episode(self, info: dict) -> None:
        self.all_rewards.append(self.ep_reward)
        self.all_solved.append(info.get("solved", False))
        self.all_ep_steps.append(self.ep_steps)
        if info.get("deadlock", False):
            self.deadlocks += 1
        self.episode += 1
        self.obs, _ = self.env.reset()
        self.ep_reward = 0.0
        self.ep_steps = 0

    def run_episode(self) -> None:
        while not self.do_one_step():
            pass

    @property
    def state(self):
        return self.env.state

    @property
    def phase(self) -> int:
        return self.env.phase

    def close(self) -> None:
        self.env.close()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("TRAIN & WATCH -- Thinker Agent Learning Sokoban")
    clock = pygame.time.Clock()

    font     = pygame.font.SysFont("consolas,dejavusansmono,monospace", 15)
    font_big = pygame.font.SysFont("consolas,dejavusansmono,monospace", 18, bold=True)

    trainer = Trainer()

    mode = "visual"
    paused = False
    last_step_time = time.time()
    running = True

    while running:
        now = time.time()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_f:
                    mode = "fast"
                elif event.key == pygame.K_s:
                    mode = "slow"
                elif event.key == pygame.K_v:
                    mode = "visual"

        # Training tick
        if not paused and trainer.episode < N_EPISODES:
            if mode == "fast":
                for _ in range(10):
                    if trainer.episode >= N_EPISODES:
                        break
                    trainer.run_episode()
            elif mode == "visual":
                trainer.run_episode()
            elif mode == "slow":
                if now - last_step_time >= SLOW_DELAY:
                    last_step_time = now
                    trainer.do_one_step()

        # Render
        screen.fill(COL_BG)

        # Grid
        draw_grid(screen, trainer.state, GRID_AREA)
        pygame.draw.rect(screen, COL_GRID_LINE, GRID_AREA, 2, border_radius=4)

        # Title under grid
        title = font_big.render("Thinker Agent -- Learning Sokoban", True, COL_WHITE)
        screen.blit(
            title,
            (GRID_AREA.centerx - title.get_width() // 2, GRID_AREA.bottom + 8),
        )

        # Stats panel (with Thinker extras)
        draw_stats(
            screen, font, font_big,
            trainer.episode,
            trainer.all_rewards,
            trainer.all_solved,
            trainer.deadlocks,
            trainer.avg_steps,
            mode, paused,
            trainer.agent,
            trainer.phase,
        )

        # Learning curve graph
        draw_graph(screen, font, trainer.all_solved, GRAPH_RECT)

        # Action probability bars
        draw_action_probs(screen, font, trainer.action_probs, PROBS_RECT)

        pygame.display.flip()
        clock.tick(60)

    trainer.close()
    pygame.quit()


if __name__ == "__main__":
    main()
