"""
Real PPO Reinforcement Learning for Sokoban — 60 Levels with Curriculum.

The agent starts knowing NOTHING and learns through trial and error
on the actual 60 Sokoban levels. Curriculum learning unlocks harder
levels as the agent improves.

Controls:
  Space  - Pause / Resume
  F      - Fast mode (200 steps/frame, no rendering delay)
  S      - Slow mode (1 step/frame, 200ms delay)
  V      - Visual mode (3 steps/frame, normal)
  ESC    - Quit and save checkpoint
"""

import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
import os
import sys
import time
import math
from collections import deque
from typing import List, Optional, Tuple, Dict

from env.sokoban import SokobanState, solve, DIR_DELTAS
from env.level_loader import LevelLoader


# ============================================================
# SOKOBAN GYM WRAPPER — Wraps SokobanState for RL training
# ============================================================

MAX_GRID = 12  # pad all observations to 12x12

class SokobanRLEnv:
    """
    Gym-like wrapper around SokobanState for RL training.

    Loads actual levels from classic_60.txt.
    Provides 5-channel observations and shaped rewards.
    """

    def __init__(self, levels: List[SokobanState], max_steps: int = 200):
        self.levels = levels
        self.max_steps = max_steps
        self.state: Optional[SokobanState] = None
        self.initial_state: Optional[SokobanState] = None
        self.steps = 0
        self.current_level_idx = 0
        self._prev_box_dist = 0.0

    def reset(self, level_idx: Optional[int] = None) -> np.ndarray:
        """Reset to a specific or random level."""
        if level_idx is not None:
            self.current_level_idx = level_idx
        else:
            self.current_level_idx = random.randint(0, len(self.levels) - 1)

        self.state = self.levels[self.current_level_idx].clone()
        self.initial_state = self.state.clone()
        self.steps = 0
        self._prev_box_dist = self.state.box_distances()
        return self._get_obs()

    def reset_from_pool(self, pool: List[int]) -> np.ndarray:
        """Reset to a random level from the given pool of level indices."""
        idx = random.choice(pool)
        return self.reset(level_idx=idx)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action. Returns (obs, reward, done, info).

        Reward shaping is CRITICAL for Sokoban RL:
        - Step penalty: -0.1 (encourages efficiency)
        - Wall bump / invalid: -1.0
        - Box pushed to target: +10.0
        - Box pushed off target: -10.0
        - Box closer to target: +1.0
        - Box farther from target: -1.0
        - Deadlock detected: -50.0
        - Puzzle solved: +100.0
        """
        reward = -0.1  # step penalty
        info = {"solved": False, "deadlock": False, "pushed": False}

        new_state = self.state.move(action)

        if new_state is None:
            # Invalid move (wall or can't push)
            reward = -1.0
            self.steps += 1
            done = self.steps >= self.max_steps
            return self._get_obs(), reward, done, info

        # Check if a box was pushed
        old_boxes = self.state.boxes
        new_boxes = new_state.boxes
        pushed = old_boxes != new_boxes

        if pushed:
            info["pushed"] = True

            # Box on/off target rewards
            old_on = len(old_boxes & self.state.targets)
            new_on = len(new_boxes & new_state.targets)
            if new_on > old_on:
                reward += 10.0  # box placed on target
            elif new_on < old_on:
                reward -= 10.0  # box removed from target

            # Distance-based reward shaping
            new_dist = new_state.box_distances()
            if new_dist < self._prev_box_dist:
                reward += 1.0  # moved closer
            elif new_dist > self._prev_box_dist:
                reward -= 1.0  # moved farther
            self._prev_box_dist = new_dist

            # Deadlock check
            if new_state.is_deadlocked():
                reward -= 50.0
                info["deadlock"] = True

        self.state = new_state
        self.steps += 1

        # Check solved
        if self.state.solved:
            reward += 100.0
            info["solved"] = True
            return self._get_obs(), reward, True, info

        done = info["deadlock"] or (self.steps >= self.max_steps)
        return self._get_obs(), reward, done, info

    def _get_obs(self) -> np.ndarray:
        """
        5-channel binary observation padded to (5, MAX_GRID, MAX_GRID).

        Channel 0: walls
        Channel 1: boxes (not on target)
        Channel 2: targets (without box)
        Channel 3: player
        Channel 4: boxes on target
        """
        obs = np.zeros((5, MAX_GRID, MAX_GRID), dtype=np.float32)
        s = self.state

        for x, y in s.walls:
            if 0 <= y < MAX_GRID and 0 <= x < MAX_GRID:
                obs[0, y, x] = 1.0

        for x, y in s.boxes:
            if 0 <= y < MAX_GRID and 0 <= x < MAX_GRID:
                if (x, y) in s.targets:
                    obs[4, y, x] = 1.0  # box on target
                else:
                    obs[1, y, x] = 1.0  # box not on target

        for x, y in s.targets:
            if 0 <= y < MAX_GRID and 0 <= x < MAX_GRID:
                if (x, y) not in s.boxes:
                    obs[2, y, x] = 1.0  # empty target

        px, py = s.player
        if 0 <= py < MAX_GRID and 0 <= px < MAX_GRID:
            obs[3, py, px] = 1.0

        return obs

    def get_render_state(self) -> SokobanState:
        """Return current state for rendering."""
        return self.state


# ============================================================
# NEURAL NETWORK — CNN Policy + Value
# ============================================================

class SokobanNet(nn.Module):
    """
    Convolutional neural network for Sokoban.

    Input: (batch, 5, 12, 12) — 5 binary channels padded to 12x12
    Output: action_probs (batch, 4), value (batch, 1)

    THIS is what learns. Starts with random weights → random actions.
    After thousands of episodes, outputs good actions.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        flat_size = 64 * MAX_GRID * MAX_GRID  # 64 * 12 * 12 = 9216

        self.fc = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(512, 4)
        self.value_head = nn.Linear(512, 1)

        # Orthogonal initialization for stable training
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                gain = 0.01 if module is self.policy_head else np.sqrt(2)
                nn.init.orthogonal_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        h = self.fc(h)
        logits = self.policy_head(h)
        probs = F.softmax(logits, dim=-1)
        value = self.value_head(h)
        return probs, value


# ============================================================
# PPO AGENT — Real Reinforcement Learning
# ============================================================

class PPOAgent:
    """
    Proximal Policy Optimization agent.

    How it works:
    1. Collect experience by running episodes in the environment
    2. Compute advantages using GAE (Generalized Advantage Estimation)
    3. Update neural network to make good actions more likely (policy gradient)
    4. Clip updates to prevent too-large changes (PPO stability trick)
    5. Repeat

    The agent starts with random weights → random actions.
    Over time: weights update → better actions → higher rewards → better weights.
    """

    def __init__(self, lr: float = 2.5e-4, entropy_coef: float = 0.05):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = SokobanNet().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        # PPO hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_eps = 0.2
        self.entropy_coef = entropy_coef  # starts high for exploration
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.update_epochs = 4
        self.mini_batch_size = 128

        # Experience buffer
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[float] = []

    def select_action(self, state_np: np.ndarray) -> Tuple[int, float, float, np.ndarray]:
        """
        Given observation, return (action, log_prob, value, action_probs).

        We SAMPLE from the distribution (exploration), not argmax.
        Early training: probs ~ [0.25, 0.25, 0.25, 0.25] → random
        Late training:  probs ~ [0.02, 0.05, 0.03, 0.90] → mostly best action
        """
        state_t = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs, value = self.net(state_t)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return (action.item(), log_prob.item(), value.item(),
                probs.squeeze().cpu().numpy())

    def store(self, state, action, reward, log_prob, value, done):
        """Store one transition in the buffer."""
        self.states.append(state.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def update(self) -> dict:
        """
        PPO update — THE LEARNING STEP.

        Called after collecting enough experience (rollout_size transitions).
        Returns loss metrics for visualization.
        """
        if len(self.states) < 2:
            self._clear()
            return {}

        # Step 1: Compute GAE advantages
        advantages = []
        returns = []
        gae = 0.0

        for t in reversed(range(len(self.rewards))):
            next_value = self.values[t + 1] if t < len(self.rewards) - 1 else 0.0
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])

        # Convert to tensors
        states_t = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_t = torch.LongTensor(self.actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(self.log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Step 2: PPO epochs with mini-batches
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.update_epochs):
            indices = torch.randperm(len(self.states)).to(self.device)

            for start in range(0, len(indices), self.mini_batch_size):
                end = min(start + self.mini_batch_size, len(indices))
                if end - start < 4:
                    continue

                idx = indices[start:end]

                b_states = states_t[idx]
                b_actions = actions_t[idx]
                b_old_lp = old_log_probs_t[idx]
                b_adv = advantages_t[idx]
                b_ret = returns_t[idx]

                # Forward pass with current network
                new_probs, new_values = self.net(b_states)
                dist = Categorical(new_probs)
                new_lp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values.squeeze(-1), b_ret)

                # Total loss = policy + value - entropy bonus
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Backpropagation — THIS IS WHERE LEARNING HAPPENS
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        self._clear()

        if n_updates == 0:
            return {}

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def _clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "net": self.net.state_dict(),
            "opt": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        if os.path.exists(path):
            data = torch.load(path, map_location=self.device, weights_only=True)
            self.net.load_state_dict(data["net"])
            self.optimizer.load_state_dict(data["opt"])
            print(f"  Loaded checkpoint: {path}")


# ============================================================
# CURRICULUM — Progressive level unlocking
# ============================================================

class Curriculum:
    """
    Manages which levels are available for training.

    Tiers:
      Tier 0: Levels 1-5   (1 box, 5x5)
      Tier 1: Levels 6-10  (2 boxes, 6x6)
      Tier 2: Levels 11-20 (2 boxes, 7x7)
      Tier 3: Levels 21-30 (3 boxes, 7x7)
      Tier 4: Levels 31-40 (3 boxes, 8x8)
      Tier 5: Levels 41-50 (4 boxes, 8x8)
      Tier 6: Levels 51-60 (4 boxes, 9x9)

    Unlocks next tier when solve rate > threshold on current tier.
    """

    TIERS = [
        (0, 5),    # Tier 0: levels 0-4 (1-5)
        (5, 10),   # Tier 1: levels 5-9 (6-10)
        (10, 20),  # Tier 2: levels 10-19 (11-20)
        (20, 30),  # Tier 3: levels 20-29 (21-30)
        (30, 40),  # Tier 4: levels 30-39 (31-40)
        (40, 50),  # Tier 5: levels 40-49 (41-50)
        (50, 60),  # Tier 6: levels 50-59 (51-60)
    ]

    TIER_NAMES = [
        "Tier 0: Tutorial (1 box)",
        "Tier 1: Easy (2 box, 6x6)",
        "Tier 2: Medium (2 box, 7x7)",
        "Tier 3: Hard (3 box, 7x7)",
        "Tier 4: Harder (3 box, 8x8)",
        "Tier 5: Very Hard (4 box, 8x8)",
        "Tier 6: Expert (4 box, 9x9)",
    ]

    UNLOCK_THRESHOLD = 0.50  # 50% solve rate to unlock next tier

    def __init__(self, total_levels: int = 60):
        self.total_levels = total_levels
        self.current_tier = 0
        self.max_unlocked_tier = 0

        # Track per-level solve stats
        self.level_attempts: Dict[int, int] = {}
        self.level_solves: Dict[int, int] = {}

        # Recent per-tier tracking
        self.tier_recent_solves: Dict[int, deque] = {}
        for i in range(len(self.TIERS)):
            self.tier_recent_solves[i] = deque(maxlen=100)

    def get_available_levels(self) -> List[int]:
        """Return list of level indices available for training."""
        start, end = self.TIERS[0]
        pool = list(range(start, min(end, self.total_levels)))

        for tier in range(1, self.max_unlocked_tier + 1):
            start, end = self.TIERS[tier]
            pool.extend(range(start, min(end, self.total_levels)))

        return pool

    def get_weighted_pool(self) -> List[int]:
        """
        Return weighted level pool — harder unsolved levels sampled more.

        Strategy:
        - Levels in current tier: weight 3 (focus on frontier)
        - Levels in previous tiers (unsolved recently): weight 2
        - Levels in previous tiers (solved recently): weight 1
        """
        pool = []

        for tier in range(self.max_unlocked_tier + 1):
            start, end = self.TIERS[tier]
            is_current = (tier == self.max_unlocked_tier)

            for lvl in range(start, min(end, self.total_levels)):
                attempts = self.level_attempts.get(lvl, 0)
                solves = self.level_solves.get(lvl, 0)
                solve_rate = solves / max(attempts, 1)

                if is_current:
                    weight = 3  # current frontier — high priority
                elif solve_rate < 0.3:
                    weight = 2  # hard unsolved from earlier tiers
                else:
                    weight = 1  # already mostly solved

                pool.extend([lvl] * weight)

        return pool if pool else [0]

    def record_episode(self, level_idx: int, solved: bool):
        """Record an episode result."""
        self.level_attempts[level_idx] = self.level_attempts.get(level_idx, 0) + 1
        if solved:
            self.level_solves[level_idx] = self.level_solves.get(level_idx, 0) + 1

        # Find which tier this level belongs to
        for tier, (start, end) in enumerate(self.TIERS):
            if start <= level_idx < end:
                self.tier_recent_solves[tier].append(1.0 if solved else 0.0)
                break

        # Check for tier advancement
        self._check_advance()

    def _check_advance(self):
        """Check if we should unlock the next tier."""
        if self.max_unlocked_tier >= len(self.TIERS) - 1:
            return  # all tiers unlocked

        current = self.max_unlocked_tier
        recent = self.tier_recent_solves[current]

        if len(recent) >= 30:  # need at least 30 attempts
            rate = sum(recent) / len(recent)
            if rate >= self.UNLOCK_THRESHOLD:
                self.max_unlocked_tier += 1

    def get_tier_solve_rate(self, tier: int) -> float:
        """Get recent solve rate for a tier."""
        recent = self.tier_recent_solves.get(tier, deque())
        if len(recent) == 0:
            return 0.0
        return sum(recent) / len(recent)

    def get_overall_solve_rate(self) -> float:
        """Get overall solve rate across all attempted levels."""
        total_attempts = sum(self.level_attempts.values())
        total_solves = sum(self.level_solves.values())
        if total_attempts == 0:
            return 0.0
        return total_solves / total_attempts

    def get_max_steps_for_level(self, level_idx: int) -> int:
        """Dynamic max steps based on difficulty tier."""
        for tier, (start, end) in enumerate(self.TIERS):
            if start <= level_idx < end:
                return [80, 120, 150, 200, 250, 300, 350][tier]
        return 200


# ============================================================
# RENDERER — Beautiful Training Visualization
# ============================================================

class GameRenderer:
    """Render Sokoban grid and training stats."""

    # Colors
    BG = (18, 18, 30)
    FLOOR_COLOR = (32, 32, 48)
    GRID_LINE = (42, 42, 58)
    WALL_MAIN = (65, 70, 85)
    WALL_TOP = (80, 85, 100)
    WALL_BOTTOM = (45, 50, 60)
    BOX_COLOR = (240, 180, 40)
    BOX_INNER = (210, 150, 30)
    BOX_ON_TARGET = (60, 200, 120)
    TARGET_COLOR = (52, 211, 153)
    PLAYER_COLOR = (99, 102, 241)
    TEXT_COLOR = (220, 225, 240)
    DIM_TEXT = (130, 140, 160)
    PANEL_BG = (25, 25, 40)

    def __init__(self, game_area_size=500):
        self.game_area = game_area_size

    def render_state(self, surface, state: SokobanState, offset_x=10, offset_y=10):
        """Render a SokobanState onto a surface."""
        w, h = state.width, state.height
        cell = min(self.game_area // max(w, h), 64)
        total_w = cell * w
        total_h = cell * h
        ox = offset_x + (self.game_area - total_w) // 2
        oy = offset_y + (self.game_area - total_h) // 2

        for y in range(h):
            for x in range(w):
                rx, ry = ox + x * cell, oy + y * cell
                pos = (x, y)
                rect = pygame.Rect(rx, ry, cell, cell)

                if pos in state.walls:
                    pygame.draw.rect(surface, self.WALL_MAIN, rect, border_radius=cell // 8)
                    pygame.draw.line(surface, self.WALL_TOP, (rx + 2, ry + 1), (rx + cell - 2, ry + 1), 2)
                    pygame.draw.line(surface, self.WALL_BOTTOM, (rx + 2, ry + cell - 2), (rx + cell - 2, ry + cell - 2), 2)
                else:
                    pygame.draw.rect(surface, self.FLOOR_COLOR, rect)
                    pygame.draw.rect(surface, self.GRID_LINE, rect, 1)

                # Target
                if pos in state.targets and pos not in state.boxes:
                    cx, cy = rx + cell // 2, ry + cell // 2
                    pulse = 0.3 + 0.1 * math.sin(time.time() * 3)
                    tr = int(cell * pulse)
                    target_surf = pygame.Surface((tr * 2, tr * 2), pygame.SRCALPHA)
                    pygame.draw.circle(target_surf, (*self.TARGET_COLOR, 80), (tr, tr), tr)
                    surface.blit(target_surf, (cx - tr, cy - tr))
                    pygame.draw.circle(surface, self.TARGET_COLOR, (cx, cy), max(cell // 8, 3))

                # Box
                if pos in state.boxes:
                    on_target = pos in state.targets
                    inner = pygame.Rect(rx + 3, ry + 3, cell - 6, cell - 6)

                    if on_target:
                        # Green glow
                        glow = pygame.Surface((cell + 8, cell + 8), pygame.SRCALPHA)
                        pygame.draw.rect(glow, (60, 200, 120, 40),
                                         pygame.Rect(0, 0, cell + 8, cell + 8), border_radius=cell // 4)
                        surface.blit(glow, (rx - 4, ry - 4))
                        pygame.draw.rect(surface, self.BOX_ON_TARGET, inner, border_radius=cell // 5)
                        # Checkmark
                        m = cell // 3
                        pygame.draw.line(surface, (30, 140, 70),
                                         (rx + m, ry + cell // 2), (rx + cell // 2, ry + cell - m), 3)
                        pygame.draw.line(surface, (30, 140, 70),
                                         (rx + cell // 2, ry + cell - m), (rx + cell - m, ry + m), 3)
                    else:
                        pygame.draw.rect(surface, self.BOX_COLOR, inner, border_radius=cell // 5)
                        pygame.draw.rect(surface, self.BOX_INNER, inner.inflate(-4, -4), border_radius=cell // 6)
                        m = cell // 4
                        pygame.draw.line(surface, (200, 140, 20),
                                         (rx + m, ry + m), (rx + cell - m, ry + cell - m), 2)
                        pygame.draw.line(surface, (200, 140, 20),
                                         (rx + cell - m, ry + m), (rx + m, ry + cell - m), 2)

        # Player
        px, py = state.player
        cx = ox + px * cell + cell // 2
        cy = oy + py * cell + cell // 2
        radius = int(cell * 0.35)
        pygame.draw.circle(surface, (70, 80, 200), (cx + 1, cy + 1), radius)
        pygame.draw.circle(surface, self.PLAYER_COLOR, (cx, cy), radius)
        eye_r = max(cell // 12, 2)
        pygame.draw.circle(surface, (255, 255, 255), (cx - eye_r * 2, cy - eye_r), eye_r)
        pygame.draw.circle(surface, (255, 255, 255), (cx + eye_r * 2, cy - eye_r), eye_r)


# ============================================================
# MAIN — Training + Visualization
# ============================================================

class TrainAndWatch:
    def __init__(self):
        pygame.init()
        self.W, self.H = 1050, 750
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Real RL — Sokoban 60 Levels (PPO + Curriculum)")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.SysFont("consolas", 28, bold=True)
        self.font_med = pygame.font.SysFont("consolas", 20)
        self.font_small = pygame.font.SysFont("consolas", 14)
        self.font_tiny = pygame.font.SysFont("consolas", 11)

        # Load real levels
        levels_path = os.path.join(os.path.dirname(__file__), "levels", "classic_60.txt")
        loader = LevelLoader(levels_path)
        self.all_levels = [loader.get_level(i) for i in range(loader.get_total_levels())]
        print(f"  Loaded {len(self.all_levels)} levels from classic_60.txt")

        # RL components
        self.env = SokobanRLEnv(self.all_levels, max_steps=200)
        self.agent = PPOAgent(lr=2.5e-4, entropy_coef=0.05)
        self.curriculum = Curriculum(total_levels=len(self.all_levels))
        self.renderer = GameRenderer(game_area_size=500)

        # Try to load existing checkpoint
        self.checkpoint_path = "checkpoints/ppo_sokoban_60.pt"
        self.agent.load(self.checkpoint_path)

        # Metrics
        self.episode_rewards = deque(maxlen=500)
        self.episode_solved = deque(maxlen=500)
        self.total_episodes = 0
        self.total_steps = 0
        self.steps_since_update = 0
        self.rollout_size = 2048
        self.last_losses: dict = {}
        self.action_probs = [0.25, 0.25, 0.25, 0.25]

        # Learning curve
        self.curve_data: List[Tuple[int, float]] = []
        self.tier_change_episodes: List[int] = []

        # UI state
        self.mode = "visual"
        self.paused = False

        # Current episode state
        pool = self.curriculum.get_weighted_pool()
        self.obs = self.env.reset_from_pool(pool)
        self.episode_reward = 0.0
        self.episode_steps = 0

        # Entropy annealing
        self.initial_entropy_coef = 0.05
        self.min_entropy_coef = 0.005
        self.entropy_anneal_episodes = 20000

        # Notification
        self.notify_text = ""
        self.notify_timer = 0

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_f:
                        self.mode = "fast"
                    elif event.key == pygame.K_s:
                        self.mode = "slow"
                    elif event.key == pygame.K_v:
                        self.mode = "visual"

            if not self.paused:
                if self.mode == "fast":
                    for _ in range(200):
                        self._train_one_step()
                elif self.mode == "slow":
                    self._train_one_step()
                    pygame.time.wait(200)
                else:
                    for _ in range(5):
                        self._train_one_step()

            self._render()
            self.clock.tick(60 if self.mode != "slow" else 10)

        # Save on exit
        self.agent.save(self.checkpoint_path)
        print(f"\n  Saved checkpoint to {self.checkpoint_path}")
        print(f"  Total episodes: {self.total_episodes}")
        print(f"  Max tier unlocked: {self.curriculum.max_unlocked_tier}")
        pygame.quit()

    def _train_one_step(self):
        """One step of real RL training."""
        action, log_prob, value, probs = self.agent.select_action(self.obs)
        self.action_probs = probs.tolist()

        next_obs, reward, done, info = self.env.step(action)

        self.agent.store(self.obs, action, reward, log_prob, value, float(done))
        self.total_steps += 1
        self.steps_since_update += 1
        self.episode_reward += reward
        self.episode_steps += 1

        if done:
            solved = info.get("solved", False)
            self.episode_rewards.append(self.episode_reward)
            self.episode_solved.append(solved)
            self.total_episodes += 1

            # Record in curriculum
            old_tier = self.curriculum.max_unlocked_tier
            self.curriculum.record_episode(self.env.current_level_idx, solved)
            new_tier = self.curriculum.max_unlocked_tier

            if new_tier > old_tier:
                self.tier_change_episodes.append(self.total_episodes)
                self.notify_text = f"TIER UP! {Curriculum.TIER_NAMES[new_tier]}"
                self.notify_timer = 180

            # Entropy annealing
            progress = min(self.total_episodes / self.entropy_anneal_episodes, 1.0)
            self.agent.entropy_coef = self.initial_entropy_coef - progress * (self.initial_entropy_coef - self.min_entropy_coef)

            # Record curve data
            if self.total_episodes % 20 == 0 and len(self.episode_solved) >= 20:
                rate = sum(list(self.episode_solved)[-100:]) / min(len(self.episode_solved), 100) * 100
                self.curve_data.append((self.total_episodes, rate))

            # Reset with weighted level selection
            self.episode_reward = 0.0
            self.episode_steps = 0
            pool = self.curriculum.get_weighted_pool()
            self.env.max_steps = self.curriculum.get_max_steps_for_level(self.env.current_level_idx)
            self.obs = self.env.reset_from_pool(pool)
        else:
            self.obs = next_obs

        # PPO update
        if self.steps_since_update >= self.rollout_size:
            self.last_losses = self.agent.update()
            self.steps_since_update = 0

    def _render(self):
        """Full render of training visualization."""
        self.screen.fill(GameRenderer.BG)

        # LEFT: Game grid
        state = self.env.get_render_state()
        if state:
            self.renderer.render_state(self.screen, state, offset_x=15, offset_y=15)

        # Level info under grid
        lvl_idx = self.env.current_level_idx
        boxes = len(state.boxes) if state else 0
        on_target = state.n_boxes_on_target if state else 0
        lvl_text = f"Level {lvl_idx + 1}/60  |  {on_target}/{boxes} boxes on target  |  Step {self.env.steps}"
        self._text(lvl_text, 15, 530, self.font_small, GameRenderer.DIM_TEXT)

        # Current tier
        tier_name = Curriculum.TIER_NAMES[self.curriculum.max_unlocked_tier]
        self._text(tier_name, 15, 550, self.font_small, (99, 102, 241))

        # RIGHT: Stats panel
        panel_x = 560
        pygame.draw.rect(self.screen, GameRenderer.PANEL_BG,
                         pygame.Rect(panel_x, 0, self.W - panel_x, self.H))

        x = panel_x + 20
        y = 20

        # Episode counter
        self._text(f"Episode: {self.total_episodes:,}", x, y, self.font_large, GameRenderer.TEXT_COLOR)
        y += 35
        self._text(f"Steps: {self.total_steps:,}", x, y, self.font_small, GameRenderer.DIM_TEXT)
        y += 25

        # Overall solve rate
        overall = self.curriculum.get_overall_solve_rate() * 100
        color = self._rate_color(overall)
        self._text("Overall Solve Rate:", x, y, self.font_small, GameRenderer.DIM_TEXT)
        y += 20
        self._text(f"{overall:.1f}%", x, y, self.font_large, color)
        y += 40

        # Per-tier solve rates
        self._text("Tier Progress:", x, y, self.font_small, GameRenderer.DIM_TEXT)
        y += 20

        for tier in range(len(Curriculum.TIERS)):
            start, end = Curriculum.TIERS[tier]
            rate = self.curriculum.get_tier_solve_rate(tier) * 100
            unlocked = tier <= self.curriculum.max_unlocked_tier
            active = tier == self.curriculum.max_unlocked_tier

            if unlocked:
                label_color = (220, 225, 240) if active else (160, 165, 180)
                bar_color = self._rate_color(rate)
            else:
                label_color = (70, 75, 90)
                bar_color = (50, 55, 70)

            # Tier label
            prefix = ">" if active else " "
            lock = "" if unlocked else " [locked]"
            self._text(f"{prefix} T{tier} (L{start + 1}-{end}){lock}", x, y, self.font_tiny, label_color)

            # Progress bar
            bar_x = x + 160
            bar_w = 200
            bar_h = 12
            pygame.draw.rect(self.screen, (40, 40, 55),
                             pygame.Rect(bar_x, y + 2, bar_w, bar_h), border_radius=3)
            if unlocked:
                fill_w = int(bar_w * min(rate / 100, 1.0))
                if fill_w > 0:
                    pygame.draw.rect(self.screen, bar_color,
                                     pygame.Rect(bar_x, y + 2, fill_w, bar_h), border_radius=3)
                self._text(f"{rate:.0f}%", bar_x + bar_w + 8, y, self.font_tiny, bar_color)

            y += 18

        y += 10

        # Learning curve graph
        graph_rect = pygame.Rect(panel_x + 15, y, self.W - panel_x - 30, 170)
        pygame.draw.rect(self.screen, (30, 30, 48), graph_rect, border_radius=8)
        pygame.draw.rect(self.screen, (50, 55, 70), graph_rect, 1, border_radius=8)
        self._text("Solve Rate (recent 100 episodes)", panel_x + 25, y + 5, self.font_tiny, (100, 110, 130))

        gy = graph_rect.y + 25
        gh = graph_rect.height - 35
        gx = graph_rect.x + 10
        gw = graph_rect.width - 20

        for pct in [25, 50, 75]:
            line_y = gy + gh - int(gh * pct / 100)
            pygame.draw.line(self.screen, (45, 45, 60), (gx, line_y), (gx + gw, line_y), 1)
            self._text(f"{pct}%", gx - 5, line_y - 6, self.font_tiny, (70, 80, 100))

        if len(self.curve_data) >= 2:
            max_ep = max(d[0] for d in self.curve_data)
            points = []
            for ep, rate in self.curve_data:
                px = gx + int(gw * ep / max(max_ep, 1))
                py = gy + gh - int(gh * min(rate, 100) / 100)
                points.append((px, py))
            if len(points) >= 2:
                pygame.draw.lines(self.screen, (99, 102, 241), False, points, 2)

        # Tier change markers
        if self.curve_data:
            max_ep = max(d[0] for d in self.curve_data)
            for ep in self.tier_change_episodes:
                if ep <= max_ep:
                    tx = gx + int(gw * ep / max(max_ep, 1))
                    pygame.draw.line(self.screen, (52, 211, 153), (tx, gy), (tx, gy + gh), 1)

        y = graph_rect.bottom + 15

        # Action probabilities
        self._text("Action Probabilities:", x, y, self.font_small, (100, 110, 130))
        y += 22
        labels = ["UP:", "DOWN:", "LEFT:", "RIGHT:"]
        for label, prob in zip(labels, self.action_probs):
            self._text(label, x, y, self.font_small, GameRenderer.DIM_TEXT)
            bar_x = x + 80
            bar_w = 150
            bar_h = 14
            pygame.draw.rect(self.screen, (40, 40, 55),
                             pygame.Rect(bar_x, y + 2, bar_w, bar_h), border_radius=3)
            fill_w = int(bar_w * prob)
            if fill_w > 0:
                pygame.draw.rect(self.screen, (99, 102, 241),
                                 pygame.Rect(bar_x, y + 2, fill_w, bar_h), border_radius=3)
            self._text(f"{prob:.2f}", bar_x + bar_w + 8, y, self.font_small, (180, 190, 210))
            y += 20

        y += 10

        # Loss info
        if self.last_losses:
            self._text(f"Policy Loss: {self.last_losses.get('policy_loss', 0):.4f}",
                       x, y, self.font_tiny, (100, 110, 130))
            y += 16
            self._text(f"Value Loss: {self.last_losses.get('value_loss', 0):.4f}",
                       x, y, self.font_tiny, (100, 110, 130))
            y += 16
            self._text(f"Entropy: {self.last_losses.get('entropy', 0):.4f}  (coef={self.agent.entropy_coef:.4f})",
                       x, y, self.font_tiny, (100, 110, 130))

        # Mode indicator
        mode_text = {"visual": "VISUAL", "fast": "FAST (200x)", "slow": "SLOW (1x)"}
        mode_color = {"visual": (52, 211, 153), "fast": (251, 191, 36), "slow": (147, 197, 253)}
        self._text(f"Mode: {mode_text[self.mode]}", x, self.H - 55,
                   self.font_med, mode_color[self.mode])

        self._text("Space=Pause  F=Fast  S=Slow  V=Visual  ESC=Quit+Save",
                   panel_x + 20, self.H - 25, self.font_tiny, (70, 80, 100))

        # Tier up notification
        if self.notify_timer > 0:
            self.notify_timer -= 1
            alpha = min(255, self.notify_timer * 3)
            overlay = pygame.Surface((540, 80), pygame.SRCALPHA)
            overlay.fill((18, 18, 30, min(200, alpha)))
            self.screen.blit(overlay, (10, 230))
            text = self.font_large.render(self.notify_text, True, (52, 211, 153))
            self.screen.blit(text, (30, 250))

        # Pause overlay
        if self.paused:
            overlay = pygame.Surface((540, 540), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            self.screen.blit(overlay, (10, 10))
            text = self.font_large.render("PAUSED", True, (255, 255, 255))
            self.screen.blit(text, (210, 270))

        pygame.display.flip()

    def _text(self, text, x, y, font, color):
        surf = font.render(str(text), True, color)
        self.screen.blit(surf, (x, y))

    def _rate_color(self, rate: float) -> Tuple[int, int, int]:
        if rate < 15:
            return (239, 68, 68)
        elif rate < 30:
            return (251, 146, 60)
        elif rate < 60:
            return (251, 191, 36)
        else:
            return (52, 211, 153)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Real RL Training — Sokoban 60 Levels")
    print("=" * 60)
    print()
    print("  The AI starts RANDOM and learns through trial and error")
    print("  on 60 real Sokoban levels with curriculum learning.")
    print()
    print("  Algorithm: PPO (Proximal Policy Optimization)")
    print("  Network:   CNN (3 conv layers + FC)")
    print("  Curriculum: 7 tiers, unlocks at 50% solve rate")
    print()
    print("  Controls:")
    print("    Space = Pause/Resume")
    print("    F = Fast mode (train 200 steps/frame)")
    print("    S = Slow mode (watch every step)")
    print("    V = Visual mode (normal speed)")
    print("    ESC = Quit and save checkpoint")
    print()

    app = TrainAndWatch()
    app.run()
