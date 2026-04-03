"""
Real PPO Reinforcement Learning for Sokoban.
The agent starts knowing NOTHING and learns through trial and error.
Watch the learning curve go from 0% to 60%+ over thousands of episodes.
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
import json
import sys
from collections import deque
import time
import math

# ============================================================
# SOKOBAN ENVIRONMENT — Simple training levels
# ============================================================

class SokobanEnv:
    """
    Sokoban environment for RL training.

    IMPORTANT: We do NOT use the classic 60 levels for training.
    Classic levels have 6-10 boxes and require 100+ optimal moves.
    That is WAY too hard for RL from scratch.

    Instead, we generate simple training levels:
    Phase 1: 5x5-6x6 grid, 1 box, 1 target
    Phase 2: 6x6-7x7 grid, 2 boxes, 2 targets
    Phase 3: 7x7-8x8 grid, 3 boxes, 3 targets
    """

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # row, col

    FLOOR, WALL, BOX, TARGET, BOX_ON_TARGET, PLAYER, PLAYER_ON_TARGET = 0, 1, 2, 3, 4, 5, 6

    def __init__(self, phase=1):
        self.phase = phase
        self.max_steps = {1: 120, 2: 200, 3: 300}[phase]
        self.levels = self._generate_levels(phase)
        self.current_level_idx = 0
        self.grid = None
        self.player_pos = None  # (row, col)
        self.steps = 0
        self.num_boxes = {1: 1, 2: 2, 3: 3}[phase]
        self.grid_size = {1: 6, 2: 7, 3: 8}[phase]  # padded size for neural network
        self.reset()

    def _generate_levels(self, phase):
        """
        Generate solvable Sokoban levels.

        Algorithm for 1-box levels:
        1. Create NxN grid with walls on border
        2. Place target at random interior position
        3. Place box at random interior position (not on target, not in corner)
        4. Place player at random interior position (not on box or target)
        5. Check solvability with BFS (state = player_pos + box_pos)
        6. If not solvable, try again

        For 2-box and 3-box: same but with multiple boxes/targets.
        BFS state = player_pos + frozenset(box_positions)
        """
        levels = []
        configs = {
            1: {"min_size": 5, "max_size": 6, "boxes": 1, "walls": 1, "count": 200},
            2: {"min_size": 6, "max_size": 7, "boxes": 2, "walls": 2, "count": 150},
            3: {"min_size": 7, "max_size": 8, "boxes": 3, "walls": 3, "count": 100},
        }
        cfg = configs[phase]

        attempts = 0
        max_attempts = cfg["count"] * 200  # don't loop forever

        while len(levels) < cfg["count"] and attempts < max_attempts:
            attempts += 1
            size = random.randint(cfg["min_size"], cfg["max_size"])
            level = self._try_generate_one(size, cfg["boxes"], cfg["walls"])
            if level is not None:
                levels.append(level)
                if len(levels) % 20 == 0:
                    print(f"  Phase {phase}: generated {len(levels)}/{cfg['count']} levels...")

        if len(levels) < 10:
            # Fallback: generate trivial levels
            for _ in range(50):
                size = cfg["min_size"]
                grid = [[self.WALL if r == 0 or c == 0 or r == size-1 or c == size-1 else self.FLOOR
                         for c in range(size)] for r in range(size)]
                # Place 1 box and 1 target in straightforward position
                grid[2][2] = self.TARGET
                grid[2][3] = self.BOX
                player = (3, 3)
                levels.append({"grid": grid, "player": player, "size": size})

        print(f"  Phase {phase}: {len(levels)} levels ready!")
        return levels

    def _try_generate_one(self, size, num_boxes, num_interior_walls):
        """Try to generate one solvable level. Return None if failed."""
        # Create empty grid with border walls
        grid = [[self.WALL if r == 0 or c == 0 or r == size-1 or c == size-1 else self.FLOOR
                 for c in range(size)] for r in range(size)]

        # Get all interior positions
        interior = [(r, c) for r in range(1, size-1) for c in range(1, size-1)]
        random.shuffle(interior)

        # Add some interior walls (but not too many)
        walls_placed = 0
        wall_positions = []
        for pos in interior[:num_interior_walls * 3]:  # try more than needed
            if walls_placed >= num_interior_walls:
                break
            r, c = pos
            # Don't place wall if it would block too much
            neighbors_wall = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                               if grid[r+dr][c+dc] == self.WALL)
            if neighbors_wall <= 1:  # at most 1 adjacent wall
                grid[r][c] = self.WALL
                wall_positions.append(pos)
                walls_placed += 1

        # Get remaining free positions
        free = [(r, c) for r in range(1, size-1) for c in range(1, size-1)
                if grid[r][c] == self.FLOOR]

        if len(free) < num_boxes * 2 + 1:
            return None

        random.shuffle(free)

        # Place targets
        targets = []
        for i in range(num_boxes):
            pos = free.pop()
            targets.append(pos)
            grid[pos[0]][pos[1]] = self.TARGET

        # Place boxes — avoid corners (deadlock from start)
        boxes = []
        for i in range(num_boxes):
            placed = False
            for j in range(len(free)):
                r, c = free[j]
                # Check not in a corner
                adj_walls = [(grid[r-1][c] == self.WALL), (grid[r+1][c] == self.WALL),
                             (grid[r][c-1] == self.WALL), (grid[r][c+1] == self.WALL)]
                corner = (adj_walls[0] and adj_walls[2]) or (adj_walls[0] and adj_walls[3]) or \
                         (adj_walls[1] and adj_walls[2]) or (adj_walls[1] and adj_walls[3])
                if not corner and (r, c) not in targets:
                    boxes.append(free.pop(j))
                    placed = True
                    break
            if not placed:
                return None

        # Place player
        if not free:
            return None
        player = free.pop()

        # Mark boxes on grid (temporarily, for BFS)
        for br, bc in boxes:
            if grid[br][bc] == self.TARGET:
                grid[br][bc] = self.BOX_ON_TARGET
            else:
                grid[br][bc] = self.BOX

        # Check solvability with BFS
        max_states = 50000 if num_boxes <= 2 else 200000
        if not self._is_solvable(grid, player, boxes, targets, size, max_states=max_states):
            return None

        # Store the level in clean form (boxes as separate data, grid has only walls/floor/targets)
        clean_grid = [[self.WALL if grid[r][c] == self.WALL else
                       (self.TARGET if (r,c) in targets else self.FLOOR)
                       for c in range(size)] for r in range(size)]

        return {
            "grid": clean_grid,
            "player": player,
            "boxes": list(boxes),
            "targets": list(targets),
            "size": size
        }

    def _is_solvable(self, grid, player, boxes, targets, size, max_states=50000):
        """BFS solvability check."""
        from collections import deque as bfs_deque

        target_set = frozenset(targets)
        initial_state = (player, frozenset(boxes))

        visited = {initial_state}
        queue = bfs_deque([initial_state])

        while queue and len(visited) < max_states:
            (pr, pc), box_set = queue.popleft()

            # Check if solved
            if box_set == target_set:
                return True

            for action in range(4):
                dr, dc = self.MOVES[action]
                nr, nc = pr + dr, pc + dc

                # Out of bounds or wall
                if nr < 0 or nr >= size or nc < 0 or nc >= size:
                    continue
                if grid[nr][nc] == self.WALL:
                    continue

                new_boxes = set(box_set)

                if (nr, nc) in box_set:
                    # Push box
                    bnr, bnc = nr + dr, nc + dc
                    if bnr < 0 or bnr >= size or bnc < 0 or bnc >= size:
                        continue
                    if grid[bnr][bnc] == self.WALL or (bnr, bnc) in box_set:
                        continue

                    # Check deadlock: box pushed into corner
                    up_wall = grid[bnr-1][bnc] == self.WALL
                    down_wall = grid[bnr+1][bnc] == self.WALL
                    left_wall = grid[bnr][bnc-1] == self.WALL
                    right_wall = grid[bnr][bnc+1] == self.WALL

                    is_corner = (up_wall and left_wall) or (up_wall and right_wall) or \
                                (down_wall and left_wall) or (down_wall and right_wall)

                    if is_corner and (bnr, bnc) not in target_set:
                        continue  # deadlock, prune

                    new_boxes.remove((nr, nc))
                    new_boxes.add((bnr, bnc))

                new_state = ((nr, nc), frozenset(new_boxes))
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append(new_state)

        return False

    def reset(self, level_idx=None):
        """Reset to a random (or specific) level."""
        if level_idx is None:
            self.current_level_idx = random.randint(0, len(self.levels) - 1)
        else:
            self.current_level_idx = level_idx % len(self.levels)

        level = self.levels[self.current_level_idx]
        size = level["size"]

        # Deep copy grid
        self.grid = [row[:] for row in level["grid"]]
        self.player_pos = level["player"]
        self.steps = 0

        # Place boxes
        if "boxes" in level:
            for br, bc in level["boxes"]:
                if self.grid[br][bc] == self.TARGET:
                    self.grid[br][bc] = self.BOX_ON_TARGET
                else:
                    self.grid[br][bc] = self.BOX

        self._size = size
        return self._get_obs()

    def _get_obs(self):
        """
        Return observation as 5 binary channels, padded to (5, grid_size, grid_size).

        Channel 0: walls
        Channel 1: boxes (not on target)
        Channel 2: targets (without box on them)
        Channel 3: player
        Channel 4: boxes on target

        This representation is crucial — the CNN needs spatial binary features.
        """
        s = self.grid_size  # padded size
        obs = np.zeros((5, s, s), dtype=np.float32)

        for r in range(self._size):
            for c in range(self._size):
                cell = self.grid[r][c]
                if cell == self.WALL:
                    obs[0, r, c] = 1.0
                elif cell == self.BOX:
                    obs[1, r, c] = 1.0
                elif cell == self.TARGET:
                    obs[2, r, c] = 1.0
                elif cell == self.BOX_ON_TARGET:
                    obs[4, r, c] = 1.0

        pr, pc = self.player_pos
        obs[3, pr, pc] = 1.0

        # If player is on target, also mark target channel
        if self.grid[pr][pc] == self.TARGET or self.grid[pr][pc] == self.PLAYER_ON_TARGET:
            obs[2, pr, pc] = 1.0

        return obs

    def step(self, action):
        """
        Execute action and return (obs, reward, done, info).

        Reward shaping is CRITICAL for Sokoban RL:
        Without distance-based rewards, the agent only gets +100 on solve,
        which happens so rarely that learning is essentially impossible.
        """
        dr, dc = self.MOVES[action]
        pr, pc = self.player_pos
        nr, nc = pr + dr, pc + dc

        reward = -0.1  # step penalty
        solved = False
        deadlock = False
        pushed = False

        # Check bounds
        if nr < 0 or nr >= self._size or nc < 0 or nc >= self._size:
            reward = -1.0  # invalid move
            self.steps += 1
            done = self.steps >= self.max_steps
            return self._get_obs(), reward, done, {"solved": False, "deadlock": False}

        target_cell = self.grid[nr][nc]

        if target_cell == self.WALL:
            reward = -1.0  # bumped into wall

        elif target_cell in (self.BOX, self.BOX_ON_TARGET):
            # Try to push box
            bnr, bnc = nr + dr, nc + dc

            if (bnr < 0 or bnr >= self._size or bnc < 0 or bnc >= self._size or
                self.grid[bnr][bnc] in (self.WALL, self.BOX, self.BOX_ON_TARGET)):
                reward = -1.0  # can't push
            else:
                # Compute distance BEFORE push
                old_box_dist = self._min_target_distance(nr, nc)

                # Move box
                box_was_on_target = (target_cell == self.BOX_ON_TARGET)
                box_lands_on_target = (self.grid[bnr][bnc] == self.TARGET)

                # Update box destination
                if box_lands_on_target:
                    self.grid[bnr][bnc] = self.BOX_ON_TARGET
                    reward += 10.0  # box on target!
                else:
                    self.grid[bnr][bnc] = self.BOX

                # Update box source
                if box_was_on_target:
                    self.grid[nr][nc] = self.TARGET
                    reward -= 10.0  # box removed from target
                else:
                    self.grid[nr][nc] = self.FLOOR

                # Move player
                self._move_player(nr, nc)
                pushed = True

                # Distance-based reward shaping
                new_box_dist = self._min_target_distance(bnr, bnc)
                if new_box_dist < old_box_dist:
                    reward += 1.0  # moved closer to target
                elif new_box_dist > old_box_dist:
                    reward -= 1.0  # moved further from target

                # Check deadlock
                if self._is_deadlocked(bnr, bnc):
                    reward -= 50.0
                    deadlock = True

                # Check solved
                if self._is_solved():
                    reward += 100.0
                    solved = True

        elif target_cell in (self.FLOOR, self.TARGET):
            # Simple move
            self._move_player(nr, nc)

        self.steps += 1
        done = solved or deadlock or (self.steps >= self.max_steps)

        return self._get_obs(), reward, done, {"solved": solved, "deadlock": deadlock, "pushed": pushed}

    def _move_player(self, nr, nc):
        """Move player to new position."""
        pr, pc = self.player_pos
        # Restore old cell
        if self.grid[pr][pc] == self.PLAYER_ON_TARGET:
            self.grid[pr][pc] = self.TARGET
        # Don't overwrite box/target cells — player overlays
        self.player_pos = (nr, nc)

    def _min_target_distance(self, br, bc):
        """Manhattan distance from box at (br,bc) to nearest target."""
        min_dist = 999
        for r in range(self._size):
            for c in range(self._size):
                if self.grid[r][c] in (self.TARGET, self.PLAYER_ON_TARGET):
                    dist = abs(br - r) + abs(bc - c)
                    min_dist = min(min_dist, dist)
        # Also check BOX_ON_TARGET positions as occupied targets
        return min_dist

    def _is_deadlocked(self, br, bc):
        """Check if box at (br,bc) is stuck in a corner (not on target)."""
        if self.grid[br][bc] == self.BOX_ON_TARGET:
            return False  # on target, fine

        up = self.grid[br-1][bc] == self.WALL if br > 0 else True
        down = self.grid[br+1][bc] == self.WALL if br < self._size-1 else True
        left = self.grid[br][bc-1] == self.WALL if bc > 0 else True
        right = self.grid[br][bc+1] == self.WALL if bc < self._size-1 else True

        return (up and left) or (up and right) or (down and left) or (down and right)

    def _is_solved(self):
        """Check if all boxes are on targets."""
        for r in range(self._size):
            for c in range(self._size):
                if self.grid[r][c] == self.BOX:  # box NOT on target
                    return False
                if self.grid[r][c] == self.TARGET:  # target without box
                    return False
        return True

    def get_render_data(self):
        """Return data needed for PyGame rendering."""
        return {
            "grid": [row[:] for row in self.grid],
            "player_pos": self.player_pos,
            "size": self._size,
            "steps": self.steps,
            "max_steps": self.max_steps,
        }


# ============================================================
# NEURAL NETWORK — CNN Policy + Value
# ============================================================

class SokobanNet(nn.Module):
    """
    Convolutional neural network for Sokoban.

    Input: (batch, 5, grid_size, grid_size) — 5 binary channels
    Output: action_probs (batch, 4), value (batch, 1)

    THIS is what learns. At the start, it outputs random probabilities.
    After thousands of episodes of training, it learns to output good actions.
    """

    def __init__(self, grid_size):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        flat_size = 64 * grid_size * grid_size

        self.fc = nn.Linear(flat_size, 256)
        self.policy_head = nn.Linear(256, 4)
        self.value_head = nn.Linear(256, 1)

        # Initialize weights with small values — IMPORTANT for stable training
        for layer in [self.conv1, self.conv2, self.conv3, self.fc, self.policy_head, self.value_head]:
            if hasattr(layer, 'weight'):
                nn.init.orthogonal_(layer.weight, gain=0.01 if layer == self.policy_head else np.sqrt(2))
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # flatten
        x = F.relu(self.fc(x))

        # Policy: probability distribution over 4 actions
        action_logits = self.policy_head(x)
        action_probs = F.softmax(action_logits, dim=-1)

        # Value: estimated future reward
        value = self.value_head(x)

        return action_probs, value


# ============================================================
# PPO AGENT — The actual RL algorithm
# ============================================================

class PPOAgent:
    """
    Proximal Policy Optimization.

    How it works:
    1. Collect experience by running episodes (agent acts in environment)
    2. Compute advantages (how much better was this action than average?)
    3. Update neural network to make good actions more likely
    4. Clip updates to prevent too-large changes (stability)
    5. Repeat

    The agent starts with random weights → random actions.
    Over time, weights update → better actions → higher rewards → better weights.
    This is the learning loop.
    """

    def __init__(self, grid_size):
        self.net = SokobanNet(grid_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=2.5e-4, eps=1e-5)

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_eps = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.update_epochs = 4
        self.mini_batch_size = 128

        # Experience storage — cleared after each PPO update
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def select_action(self, state_np):
        """
        Given state as numpy array, return (action, log_prob, value).

        CRITICAL: We SAMPLE from the probability distribution, not argmax.
        Early training: probs ≈ [0.25, 0.25, 0.25, 0.25] → random action
        Late training: probs ≈ [0.02, 0.05, 0.03, 0.90] → almost always RIGHT

        This sampling IS the exploration mechanism.
        """
        state_tensor = torch.FloatTensor(state_np).unsqueeze(0)  # (1, 5, H, W)

        with torch.no_grad():
            action_probs, value = self.net(state_tensor)

        dist = Categorical(action_probs)
        action = dist.sample()  # SAMPLE, not argmax!
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item(), action_probs.squeeze().numpy()

    def store(self, state, action, reward, log_prob, value, done):
        """Store one transition."""
        self.states.append(state.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def update(self):
        """
        PPO update — THE LEARNING STEP.

        This is where the neural network weights actually change.
        Called after collecting enough experience (steps_per_update transitions).
        """
        if len(self.states) < 2:
            self._clear()
            return {}

        # Step 1: Compute GAE advantages
        advantages = []
        returns = []
        gae = 0

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])

        # Convert to tensors
        states_t = torch.FloatTensor(np.array(self.states))
        actions_t = torch.LongTensor(self.actions)
        old_log_probs_t = torch.FloatTensor(self.log_probs)
        advantages_t = torch.FloatTensor(advantages)
        returns_t = torch.FloatTensor(returns)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Step 2: PPO update epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(self.update_epochs):
            # Shuffle and create mini-batches
            indices = torch.randperm(len(self.states))

            for start in range(0, len(indices), self.mini_batch_size):
                end = start + self.mini_batch_size
                if end > len(indices):
                    break

                batch_idx = indices[start:end]

                b_states = states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_log_probs = old_log_probs_t[batch_idx]
                b_advantages = advantages_t[batch_idx]
                b_returns = returns_t[batch_idx]

                # Forward pass with CURRENT network (weights have been updating)
                new_probs, new_values = self.net(b_states)
                dist = Categorical(new_probs)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values.squeeze(-1), b_returns)

                # Total loss
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

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({"net": self.net.state_dict(), "opt": self.optimizer.state_dict()}, path)

    def load(self, path):
        if os.path.exists(path):
            data = torch.load(path, map_location="cpu")
            self.net.load_state_dict(data["net"])
            self.optimizer.load_state_dict(data["opt"])


# ============================================================
# RENDERER — Beautiful Sokoban Visualization
# ============================================================

class GameRenderer:
    """Render Sokoban grid beautifully."""

    # Colors
    BG = (18, 18, 30)
    FLOOR_COLOR = (32, 32, 48)
    GRID_LINE = (42, 42, 58)
    WALL_MAIN = (65, 70, 85)
    WALL_TOP = (80, 85, 100)
    WALL_BOTTOM = (45, 50, 60)
    BOX_COLOR = (240, 180, 40)
    BOX_INNER = (210, 150, 30)
    BOX_ON_TARGET_COLOR = (60, 200, 120)
    TARGET_COLOR = (52, 211, 153)
    PLAYER_COLOR = (99, 102, 241)
    TEXT_COLOR = (220, 225, 240)
    DIM_TEXT = (130, 140, 160)
    PANEL_BG = (25, 25, 40)

    def __init__(self, game_area_size=550):
        self.game_area = game_area_size

    def render_grid(self, surface, grid_data, offset_x=10, offset_y=10):
        """Render the Sokoban grid onto a surface."""
        grid = grid_data["grid"]
        player = grid_data["player_pos"]
        size = grid_data["size"]

        cell = min(self.game_area // size, 80)
        total = cell * size
        ox = offset_x + (self.game_area - total) // 2
        oy = offset_y + (self.game_area - total) // 2

        # Draw floor
        for r in range(size):
            for c in range(size):
                x, y = ox + c * cell, oy + r * cell
                rect = pygame.Rect(x, y, cell, cell)

                val = grid[r][c]

                if val == SokobanEnv.WALL:
                    pygame.draw.rect(surface, self.WALL_MAIN, rect, border_radius=cell//8)
                    pygame.draw.line(surface, self.WALL_TOP, (x+2, y+1), (x+cell-2, y+1), 2)
                    pygame.draw.line(surface, self.WALL_BOTTOM, (x+2, y+cell-2), (x+cell-2, y+cell-2), 2)
                else:
                    pygame.draw.rect(surface, self.FLOOR_COLOR, rect)
                    pygame.draw.rect(surface, self.GRID_LINE, rect, 1)

                if val == SokobanEnv.TARGET:
                    # Pulsing target
                    pulse = 0.3 + 0.1 * math.sin(time.time() * 3)
                    tr = int(cell * pulse)
                    cx, cy = x + cell // 2, y + cell // 2
                    target_surf = pygame.Surface((tr*2, tr*2), pygame.SRCALPHA)
                    pygame.draw.circle(target_surf, (*self.TARGET_COLOR, 80), (tr, tr), tr)
                    surface.blit(target_surf, (cx - tr, cy - tr))
                    pygame.draw.circle(surface, self.TARGET_COLOR, (cx, cy), max(cell // 8, 3))

                elif val == SokobanEnv.BOX:
                    inner = pygame.Rect(x + 3, y + 3, cell - 6, cell - 6)
                    pygame.draw.rect(surface, self.BOX_COLOR, inner, border_radius=cell//5)
                    pygame.draw.rect(surface, self.BOX_INNER, inner.inflate(-4, -4), border_radius=cell//6)
                    # X mark
                    m = cell // 4
                    pygame.draw.line(surface, (200, 140, 20), (x+m, y+m), (x+cell-m, y+cell-m), 2)
                    pygame.draw.line(surface, (200, 140, 20), (x+cell-m, y+m), (x+m, y+cell-m), 2)

                elif val == SokobanEnv.BOX_ON_TARGET:
                    # Green glow
                    glow_surf = pygame.Surface((cell+8, cell+8), pygame.SRCALPHA)
                    pygame.draw.rect(glow_surf, (60, 200, 120, 40),
                                   pygame.Rect(0, 0, cell+8, cell+8), border_radius=cell//4)
                    surface.blit(glow_surf, (x-4, y-4))
                    # Green box
                    inner = pygame.Rect(x + 3, y + 3, cell - 6, cell - 6)
                    pygame.draw.rect(surface, self.BOX_ON_TARGET_COLOR, inner, border_radius=cell//5)
                    # Checkmark
                    m = cell // 3
                    pygame.draw.line(surface, (30, 140, 70), (x+m, y+cell//2), (x+cell//2, y+cell-m), 3)
                    pygame.draw.line(surface, (30, 140, 70), (x+cell//2, y+cell-m), (x+cell-m, y+m), 3)

        # Draw player
        pr, pc = player
        px = ox + pc * cell + cell // 2
        py = oy + pr * cell + cell // 2
        radius = int(cell * 0.35)
        pygame.draw.circle(surface, self.PLAYER_COLOR, (px, py), radius)
        # Eyes (simple white dots)
        eye_r = max(cell // 12, 2)
        pygame.draw.circle(surface, (255, 255, 255), (px - eye_r*2, py - eye_r), eye_r)
        pygame.draw.circle(surface, (255, 255, 255), (px + eye_r*2, py - eye_r), eye_r)


# ============================================================
# MAIN TRAINING VISUALIZATION
# ============================================================

class TrainAndWatch:
    def __init__(self):
        pygame.init()
        self.W, self.H = 1000, 700
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Real RL - Watch AI Learn Sokoban From Scratch")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("consolas", 28, bold=True)
        self.font_med = pygame.font.SysFont("consolas", 20)
        self.font_small = pygame.font.SysFont("consolas", 14)
        self.font_tiny = pygame.font.SysFont("consolas", 11)

        # RL components
        self.phase = 1
        self.env = SokobanEnv(phase=self.phase)
        self.agent = PPOAgent(grid_size=self.env.grid_size)
        self.renderer = GameRenderer(game_area_size=550)

        # Metrics
        self.episode_rewards = deque(maxlen=500)
        self.episode_solved = deque(maxlen=500)
        self.episode_deadlocks = deque(maxlen=500)
        self.episode_steps_list = deque(maxlen=500)
        self.total_episodes = 0
        self.total_steps = 0
        self.steps_since_update = 0
        self.steps_per_update = 512
        self.last_losses = {}
        self.action_probs = [0.25, 0.25, 0.25, 0.25]

        # Learning curve
        self.curve_data = []
        self.phase_changes = []

        # State
        self.mode = "visual"  # visual, fast, slow
        self.paused = False
        self.obs = self.env.reset()
        self.episode_reward = 0
        self.episode_steps = 0

        # Phase up text
        self.phase_up_text = ""
        self.phase_up_timer = 0

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
                    for _ in range(3):
                        self._train_one_step()

            self._render()
            self.clock.tick(60 if self.mode != "slow" else 10)

        # Save on exit
        self.agent.save(f"checkpoints/ppo_phase{self.phase}.pt")
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
            self.episode_rewards.append(self.episode_reward)
            self.episode_solved.append(info.get("solved", False))
            self.episode_deadlocks.append(info.get("deadlock", False))
            self.episode_steps_list.append(self.episode_steps)
            self.total_episodes += 1
            self.episode_reward = 0
            self.episode_steps = 0
            self.obs = self.env.reset()

            # Record learning curve
            if self.total_episodes % 20 == 0 and len(self.episode_solved) >= 20:
                rate = sum(list(self.episode_solved)[-100:]) / min(len(self.episode_solved), 100) * 100
                self.curve_data.append((self.total_episodes, rate))

            # Check phase advancement
            if len(self.episode_solved) >= 100:
                recent_rate = sum(list(self.episode_solved)[-100:]) / 100
                if recent_rate > 0.70 and self.phase < 3:
                    self._advance_phase()
        else:
            self.obs = next_obs

        # PPO update
        if self.steps_since_update >= self.steps_per_update:
            self.last_losses = self.agent.update()
            self.steps_since_update = 0

    def _advance_phase(self):
        """Move to next difficulty phase."""
        self.phase += 1
        self.phase_up_text = f"PHASE {self.phase}: {['', 'Baby Steps', 'Two Boxes!', 'Three Boxes!'][self.phase]}"
        self.phase_up_timer = 120  # frames
        self.phase_changes.append(self.total_episodes)

        self.env = SokobanEnv(phase=self.phase)
        self.obs = self.env.reset()
        self.episode_solved.clear()
        self.episode_rewards.clear()

    def _render(self):
        """Full render of training visualization."""
        self.screen.fill((18, 18, 30))

        # LEFT: Game grid
        render_data = self.env.get_render_data()
        self.renderer.render_grid(self.screen, render_data, offset_x=15, offset_y=15)

        # Phase label
        phase_names = {1: "Phase 1: Baby Steps (1 box)", 2: "Phase 2: Two Boxes", 3: "Phase 3: Three Boxes"}
        phase_text = self.font_small.render(phase_names.get(self.phase, ""), True, (150, 160, 180))
        self.screen.blit(phase_text, (15, 580))

        # RIGHT: Stats panel
        panel_x = 600
        pygame.draw.rect(self.screen, (25, 25, 40), pygame.Rect(panel_x, 0, 400, self.H))

        x = panel_x + 20
        y = 20

        # Episode
        self._text(f"Episode: {self.total_episodes:,}", x, y, self.font_large, (220, 225, 240))
        y += 40
        self._text(f"Total Steps: {self.total_steps:,}", x, y, self.font_small, (130, 140, 160))
        y += 30

        # Solve rate
        if len(self.episode_solved) > 0:
            solve_rate = sum(self.episode_solved) / len(self.episode_solved) * 100
        else:
            solve_rate = 0

        color = (239, 68, 68) if solve_rate < 15 else \
                (251, 146, 60) if solve_rate < 30 else \
                (251, 191, 36) if solve_rate < 60 else (52, 211, 153)

        self._text("Solve Rate:", x, y, self.font_small, (130, 140, 160))
        y += 20
        self._text(f"{solve_rate:.1f}%", x, y, self.font_large, color)
        y += 40

        # Other stats
        avg_reward = sum(self.episode_rewards) / max(len(self.episode_rewards), 1)
        avg_steps = sum(self.episode_steps_list) / max(len(self.episode_steps_list), 1)
        deadlock_rate = sum(self.episode_deadlocks) / max(len(self.episode_deadlocks), 1) * 100

        self._text(f"Avg Reward: {avg_reward:.1f}", x, y, self.font_med, (180, 190, 210))
        y += 25
        self._text(f"Avg Steps: {avg_steps:.0f}", x, y, self.font_med, (180, 190, 210))
        y += 25
        dl_color = (52, 211, 153) if deadlock_rate < 10 else (251, 191, 36) if deadlock_rate < 30 else (239, 68, 68)
        self._text(f"Deadlock Rate: {deadlock_rate:.0f}%", x, y, self.font_med, dl_color)
        y += 40

        # Learning curve graph
        graph_rect = pygame.Rect(panel_x + 15, y, 370, 200)
        pygame.draw.rect(self.screen, (30, 30, 48), graph_rect, border_radius=8)
        pygame.draw.rect(self.screen, (50, 55, 70), graph_rect, 1, border_radius=8)

        self._text("Solve Rate (rolling 100)", panel_x + 25, y + 5, self.font_tiny, (100, 110, 130))

        # Grid lines
        gy = graph_rect.y + 25
        gh = graph_rect.height - 35
        gx = graph_rect.x + 10
        gw = graph_rect.width - 20

        for pct in [25, 50, 75]:
            line_y = gy + gh - int(gh * pct / 100)
            pygame.draw.line(self.screen, (45, 45, 60), (gx, line_y), (gx + gw, line_y), 1)
            self._text(f"{pct}%", gx - 5, line_y - 6, self.font_tiny, (70, 80, 100))

        # Plot data
        if len(self.curve_data) >= 2:
            max_ep = max(d[0] for d in self.curve_data)
            points = []
            for ep, rate in self.curve_data:
                px = gx + int(gw * ep / max(max_ep, 1))
                py = gy + gh - int(gh * min(rate, 100) / 100)
                points.append((px, py))

            if len(points) >= 2:
                pygame.draw.lines(self.screen, (99, 102, 241), False, points, 2)

        y = graph_rect.bottom + 15

        # Action probabilities
        self._text("Action Probabilities:", x, y, self.font_small, (100, 110, 130))
        y += 22
        labels = ["UP:", "DOWN:", "LEFT:", "RIGHT:"]
        for i, (label, prob) in enumerate(zip(labels, self.action_probs)):
            self._text(label, x, y, self.font_small, (130, 140, 160))
            bar_x = x + 90
            bar_w = 150
            bar_h = 14
            pygame.draw.rect(self.screen, (40, 40, 55), pygame.Rect(bar_x, y+2, bar_w, bar_h), border_radius=3)
            fill_w = int(bar_w * prob)
            if fill_w > 0:
                pygame.draw.rect(self.screen, (99, 102, 241), pygame.Rect(bar_x, y+2, fill_w, bar_h), border_radius=3)
            self._text(f"{prob:.2f}", bar_x + bar_w + 8, y, self.font_small, (180, 190, 210))
            y += 22

        y += 10

        # Loss info
        if self.last_losses:
            self._text(f"Policy Loss: {self.last_losses.get('policy_loss', 0):.4f}", x, y, self.font_tiny, (100, 110, 130))
            y += 16
            self._text(f"Value Loss: {self.last_losses.get('value_loss', 0):.4f}", x, y, self.font_tiny, (100, 110, 130))
            y += 16
            self._text(f"Entropy: {self.last_losses.get('entropy', 0):.4f}", x, y, self.font_tiny, (100, 110, 130))

        # Mode
        mode_text = {"visual": "VISUAL", "fast": "FAST", "slow": "SLOW"}
        mode_color = {"visual": (52, 211, 153), "fast": (251, 191, 36), "slow": (147, 197, 253)}
        self._text(f"Mode: {mode_text[self.mode]}", x, self.H - 60, self.font_med, mode_color[self.mode])

        # Controls
        self._text("Space=Pause  F=Fast  S=Slow  V=Visual  ESC=Quit",
                   panel_x + 20, self.H - 25, self.font_tiny, (70, 80, 100))

        # Phase up notification
        if self.phase_up_timer > 0:
            self.phase_up_timer -= 1
            overlay = pygame.Surface((580, 100), pygame.SRCALPHA)
            overlay.fill((18, 18, 30, 200))
            self.screen.blit(overlay, (10, 250))
            text = self.font_large.render(f"PHASE UP: {self.phase_up_text}", True, (99, 102, 241))
            self.screen.blit(text, (60, 280))

        # Pause overlay
        if self.paused:
            overlay = pygame.Surface((580, 580), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            self.screen.blit(overlay, (10, 10))
            text = self.font_large.render("PAUSED", True, (255, 255, 255))
            self.screen.blit(text, (220, 280))

        pygame.display.flip()

    def _text(self, text, x, y, font, color):
        surf = font.render(str(text), True, color)
        self.screen.blit(surf, (x, y))


if __name__ == "__main__":
    print("=" * 50)
    print("  Real RL Training — Sokoban")
    print("  The AI starts RANDOM and learns over time.")
    print("  Be patient — real learning takes thousands")
    print("  of episodes!")
    print("=" * 50)
    print()
    print("Generating training levels (this may take a moment)...")

    app = TrainAndWatch()

    print("Training levels generated! Starting visualization...")
    print()
    print("Controls:")
    print("  Space = Pause/Resume")
    print("  F = Fast mode (train without rendering)")
    print("  S = Slow mode (watch every step)")
    print("  V = Visual mode (normal)")
    print("  ESC = Quit and save")

    app.run()
