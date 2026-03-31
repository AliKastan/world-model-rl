"""Gymnasium environment for Sokoban RL training.

Provides 5-channel binary observation and shaped rewards.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.sokoban import SokobanState, solve
from env.level_loader import LevelLoader, generate_training_level


class SokobanEnv(gym.Env):
    """Sokoban RL environment with reward shaping.

    Observation: (5, max_h, max_w) float32 binary channels:
      0: walls, 1: boxes, 2: targets, 3: player, 4: boxes on targets

    Reward shaping (critical for RL to learn):
      +100  puzzle solved
      +10   box pushed onto target
      -10   box pushed off target
      +1    box moved closer to nearest target
      -1    box moved further from nearest target
      -0.1  each step (time pressure)
      -1    invalid move (into wall)
      -50   deadlock detected (episode ends)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        level_set: str = "training",
        max_h: int = 10,
        max_w: int = 10,
        max_steps: int = 200,
    ) -> None:
        super().__init__()
        self.max_h = max_h
        self.max_w = max_w
        self.max_steps = max_steps
        self.level_set = level_set

        # Load levels
        if level_set == "training":
            self._training_params = True
            self._levels: List[SokobanState] = []  # generated on demand
        elif level_set == "classic":
            self._training_params = False
            loader = LevelLoader("levels/classic_60.txt")
            self._levels = [loader.get_level(i) for i in range(loader.get_total_levels())]
        else:
            self._training_params = False
            self._levels = []

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, max_h, max_w), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self._state: Optional[SokobanState] = None
        self._steps = 0
        self._prev_dist = 0.0
        self._level_idx = 0
        self._gen_seed = 2000

    @property
    def state(self) -> Optional[SokobanState]:
        return self._state

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._gen_seed = seed

        if self._training_params:
            # Generate a random simple level
            result = None
            while result is None:
                w = random.choice([5, 6, 7])
                h = random.choice([5, 6, 7])
                result = generate_training_level(
                    width=w, height=h, n_boxes=1,
                    seed=self._gen_seed, min_solution=3, max_solution=30,
                )
                self._gen_seed += 1
            self._state = result[0]
        else:
            if self._levels:
                idx = options.get("level", self._level_idx) if options else self._level_idx
                self._level_idx = idx % len(self._levels)
                self._state = self._levels[self._level_idx].clone()
                self._level_idx = (self._level_idx + 1) % len(self._levels)
            else:
                result = generate_training_level(width=6, height=6, n_boxes=1, seed=self._gen_seed)
                self._gen_seed += 1
                self._state = result[0] if result else SokobanState(
                    frozenset(), frozenset(), frozenset(), (1, 1), 5, 5
                )

        self._steps = 0
        self._prev_dist = self._state.box_distances()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self._state is not None
        self._steps += 1

        old_on = self._state.n_boxes_on_target
        new_state = self._state.move(action)

        if new_state is None:
            # Invalid move
            reward = -1.0
            obs = self._get_obs()
            truncated = self._steps >= self.max_steps
            return obs, reward, False, truncated, self._get_info(invalid=True)

        self._state = new_state
        new_on = self._state.n_boxes_on_target
        new_dist = self._state.box_distances()

        # Reward computation
        reward = -0.1  # time penalty

        if self._state.solved:
            reward += 100.0
            obs = self._get_obs()
            return obs, reward, True, False, self._get_info(solved=True)

        if self._state.is_deadlocked():
            reward += -50.0
            obs = self._get_obs()
            return obs, reward, True, False, self._get_info(deadlock=True)

        # Box-on-target changes
        if new_on > old_on:
            reward += 10.0
        elif new_on < old_on:
            reward += -10.0

        # Distance shaping
        dist_diff = self._prev_dist - new_dist
        if dist_diff > 0.5:
            reward += 1.0
        elif dist_diff < -0.5:
            reward += -1.0
        self._prev_dist = new_dist

        truncated = self._steps >= self.max_steps
        obs = self._get_obs()
        return obs, reward, False, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        """5-channel binary observation, padded to (max_h, max_w)."""
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

    def _get_info(self, solved=False, deadlock=False, invalid=False) -> Dict[str, Any]:
        s = self._state
        return {
            "solved": solved,
            "deadlock": deadlock,
            "invalid_move": invalid,
            "steps": self._steps,
            "boxes_on_target": s.n_boxes_on_target if s else 0,
            "total_boxes": len(s.boxes) if s else 0,
        }
