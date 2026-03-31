"""Gymnasium-compatible wrapper for the puzzle environment.

Provides :class:`PuzzleEnv`, a standard ``gymnasium.Env`` with Dict and
flat observation modes, automatic level generation via
:class:`~env.level_generator.LevelGenerator`, and optional PyGame or
rgb_array rendering.

Registered environments
-----------------------
- ``ThinkPuzzle-v0``        — difficulty 1 (default)
- ``ThinkPuzzle-Easy-v0``   — difficulty 2
- ``ThinkPuzzle-Medium-v0`` — difficulty 5
- ``ThinkPuzzle-Hard-v0``   — difficulty 8
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from env.level_generator import LevelGenerator, _TIERS
from env.objects import Door, Key
from env.puzzle_world import PuzzleWorld

# ---------------------------------------------------------------------------
# Constants derived from tier specifications
# ---------------------------------------------------------------------------

_MAX_GRID_W = max(t.grid_w for t in _TIERS.values())  # 12
_MAX_GRID_H = max(t.grid_h for t in _TIERS.values())  # 12
_MAX_BOXES = max(t.num_boxes for t in _TIERS.values())  # 3
_MAX_KEYS = 4  # generous upper bound for future tiers


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def _pad_grid(grid: np.ndarray, max_h: int, max_w: int) -> np.ndarray:
    """Pad *grid* to (max_h, max_w) with 1 (wall id) so smaller grids fit."""
    h, w = grid.shape
    padded = np.ones((max_h, max_w), dtype=np.int8)  # walls
    padded[:h, :w] = grid.astype(np.int8)
    return padded


def get_flat_obs(obs_dict: Dict[str, Any]) -> np.ndarray:
    """Flatten a Dict observation into a single float32 vector.

    Layout::

        [grid_flat, agent_x, agent_y, agent_dir,
         inventory_bits, boxes_on_targets, total_targets]

    Suitable for simple MLP policies.
    """
    grid_flat = obs_dict["grid"].flatten().astype(np.float32)
    agent = obs_dict["agent_pos"].astype(np.float32)
    direction = np.array([obs_dict["agent_dir"]], dtype=np.float32)
    inv = obs_dict["inventory"].astype(np.float32)
    bot = obs_dict["boxes_on_targets"].astype(np.float32)
    tt = obs_dict["total_targets"].astype(np.float32)
    return np.concatenate([grid_flat, agent, direction, inv, bot, tt])


# ---------------------------------------------------------------------------
# PuzzleEnv
# ---------------------------------------------------------------------------

class PuzzleEnv(gymnasium.Env):
    """Gymnasium wrapper around :class:`~env.puzzle_world.PuzzleWorld`.

    Parameters
    ----------
    difficulty : int
        Starting difficulty tier (1-10).
    max_steps : int
        Step budget per episode.
    render_mode : str or None
        ``"human"`` for a PyGame window, ``"rgb_array"`` for a numpy frame.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        difficulty: int = 1,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.difficulty = difficulty
        self._max_steps = max_steps
        self.render_mode = render_mode

        self._generator = LevelGenerator()
        self._world: Optional[PuzzleWorld] = None
        self._renderer = None  # lazy init
        self._optimal_steps: int = 0

        # -- spaces ---------------------------------------------------------
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=0, high=13,
                shape=(_MAX_GRID_H, _MAX_GRID_W),
                dtype=np.int8,
            ),
            "agent_pos": spaces.Box(
                low=0, high=max(_MAX_GRID_W, _MAX_GRID_H) - 1,
                shape=(2,),
                dtype=np.int32,
            ),
            "agent_dir": spaces.Discrete(4),
            "inventory": spaces.MultiBinary(_MAX_KEYS),
            "boxes_on_targets": spaces.Box(
                low=0, high=_MAX_BOXES,
                shape=(1,),
                dtype=np.int32,
            ),
            "total_targets": spaces.Box(
                low=0, high=_MAX_BOXES,
                shape=(1,),
                dtype=np.int32,
            ),
        })

        self.action_space = spaces.Discrete(4)

    # ------------------------------------------------------------------
    # Observation conversion
    # ------------------------------------------------------------------

    def _make_obs(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a raw :meth:`PuzzleWorld.get_observation` dict to a
        gymnasium-compatible observation with fixed shapes."""
        grid = _pad_grid(raw["grid"], _MAX_GRID_H, _MAX_GRID_W)

        agent_pos = np.array(raw["agent_pos"], dtype=np.int32)
        agent_dir = int(raw["agent_dir"])

        inv = np.zeros(_MAX_KEYS, dtype=np.int8)
        for kid in raw["inventory"]:
            if kid < _MAX_KEYS:
                inv[kid] = 1

        bot = np.array([raw["boxes_on_targets"]], dtype=np.int32)
        tt = np.array([raw["total_targets"]], dtype=np.int32)

        return {
            "grid": grid,
            "agent_pos": agent_pos,
            "agent_dir": agent_dir,
            "inventory": inv,
            "boxes_on_targets": bot,
            "total_targets": tt,
        }

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)

        difficulty = self.difficulty
        level_seed = None
        if options:
            difficulty = options.get("difficulty", difficulty)
            level_seed = options.get("level_seed", None)

        if level_seed is None and seed is not None:
            level_seed = seed

        world = self._generator.generate(difficulty=difficulty, seed=level_seed)
        world.max_steps = self._max_steps
        self._world = world
        self._optimal_steps = getattr(world, "_optimal_steps", 0)

        obs = self._make_obs(world.get_observation())
        info = self._build_info(world, first=True)
        return obs, info

    def step(
        self, action: int,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        assert self._world is not None, "Call reset() before step()"
        raw_obs, reward, terminated, truncated, raw_info = self._world.step(action)
        obs = self._make_obs(raw_obs)
        info = self._build_info(self._world)
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None or self._world is None:
            return None

        if self.render_mode == "human":
            if self._renderer is None:
                from env.renderer import PuzzleRenderer
                self._renderer = PuzzleRenderer(cell_size=64)
            self._renderer.render(self._world)
            return None

        if self.render_mode == "rgb_array":
            return self._render_rgb_array()

        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Info dict
    # ------------------------------------------------------------------

    def _build_info(self, world: PuzzleWorld, first: bool = False) -> Dict[str, Any]:
        # Count doors opened
        doors_opened = 0
        for y in range(world.height):
            for x in range(world.width):
                for obj in world.get_cell(x, y):
                    if isinstance(obj, Door) and not obj.locked:
                        doors_opened += 1

        raw_obs = world.get_observation()
        return {
            "is_deadlock": world.is_deadlock(),
            "steps_taken": world.steps,
            "boxes_placed": raw_obs["boxes_on_targets"],
            "keys_collected": len(world.inventory),
            "doors_opened": doors_opened,
            "solved": world.solved,
            "optimal_steps": self._optimal_steps,
        }

    # ------------------------------------------------------------------
    # RGB array rendering
    # ------------------------------------------------------------------

    def _render_rgb_array(self) -> np.ndarray:
        """Render the current world to a numpy RGB array via an off-screen
        PyGame surface."""
        import pygame

        if not pygame.get_init():
            pygame.init()

        assert self._world is not None
        world = self._world
        cs = 64
        w = world.width * cs
        h = world.height * cs

        surf = pygame.Surface((w, h))
        surf.fill((22, 22, 35))

        from env.objects import (
            Box, Door, Floor, IceTile, Key, OneWayTile,
            PressureSwitch, SwitchWall, Target, Wall,
        )

        for gy in range(world.height):
            for gx in range(world.width):
                rect = pygame.Rect(gx * cs, gy * cs, cs, cs)
                cell = world.get_cell(gx, gy)

                # Background layer
                for obj in cell:
                    if isinstance(obj, Wall):
                        pygame.draw.rect(surf, (75, 85, 99), rect, border_radius=4)
                        break
                    elif isinstance(obj, Target):
                        cx, cy = rect.centerx, rect.centery
                        pygame.draw.circle(surf, (52, 211, 153), (cx, cy), cs // 4, 2)
                    elif isinstance(obj, IceTile):
                        s = pygame.Surface((cs, cs), pygame.SRCALPHA)
                        pygame.draw.rect(s, (147, 197, 253, 60), (0, 0, cs, cs), border_radius=4)
                        surf.blit(s, rect.topleft)
                    elif isinstance(obj, PressureSwitch):
                        col = (236, 72, 153) if obj.activated else (168, 85, 247)
                        pad = cs // 5
                        pygame.draw.rect(surf, col, rect.inflate(-pad * 2, -pad * 2), border_radius=4)
                    elif isinstance(obj, SwitchWall) and not obj.open:
                        s = pygame.Surface((cs, cs), pygame.SRCALPHA)
                        pygame.draw.rect(s, (168, 85, 247, 180), (0, 0, cs, cs), border_radius=4)
                        surf.blit(s, rect.topleft)
                    elif isinstance(obj, OneWayTile):
                        pygame.draw.circle(surf, (56, 189, 248), rect.center, cs // 5)

                # Object layer
                for obj in cell:
                    if isinstance(obj, Box):
                        inset = rect.inflate(-8, -8)
                        col = (52, 211, 153) if obj.on_target else (251, 191, 36)
                        pygame.draw.rect(surf, col, inset, border_radius=6)
                        break
                    elif isinstance(obj, Key):
                        pygame.draw.circle(surf, (251, 146, 60), rect.center, cs // 5)
                        break
                    elif isinstance(obj, Door):
                        col = (239, 68, 68) if obj.locked else (74, 222, 128)
                        inset = rect.inflate(-6, -6)
                        pygame.draw.rect(surf, col, inset, border_radius=6)
                        break

                # Agent
                if (gx, gy) == world.agent_pos:
                    pygame.draw.circle(surf, (99, 102, 241), rect.center, cs // 3)

        # Convert to numpy (H, W, 3)
        arr = pygame.surfarray.array3d(surf)  # (W, H, 3)
        return np.transpose(arr, (1, 0, 2))  # (H, W, 3)


# ---------------------------------------------------------------------------
# Gymnasium registration
# ---------------------------------------------------------------------------

def _register() -> None:
    """Register puzzle environments with Gymnasium."""
    envs = {
        "ThinkPuzzle-v0": {"difficulty": 1},
        "ThinkPuzzle-Easy-v0": {"difficulty": 2},
        "ThinkPuzzle-Medium-v0": {"difficulty": 5},
        "ThinkPuzzle-Hard-v0": {"difficulty": 8},
    }
    for env_id, kwargs in envs.items():
        if env_id not in gymnasium.envs.registry:
            gymnasium.register(
                id=env_id,
                entry_point="env.gym_env:PuzzleEnv",
                kwargs=kwargs,
            )


_register()
