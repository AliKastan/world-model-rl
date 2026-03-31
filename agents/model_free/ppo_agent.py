"""PPO agent backed by Stable-Baselines3.

Provides a custom CNN+MLP feature extractor that processes the Dict
observation space from :class:`~env.gym_env.PuzzleEnv`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Type

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import yaml
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from agents.base_agent import BaseAgent

# ---------------------------------------------------------------------------
# Default hyperparameters (overridden by configs/default.yaml if present)
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

_MAX_KEYS = 4  # must match env.gym_env._MAX_KEYS


def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Merge YAML config with built-in defaults."""
    cfg = dict(_DEFAULTS)
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, "configs", "default.yaml"
        )
    config_path = os.path.normpath(config_path)
    if os.path.isfile(config_path):
        with open(config_path) as f:
            data = yaml.safe_load(f)
        training = data.get("training", {})
        for key in _DEFAULTS:
            if key in training:
                cfg[key] = training[key]
    return cfg


# ---------------------------------------------------------------------------
# Custom feature extractor
# ---------------------------------------------------------------------------

class PuzzleFeatureExtractor(BaseFeaturesExtractor):
    """CNN for the grid + small MLPs for auxiliary inputs.

    Architecture
    ------------
    Grid branch:
        Conv2d(1, 16, 3, padding=1) -> ReLU
        Conv2d(16, 32, 3, padding=1) -> ReLU
        Conv2d(32, 64, 3, padding=1) -> ReLU
        Flatten

    Auxiliary branches:
        agent_pos   : Linear(2, 16)  -> ReLU
        inventory   : Linear(4, 8)   -> ReLU
        box_progress: Linear(2, 8)   -> ReLU

    Fusion:
        Concat(grid_flat, pos, inv, box) -> Linear(combined, 256) -> ReLU
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256) -> None:
        # Must call super with the final features_dim *before* building layers
        super().__init__(observation_space, features_dim)

        grid_space = observation_space["grid"]
        grid_h, grid_w = grid_space.shape  # (12, 12)

        # Grid CNN
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out_dim = 64 * grid_h * grid_w  # 64 * 12 * 12 = 9216

        # Auxiliary branches
        self.pos_mlp = nn.Sequential(nn.Linear(2, 16), nn.ReLU())
        self.inv_mlp = nn.Sequential(nn.Linear(_MAX_KEYS, 8), nn.ReLU())
        self.box_mlp = nn.Sequential(nn.Linear(2, 8), nn.ReLU())

        combined_dim = cnn_out_dim + 16 + 8 + 8
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Grid: (B, H, W) int8 -> (B, 1, H, W) float
        grid = observations["grid"].float().unsqueeze(1) / 13.0
        grid_feat = self.grid_cnn(grid)

        # Agent pos: (B, 2) int32 -> float, normalise to [0, 1]
        pos = observations["agent_pos"].float() / 12.0
        pos_feat = self.pos_mlp(pos)

        # Inventory: (B, 4) int8 -> float
        inv = observations["inventory"].float()
        inv_feat = self.inv_mlp(inv)

        # Box progress: concat boxes_on_targets and total_targets (B, 1) each
        bot = observations["boxes_on_targets"].float()
        tt = observations["total_targets"].float()
        box_input = torch.cat([bot, tt], dim=-1)  # (B, 2)
        box_feat = self.box_mlp(box_input)

        combined = torch.cat([grid_feat, pos_feat, inv_feat, box_feat], dim=-1)
        return self.fusion(combined)


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------

class PPOAgent(BaseAgent):
    """PPO agent using Stable-Baselines3 with a custom feature extractor.

    Parameters
    ----------
    env : gymnasium.Env
        The training environment.
    config_path : str or None
        Path to a YAML config file.  Falls back to ``configs/default.yaml``.
    seed : int
        Random seed.
    device : str
        ``"auto"``, ``"cpu"``, or ``"cuda"``.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        config_path: Optional[str] = None,
        seed: int = 42,
        device: str = "auto",
        **overrides: Any,
    ) -> None:
        self._cfg = _load_config(config_path)
        # Apply any explicit overrides (e.g. n_steps, learning_rate)
        for k, v in overrides.items():
            if v is not None and k in self._cfg:
                self._cfg[k] = v
        self._env = env

        policy_kwargs = {
            "features_extractor_class": PuzzleFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": dict(pi=[128, 64], vf=[128, 64]),
        }

        self._model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=self._cfg["learning_rate"],
            n_steps=self._cfg["n_steps"],
            batch_size=self._cfg["batch_size"],
            n_epochs=self._cfg["n_epochs"],
            gamma=self._cfg["gamma"],
            gae_lambda=self._cfg["gae_lambda"],
            clip_range=self._cfg["clip_range"],
            ent_coef=self._cfg["ent_coef"],
            vf_coef=self._cfg["vf_coef"],
            max_grad_norm=self._cfg["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            verbose=0,
        )

    # -- BaseAgent interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "PPO"

    def act(self, observation: Dict[str, Any]) -> int:
        action, _ = self._model.predict(observation, deterministic=True)
        return int(action)

    def learn(self, **kwargs: Any) -> Dict[str, Any]:
        """Train for *total_timesteps* steps (default from config).

        Keyword arguments are forwarded to ``PPO.learn()``.
        """
        total_timesteps = kwargs.pop(
            "total_timesteps", self._cfg.get("total_timesteps", 100_000)
        )
        self._model.learn(total_timesteps=total_timesteps, **kwargs)
        # Return latest logged metrics
        logger = self._model.logger
        metrics: Dict[str, Any] = {}
        if hasattr(logger, "name_to_value"):
            metrics = dict(logger.name_to_value)
        return metrics

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._model.save(path)

    def load(self, path: str) -> None:
        self._model = PPO.load(path, env=self._env)

    # -- Convenience --------------------------------------------------------

    @property
    def model(self) -> PPO:
        """Direct access to the underlying SB3 model."""
        return self._model
