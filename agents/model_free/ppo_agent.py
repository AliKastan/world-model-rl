"""Real PPO agent — from-scratch PyTorch implementation (no Stable-Baselines3).

The agent starts DUMB (random moves) and learns from EXPERIENCE through
thousands of episodes of trial and error.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from agents.base_agent import BaseAgent

_MAX_KEYS = 4
N_ACTIONS = 4
N_OBJECT_TYPES = 14


# ═══════════════════════════════════════════════════════════════════════
# Network
# ═══════════════════════════════════════════════════════════════════════


class PPONetwork(nn.Module):
    """Neural network: puzzle observation -> (action probs, state value).

    Architecture::

        Grid (H*W*14 one-hot) -> Linear(grid_dim, 256) -> ReLU --+
        Agent pos (2)         -> Linear(2, 32) -> ReLU -----------+-- Concat(320)
        Inventory (4)         -> Linear(4, 32) -> ReLU -----------+        |
                                                            Linear(320, 256) -> ReLU
                                                            Linear(256, 128) -> ReLU
                                                               /                \\
                                                      Policy head          Value head
                                                      Linear(128,4)        Linear(128,1)
                                                      Softmax              (raw value)
    """

    def __init__(self, grid_height: int = 12, grid_width: int = 12,
                 n_object_types: int = N_OBJECT_TYPES) -> None:
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.n_object_types = n_object_types
        grid_dim = grid_height * grid_width * n_object_types

        self.grid_fc = nn.Sequential(nn.Linear(grid_dim, 256), nn.ReLU())
        self.pos_fc = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
        self.inv_fc = nn.Sequential(nn.Linear(_MAX_KEYS, 32), nn.ReLU())

        self.trunk = nn.Sequential(
            nn.Linear(256 + 32 + 32, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )

        self.policy_head = nn.Linear(128, N_ACTIONS)
        self.value_head = nn.Linear(128, 1)

    def forward(self, grid: torch.Tensor, agent_pos: torch.Tensor,
                inventory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = grid.shape[0]
        H, W = grid.shape[1], grid.shape[2]

        # Pad smaller grids to expected size (wall = type 1)
        if H < self.grid_height or W < self.grid_width:
            padded = torch.ones(B, self.grid_height, self.grid_width,
                                dtype=grid.dtype, device=grid.device)
            padded[:, :H, :W] = grid
            grid = padded

        grid_oh = F.one_hot(grid.long(), num_classes=self.n_object_types).float()
        grid_flat = grid_oh.reshape(B, -1)

        feat = torch.cat([
            self.grid_fc(grid_flat),
            self.pos_fc(agent_pos.float()),
            self.inv_fc(inventory.float()),
        ], dim=-1)

        trunk_out = self.trunk(feat)
        action_probs = F.softmax(self.policy_head(trunk_out), dim=-1)
        state_value = self.value_head(trunk_out)
        return action_probs, state_value


# ═══════════════════════════════════════════════════════════════════════
# Memory
# ═══════════════════════════════════════════════════════════════════════


def _extract_obs(obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull (grid, normalised_pos, binary_inv) from an observation dict."""
    grid = np.asarray(obs["grid"])
    pos = np.array(obs["agent_pos"], dtype=np.float32) / 12.0

    raw_inv = obs.get("inventory", [])
    if isinstance(raw_inv, np.ndarray) and raw_inv.shape == (_MAX_KEYS,):
        inv = raw_inv.astype(np.float32)
    else:
        inv = np.zeros(_MAX_KEYS, dtype=np.float32)
        for kid in (raw_inv if hasattr(raw_inv, "__iter__") else []):
            kid = int(kid)
            if 0 <= kid < _MAX_KEYS:
                inv[kid] = 1.0
    return grid, pos, inv


class PPOMemory:
    """Stores rollout transitions for a PPO update."""

    def __init__(self) -> None:
        self.grids: List[np.ndarray] = []
        self.positions: List[np.ndarray] = []
        self.inventories: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def store(self, obs, action, reward, log_prob, value, done) -> None:
        """Store one step. *obs* is an observation dict."""
        grid, pos, inv = _extract_obs(obs)
        self.grids.append(grid)
        self.positions.append(pos)
        self.inventories.append(inv)
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def clear(self) -> None:
        for lst in (self.grids, self.positions, self.inventories,
                    self.actions, self.rewards, self.log_probs,
                    self.values, self.dones):
            lst.clear()

    def __len__(self) -> int:
        return len(self.actions)

    def get_batches(self, batch_size: int) -> List[np.ndarray]:
        n = len(self)
        indices = np.arange(n)
        np.random.shuffle(indices)
        return [indices[i:i + batch_size] for i in range(0, n, batch_size)]


# ═══════════════════════════════════════════════════════════════════════
# Observation -> Tensor
# ═══════════════════════════════════════════════════════════════════════


def _obs_to_tensors(obs: Dict[str, Any], device: torch.device):
    grid, pos, inv = _extract_obs(obs)
    return (
        torch.tensor(grid, dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor(pos, device=device).unsqueeze(0),
        torch.tensor(inv, device=device).unsqueeze(0),
    )


# ═══════════════════════════════════════════════════════════════════════
# PPOAgent
# ═══════════════════════════════════════════════════════════════════════


class PPOAgent(BaseAgent):
    """From-scratch PPO that learns to solve puzzles through trial and error."""

    def __init__(
        self,
        grid_height: int = 12,
        grid_width: int = 12,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        epochs_per_update: int = 4,
        batch_size: int = 64,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self.network = PPONetwork(grid_height, grid_width).to(self._device)
        self.optimizer = Adam(self.network.parameters(), lr=lr)
        self.memory = PPOMemory()

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

    @property
    def name(self) -> str:
        return "PPO"

    # -- Action selection ---------------------------------------------------

    def act(self, observation: Dict[str, Any]) -> int:
        """Deterministic greedy action (evaluation)."""
        self.network.eval()
        with torch.no_grad():
            probs, _ = self.network(*_obs_to_tensors(observation, self._device))
        return int(probs.argmax(dim=-1).item())

    def select_action(self, observation: Dict[str, Any]) -> Tuple[int, float, float]:
        """Sample action from policy (training — exploration).

        Returns (action, log_prob, state_value).
        """
        self.network.eval()
        with torch.no_grad():
            probs, value = self.network(*_obs_to_tensors(observation, self._device))
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def get_action_probs(self, observation: Dict[str, Any]) -> np.ndarray:
        """Return action probability vector (4,) for visualization."""
        self.network.eval()
        with torch.no_grad():
            probs, _ = self.network(*_obs_to_tensors(observation, self._device))
        return probs.squeeze(0).cpu().numpy()

    # -- PPO update ---------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """PPO update: GAE advantages -> clipped surrogate loss -> backprop."""
        if len(self.memory) == 0:
            return {}

        self.network.train()
        dev = self._device

        grids = torch.tensor(np.array(self.memory.grids), dtype=torch.long, device=dev)
        positions = torch.tensor(np.array(self.memory.positions), dtype=torch.float32, device=dev)
        inventories = torch.tensor(np.array(self.memory.inventories), dtype=torch.float32, device=dev)
        actions = torch.tensor(self.memory.actions, dtype=torch.long, device=dev)
        old_log_probs = torch.tensor(self.memory.log_probs, dtype=torch.float32, device=dev)
        rewards = np.array(self.memory.rewards, dtype=np.float32)
        values = np.array(self.memory.values, dtype=np.float32)
        dones = np.array(self.memory.dones, dtype=np.float32)

        # GAE
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_val = 0.0 if t == n - 1 else values[t + 1]
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=dev)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=dev)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Optimise
        total_pl = total_vl = total_ent = 0.0
        n_updates = 0

        for _ in range(self.epochs_per_update):
            for idx in self.memory.get_batches(self.batch_size):
                probs, vals = self.network(grids[idx], positions[idx], inventories[idx])
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_log_probs[idx])
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                    1 + self.clip_epsilon) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(vals.squeeze(-1), ret_t[idx])

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_ent += entropy.item()
                n_updates += 1

        self.memory.clear()
        n_updates = max(n_updates, 1)
        return {
            "policy_loss": total_pl / n_updates,
            "value_loss": total_vl / n_updates,
            "entropy": total_ent / n_updates,
        }

    def learn(self, **kwargs: Any) -> Dict[str, Any]:
        return self.update()

    # -- Save / Load --------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)

    def load(self, path: str) -> None:
        self.network.load_state_dict(torch.load(path, weights_only=True))
