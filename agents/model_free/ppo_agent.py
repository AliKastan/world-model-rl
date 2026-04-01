"""CNN-based PPO agent for Sokoban.

5-channel spatial input (walls, boxes, targets, player, boxes_on_targets)
processed through a CNN for spatial reasoning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# ═══════════════════════════════════════════════════════════════════════
# Network
# ═══════════════════════════════════════════════════════════════════════


class SokobanNet(nn.Module):
    """CNN policy+value network for Sokoban.

    Input: (B, 5, H, W) binary channels.
    Output: action_probs (B, 4), value (B, 1).
    """

    def __init__(self, grid_h: int = 12, grid_w: int = 12) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        flat_dim = 64 * grid_h * grid_w
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 512), nn.ReLU(),
        )
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


# ═══════════════════════════════════════════════════════════════════════
# PPO Agent
# ═══════════════════════════════════════════════════════════════════════


class PPOAgent:
    """Proximal Policy Optimization agent for Sokoban.

    The agent starts knowing NOTHING. Over thousands of episodes it learns
    from pure experience:
    - Push boxes toward targets (good)
    - Avoid corners (deadlock = bad)
    - Minimize wasted moves
    """

    def __init__(
        self,
        grid_h: int = 12,
        grid_w: int = 12,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        mini_batch_size: int = 256,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self.net = SokobanNet(grid_h, grid_w).to(self._device)
        self.optimizer = Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size

        # Rollout storage
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        self.memory = self  # allow agent.memory.store() calls

    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """Sample action from policy. Returns (action, log_prob, value)."""
        self.net.eval()
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)
            probs, value = self.net(t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def get_action_probs(self, obs: np.ndarray) -> np.ndarray:
        """Return action probability vector (4,)."""
        self.net.eval()
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)
            probs, _ = self.net(t)
        return probs.squeeze(0).cpu().numpy()

    def store_transition(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def store(self, state, action, reward, log_prob, value, done):
        self.store_transition(state, action, reward, log_prob, value, done)

    def update(self) -> Dict[str, float]:
        """PPO update on collected rollout data."""
        if not self.states:
            return {}

        self.net.train()
        dev = self._device

        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=dev)
        actions = torch.tensor(self.actions, dtype=torch.long, device=dev)
        old_lp = torch.tensor(self.log_probs, dtype=torch.float32, device=dev)
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

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

        # Mini-batch PPO epochs
        n_samples = len(self.states)
        indices = np.arange(n_samples)
        total_pl = total_vl = total_ent = total_kl = 0.0
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

                probs, vals = self.net(b_states)
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - b_old_lp)
                s1 = ratio * b_adv
                s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = F.mse_loss(vals.squeeze(-1), b_ret)

                loss = (policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (b_old_lp - new_lp).mean().item()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_ent += entropy.item()
                total_kl += approx_kl
                n_updates += 1

        # Clear storage
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
            "approx_kl": total_kl / n_updates,
        }

    def save(self, path: str) -> None:
        torch.save({
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, weights_only=False)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
