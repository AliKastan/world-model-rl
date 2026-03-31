"""Standalone PPO agent — no Stable-Baselines3 dependency.

A from-scratch Proximal Policy Optimisation implementation that learns
to solve puzzles directly from :class:`~env.puzzle_world.PuzzleWorld`
observations.

Architecture (see PPONetwork docstring for the full diagram).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import gymnasium

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

        Grid input (H*W*14 one-hot) ─> Linear(grid_dim, 256) ─> ReLU ─┐
                                                                       │
        Agent pos (2) ─> Linear(2, 32) ─> ReLU ───────────────────────>├─ Concat(320)
                                                                       │
        Inventory (4) ─> Linear(4, 32) ─> ReLU ──────────────────────>─┘
                                                                       │
                                                          Linear(320, 256) ─> ReLU
                                                          Linear(256, 128) ─> ReLU
                                                              /                  \\
                                                     Policy head            Value head
                                                     Linear(128, 4)         Linear(128, 1)
                                                     Softmax                (raw value)
    """

    def __init__(self, grid_height: int = 12, grid_width: int = 12,
                 n_object_types: int = N_OBJECT_TYPES) -> None:
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.n_object_types = n_object_types
        grid_dim = grid_height * grid_width * n_object_types

        # Grid branch
        self.grid_fc = nn.Sequential(
            nn.Linear(grid_dim, 256),
            nn.ReLU(),
        )

        # Auxiliary branches
        self.pos_fc = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
        self.inv_fc = nn.Sequential(nn.Linear(_MAX_KEYS, 32), nn.ReLU())

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(256 + 32 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Linear(128, N_ACTIONS)

        # Value head
        self.value_head = nn.Linear(128, 1)

    def forward(
        self,
        grid: torch.Tensor,
        agent_pos: torch.Tensor,
        inventory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        grid : (B, H, W) int / long — cell type IDs
        agent_pos : (B, 2) float — normalised agent (x, y)
        inventory : (B, 4) float — binary key flags

        Returns
        -------
        action_probs : (B, 4) — softmax probabilities
        state_value  : (B, 1) — estimated V(s)
        """
        B = grid.shape[0]
        H, W = grid.shape[1], grid.shape[2]

        # Pad grid to expected (grid_height, grid_width) if smaller
        if H < self.grid_height or W < self.grid_width:
            padded = torch.ones(B, self.grid_height, self.grid_width,
                                dtype=grid.dtype, device=grid.device)  # 1 = wall
            padded[:, :H, :W] = grid
            grid = padded

        # One-hot encode grid: (B, H, W) -> (B, H, W, 14) -> (B, H*W*14)
        grid_oh = F.one_hot(grid.long(), num_classes=self.n_object_types).float()
        grid_flat = grid_oh.reshape(B, -1)
        grid_feat = self.grid_fc(grid_flat)

        pos_feat = self.pos_fc(agent_pos.float())
        inv_feat = self.inv_fc(inventory.float())

        combined = torch.cat([grid_feat, pos_feat, inv_feat], dim=-1)
        trunk_out = self.trunk(combined)

        action_probs = F.softmax(self.policy_head(trunk_out), dim=-1)
        state_value = self.value_head(trunk_out)

        return action_probs, state_value


# ═══════════════════════════════════════════════════════════════════════
# Rollout Memory
# ═══════════════════════════════════════════════════════════════════════


class PPOMemory:
    """Stores experiences from rollouts for a PPO update.

    Accepts either raw observation dicts (from ``PuzzleEnv``) or
    pre-extracted arrays.  The 6-argument form used by the training loop::

        memory.store(obs, action, reward, log_prob, value, done)

    automatically extracts grid / agent_pos / inventory from the obs dict.
    """

    def __init__(self) -> None:
        self.grids: List[np.ndarray] = []
        self.positions: List[np.ndarray] = []
        self.inventories: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def store(self, obs_or_grid, action, reward, log_prob, value, done) -> None:
        """Store one transition.

        Parameters
        ----------
        obs_or_grid : dict or np.ndarray
            Either a full observation dict (with keys ``"grid"``,
            ``"agent_pos"``, ``"inventory"``) **or** a raw grid array
            (legacy 8-arg form is no longer supported — use the dict form).
        """
        if isinstance(obs_or_grid, dict):
            obs = obs_or_grid
            grid = np.asarray(obs["grid"])
            pos = np.array(obs["agent_pos"], dtype=np.float32) / 12.0
            # inventory may be a list of key-ids OR a binary array
            raw_inv = obs.get("inventory", [])
            if isinstance(raw_inv, np.ndarray) and raw_inv.shape == (_MAX_KEYS,):
                inv = raw_inv.astype(np.float32)
            else:
                inv = np.zeros(_MAX_KEYS, dtype=np.float32)
                for kid in (raw_inv if hasattr(raw_inv, "__iter__") else []):
                    kid = int(kid)
                    if 0 <= kid < _MAX_KEYS:
                        inv[kid] = 1.0
        else:
            # Caller passed a raw grid array (backwards compat)
            grid = np.asarray(obs_or_grid)
            pos = np.zeros(2, dtype=np.float32)
            inv = np.zeros(_MAX_KEYS, dtype=np.float32)

        self.grids.append(grid)
        self.positions.append(pos)
        self.inventories.append(inv)
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def clear(self) -> None:
        self.grids.clear()
        self.positions.clear()
        self.inventories.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.actions)

    def get_batches(self, batch_size: int) -> List[np.ndarray]:
        """Return shuffled index batches for mini-batch SGD."""
        n = len(self)
        indices = np.arange(n)
        np.random.shuffle(indices)
        batches = []
        for start in range(0, n, batch_size):
            batches.append(indices[start : start + batch_size])
        return batches


# ═══════════════════════════════════════════════════════════════════════
# Observation helpers
# ═══════════════════════════════════════════════════════════════════════


def _obs_to_tensors(
    obs: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a single PuzzleWorld observation dict to batched tensors."""
    grid = torch.tensor(obs["grid"], dtype=torch.long, device=device).unsqueeze(0)

    pos = np.array(obs["agent_pos"], dtype=np.float32) / 12.0
    pos_t = torch.tensor(pos, device=device).unsqueeze(0)

    inv_list = obs.get("inventory", [])
    inv = np.zeros(_MAX_KEYS, dtype=np.float32)
    for kid in inv_list:
        if 0 <= kid < _MAX_KEYS:
            inv[kid] = 1.0
    inv_t = torch.tensor(inv, device=device).unsqueeze(0)

    return grid, pos_t, inv_t


# ═══════════════════════════════════════════════════════════════════════
# PPOAgent
# ═══════════════════════════════════════════════════════════════════════


class PPOAgent(BaseAgent):
    """From-scratch PPO agent that learns to solve puzzles.

    Parameters
    ----------
    grid_height, grid_width : int
        Maximum grid dimensions (observations are padded / clipped to fit).
    lr : float
        Learning rate for Adam.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda for advantage estimation.
    clip_epsilon : float
        PPO clipping parameter.
    epochs_per_update : int
        Number of optimisation epochs per update.
    batch_size : int
        Mini-batch size for SGD.
    ent_coef : float
        Entropy bonus coefficient (encourages exploration).
    vf_coef : float
        Value-function loss coefficient.
    device : str
        ``"auto"``, ``"cpu"``, or ``"cuda"``.
    """

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

    # -- BaseAgent interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "PPO-Standalone"

    def act(self, observation: Dict[str, Any]) -> int:
        """Deterministic greedy action (for evaluation)."""
        self.network.eval()
        with torch.no_grad():
            grid, pos, inv = _obs_to_tensors(observation, self._device)
            probs, _ = self.network(grid, pos, inv)
        return int(probs.argmax(dim=-1).item())

    def select_action(
        self, observation: Dict[str, Any]
    ) -> Tuple[int, float, float]:
        """Sample an action for training (with exploration).

        Returns
        -------
        action : int
        log_prob : float
        state_value : float
        """
        self.network.eval()
        with torch.no_grad():
            grid, pos, inv = _obs_to_tensors(observation, self._device)
            probs, value = self.network(grid, pos, inv)

            # Sample from the categorical distribution (NOT argmax)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item())

    # -- Training -----------------------------------------------------------

    def update(self) -> Dict[str, float]:
        """Run a PPO update on the collected memory.

        1. Compute GAE advantages and discounted returns.
        2. Run ``epochs_per_update`` optimisation epochs with mini-batches.
        3. Clear memory.

        Returns dict with ``policy_loss``, ``value_loss``, ``entropy``.
        """
        if len(self.memory) == 0:
            return {}

        self.network.train()

        # ── Convert memory to tensors ─────────────────────────────────
        grids = torch.tensor(np.array(self.memory.grids), dtype=torch.long,
                             device=self._device)
        positions = torch.tensor(np.array(self.memory.positions), dtype=torch.float32,
                                 device=self._device)
        inventories = torch.tensor(np.array(self.memory.inventories), dtype=torch.float32,
                                   device=self._device)
        actions = torch.tensor(self.memory.actions, dtype=torch.long,
                               device=self._device)
        old_log_probs = torch.tensor(self.memory.log_probs, dtype=torch.float32,
                                     device=self._device)
        rewards = np.array(self.memory.rewards, dtype=np.float32)
        values = np.array(self.memory.values, dtype=np.float32)
        dones = np.array(self.memory.dones, dtype=np.float32)

        # ── GAE advantage estimation ──────────────────────────────────
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values  # discounted returns = advantages + V(s)

        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self._device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self._device)

        # Normalise advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # ── PPO epochs ────────────────────────────────────────────────
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.epochs_per_update):
            batches = self.memory.get_batches(self.batch_size)
            for batch_idx in batches:
                b_grids = grids[batch_idx]
                b_pos = positions[batch_idx]
                b_inv = inventories[batch_idx]
                b_actions = actions[batch_idx]
                b_old_lp = old_log_probs[batch_idx]
                b_advantages = advantages_t[batch_idx]
                b_returns = returns_t[batch_idx]

                # Forward pass with current network
                probs, new_values = self.network(b_grids, b_pos, b_inv)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                    1.0 + self.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values.squeeze(-1), b_returns)

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        self.memory.clear()

        n_updates = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def learn(self, **kwargs: Any) -> Dict[str, Any]:
        """Alias for :meth:`update` to satisfy BaseAgent interface."""
        return self.update()

    # -- Save / Load --------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)

    def load(self, path: str) -> None:
        self.network.load_state_dict(torch.load(path, weights_only=True))


# ═══════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════


def train(
    agent: PPOAgent,
    env: "gymnasium.Env",
    n_episodes: int = 10_000,
    update_every: int = 10,
) -> Tuple[List[float], List[bool]]:
    """Train a PPOAgent on a gymnasium environment.

    The agent is *terrible* at first — that's the point!

    - Episodes 1-100:     Random moves, wall bumping, deadlocks everywhere.
    - Episodes 100-500:   Starts learning to avoid walls.
    - Episodes 500-2000:  Learns to push boxes sometimes.
    - Episodes 2000-5000: Consistently solves easy puzzles.
    - Episodes 5000+:     Gets efficient, avoids deadlocks.

    Parameters
    ----------
    agent : PPOAgent
        The agent to train (modified in-place).
    env : gymnasium.Env
        A ``PuzzleEnv`` (or any gym env with matching obs/action spaces).
    n_episodes : int
        Total training episodes.
    update_every : int
        Run a PPO update every N episodes.

    Returns
    -------
    all_rewards : list[float]
        Per-episode cumulative rewards.
    all_solved : list[bool]
        Per-episode solved flags.
    """
    all_rewards: List[float] = []
    all_solved: List[bool] = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.memory.store(obs, action, reward, log_prob, value, done)
            obs = next_obs
            episode_reward += reward

        all_rewards.append(episode_reward)
        all_solved.append(info.get("solved", False))

        # Update policy every N episodes
        if (episode + 1) % update_every == 0:
            agent.update()

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            recent_rewards = all_rewards[-100:]
            recent_solved = all_solved[-100:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            solve_rate = sum(recent_solved) / len(recent_solved) * 100
            if solve_rate > 50:
                indicator = "[+++]"
            elif solve_rate > 20:
                indicator = "[++ ]"
            else:
                indicator = "[+  ]"
            print(
                f"  Episode {episode + 1:>6}/{n_episodes} | "
                f"Avg Reward: {avg_reward:>7.1f} | "
                f"Solve Rate: {solve_rate:>5.1f}% | "
                f"{indicator}"
            )

    return all_rewards, all_solved


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train standalone PPO agent")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--update-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save", type=str, default=None, help="Path to save model")
    args = parser.parse_args()

    from env.gym_env import PuzzleEnv

    print("=" * 60)
    print("  PPO Standalone Training")
    print(f"  difficulty={args.difficulty}  episodes={args.episodes}")
    print("=" * 60)

    env = PuzzleEnv(difficulty=args.difficulty, max_steps=args.max_steps)
    agent = PPOAgent()

    all_rewards, all_solved = train(
        agent=agent,
        env=env,
        n_episodes=args.episodes,
        update_every=args.update_every,
    )

    env.close()

    total_solved = sum(all_solved)
    print(f"\n  Final: {total_solved}/{len(all_solved)} solved "
          f"({total_solved / max(len(all_solved), 1) * 100:.1f}%)")

    if args.save:
        agent.save(args.save)
        print(f"  Model saved to {args.save}")

    print("  Done.")


if __name__ == "__main__":
    main()
