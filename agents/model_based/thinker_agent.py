"""ThinkerAgent — learns a world model, then thinks before it acts.

Three learning mechanisms:
1. **Real experience** — acts in environment, stores transitions
2. **World model training** — learns to predict transitions from experience
3. **Dreaming** — imagines episodes using the world model, trains policy on them

When the world model is accurate enough, uses beam search planning (mental simulation)
to look ahead instead of reacting with a raw policy network.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from agents.base_agent import BaseAgent
from agents.model_free.ppo_agent import PPONetwork, PPOMemory, _extract_obs, _obs_to_tensors
from agents.model_based.world_model import LearnedWorldModel, ReplayBuffer, WorldModelTrainer
from agents.model_based.mental_sim import MentalSimulator, ThoughtStep

_MAX_KEYS = 4
_GRID_H = 12
_GRID_W = 12


class ThinkerAgent(BaseAgent):
    """The "Think Before You Act" agent.

    Uses a learned world model + beam search planning to solve puzzles
    more sample-efficiently than pure PPO.
    """

    def __init__(
        self,
        grid_height: int = _GRID_H,
        grid_width: int = _GRID_W,
        planning_depth: int = 8,
        beam_width: int = 3,
        wm_lr: float = 1e-3,
        policy_lr: float = 3e-4,
        planning_threshold: float = 0.7,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # World model
        self.world_model = LearnedWorldModel(
            height=grid_height, width=grid_width
        ).to(self._device)
        self.replay_buffer = ReplayBuffer(max_size=50_000)
        self.wm_trainer = WorldModelTrainer(
            self.world_model, lr=wm_lr, device=str(self._device)
        )
        # Use the trainer's internal buffer as our replay buffer
        self.wm_trainer.buffer = self.replay_buffer

        # Policy network (PPO-style fallback)
        self.policy_network = PPONetwork(grid_height, grid_width).to(self._device)
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=policy_lr
        )
        self.policy_memory = PPOMemory()

        # Mental simulation
        self.mental_sim = MentalSimulator()
        self.planning_depth = planning_depth
        self.beam_width = beam_width
        self.planning_threshold = planning_threshold

        # Tracking
        self.world_model_accuracy = 0.0
        self.planning_active = False
        self.total_dreams = 0
        self.last_thought: Optional[ThoughtStep] = None

    @property
    def name(self) -> str:
        return "ThinkerAgent"

    # -- Action selection ---------------------------------------------------

    def select_action(
        self, observation: Dict[str, Any]
    ) -> Tuple[int, Optional[ThoughtStep]]:
        """Pick an action. Uses planning if world model is accurate, else policy.

        Returns (action, thought_info_or_None).
        """
        if (self.world_model_accuracy >= self.planning_threshold
                and len(self.replay_buffer) >= 500):
            # Plan ahead with world model
            self.planning_active = True
            action, thought = self.mental_sim.think_ahead(
                observation, self.world_model,
                depth=self.planning_depth, beam_width=self.beam_width,
            )
            self.last_thought = thought
            return action, thought

        # Fallback: use policy network (like PPO)
        self.planning_active = False
        self.last_thought = None
        action, log_prob, value = self._policy_select(observation)
        return action, None

    def act(self, observation: Dict[str, Any]) -> int:
        action, _ = self.select_action(observation)
        return action

    def _policy_select(
        self, obs: Dict[str, Any]
    ) -> Tuple[int, float, float]:
        """Sample from policy network (training mode)."""
        self.policy_network.eval()
        with torch.no_grad():
            probs, value = self.policy_network(
                *_obs_to_tensors(obs, self._device)
            )
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def get_action_probs(self, observation: Dict[str, Any]) -> np.ndarray:
        self.policy_network.eval()
        with torch.no_grad():
            probs, _ = self.policy_network(
                *_obs_to_tensors(observation, self._device)
            )
        return probs.squeeze(0).cpu().numpy()

    # -- Experience collection ----------------------------------------------

    def store_experience(
        self, obs: Dict[str, Any], action: int,
        next_obs: Dict[str, Any], reward: float, done: bool
    ) -> None:
        """Store a real transition in the replay buffer."""
        grid = np.asarray(obs["grid"])
        pos = np.array(obs["agent_pos"], dtype=np.float32)
        next_grid = np.asarray(next_obs["grid"])
        next_pos = np.array(next_obs["agent_pos"], dtype=np.float32)

        self.replay_buffer.add(
            grid, pos, action, next_grid, next_pos, reward, done
        )

        # Also store in policy memory for PPO-style updates (when not planning)
        if not self.planning_active:
            _, log_prob, value = self._policy_select(obs)
            self.policy_memory.store(obs, action, reward, log_prob, value, done)

    # -- World model learning -----------------------------------------------

    def learn_world_model(self, train_steps: int = 10) -> Dict[str, float]:
        """Train the world model on the replay buffer."""
        if len(self.replay_buffer) < 128:
            return {}

        metrics = {}
        for _ in range(train_steps):
            step_metrics = self.wm_trainer.train_step(batch_size=64)
            if step_metrics:
                metrics = step_metrics

        # Compute accuracy
        self._update_accuracy()
        return metrics

    def _update_accuracy(self) -> None:
        """Estimate world model accuracy on a sample from the replay buffer."""
        if len(self.replay_buffer) < 100:
            self.world_model_accuracy = 0.0
            return

        grids, positions, actions, next_grids, _, _, _ = self.replay_buffer.sample(
            min(200, len(self.replay_buffer))
        )
        dev = self._device
        g = torch.tensor(grids, dtype=torch.long, device=dev)
        p = torch.tensor(positions, dtype=torch.float32, device=dev)
        a = torch.tensor(actions, dtype=torch.long, device=dev)

        pred_g, _, _, _ = self.world_model.predict(g, p, a)
        self.world_model_accuracy = float(
            (pred_g.cpu().numpy() == next_grids).mean()
        )

    # -- Dreaming (model-based data augmentation) ---------------------------

    def dream_and_learn(self, n_episodes: int = 50, max_steps: int = 30) -> None:
        """Generate imagined episodes using the world model.

        Then train the policy network on this imagined data.
        This is what makes us more sample-efficient than PPO.
        """
        if self.world_model_accuracy < 0.5 or len(self.replay_buffer) < 500:
            return

        dream_memory = PPOMemory()
        dev = self._device

        for _ in range(n_episodes):
            # Sample a starting state from the replay buffer
            grids, positions, _, _, _, _, _ = self.replay_buffer.sample(1)
            grid_t = torch.tensor(grids[0], dtype=torch.long, device=dev)
            pos_t = torch.tensor(positions[0], dtype=torch.float32, device=dev)

            for step in range(max_steps):
                # Build observation dict for policy
                obs = {
                    "grid": grid_t.cpu().numpy(),
                    "agent_pos": pos_t.cpu().numpy().astype(np.float32),
                    "inventory": np.zeros(_MAX_KEYS, dtype=np.int8),
                }

                action, log_prob, value = self._policy_select(obs)

                # Predict next state with world model
                g = grid_t.unsqueeze(0)
                p = pos_t.unsqueeze(0)
                a = torch.tensor([action], device=dev)
                pred_grid, pred_pos, pred_reward, pred_done = \
                    self.world_model.predict(g, p, a)

                reward = pred_reward.squeeze().item()
                done = pred_done.squeeze().item() > 0.5

                dream_memory.store(obs, action, reward, log_prob, value, done)

                grid_t = pred_grid.squeeze(0)
                pos_t = pred_pos.squeeze(0)

                if done:
                    break

        self.total_dreams += n_episodes

        # PPO update on dreamed data
        if len(dream_memory) > 0:
            self._ppo_update(dream_memory)

        # Also update on real data if available
        if len(self.policy_memory) > 0:
            self._ppo_update(self.policy_memory)

    def _ppo_update(
        self, memory: PPOMemory,
        gamma: float = 0.99, gae_lambda: float = 0.95,
        clip_eps: float = 0.2, epochs: int = 4, batch_size: int = 64,
    ) -> None:
        """Run a PPO update on the given memory buffer."""
        if len(memory) == 0:
            return

        import torch.nn.functional as F

        dev = self._device
        self.policy_network.train()

        grids = torch.tensor(np.array(memory.grids), dtype=torch.long, device=dev)
        positions = torch.tensor(np.array(memory.positions), dtype=torch.float32, device=dev)
        inventories = torch.tensor(np.array(memory.inventories), dtype=torch.float32, device=dev)
        actions = torch.tensor(memory.actions, dtype=torch.long, device=dev)
        old_lp = torch.tensor(memory.log_probs, dtype=torch.float32, device=dev)
        rewards = np.array(memory.rewards, dtype=np.float32)
        values = np.array(memory.values, dtype=np.float32)
        dones = np.array(memory.dones, dtype=np.float32)

        # GAE
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            nv = 0.0 if t == n - 1 else values[t + 1]
            nt = 1.0 - dones[t]
            delta = rewards[t] + gamma * nv * nt - values[t]
            last_gae = delta + gamma * gae_lambda * nt * last_gae
            advantages[t] = last_gae
        returns = advantages + values

        adv_t = torch.tensor(advantages, dtype=torch.float32, device=dev)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=dev)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(epochs):
            for idx in memory.get_batches(batch_size):
                probs, vals = self.policy_network(
                    grids[idx], positions[idx], inventories[idx]
                )
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_lp[idx])
                s1 = ratio * adv_t[idx]
                s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[idx]
                p_loss = -torch.min(s1, s2).mean()
                v_loss = F.mse_loss(vals.squeeze(-1), ret_t[idx])
                loss = p_loss + 0.5 * v_loss - 0.01 * entropy

                self.policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
                self.policy_optimizer.step()

        memory.clear()

    # -- BaseAgent interface ------------------------------------------------

    def learn(self, **kwargs: Any) -> Dict[str, Any]:
        metrics = self.learn_world_model()
        self.dream_and_learn()
        return metrics

    def save(self, path: str) -> None:
        torch.save({
            "world_model": self.world_model.state_dict(),
            "policy_network": self.policy_network.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        state = torch.load(path, weights_only=False)
        self.world_model.load_state_dict(state["world_model"])
        self.policy_network.load_state_dict(state["policy_network"])
