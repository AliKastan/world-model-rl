"""World model — predicts what happens next given current state + action.

Two implementations for comparison:
  - PerfectWorldModel: uses PuzzleWorld.clone() for exact simulation (upper bound)
  - LearnedWorldModel: neural network that learns transition dynamics from experience
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env.puzzle_world import PuzzleWorld, TYPE_IDS

NUM_TYPES = len(TYPE_IDS)  # 14


# ═══════════════════════════════════════════════════════════════════════
# Layer 1: PerfectWorldModel
# ═══════════════════════════════════════════════════════════════════════


class PerfectWorldModel:
    """Uses PuzzleWorld.clone() to make an exact copy and simulate.

    This is "cheating" — the agent knows the exact rules.
    But this is realistic: humans ALSO know Sokoban rules
    (push box, can't pull, walls block).
    Used as an upper-bound baseline.
    """

    def predict(
        self, state: PuzzleWorld, action: int
    ) -> Tuple[PuzzleWorld, float, bool]:
        """Clone the world, execute action, return (next_world, reward, done)."""
        clone = state.clone()
        _obs, reward, terminated, truncated, _info = clone.step(action)
        done = terminated or truncated
        return clone, reward, done

    def imagine_trajectory(
        self, state: PuzzleWorld, actions: List[int]
    ) -> List[Tuple[PuzzleWorld, float, bool]]:
        """Apply sequence of actions in imagination.

        CRITICAL FUNCTION: This is the agent "thinking ahead" without
        touching the real environment.

        Returns:
            List of (state, reward, done) at each step.
        """
        trajectory: List[Tuple[PuzzleWorld, float, bool]] = []
        current = state.clone()
        for action in actions:
            _obs, reward, terminated, truncated, _info = current.step(action)
            done = terminated or truncated
            trajectory.append((current.clone(), reward, done))
            if done:
                break
        return trajectory

    def accuracy(self) -> float:
        """Always 1.0 — perfect by definition."""
        return 1.0


# ═══════════════════════════════════════════════════════════════════════
# Layer 2: LearnedWorldModel
# ═══════════════════════════════════════════════════════════════════════


class LearnedWorldModel(nn.Module):
    """Learns the transition rules from EXPERIENCE — doesn't know them in advance.

    This is the real research contribution.

    Input:  (grid_state: Tensor, agent_pos: Tensor, action: int)
    Output: (predicted_next_grid, predicted_next_pos, predicted_reward, predicted_done)
    """

    def __init__(self, height: int, width: int, num_types: int = NUM_TYPES) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_types = num_types

        # --- Encoder ---
        # Grid encoder: one-hot grid → convolutional features
        self.grid_conv = nn.Sequential(
            nn.Conv2d(num_types, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

        # Position and action embeddings
        self.pos_encoder = nn.Linear(2, 16)
        self.action_encoder = nn.Embedding(4, 16)

        # Fusion: flatten grid features → concat with pos_emb + action_emb → latent
        grid_flat_size = 64 * height * width
        self.fusion = nn.Sequential(
            nn.Linear(grid_flat_size + 16 + 16, 256),
            nn.ReLU(),
        )

        # --- Decoder ---
        # Grid prediction: latent → logits over cell types for each (H, W)
        self.grid_decoder_fc = nn.Linear(256, 64 * height * width)
        self.grid_decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_types, 3, padding=1),
        )

        # Scalar prediction heads
        self.pos_decoder = nn.Linear(256, 2)
        self.reward_decoder = nn.Linear(256, 1)
        self.done_decoder = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        grid: torch.Tensor,
        agent_pos: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            grid: (B, H, W) int tensor of type IDs.
            agent_pos: (B, 2) float tensor of (x, y) positions.
            action: (B,) int tensor of action indices.

        Returns:
            grid_logits: (B, num_types, H, W) — class logits per cell.
            pos_pred: (B, 2) — predicted agent position.
            reward_pred: (B, 1) — predicted reward.
            done_pred: (B, 1) — predicted done probability.
        """
        batch_size = grid.shape[0]

        # One-hot encode grid: (B, H, W) → (B, num_types, H, W)
        grid_onehot = F.one_hot(grid.long(), self.num_types)  # (B, H, W, C)
        grid_onehot = grid_onehot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Encode
        grid_features = self.grid_conv(grid_onehot)  # (B, 64, H, W)
        grid_flat = grid_features.reshape(batch_size, -1)  # (B, 64*H*W)

        pos_emb = F.relu(self.pos_encoder(agent_pos.float()))  # (B, 16)
        action_emb = self.action_encoder(action.long())  # (B, 16)

        # Fusion
        fused = torch.cat([grid_flat, pos_emb, action_emb], dim=1)
        latent = self.fusion(fused)  # (B, 256)

        # Decode grid
        grid_dec = self.grid_decoder_fc(latent)  # (B, 64*H*W)
        grid_dec = grid_dec.reshape(batch_size, 64, self.height, self.width)
        grid_logits = self.grid_decoder_conv(grid_dec)  # (B, num_types, H, W)

        # Decode scalars
        pos_pred = self.pos_decoder(latent)  # (B, 2)
        reward_pred = self.reward_decoder(latent)  # (B, 1)
        done_pred = self.done_decoder(latent)  # (B, 1)

        return grid_logits, pos_pred, reward_pred, done_pred

    def predict(
        self,
        grid: torch.Tensor,
        agent_pos: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step prediction (inference mode).

        Returns:
            predicted_grid: (B, H, W) — argmax type IDs.
            predicted_pos: (B, 2)
            predicted_reward: (B, 1)
            predicted_done: (B, 1)
        """
        self.eval()
        with torch.no_grad():
            grid_logits, pos_pred, reward_pred, done_pred = self.forward(
                grid, agent_pos, action
            )
            predicted_grid = grid_logits.argmax(dim=1)  # (B, H, W)
        return predicted_grid, pos_pred, reward_pred, done_pred

    def imagine_trajectory(
        self,
        grid: torch.Tensor,
        agent_pos: torch.Tensor,
        actions: List[int],
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Apply actions sequentially using own predictions (autoregressive).

        NOTE: Errors compound over steps — 1-step is easy, 20-step is hard.

        Args:
            grid: (H, W) int tensor — initial grid state.
            agent_pos: (2,) float tensor — initial agent position.
            actions: list of action ints.

        Returns:
            List of (predicted_grid, predicted_pos, predicted_reward, predicted_done)
            at each step.
        """
        self.eval()
        trajectory = []
        current_grid = grid.unsqueeze(0)  # (1, H, W)
        current_pos = agent_pos.unsqueeze(0)  # (1, 2)

        with torch.no_grad():
            for action in actions:
                action_t = torch.tensor([action], device=grid.device)
                grid_logits, pos_pred, reward_pred, done_pred = self.forward(
                    current_grid, current_pos, action_t
                )
                predicted_grid = grid_logits.argmax(dim=1)  # (1, H, W)
                trajectory.append((
                    predicted_grid.squeeze(0),
                    pos_pred.squeeze(0),
                    reward_pred.squeeze(0),
                    done_pred.squeeze(0),
                ))
                # Feed predictions back in (autoregressive)
                current_grid = predicted_grid
                current_pos = pos_pred
                if done_pred.item() > 0.5:
                    break

        return trajectory


# ═══════════════════════════════════════════════════════════════════════
# Replay Buffer & Trainer
# ═══════════════════════════════════════════════════════════════════════


class ReplayBuffer:
    """Stores (state, action, next_state, reward, done) tuples from real experience."""

    def __init__(self, max_size: int = 100_000) -> None:
        self.buffer: deque = deque(maxlen=max_size)

    def add(
        self,
        grid: np.ndarray,
        agent_pos: np.ndarray,
        action: int,
        next_grid: np.ndarray,
        next_agent_pos: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self.buffer.append((
            grid.copy(),
            agent_pos.copy(),
            action,
            next_grid.copy(),
            next_agent_pos.copy(),
            reward,
            done,
        ))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        grids, positions, actions, next_grids, next_positions, rewards, dones = zip(
            *batch
        )
        return (
            np.array(grids),
            np.array(positions, dtype=np.float32),
            np.array(actions),
            np.array(next_grids),
            np.array(next_positions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class WorldModelTrainer:
    """Trains a LearnedWorldModel from replay buffer experience."""

    def __init__(
        self,
        model: LearnedWorldModel,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self._train_steps = 0

    def add_experience(
        self,
        grid: np.ndarray,
        agent_pos: np.ndarray,
        action: int,
        next_grid: np.ndarray,
        next_agent_pos: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self.buffer.add(grid, agent_pos, action, next_grid, next_agent_pos, reward, done)

    def train_step(self, batch_size: int = 128) -> Dict[str, float]:
        """Sample batch from buffer, compute loss, update model."""
        if len(self.buffer) < batch_size:
            return {}

        self.model.train()
        grids, positions, actions, next_grids, next_positions, rewards, dones = (
            self.buffer.sample(batch_size)
        )

        # Convert to tensors
        grids_t = torch.tensor(grids, dtype=torch.long, device=self.device)
        positions_t = torch.tensor(positions, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        next_grids_t = torch.tensor(next_grids, dtype=torch.long, device=self.device)
        next_positions_t = torch.tensor(
            next_positions, dtype=torch.float32, device=self.device
        )
        rewards_t = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        dones_t = torch.tensor(
            dones, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        # Forward pass
        grid_logits, pos_pred, reward_pred, done_pred = self.model(
            grids_t, positions_t, actions_t
        )

        # Losses
        grid_loss = F.cross_entropy(grid_logits, next_grids_t)
        pos_loss = F.mse_loss(pos_pred, next_positions_t)
        reward_loss = F.mse_loss(reward_pred, rewards_t)
        done_loss = F.binary_cross_entropy(done_pred, dones_t)

        total_loss = grid_loss + pos_loss + reward_loss + done_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self._train_steps += 1

        return {
            "total_loss": total_loss.item(),
            "grid_loss": grid_loss.item(),
            "pos_loss": pos_loss.item(),
            "reward_loss": reward_loss.item(),
            "done_loss": done_loss.item(),
        }

    def get_metrics(
        self,
        world: PuzzleWorld,
        num_random_steps: int = 200,
        multi_step_horizons: Tuple[int, ...] = (1, 5, 10, 20),
    ) -> Dict[str, Any]:
        """Evaluate the model's prediction accuracy.

        Returns:
            dict with grid_accuracy, pos_accuracy, reward_accuracy,
            and multi_step_accuracy at various horizons.
        """
        self.model.eval()
        device = self.device

        # --- 1-step accuracy ---
        grid_correct_total = 0
        grid_cells_total = 0
        pos_correct_total = 0
        reward_mse_total = 0.0
        n_samples = 0

        test_world = world.clone()
        for _ in range(num_random_steps):
            obs = test_world.get_observation()
            grid = obs["grid"]
            agent_pos = np.array(obs["agent_pos"], dtype=np.float32)
            action = random.randint(0, 3)

            _obs_next, reward, terminated, truncated, _info = test_world.step(action)
            done = terminated or truncated
            next_obs = test_world.get_observation()
            next_grid = next_obs["grid"]
            next_agent_pos = np.array(next_obs["agent_pos"], dtype=np.float32)

            # Predict
            grid_t = torch.tensor(grid, dtype=torch.long, device=device).unsqueeze(0)
            pos_t = torch.tensor(agent_pos, device=device).unsqueeze(0)
            action_t = torch.tensor([action], device=device)

            pred_grid, pred_pos, pred_reward, pred_done = self.model.predict(
                grid_t, pos_t, action_t
            )
            pred_grid_np = pred_grid.squeeze(0).cpu().numpy()
            pred_pos_np = pred_pos.squeeze(0).cpu().numpy()

            grid_correct_total += (pred_grid_np == next_grid).sum()
            grid_cells_total += next_grid.size
            pos_correct_total += int(
                np.allclose(np.round(pred_pos_np), next_agent_pos, atol=0.5)
            )
            reward_mse_total += (pred_reward.item() - reward) ** 2
            n_samples += 1

            if done:
                test_world = world.clone()

        grid_accuracy = grid_correct_total / max(grid_cells_total, 1)
        pos_accuracy = pos_correct_total / max(n_samples, 1)
        reward_mse = reward_mse_total / max(n_samples, 1)

        # --- Multi-step accuracy ---
        multi_step_accuracy: Dict[int, float] = {}
        for horizon in multi_step_horizons:
            correct_cells = 0
            total_cells = 0
            num_trials = 20

            for _ in range(num_trials):
                trial_world = world.clone()
                # Collect a random action sequence
                actions = [random.randint(0, 3) for _ in range(horizon)]

                # Ground truth trajectory
                gt_world = trial_world.clone()
                gt_grids = []
                for a in actions:
                    _o, _r, term, trunc, _i = gt_world.step(a)
                    gt_grids.append(gt_world.get_observation()["grid"])
                    if term or trunc:
                        break

                # Predicted trajectory
                obs = trial_world.get_observation()
                grid_init = torch.tensor(
                    obs["grid"], dtype=torch.long, device=device
                )
                pos_init = torch.tensor(
                    np.array(obs["agent_pos"], dtype=np.float32), device=device
                )
                predicted = self.model.imagine_trajectory(
                    grid_init, pos_init, actions[: len(gt_grids)]
                )

                # Compare final step
                if predicted and gt_grids:
                    final_pred = predicted[-1][0].cpu().numpy()
                    final_gt = gt_grids[-1]
                    correct_cells += (final_pred == final_gt).sum()
                    total_cells += final_gt.size

            multi_step_accuracy[horizon] = (
                correct_cells / max(total_cells, 1)
            )

        return {
            "grid_accuracy": float(grid_accuracy),
            "pos_accuracy": float(pos_accuracy),
            "reward_mse": float(reward_mse),
            "multi_step_accuracy": multi_step_accuracy,
        }


# ═══════════════════════════════════════════════════════════════════════
# Test / demo
# ═══════════════════════════════════════════════════════════════════════


def _collect_experience(
    world: PuzzleWorld, trainer: WorldModelTrainer, num_steps: int
) -> None:
    """Play random steps and store transitions in the trainer's replay buffer."""
    current = world.clone()
    for _ in range(num_steps):
        obs = current.get_observation()
        grid = obs["grid"]
        agent_pos = np.array(obs["agent_pos"], dtype=np.float32)
        action = random.randint(0, 3)

        _obs_next, reward, terminated, truncated, _info = current.step(action)
        done = terminated or truncated
        next_obs = current.get_observation()

        trainer.add_experience(
            grid=grid,
            agent_pos=agent_pos,
            action=action,
            next_grid=next_obs["grid"],
            next_agent_pos=np.array(next_obs["agent_pos"], dtype=np.float32),
            reward=reward,
            done=done,
        )

        if done:
            current = world.clone()


def main() -> None:
    """Generate a level, train the learned world model, and compare with perfect."""
    from env.level_generator import LevelGenerator

    print("=" * 60)
    print("World Model — Training & Evaluation")
    print("=" * 60)

    # Generate a level
    gen = LevelGenerator()
    world = gen.generate(difficulty=3, seed=42)
    H, W = world.height, world.width
    print(f"\nLevel: {W}x{H}, difficulty=3")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Set up models ---
    perfect_model = PerfectWorldModel()
    learned_model = LearnedWorldModel(height=H, width=W, num_types=NUM_TYPES)
    trainer = WorldModelTrainer(learned_model, lr=1e-3, device=device)

    # --- Collect experience: 500 random steps ---
    print("\nCollecting 500 random steps...")
    _collect_experience(world, trainer, num_steps=500)
    print(f"Replay buffer size: {len(trainer.buffer)}")

    # --- Train for 200 epochs ---
    print("\nTraining LearnedWorldModel for 200 epochs...")
    for epoch in range(200):
        losses = trainer.train_step(batch_size=128)
        if losses and (epoch + 1) % 50 == 0:
            print(
                f"  Epoch {epoch + 1:3d}: "
                f"total={losses['total_loss']:.4f}  "
                f"grid={losses['grid_loss']:.4f}  "
                f"pos={losses['pos_loss']:.4f}  "
                f"reward={losses['reward_loss']:.4f}  "
                f"done={losses['done_loss']:.4f}"
            )

    # --- Evaluate ---
    print("\nEvaluating...")
    metrics = trainer.get_metrics(world, num_random_steps=200)

    grid_1 = metrics["multi_step_accuracy"].get(1, metrics["grid_accuracy"])
    grid_5 = metrics["multi_step_accuracy"].get(5, 0.0)
    grid_10 = metrics["multi_step_accuracy"].get(10, 0.0)
    grid_20 = metrics["multi_step_accuracy"].get(20, 0.0)

    print(f"\n{'Metric':<25} {'Learned':>10} {'Perfect':>10}")
    print("-" * 47)
    print(f"{'1-step grid accuracy':<25} {grid_1:>9.0%} {'100%':>10}")
    print(f"{'5-step grid accuracy':<25} {grid_5:>9.0%} {'100%':>10}")
    print(f"{'10-step grid accuracy':<25} {grid_10:>9.0%} {'100%':>10}")
    print(f"{'20-step grid accuracy':<25} {grid_20:>9.0%} {'100%':>10}")
    print(f"{'Position accuracy':<25} {metrics['pos_accuracy']:>9.0%} {'100%':>10}")
    print(f"{'Reward MSE':<25} {metrics['reward_mse']:>10.4f} {'0.0000':>10}")
    print(f"{'Perfect model accuracy':<25} {'':>10} {perfect_model.accuracy():>9.0%}")

    print(
        f"\n1-step: {grid_1:.0%}, 5-step: {grid_5:.0%}, "
        f"10-step: {grid_10:.0%} (Learned) vs 100% (Perfect)"
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
