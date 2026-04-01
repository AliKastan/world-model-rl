"""Training pipeline for Sokoban RL with curriculum learning.

Uses SokobanEnv (pure Sokoban) + CNN-based PPOAgent with progressive
difficulty from 1-box to 3-box puzzles.

Usage::

    python -m training.train_sokoban                     # default 50k episodes
    python -m training.train_sokoban --episodes 10000    # quick run
    python -m training.train_sokoban --resume ckpt.pt    # resume training
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.sokoban_env import SokobanEnv
from agents.model_free.ppo_agent import PPOAgent
from training.curriculum import CurriculumManager


GRID_SIZE = 12  # max grid dimension (obs padded to this)


def train(
    n_episodes: int = 50_000,
    checkpoint_dir: str = "checkpoints",
    resume_path: str | None = None,
    update_every: int = 20,
    log_every: int = 100,
) -> Tuple[List[float], List[bool]]:
    """Train PPO agent on Sokoban with curriculum learning."""

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Curriculum: difficulty 1-5 controls n_boxes and grid size
    curriculum = CurriculumManager(
        start_difficulty=1,
        max_difficulty=5,
        promote_threshold=0.7,
        demote_threshold=0.15,
        eval_window=200,
    )

    # Map difficulty -> n_boxes
    def difficulty_to_boxes(d: int) -> int:
        return min(d, 3)  # 1->1, 2->2, 3->3, 4->3, 5->3

    # Environment starts at difficulty 1 (1 box)
    env = SokobanEnv(
        level_set="training",
        max_h=GRID_SIZE,
        max_w=GRID_SIZE,
        max_steps=200,
        n_boxes=1,
    )

    # Agent with matching grid dimensions
    agent = PPOAgent(
        grid_h=GRID_SIZE,
        grid_w=GRID_SIZE,
        lr=2.5e-4,
        entropy_coef=0.02,
        update_epochs=4,
        mini_batch_size=256,
    )

    if resume_path and os.path.exists(resume_path):
        agent.load(resume_path)
        print(f"  Resumed from {resume_path}")

    all_rewards: List[float] = []
    all_solved: List[bool] = []
    best_solve_rate = 0.0

    print()
    print("=" * 65)
    print("  SOKOBAN RL TRAINING")
    print("=" * 65)
    print(f"  Episodes: {n_episodes}")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Update every: {update_every} episodes")
    print(f"  Curriculum: 1-box -> 2-box -> 3-box (auto-promotion)")
    print("=" * 65)
    print()

    t0 = time.monotonic()
    prev_difficulty = curriculum.difficulty

    for episode in range(n_episodes):
        # Adjust environment based on curriculum
        nb = difficulty_to_boxes(curriculum.difficulty)
        env.n_boxes = nb
        env.max_steps = 150 + nb * 50  # more steps for more boxes

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
        solved = info.get("solved", False)
        all_solved.append(solved)

        # Update curriculum
        old_diff = curriculum.difficulty
        curriculum.record(solved)
        if curriculum.difficulty != old_diff:
            new_nb = difficulty_to_boxes(curriculum.difficulty)
            direction = "UP" if curriculum.difficulty > old_diff else "DOWN"
            print(f"\n  >>> CURRICULUM {direction}: "
                  f"difficulty {old_diff} -> {curriculum.difficulty} "
                  f"({new_nb} boxes) <<<\n")
            # Save checkpoint on curriculum change
            ckpt_path = os.path.join(
                checkpoint_dir, f"ppo_diff{curriculum.difficulty}.pt"
            )
            agent.save(ckpt_path)

        # PPO update
        if (episode + 1) % update_every == 0:
            agent.update()

        # Logging
        if (episode + 1) % log_every == 0:
            recent_r = all_rewards[-log_every:]
            recent_s = all_solved[-log_every:]
            sr = sum(recent_s) / len(recent_s) * 100
            ar = sum(recent_r) / len(recent_r)
            nb = difficulty_to_boxes(curriculum.difficulty)
            elapsed = time.monotonic() - t0
            eps_per_sec = (episode + 1) / elapsed

            if sr > best_solve_rate:
                best_solve_rate = sr

            bar = "#" * int(sr / 5) + "-" * (20 - int(sr / 5))
            print(f"  Ep {episode + 1:>6}/{n_episodes} | "
                  f"Reward: {ar:>7.1f} | "
                  f"Solve: {sr:>5.1f}% [{bar}] | "
                  f"Boxes: {nb} | "
                  f"{eps_per_sec:.0f} ep/s")

    elapsed = time.monotonic() - t0
    total_solved = sum(all_solved)
    final_rate = total_solved / max(len(all_solved), 1) * 100

    print()
    print("=" * 65)
    print(f"  TRAINING COMPLETE")
    print(f"  Time: {elapsed:.1f}s ({n_episodes / elapsed:.0f} ep/s)")
    print(f"  Total solved: {total_solved}/{len(all_solved)} ({final_rate:.1f}%)")
    print(f"  Best solve rate: {best_solve_rate:.1f}%")
    print(f"  Final difficulty: {curriculum.difficulty} "
          f"({difficulty_to_boxes(curriculum.difficulty)} boxes)")
    print("=" * 65)

    # Save final model
    final_path = os.path.join(checkpoint_dir, "ppo_final.pt")
    agent.save(final_path)
    print(f"  Model saved to {final_path}")

    env.close()
    return all_rewards, all_solved


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Sokoban RL agent")
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--update-every", type=int, default=20)
    args = parser.parse_args()

    train(
        n_episodes=args.episodes,
        checkpoint_dir=args.checkpoint_dir,
        resume_path=args.resume,
        update_every=args.update_every,
    )


if __name__ == "__main__":
    main()
