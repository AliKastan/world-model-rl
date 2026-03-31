"""Training loops for PPO and ThinkerAgent.

Usage::

    python -m training.train --agent ppo --difficulty 1 --episodes 5000
    python -m training.train --agent thinker --difficulty 1 --episodes 2000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import env.gym_env  # noqa: F401 — triggers registration


# ═══════════════════════════════════════════════════════════════════════
# PPO training
# ═══════════════════════════════════════════════════════════════════════


def train_ppo(
    agent: Any,
    env: Any,
    n_episodes: int = 10_000,
    update_every: int = 10,
) -> Tuple[List[float], List[bool]]:
    """Train a PPOAgent through trial and error.

    The agent starts random and gradually learns from experience.
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

        if (episode + 1) % update_every == 0:
            agent.update()

        if (episode + 1) % 100 == 0:
            recent_r = all_rewards[-100:]
            recent_s = all_solved[-100:]
            sr = sum(recent_s) / len(recent_s) * 100
            ar = sum(recent_r) / len(recent_r)
            tag = "[+++]" if sr > 50 else "[++ ]" if sr > 20 else "[+  ]"
            print(f"  Episode {episode + 1:>6}/{n_episodes} | "
                  f"Avg Reward: {ar:>7.1f} | Solve Rate: {sr:>5.1f}% | {tag}")

    return all_rewards, all_solved


# ═══════════════════════════════════════════════════════════════════════
# ThinkerAgent training
# ═══════════════════════════════════════════════════════════════════════


def train_thinker(
    agent: Any,
    env: Any,
    n_episodes: int = 2_000,
    wm_train_every: int = 10,
    dream_every: int = 50,
) -> Tuple[List[float], List[bool]]:
    """Train a ThinkerAgent with world model + dreaming.

    Much more sample-efficient than PPO thanks to planning ahead.
    """
    all_rewards: List[float] = []
    all_solved: List[bool] = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action, thought_info = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_experience(obs, action, next_obs, reward, done)
            obs = next_obs
            episode_reward += reward

        all_rewards.append(episode_reward)
        all_solved.append(info.get("solved", False))

        # Train world model periodically
        if (episode + 1) % wm_train_every == 0:
            agent.learn_world_model()

        # Dream periodically (once world model is decent)
        if (episode + 1) % dream_every == 0:
            agent.dream_and_learn()

        if (episode + 1) % 100 == 0:
            recent_r = all_rewards[-100:]
            recent_s = all_solved[-100:]
            sr = sum(recent_s) / len(recent_s) * 100
            ar = sum(recent_r) / len(recent_r)
            wm_acc = getattr(agent, "world_model_accuracy", 0.0)
            planning = "ON" if getattr(agent, "planning_active", False) else "OFF"
            print(f"  Episode {episode + 1:>6}/{n_episodes} | "
                  f"Reward: {ar:>7.1f} | Solve: {sr:>5.1f}% | "
                  f"WM: {wm_acc:.0%} | Plan: {planning}")

    return all_rewards, all_solved


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Train puzzle agents")
    parser.add_argument("--agent", default="ppo", choices=["ppo", "thinker"])
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    from env.gym_env import PuzzleEnv

    print("=" * 60)
    print(f"  Training {args.agent.upper()} | difficulty={args.difficulty} | "
          f"episodes={args.episodes}")
    print("=" * 60)

    env = PuzzleEnv(difficulty=args.difficulty, max_steps=args.max_steps)

    t0 = time.monotonic()

    if args.agent == "ppo":
        from agents.model_free.ppo_agent import PPOAgent
        agent = PPOAgent()
        all_rewards, all_solved = train_ppo(agent, env, n_episodes=args.episodes)
    else:
        from agents.model_based.thinker_agent import ThinkerAgent
        agent = ThinkerAgent()
        all_rewards, all_solved = train_thinker(agent, env, n_episodes=args.episodes)

    elapsed = time.monotonic() - t0
    total_solved = sum(all_solved)
    print(f"\n  Done in {elapsed:.1f}s — {total_solved}/{len(all_solved)} solved "
          f"({total_solved / max(len(all_solved), 1) * 100:.1f}%)")

    if args.save:
        agent.save(args.save)
        print(f"  Model saved to {args.save}")

    env.close()


if __name__ == "__main__":
    main()
