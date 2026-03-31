"""Main training script for puzzle agents.

Usage::

    python -m training.train --agent ppo --difficulty 1 --timesteps 100000

See ``--help`` for the full set of options.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium
import env.gym_env  # noqa: F401  — triggers registration

from stable_baselines3.common.callbacks import BaseCallback


# ---------------------------------------------------------------------------
# Evaluation callback (runs inside SB3 .learn())
# ---------------------------------------------------------------------------

class _EvalCallback(BaseCallback):
    """Periodic evaluation + curriculum + checkpointing inside SB3 training.

    This single callback handles eval, checkpointing, curriculum adjustment,
    console progress, and metric collection so that SB3's ``model.learn()``
    does everything in one call.
    """

    def __init__(
        self,
        eval_env: gymnasium.Env,
        eval_freq: int = 4096,
        eval_episodes: int = 10,
        checkpoint_dir: str = "checkpoints",
        checkpoint_freq: int = 20480,
        curriculum: bool = False,
        difficulty: int = 1,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.curriculum = curriculum
        self.difficulty = difficulty

        # Metric history (for plotting)
        self.eval_timesteps: List[int] = []
        self.eval_rewards: List[float] = []
        self.eval_solve_rates: List[float] = []
        self.eval_avg_steps: List[float] = []

        # Curriculum tracking
        self._recent_solves: deque = deque(maxlen=100)

        # Episode-level tracking from training rollouts
        self.train_rewards: List[float] = []
        self.train_solves: List[bool] = []
        self.train_steps_list: List[int] = []
        self._ep_reward = 0.0

        self._last_eval_step = 0
        self._last_ckpt_step = 0
        self._t0 = time.monotonic()

    def _on_step(self) -> bool:
        # Accumulate episode reward from info buffers
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                self.train_rewards.append(ep_info["r"])
                self.train_steps_list.append(int(ep_info["l"]))
            if info.get("solved", False):
                self.train_solves.append(True)
                self._recent_solves.append(True)
            elif "TimeLimit.truncated" in info or info.get("is_deadlock", False):
                self.train_solves.append(False)
                self._recent_solves.append(False)

        # Periodic evaluation
        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            self._last_eval_step = self.num_timesteps
            self._run_eval()

        # Periodic checkpoint
        if self.num_timesteps - self._last_ckpt_step >= self.checkpoint_freq:
            self._last_ckpt_step = self.num_timesteps
            self._save_checkpoint()

        return True

    def _run_eval(self) -> None:
        rewards = []
        steps = []
        solves = 0
        for ep in range(self.eval_episodes):
            obs, info = self.eval_env.reset(seed=10000 + ep)
            ep_reward = 0.0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(int(action))
                ep_reward += reward
                done = terminated or truncated
            rewards.append(ep_reward)
            steps.append(info.get("steps_taken", 0))
            if info.get("solved", False):
                solves += 1

        avg_reward = float(np.mean(rewards))
        avg_steps = float(np.mean(steps))
        solve_rate = solves / self.eval_episodes

        self.eval_timesteps.append(self.num_timesteps)
        self.eval_rewards.append(avg_reward)
        self.eval_solve_rates.append(solve_rate)
        self.eval_avg_steps.append(avg_steps)

        elapsed = time.monotonic() - self._t0
        print(
            f"  [{self.num_timesteps:>8d} steps | {elapsed:>6.0f}s] "
            f"eval reward={avg_reward:>7.1f}  solve={solve_rate:>5.0%}  "
            f"steps={avg_steps:>5.1f}  diff={self.difficulty}"
        )

        # Curriculum adjustment
        if self.curriculum and len(self._recent_solves) >= 20:
            recent_rate = sum(self._recent_solves) / len(self._recent_solves)
            if recent_rate > 0.80 and self.difficulty < 10:
                self.difficulty += 1
                self.eval_env.difficulty = self.difficulty
                print(f"  >>> Curriculum: difficulty UP to {self.difficulty}")
                self._recent_solves.clear()
            elif recent_rate < 0.20 and self.difficulty > 1:
                self.difficulty -= 1
                self.eval_env.difficulty = self.difficulty
                print(f"  >>> Curriculum: difficulty DOWN to {self.difficulty}")
                self._recent_solves.clear()

    def _save_checkpoint(self) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"model_{self.num_timesteps}")
        self.model.save(path)
        if self.verbose:
            print(f"  [checkpoint] saved to {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_plots(
    callback: _EvalCallback,
    output_dir: str,
    agent_name: str,
) -> None:
    """Save training curve plots to *output_dir*."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    ts = callback.eval_timesteps

    if not ts:
        print("  No eval data to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{agent_name} Training Curves", fontsize=14)

    # Reward
    axes[0].plot(ts, callback.eval_rewards, color="#6366f1", linewidth=1.5)
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Avg Eval Reward")
    axes[0].set_title("Reward")
    axes[0].grid(alpha=0.3)

    # Solve rate
    axes[1].plot(ts, callback.eval_solve_rates, color="#34d399", linewidth=1.5)
    axes[1].set_xlabel("Timesteps")
    axes[1].set_ylabel("Solve Rate")
    axes[1].set_title("Solve Rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)

    # Steps per episode
    axes[2].plot(ts, callback.eval_avg_steps, color="#fbbf24", linewidth=1.5)
    axes[2].set_xlabel("Timesteps")
    axes[2].set_ylabel("Avg Steps")
    axes[2].set_title("Steps / Episode")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{agent_name.lower()}_training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Training curves saved to {path}")


# ---------------------------------------------------------------------------
# WandB helpers
# ---------------------------------------------------------------------------

def _maybe_init_wandb(args: argparse.Namespace) -> bool:
    if not args.wandb:
        return False
    try:
        import wandb
        wandb.init(
            project="world-model-rl",
            config=vars(args),
            name=f"{args.agent}_d{args.difficulty}_{args.seed}",
        )
        return True
    except ImportError:
        print("  wandb not installed, skipping.")
        return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train puzzle agents")
    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo", "thinker"])
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total environment timesteps to train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-freq", type=int, default=20480)
    parser.add_argument("--eval-freq", type=int, default=4096)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="PPO rollout buffer size (default: from config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate override")
    args = parser.parse_args(argv)

    print("=" * 60)
    print(f"  Agent:      {args.agent}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Timesteps:  {args.timesteps:,}")
    print(f"  Seed:       {args.seed}")
    print(f"  Curriculum: {args.curriculum}")
    print(f"  Device:     {args.device}")
    print("=" * 60)

    use_wandb = _maybe_init_wandb(args)

    # Use tier-appropriate max_steps for faster episode turnover
    from env.level_generator import _TIERS
    tier_max_steps = _TIERS.get(args.difficulty, _TIERS[1]).max_steps

    # Create environments
    render_mode = "human" if args.render else None
    train_env = gymnasium.make(
        "ThinkPuzzle-v0",
        difficulty=args.difficulty,
        max_steps=tier_max_steps,
        render_mode=render_mode,
    )
    eval_env = gymnasium.make(
        "ThinkPuzzle-v0",
        difficulty=args.difficulty,
        max_steps=tier_max_steps,
    )

    # Create agent
    if args.agent == "ppo":
        from agents.model_free.ppo_agent import PPOAgent
        ppo_overrides = {}
        if args.n_steps is not None:
            ppo_overrides["n_steps"] = args.n_steps
        if args.lr is not None:
            ppo_overrides["learning_rate"] = args.lr
        agent = PPOAgent(
            env=train_env,
            seed=args.seed,
            device=args.device,
            **ppo_overrides,
        )
    else:
        raise NotImplementedError(f"Agent '{args.agent}' not yet implemented")

    print(f"\n  Training {agent.name} for {args.timesteps:,} timesteps ...\n")

    # Build callback
    callback = _EvalCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        curriculum=args.curriculum,
        difficulty=args.difficulty,
    )

    # Train via SB3 .learn() with our callback
    t0 = time.monotonic()
    agent.model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=True,
    )
    elapsed = time.monotonic() - t0

    print(f"\n  Training complete in {elapsed:.1f}s")

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "model_final")
    agent.save(final_path)
    print(f"  Final model saved to {final_path}")

    # Save plots
    results_dir = os.path.join("experiments", "results")
    _save_plots(callback, results_dir, agent.name)

    # Final eval summary
    if callback.eval_solve_rates:
        best_sr = max(callback.eval_solve_rates)
        final_sr = callback.eval_solve_rates[-1]
        print(f"\n  Best solve rate:  {best_sr:.0%}")
        print(f"  Final solve rate: {final_sr:.0%}")

    if use_wandb:
        import wandb
        wandb.finish()

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
