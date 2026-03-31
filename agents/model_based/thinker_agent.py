"""ThinkerAgent — the "Think Before You Act" agent.

Combines world model, affordance analysis, mental simulation, and planning
into a single agent that reasons about actions before taking them.

Three operating modes:
  1. "perfect": PerfectWorldModel + AStarPlanner (upper bound)
  2. "learned": LearnedWorldModel + BeamSearchPlanner (research mode)
  3. "no_thinking": skip mental sim, use simple heuristic (ablation baseline)
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from agents.base_agent import BaseAgent
from agents.model_based.affordance import AffordanceNet, ContextualAffordance
from agents.model_based.mental_sim import MentalSimulator, ThoughtStep
from agents.model_based.planner import (
    AStarPlanner,
    BeamSearchPlanner,
    BFSPlanner,
    create_planner,
    _extract_state,
    _is_goal,
)
from agents.model_based.world_model import (
    LearnedWorldModel,
    PerfectWorldModel,
    WorldModelTrainer,
    NUM_TYPES,
)
from env.puzzle_world import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    ACTION_NAMES,
    PuzzleWorld,
)


ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]


class ThinkerAgent(BaseAgent):
    """The "Think Before You Act" agent.

    Uses a world model to plan ahead, mental simulation to strategise,
    and affordance analysis to understand scene context.
    """

    def __init__(
        self,
        use_perfect_model: bool = True,
        planner_type: str = "astar",
        mode: str = "perfect",
        grid_height: int = 12,
        grid_width: int = 12,
    ) -> None:
        self._mode = mode  # "perfect", "learned", "no_thinking"
        self._use_perfect_model = use_perfect_model if mode != "learned" else False

        # --- World model ---
        self._grid_height = grid_height
        self._grid_width = grid_width
        self._world_model = PerfectWorldModel()
        self._learned_model: Optional[LearnedWorldModel] = None
        self._trainer: Optional[WorldModelTrainer] = None
        self._learned_model_initialized = False

        if not self._use_perfect_model:
            # Defer learned model creation until first observation
            # (grid dimensions depend on the level)
            pass

        # --- Affordance ---
        self._affordance_net = AffordanceNet()
        self._affordance = ContextualAffordance(net=self._affordance_net)

        # --- Mental simulator ---
        self._mental_sim = MentalSimulator(
            world_model=self._world_model,
            affordance_net=self._affordance,
        )

        # --- Planner ---
        self._planner_type = planner_type
        self._planner = create_planner(planner_type)

        # --- Plan state ---
        self._current_plan: Optional[List[int]] = None
        self._plan_step: int = 0
        self._expected_state: Optional[Any] = None  # PlannerState after expected action
        self._current_world: Optional[PuzzleWorld] = None

        # --- Thought log ---
        self.thought_log: List[ThoughtStep] = []

        # --- Metrics ---
        self._total_steps = 0
        self._total_thinks = 0
        self._total_replans = 0
        self._total_plan_completions = 0
        self._plan_lengths: List[int] = []
        self._strategies_per_think: List[int] = []
        self._thinking_times_ms: List[float] = []
        self._learn_step_counter = 0

    @property
    def name(self) -> str:
        if self._mode == "no_thinking":
            return "ThinkerAgent-NoThinking"
        model_tag = "Perfect" if self._use_perfect_model else "Learned"
        return f"ThinkerAgent-{model_tag}"

    # ------------------------------------------------------------------
    # Act
    # ------------------------------------------------------------------

    def act(self, observation: Dict[str, Any]) -> int:
        self._total_steps += 1

        if self._mode == "no_thinking":
            return self._act_no_thinking(observation)

        return self._act_thinking(observation)

    def _act_thinking(self, observation: Dict[str, Any]) -> int:
        """Smart re-planning: only re-think when needed."""

        # Check if current plan is still valid
        if self._current_plan is not None and self._plan_step < len(self._current_plan):
            # Verify world state matches expectation
            if self._expected_state is not None and self._current_world is not None:
                actual_state = _extract_state(self._current_world)
                if actual_state == self._expected_state:
                    # Plan still valid — execute next step
                    action = self._current_plan[self._plan_step]
                    self._plan_step += 1
                    self._advance_expected_state(action)

                    # Check if plan completed
                    if self._plan_step >= len(self._current_plan):
                        self._total_plan_completions += 1
                        self._current_plan = None

                    return action
                # State mismatch — need to replan
                self._total_replans += 1
            else:
                # First step of plan — just execute
                action = self._current_plan[self._plan_step]
                self._plan_step += 1
                self._advance_expected_state(action)
                return action

        # --- Need to think ---
        t0 = time.perf_counter()

        # Try planner first (fast, optimal for perfect model)
        plan = None
        if self._current_world is not None:
            if isinstance(self._planner, BeamSearchPlanner):
                plan = self._planner.plan(
                    self._world_model, self._current_world,
                    beam_width=5, max_depth=30,
                )
            else:
                plan = self._planner.plan(
                    self._world_model, self._current_world, max_depth=50,
                )

        if plan is not None and len(plan) > 0:
            # Planner found a solution
            self._current_plan = plan
            self._plan_step = 0
            self._plan_lengths.append(len(plan))

            # Generate a thought step for logging
            if self._current_world is not None:
                action, thought = self._mental_sim.think(observation, self._current_world)
                self.thought_log.append(thought)
                self._strategies_per_think.append(len(thought.strategies_considered))
                self._total_thinks += 1

            dt = (time.perf_counter() - t0) * 1000
            self._thinking_times_ms.append(dt)

            action = self._current_plan[self._plan_step]
            self._plan_step += 1
            self._advance_expected_state(action)
            return action

        # Planner failed — fall back to mental sim
        if self._current_world is not None:
            action, thought = self._mental_sim.think(observation, self._current_world)
            self.thought_log.append(thought)
            self._total_thinks += 1
            self._strategies_per_think.append(len(thought.strategies_considered))

            if thought.chosen_strategy and thought.chosen_strategy.action_sequence:
                self._current_plan = thought.chosen_strategy.action_sequence
                self._plan_step = 1  # already returning action 0
                self._plan_lengths.append(len(self._current_plan))
                self._advance_expected_state(action)

            dt = (time.perf_counter() - t0) * 1000
            self._thinking_times_ms.append(dt)
            return action

        # Total fallback — random valid move
        return random.choice(ACTIONS)

    def _act_no_thinking(self, observation: Dict[str, Any]) -> int:
        """Ablation: no mental simulation, just a simple heuristic."""
        if self._current_world is None:
            return random.choice(ACTIONS)

        # Simple heuristic: try each action, pick the one with best reward
        best_action = random.choice(ACTIONS)
        best_reward = float("-inf")

        for action in ACTIONS:
            clone = self._current_world.clone()
            _obs, reward, _term, _trunc, _info = clone.step(action)
            # Penalize deadlocks heavily
            if clone.is_deadlock():
                reward -= 100.0
            if reward > best_reward:
                best_reward = reward
                best_action = action

        return best_action

    def _advance_expected_state(self, action: int) -> None:
        """Update expected state by simulating the action in the world model."""
        if self._current_world is not None:
            next_world, _, _ = self._world_model.predict(self._current_world, action)
            self._expected_state = _extract_state(next_world)

    def set_world(self, world: PuzzleWorld) -> None:
        """Provide the current world state (called by the game loop)."""
        self._current_world = world

    def _ensure_learned_model(self, grid: np.ndarray) -> None:
        """Lazily initialize the learned world model from actual grid dimensions."""
        if self._learned_model_initialized:
            return
        h, w = grid.shape
        self._learned_model = LearnedWorldModel(
            height=h, width=w, num_types=NUM_TYPES
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._trainer = WorldModelTrainer(self._learned_model, lr=1e-3, device=device)
        self._learned_model_initialized = True

    # ------------------------------------------------------------------
    # Learn
    # ------------------------------------------------------------------

    def learn(self, **kwargs: Any) -> Dict[str, Any]:
        """Run one round of learning.

        Expected kwargs:
            state: observation dict
            action: int
            next_state: observation dict
            reward: float
            done: bool
            world: PuzzleWorld (current world for state extraction)
        """
        metrics: Dict[str, Any] = {}

        if self._mode == "perfect" or self._mode == "no_thinking":
            # No learning needed for perfect model
            return metrics

        # Learned mode: accumulate experience
        state = kwargs.get("state")
        action = kwargs.get("action")
        next_state = kwargs.get("next_state")
        reward = kwargs.get("reward")
        done = kwargs.get("done")

        if all(v is not None for v in [state, action, next_state, reward, done]):
            self._ensure_learned_model(state["grid"])
            self._trainer.add_experience(
                grid=state["grid"],
                agent_pos=np.array(state["agent_pos"], dtype=np.float32),
                action=action,
                next_grid=next_state["grid"],
                next_agent_pos=np.array(next_state["agent_pos"], dtype=np.float32),
                reward=reward,
                done=done,
            )
            self._learn_step_counter += 1

            # Train every 100 steps
            if self._learn_step_counter % 100 == 0 and len(self._trainer.buffer) >= 128:
                for _ in range(10):
                    loss = self._trainer.train_step(batch_size=128)
                if loss:
                    metrics.update(loss)

        return metrics

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_thought_log(self) -> List[ThoughtStep]:
        return self.thought_log

    def get_metrics(self) -> Dict[str, Any]:
        total = max(self._total_steps, 1)
        return {
            "total_thinks": self._total_thinks,
            "avg_strategies_per_think": (
                sum(self._strategies_per_think) / max(len(self._strategies_per_think), 1)
            ),
            "avg_plan_length": (
                sum(self._plan_lengths) / max(len(self._plan_lengths), 1)
            ),
            "replan_rate": self._total_replans / total,
            "plan_completion_rate": (
                self._total_plan_completions / max(self._total_thinks, 1)
            ),
            "thinking_time_ms": (
                sum(self._thinking_times_ms) / max(len(self._thinking_times_ms), 1)
            ),
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        state = {
            "mode": self._mode,
            "planner_type": self._planner_type,
            "affordance_net": self._affordance_net.state_dict(),
        }
        if self._learned_model is not None:
            state["learned_model"] = self._learned_model.state_dict()
        torch.save(state, path)

    def load(self, path: str) -> None:
        state = torch.load(path, weights_only=False)
        self._affordance_net.load_state_dict(state["affordance_net"])
        if "learned_model" in state and self._learned_model is not None:
            self._learned_model.load_state_dict(state["learned_model"])


# ═══════════════════════════════════════════════════════════════════════
# Random baseline agent
# ═══════════════════════════════════════════════════════════════════════


class RandomAgent(BaseAgent):
    """Uniform random action selection — the simplest possible baseline."""

    @property
    def name(self) -> str:
        return "RandomAgent"

    def act(self, observation: Dict[str, Any]) -> int:
        return random.choice(ACTIONS)

    def learn(self, **kwargs: Any) -> Dict[str, Any]:
        return {}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════
# Episode runner
# ═══════════════════════════════════════════════════════════════════════


def run_episode(
    agent: BaseAgent,
    world: PuzzleWorld,
    max_steps: int = 200,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one episode and return metrics."""
    world = world.clone()
    total_reward = 0.0
    steps = 0

    # Give ThinkerAgent access to world
    if isinstance(agent, ThinkerAgent):
        agent.set_world(world)

    for step in range(max_steps):
        obs = world.get_observation()
        action = agent.act(obs)

        _obs_next, reward, terminated, truncated, _info = world.step(action)
        total_reward += reward
        steps += 1

        # Update world reference for ThinkerAgent
        if isinstance(agent, ThinkerAgent):
            agent.set_world(world)
            next_obs = world.get_observation()
            agent.learn(
                state=obs, action=action, next_state=next_obs,
                reward=reward, done=terminated or truncated, world=world,
            )

        if verbose and step < 5:
            print(f"  Step {step + 1}: action={ACTION_NAMES[action]}, reward={reward:.1f}")

        if terminated or truncated:
            break

    solved = _is_goal(world) or world.solved
    return {
        "solved": solved,
        "steps": steps,
        "total_reward": total_reward,
    }


# ═══════════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    from env.level_generator import LevelGenerator

    gen = LevelGenerator()

    print("=" * 70)
    print("ThinkerAgent — Comprehensive Test")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Test 1: Difficulty 1-5 with ThinkerAgent (perfect model)
    # ------------------------------------------------------------------
    print("\n--- Test 1: ThinkerAgent (perfect) across difficulties ---")
    print(f"{'Diff':>4} {'Solved':>8} {'Avg Steps':>10} {'Avg Reward':>11}")
    print("-" * 37)

    for difficulty in range(1, 6):
        solved_count = 0
        total_steps = 0
        total_reward = 0.0
        n_levels = 10

        for seed in range(n_levels):
            try:
                world = gen.generate(difficulty=difficulty, seed=seed + 100)
            except RuntimeError:
                continue

            agent = ThinkerAgent(use_perfect_model=True, planner_type="astar", mode="perfect")
            result = run_episode(agent, world, max_steps=world.max_steps)
            if result["solved"]:
                solved_count += 1
            total_steps += result["steps"]
            total_reward += result["total_reward"]

        avg_steps = total_steps / n_levels
        avg_reward = total_reward / n_levels
        print(f"  {difficulty:>2}   {solved_count:>2}/{n_levels:<2}   {avg_steps:>8.1f}   {avg_reward:>9.1f}")

    # ------------------------------------------------------------------
    # Test 2: Compare ThinkerAgent vs RandomAgent on Level 4
    # ------------------------------------------------------------------
    print("\n--- Test 2: ThinkerAgent vs RandomAgent on difficulty 4 ---")

    world4 = gen.generate(difficulty=4, seed=42)

    # ThinkerAgent
    thinker = ThinkerAgent(use_perfect_model=True, planner_type="astar", mode="perfect")
    thinker_result = run_episode(thinker, world4, max_steps=world4.max_steps)
    thinker_metrics = thinker.get_metrics()

    # RandomAgent — average over 10 runs
    random_solved = 0
    random_steps_total = 0
    n_random = 10
    for _ in range(n_random):
        rand_agent = RandomAgent()
        r = run_episode(rand_agent, world4, max_steps=world4.max_steps)
        if r["solved"]:
            random_solved += 1
        random_steps_total += r["steps"]

    print(f"  ThinkerAgent: {'SOLVED' if thinker_result['solved'] else 'FAILED'} "
          f"in {thinker_result['steps']} steps, reward={thinker_result['total_reward']:.1f}")
    print(f"  RandomAgent:  {random_solved}/{n_random} solved, "
          f"avg {random_steps_total / n_random:.0f} steps")
    print(f"  Thinker metrics: thinks={thinker_metrics['total_thinks']}, "
          f"avg_plan_len={thinker_metrics['avg_plan_length']:.1f}, "
          f"think_time={thinker_metrics['thinking_time_ms']:.1f}ms")

    # ------------------------------------------------------------------
    # Test 3: Thought log for one Level 5 solve
    # ------------------------------------------------------------------
    print("\n--- Test 3: Thought log for Level 5 solve ---")

    world5 = gen.generate(difficulty=5, seed=42)
    thinker5 = ThinkerAgent(use_perfect_model=True, planner_type="astar", mode="perfect")
    result5 = run_episode(thinker5, world5, max_steps=world5.max_steps)

    print(f"  Result: {'SOLVED' if result5['solved'] else 'FAILED'} "
          f"in {result5['steps']} steps, reward={result5['total_reward']:.1f}")

    log = thinker5.get_thought_log()
    print(f"  Thought steps: {len(log)}")
    for i, thought in enumerate(log[:3]):  # Show first 3 thoughts
        print(f"\n  --- Thought {i + 1} ---")
        summary = MentalSimulator.get_thought_summary(thought)
        for line in summary.split("\n"):
            print(f"  {line}")

    if len(log) > 3:
        print(f"\n  ... ({len(log) - 3} more thought steps)")

    # ------------------------------------------------------------------
    # Test 4: Ablation — no_thinking mode
    # ------------------------------------------------------------------
    print("\n--- Test 4: Ablation — no_thinking mode ---")

    no_think = ThinkerAgent(mode="no_thinking")
    nt_result = run_episode(no_think, world4, max_steps=world4.max_steps)
    print(f"  no_thinking: {'SOLVED' if nt_result['solved'] else 'FAILED'} "
          f"in {nt_result['steps']} steps, reward={nt_result['total_reward']:.1f}")
    print(f"  vs ThinkerAgent-Perfect: {'SOLVED' if thinker_result['solved'] else 'FAILED'} "
          f"in {thinker_result['steps']} steps")

    print("\n" + "=" * 70)
    print("All tests complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
