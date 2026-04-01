"""ThinkerAgent — learns a world model, then thinks before it acts.

Two operating modes:
1. **Perfect mode** — uses real game rules (PerfectWorldModel) + A* search
   to solve puzzles optimally. Used by WATCH_AI.py for visualization.
2. **Learning mode** — learns a neural world model from experience, trains
   a policy network via PPO + dreaming. Used for RL training.

Three learning mechanisms (learning mode):
1. **Real experience** — acts in environment, stores transitions
2. **World model training** — learns to predict transitions from experience
3. **Dreaming** — imagines episodes using the world model, trains policy on them
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from agents.base_agent import BaseAgent
from agents.model_based.world_model import PerfectWorldModel, LearnedWorldModel, ReplayBuffer, WorldModelTrainer
from agents.model_based.mental_sim import MentalSimulator, ThoughtStep, Strategy, SimulationResult
from agents.model_based.planner import create_planner, AStarPlanner, _is_goal, _heuristic
from env.puzzle_world import PuzzleWorld, ACTION_NAMES


# ═══════════════════════════════════════════════════════════════════════
# PerfectMentalSimulator — thinks using real game rules + search
# ═══════════════════════════════════════════════════════════════════════


class PerfectMentalSimulator:
    """Plans ahead using the REAL game engine (PerfectWorldModel) + A* search.

    Produces ThoughtStep objects compatible with WATCH_AI visualization.
    """

    def __init__(self, planner_type: str = "astar") -> None:
        self._step_counter = 0
        self._planner = create_planner(planner_type)
        self._world_model = PerfectWorldModel()
        self._cached_plan: Optional[List[int]] = None
        self._plan_step: int = 0

    def think(
        self,
        obs: Dict[str, Any],
        world: PuzzleWorld,
    ) -> Tuple[int, ThoughtStep]:
        """Plan using A* search and return (action, thought_step).

        Returns the next action from the cached plan, or replans if needed.
        """
        self._step_counter += 1

        # Perception
        perception = self._perceive(obs)

        # Try to use cached plan first
        if self._cached_plan and self._plan_step < len(self._cached_plan):
            action = self._cached_plan[self._plan_step]
            self._plan_step += 1

            # Build thought step from cached plan
            remaining = self._cached_plan[self._plan_step - 1:]
            thought = self._build_thought(
                perception, remaining, action,
                strategy_name="cached_plan",
                reasoning=f"Executing step {self._plan_step} of {len(self._cached_plan)} in cached plan",
            )
            return action, thought

        # Need to (re)plan
        plan = self._planner.plan(
            self._world_model, world,
            max_depth=200, max_states=500_000,
        )

        if plan and len(plan) > 0:
            self._cached_plan = plan
            self._plan_step = 1
            action = plan[0]

            # Also try alternative first moves to show "thinking"
            strategies, simulations = self._evaluate_alternatives(world, plan)

            thought = ThoughtStep(
                step_number=self._step_counter,
                perception=perception,
                affordances=[],
                strategies_considered=strategies,
                simulations=simulations,
                chosen_strategy=strategies[0] if strategies else None,
                confidence=0.95,
                reasoning=f"A* found optimal solution: {len(plan)} steps",
            )
            return action, thought

        # No solution found — try random
        action = random.randint(0, 3)
        self._cached_plan = None
        self._plan_step = 0

        thought = ThoughtStep(
            step_number=self._step_counter,
            perception=perception,
            affordances=[],
            strategies_considered=[],
            simulations=[],
            chosen_strategy=None,
            confidence=0.0,
            reasoning="No solution found by A* — random fallback",
        )
        return action, thought

    def _evaluate_alternatives(
        self, world: PuzzleWorld, best_plan: List[int]
    ) -> Tuple[List[Strategy], List[SimulationResult]]:
        """Simulate a few alternative first-moves to show comparison."""
        strategies: List[Strategy] = []
        simulations: List[SimulationResult] = []

        # Best plan as primary strategy
        action_names = " -> ".join(
            ACTION_NAMES.get(a, "?") for a in best_plan[:6]
        )
        if len(best_plan) > 6:
            action_names += "..."

        best_strat = Strategy(
            name="optimal_path",
            description=f"{action_names} ({len(best_plan)} steps)",
            action_sequence=best_plan,
            expected_reward=100.0 - len(best_plan) * 0.1,
            risk_level=0.05,
            prerequisites=[],
            estimated_steps=len(best_plan),
        )
        strategies.append(best_strat)
        simulations.append(SimulationResult(
            strategy=best_strat,
            imagined_states=[],
            final_reward=100.0 - len(best_plan) * 0.1,
            success=True,
            failure_reason=None,
            steps_simulated=len(best_plan),
        ))

        # Try each alternative first action
        for alt_action in range(4):
            if alt_action == best_plan[0]:
                continue

            clone = world.clone()
            _obs, reward, terminated, truncated, info = clone.step(alt_action)

            if clone.is_deadlock():
                failure = "deadlock detected"
                alt_reward = -50.0
                success = False
            elif terminated:
                failure = None
                alt_reward = reward
                success = True
            else:
                # Quick heuristic score
                h = _heuristic(clone)
                alt_reward = reward - h
                failure = None if reward >= 0 else "negative reward"
                success = False

            action_name = ACTION_NAMES.get(alt_action, "?")
            alt_strat = Strategy(
                name=f"start_{action_name}",
                description=f"Start with {action_name}",
                action_sequence=[alt_action],
                expected_reward=alt_reward,
                risk_level=0.5 if failure else 0.3,
                prerequisites=[],
                estimated_steps=1,
            )
            strategies.append(alt_strat)
            simulations.append(SimulationResult(
                strategy=alt_strat,
                imagined_states=[],
                final_reward=alt_reward,
                success=success,
                failure_reason=failure,
                steps_simulated=1,
            ))

        return strategies, simulations

    def _perceive(self, obs: Dict[str, Any]) -> str:
        """Build a human-readable perception string."""
        grid = np.asarray(obs["grid"])
        counts: Dict[str, int] = {}
        _names = {
            2: "box", 3: "target", 4: "box_on_target",
        }
        for val in grid.flat:
            name = _names.get(int(val))
            if name:
                counts[name] = counts.get(name, 0) + 1

        if not counts:
            return "I see: empty room"
        parts = [f"{c} {n}{'s' if c > 1 else ''}" for n, c in counts.items()]

        placed = counts.get("box_on_target", 0)
        total_targets = placed + counts.get("target", 0)
        remaining = total_targets - placed
        if remaining > 0:
            parts.append(f"{remaining} box{'es' if remaining > 1 else ''} to place")

        return f"I see: {', '.join(parts)}"


# ═══════════════════════════════════════════════════════════════════════
# ThinkerAgent
# ═══════════════════════════════════════════════════════════════════════


class ThinkerAgent(BaseAgent):
    """The 'Think Before You Act' agent.

    Supports two modes:
    - perfect: Uses real game rules + A* for optimal solving (demo/viz)
    - learning: Uses learned world model + PPO for RL training
    """

    def __init__(
        self,
        grid_height: int = 12,
        grid_width: int = 12,
        use_perfect_model: bool = False,
        planner_type: str = "astar",
        mode: str = "learning",
        planning_depth: int = 8,
        beam_width: int = 3,
        planning_threshold: float = 0.7,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self.mode = mode
        self.use_perfect_model = use_perfect_model or (mode == "perfect")
        self._world: Optional[PuzzleWorld] = None

        if self.use_perfect_model:
            # Perfect mode: real world model + search planner
            self._mental_sim = PerfectMentalSimulator(planner_type=planner_type)
            self.world_model = PerfectWorldModel()
        else:
            # Learning mode: neural world model + beam search
            self.world_model = LearnedWorldModel(
                height=grid_height, width=grid_width
            ).to(self._device)
            self.replay_buffer = ReplayBuffer(max_size=50_000)
            self.wm_trainer = WorldModelTrainer(
                self.world_model, lr=1e-3, device=str(self._device)
            )
            self.wm_trainer.buffer = self.replay_buffer
            self._mental_sim_learned = MentalSimulator()
            self.planning_depth = planning_depth
            self.beam_width = beam_width
            self.planning_threshold = planning_threshold

        # Tracking
        self.world_model_accuracy = 1.0 if self.use_perfect_model else 0.0
        self.planning_active = False
        self.total_dreams = 0
        self.last_thought: Optional[ThoughtStep] = None

    @property
    def name(self) -> str:
        return "ThinkerAgent"

    def set_world(self, world: PuzzleWorld) -> None:
        """Set the current world reference (used in perfect mode)."""
        self._world = world
        if self.use_perfect_model:
            # Invalidate cached plan when world changes
            self._mental_sim._cached_plan = None
            self._mental_sim._plan_step = 0

    # -- Action selection ---------------------------------------------------

    def select_action(
        self, observation: Dict[str, Any]
    ) -> Tuple[int, Optional[ThoughtStep]]:
        """Pick an action. Uses planning if world model is accurate, else random.

        Returns (action, thought_info_or_None).
        """
        if self.use_perfect_model and self._world is not None:
            action, thought = self._mental_sim.think(observation, self._world)
            self.last_thought = thought
            self.planning_active = True
            return action, thought

        # Learning mode with learned world model
        if hasattr(self, '_mental_sim_learned') and hasattr(self, 'replay_buffer'):
            if (self.world_model_accuracy >= self.planning_threshold
                    and len(self.replay_buffer) >= 500):
                self.planning_active = True
                action, thought = self._mental_sim_learned.think_ahead(
                    observation, self.world_model,
                    depth=self.planning_depth, beam_width=self.beam_width,
                )
                self.last_thought = thought
                return action, thought

        # Fallback: random
        self.planning_active = False
        self.last_thought = None
        return random.randint(0, 3), None

    def act(self, observation: Dict[str, Any]) -> int:
        action, _ = self.select_action(observation)
        return action

    def get_action_probs(self, observation: Dict[str, Any]) -> np.ndarray:
        """Return uniform probs (no policy network in simplified version)."""
        return np.array([0.25, 0.25, 0.25, 0.25])

    # -- Experience collection (learning mode) ------------------------------

    def store_experience(
        self, obs: Dict[str, Any], action: int,
        next_obs: Dict[str, Any], reward: float, done: bool
    ) -> None:
        """Store a real transition in the replay buffer (learning mode only)."""
        if not hasattr(self, 'replay_buffer'):
            return

        grid = np.asarray(obs["grid"])
        pos = np.array(obs["agent_pos"], dtype=np.float32)
        next_grid = np.asarray(next_obs["grid"])
        next_pos = np.array(next_obs["agent_pos"], dtype=np.float32)

        self.replay_buffer.add(
            grid, pos, action, next_grid, next_pos, reward, done
        )

    # -- World model learning (learning mode) -------------------------------

    def learn_world_model(self, train_steps: int = 10) -> Dict[str, float]:
        """Train the world model on the replay buffer."""
        if not hasattr(self, 'replay_buffer') or len(self.replay_buffer) < 128:
            return {}

        metrics = {}
        for _ in range(train_steps):
            step_metrics = self.wm_trainer.train_step(batch_size=64)
            if step_metrics:
                metrics = step_metrics

        self._update_accuracy()
        return metrics

    def _update_accuracy(self) -> None:
        """Estimate world model accuracy on a sample from the replay buffer."""
        if not hasattr(self, 'replay_buffer') or len(self.replay_buffer) < 100:
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

    # -- BaseAgent interface ------------------------------------------------

    def learn(self, **kwargs: Any) -> Dict[str, Any]:
        if self.use_perfect_model:
            return {}  # Nothing to learn in perfect mode
        return self.learn_world_model()

    def save(self, path: str) -> None:
        state: Dict[str, Any] = {"mode": self.mode}
        if hasattr(self, 'world_model') and isinstance(self.world_model, LearnedWorldModel):
            state["world_model"] = self.world_model.state_dict()
        torch.save(state, path)

    def load(self, path: str) -> None:
        state = torch.load(path, weights_only=False)
        if "world_model" in state and isinstance(self.world_model, LearnedWorldModel):
            self.world_model.load_state_dict(state["world_model"])
