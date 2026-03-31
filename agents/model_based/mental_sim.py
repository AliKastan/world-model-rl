"""MentalSimulator — beam search planning with the LEARNED world model.

Uses the learned world model (neural network) to imagine future states,
NOT the real game engine. This means planning quality improves as the
world model gets better.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from env.puzzle_world import ACTION_NAMES


# ═══════════════════════════════════════════════════════════════════════
# Data classes (kept compatible with dashboard / viz code)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Strategy:
    name: str
    description: str
    action_sequence: List[int]
    expected_reward: float
    risk_level: float
    prerequisites: List[str]
    estimated_steps: int


@dataclass
class SimulationResult:
    strategy: Strategy
    imagined_states: List[dict]
    final_reward: float
    success: bool
    failure_reason: Optional[str]
    steps_simulated: int


@dataclass
class ThoughtStep:
    step_number: int
    perception: str
    affordances: List[str]
    strategies_considered: List[Strategy]
    simulations: List[SimulationResult]
    chosen_strategy: Optional[Strategy]
    confidence: float
    reasoning: str


# ═══════════════════════════════════════════════════════════════════════
# MentalSimulator — beam search through imagined futures
# ═══════════════════════════════════════════════════════════════════════


class MentalSimulator:
    """Plans ahead using a LEARNED world model (not the real game engine).

    think_ahead() performs beam search:
    1. From current state, try all 4 actions
    2. Use world model to predict next state for each
    3. Keep top beam_width paths by cumulative predicted reward
    4. Repeat for depth steps
    5. Return the first action of the highest-scoring path
    """

    def __init__(self) -> None:
        self._step_counter = 0

    def think_ahead(
        self,
        obs: Dict[str, Any],
        world_model: "torch.nn.Module",
        depth: int = 10,
        beam_width: int = 3,
    ) -> Tuple[int, ThoughtStep]:
        """Beam search through imagined futures using the learned world model.

        Returns (best_action, thought_step).
        """
        self._step_counter += 1
        device = next(world_model.parameters()).device

        grid_np = np.asarray(obs["grid"])
        pos_np = np.array(obs["agent_pos"], dtype=np.float32)

        grid_t = torch.tensor(grid_np, dtype=torch.long, device=device)
        pos_t = torch.tensor(pos_np, dtype=torch.float32, device=device)

        # Perception
        perception = self._perceive(obs)

        # Each beam: (grid_tensor, pos_tensor, action_list, cumulative_reward)
        beams: List[Tuple[torch.Tensor, torch.Tensor, List[int], float]] = [
            (grid_t, pos_t, [], 0.0)
        ]
        completed: List[Tuple[torch.Tensor, torch.Tensor, List[int], float]] = []
        all_paths: List[Tuple[List[int], float]] = []

        world_model.eval()
        with torch.no_grad():
            for d in range(depth):
                if not beams:
                    break

                n_beams = len(beams)
                # Batch all beams x 4 actions into one forward pass
                batch_grids = torch.stack([b[0] for b in beams for _ in range(4)])
                batch_pos = torch.stack([b[1] for b in beams for _ in range(4)])
                batch_actions = torch.tensor(
                    [a for _ in beams for a in range(4)], device=device
                )

                pred_grids, pred_pos, pred_rewards, pred_dones = world_model.predict(
                    batch_grids, batch_pos, batch_actions
                )

                candidates: List[Tuple[torch.Tensor, torch.Tensor, List[int], float]] = []
                idx = 0
                for beam_idx in range(n_beams):
                    b_actions = beams[beam_idx][2]
                    b_reward = beams[beam_idx][3]
                    for action in range(4):
                        new_reward = b_reward + pred_rewards[idx].item()
                        new_actions = b_actions + [action]

                        if pred_dones[idx].item() > 0.5:
                            completed.append((
                                pred_grids[idx], pred_pos[idx],
                                new_actions, new_reward
                            ))
                            all_paths.append((new_actions, new_reward))
                        else:
                            candidates.append((
                                pred_grids[idx], pred_pos[idx],
                                new_actions, new_reward
                            ))
                        idx += 1

                # Keep top beam_width candidates
                candidates.sort(key=lambda x: x[3], reverse=True)
                beams = candidates[:beam_width]

        # Merge all paths
        for b in beams:
            all_paths.append((b[2], b[3]))
        for c in completed:
            all_paths.append((c[2], c[3]))

        # Build strategies from top paths
        all_paths.sort(key=lambda x: x[1], reverse=True)
        strategies: List[Strategy] = []
        simulations: List[SimulationResult] = []

        for i, (actions, reward) in enumerate(all_paths[:5]):
            if not actions:
                continue
            action_names = " -> ".join(ACTION_NAMES.get(a, "?") for a in actions[:5])
            strat = Strategy(
                name=f"path_{i + 1}",
                description=f"{action_names}{'...' if len(actions) > 5 else ''}",
                action_sequence=actions,
                expected_reward=reward,
                risk_level=max(0.0, min(1.0, 1.0 - reward / 20.0)),
                prerequisites=[],
                estimated_steps=len(actions),
            )
            strategies.append(strat)
            sim = SimulationResult(
                strategy=strat,
                imagined_states=[],
                final_reward=reward,
                success=reward > 50.0,
                failure_reason=None if reward > 0 else "negative reward path",
                steps_simulated=len(actions),
            )
            simulations.append(sim)

        # Choose best
        chosen = strategies[0] if strategies else None
        if chosen:
            best_action = chosen.action_sequence[0]
            second_best = strategies[1].expected_reward if len(strategies) > 1 else 0.0
            gap = chosen.expected_reward - second_best
            confidence = min(1.0, 0.5 + gap / 20.0)
            reasoning = (f"Beam search depth={depth}: best path reward "
                         f"{chosen.expected_reward:.1f}, {len(all_paths)} paths explored")
        else:
            best_action = random.randint(0, 3)
            confidence = 0.0
            reasoning = "No paths found — random fallback"

        thought = ThoughtStep(
            step_number=self._step_counter,
            perception=perception,
            affordances=[],
            strategies_considered=strategies,
            simulations=simulations,
            chosen_strategy=chosen,
            confidence=confidence,
            reasoning=reasoning,
        )

        return best_action, thought

    def _perceive(self, obs: Dict[str, Any]) -> str:
        """Build a human-readable perception string."""
        grid = np.asarray(obs["grid"])
        counts: Dict[str, int] = {}
        _names = {
            2: "box", 3: "target", 4: "box_on_target", 5: "key",
            6: "locked_door", 7: "open_door", 8: "ice",
            9: "switch", 10: "active_switch", 11: "switch_wall",
        }
        for val in grid.flat:
            name = _names.get(int(val))
            if name:
                counts[name] = counts.get(name, 0) + 1

        if not counts:
            return "I see: empty room"
        parts = [f"{c} {n}{'s' if c > 1 else ''}" for n, c in counts.items()]
        return f"I see: {', '.join(parts)}"

    @staticmethod
    def get_thought_summary(thought: ThoughtStep) -> str:
        """Human-readable summary of one thinking step."""
        lines = [f"Thinking... {thought.perception}"]

        for i, sim in enumerate(thought.simulations[:5]):
            s = sim.strategy
            mark = "[OK]" if sim.success else "[--]" if sim.final_reward < 0 else "[??]"
            lines.append(
                f"  {i + 1}. {s.description} "
                f"(reward: {sim.final_reward:.0f}) {mark}"
            )

        if thought.chosen_strategy:
            lines.append(
                f"Decision: {thought.chosen_strategy.name}, "
                f"confidence {thought.confidence:.2f}"
            )
        lines.append(f"Reasoning: {thought.reasoning}")
        return "\n".join(lines)
