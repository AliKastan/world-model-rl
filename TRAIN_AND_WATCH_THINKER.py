"""
Thinker RL: PPO + Learned World Model + Beam Search for Sokoban.

The Thinker agent learns TWO neural networks:
1. Policy+Value network (same as PPO) — picks actions
2. World Model — predicts next state and reward given (state, action)

During action selection, the Thinker uses beam search over IMAGINED futures
(predicted by the world model) to pick the best action sequence.

This should learn FASTER than vanilla PPO because the world model lets the
agent "think ahead" without actually taking steps in the real environment.
"""

import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
import os
import sys
from collections import deque
import time
import math

# Import the shared environment and renderer from TRAIN_AND_WATCH
from TRAIN_AND_WATCH import SokobanEnv, SokobanNet, GameRenderer, PPOAgent


# ============================================================
# WORLD MODEL — Learns to predict state transitions
# ============================================================

class WorldModel(nn.Module):
    """
    Learns to predict: given (state, action) -> (next_state, reward, done)

    Architecture:
    - Encode state with CNN (shared backbone)
    - Concatenate with action embedding
    - Decode to predicted next state + reward + done probability

    This is trained on REAL transitions collected during PPO training.
    Over time, its predictions become accurate enough to plan with.
    """

    def __init__(self, grid_size):
        super().__init__()
        self.grid_size = grid_size

        # State encoder (CNN)
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        flat_size = 64 * grid_size * grid_size

        # Action embedding — 4 actions embedded to 32 dims
        self.action_embed = nn.Embedding(4, 32)

        # Transition predictor
        self.transition_fc = nn.Sequential(
            nn.Linear(flat_size + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # State decoder — predict next state channels
        self.state_decoder = nn.Sequential(
            nn.Linear(512, flat_size),
            nn.ReLU(),
        )
        # Reshape back to (5, grid_size, grid_size) with per-channel conv
        self.decode_conv = nn.Conv2d(64, 5, kernel_size=1)

        # Reward predictor
        self.reward_head = nn.Linear(512, 1)

        # Done predictor
        self.done_head = nn.Linear(512, 1)

        # Initialize with small weights for stability
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        """
        state: (batch, 5, H, W)
        action: (batch,) long tensor

        Returns: predicted_next_state (batch, 5, H, W), reward (batch,), done_prob (batch,)
        """
        batch_size = state.size(0)

        # Encode state
        encoded = self.encoder(state)
        flat = encoded.reshape(batch_size, -1)

        # Embed action
        act_emb = self.action_embed(action)

        # Combine
        combined = torch.cat([flat, act_emb], dim=-1)
        hidden = self.transition_fc(combined)

        # Predict next state
        state_flat = self.state_decoder(hidden)
        state_spatial = state_flat.reshape(batch_size, 64, self.grid_size, self.grid_size)
        pred_next_state = torch.sigmoid(self.decode_conv(state_spatial))

        # Predict reward and done
        pred_reward = self.reward_head(hidden).squeeze(-1)
        pred_done = torch.sigmoid(self.done_head(hidden)).squeeze(-1)

        return pred_next_state, pred_reward, pred_done


# ============================================================
# THINKER AGENT — PPO + World Model + Beam Search
# ============================================================

class ThinkerAgent:
    """
    Thinker = PPO + World Model + Planning via beam search.

    The key insight: once the world model is accurate enough,
    the agent can "imagine" future states and pick the action
    sequence that leads to the best imagined outcome.

    Training loop:
    1. Collect real transitions (same as PPO)
    2. Train policy+value with PPO
    3. Train world model on collected transitions
    4. During action selection: use beam search over imagined states
       to pick the best first action

    The world model starts terrible (random predictions).
    After ~500 episodes, it becomes accurate enough to help.
    After ~2000 episodes, planning significantly improves performance.
    """

    def __init__(self, grid_size):
        self.grid_size = grid_size

        # PPO components (same as vanilla PPO)
        self.net = SokobanNet(grid_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=2.5e-4, eps=1e-5)

        # World model
        self.world_model = WorldModel(grid_size)
        self.wm_optimizer = optim.Adam(self.world_model.parameters(), lr=3e-4)

        # PPO hyperparams
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_eps = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.update_epochs = 4
        self.mini_batch_size = 128

        # Experience storage for PPO
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

        # Transition buffer for world model training
        self.wm_buffer_size = 50000
        self.wm_states = deque(maxlen=self.wm_buffer_size)
        self.wm_actions = deque(maxlen=self.wm_buffer_size)
        self.wm_next_states = deque(maxlen=self.wm_buffer_size)
        self.wm_rewards_buf = deque(maxlen=self.wm_buffer_size)
        self.wm_dones_buf = deque(maxlen=self.wm_buffer_size)

        # Planning config
        self.beam_width = 8
        self.beam_depth = 5
        self.use_planning = False  # enabled after world model is trained enough
        self.wm_train_steps = 0
        self.wm_loss_avg = 1.0  # track world model quality

    def select_action(self, state_np):
        """
        Select action using policy + optional beam search planning.

        If world model is accurate enough (use_planning=True):
          - Use beam search to evaluate all 4 first actions
          - Pick the one with best imagined total reward
          - But still sample from a softmax over beam scores (for exploration)

        If world model is not ready yet:
          - Just use PPO policy sampling (same as vanilla)
        """
        state_tensor = torch.FloatTensor(state_np).unsqueeze(0)

        with torch.no_grad():
            action_probs, value = self.net(state_tensor)

        if self.use_planning and self.wm_loss_avg < 0.5:
            # Beam search planning
            beam_scores = self._beam_search(state_np)

            # Combine policy probs with beam search scores
            # Temperature-scaled softmax of beam scores
            beam_tensor = torch.FloatTensor(beam_scores)
            beam_probs = F.softmax(beam_tensor / 2.0, dim=0)

            # Blend: 60% beam, 40% policy (keep some policy exploration)
            blended = 0.6 * beam_probs + 0.4 * action_probs.squeeze()
            blended = blended / blended.sum()  # renormalize

            dist = Categorical(blended)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action.item(), log_prob.item(), value.item(), blended.numpy()
        else:
            # Pure PPO sampling
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action.item(), log_prob.item(), value.item(), action_probs.squeeze().numpy()

    def _beam_search(self, state_np):
        """
        Beam search over imagined futures using the world model.

        For each of the 4 possible first actions:
        1. Predict next state using world model
        2. From predicted state, expand top-k actions (by policy)
        3. Continue for beam_depth steps
        4. Score = sum of predicted rewards + value estimate at leaf

        Returns: scores[4] — estimated value of taking each first action
        """
        scores = np.zeros(4, dtype=np.float32)

        with torch.no_grad():
            for first_action in range(4):
                # Start beam with this first action
                state_t = torch.FloatTensor(state_np).unsqueeze(0)
                action_t = torch.LongTensor([first_action])

                total_reward = 0.0
                current_states = state_t
                discount = 1.0

                for depth in range(self.beam_depth):
                    # Predict next state
                    pred_next, pred_reward, pred_done = self.world_model(
                        current_states, action_t
                    )

                    total_reward += discount * pred_reward.item()
                    discount *= self.gamma

                    if pred_done.item() > 0.5:
                        break

                    # Use policy to pick next action from predicted state
                    pred_probs, pred_value = self.net(pred_next)

                    if depth == self.beam_depth - 1:
                        # At leaf, add value estimate
                        total_reward += discount * pred_value.item()
                    else:
                        # Pick most likely action for next step
                        action_t = pred_probs.argmax(dim=-1)
                        current_states = pred_next

                scores[first_action] = total_reward

        return scores

    def store(self, state, action, reward, log_prob, value, done):
        """Store transition for PPO and world model."""
        self.states.append(state.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def store_transition(self, state, action, next_state, reward, done):
        """Store transition specifically for world model training."""
        self.wm_states.append(state.copy())
        self.wm_actions.append(action)
        self.wm_next_states.append(next_state.copy())
        self.wm_rewards_buf.append(reward)
        self.wm_dones_buf.append(float(done))

    def update(self):
        """PPO update + World Model update."""
        losses = self._ppo_update()
        wm_loss = self._world_model_update()
        if wm_loss is not None:
            losses["wm_loss"] = wm_loss
        return losses

    def _ppo_update(self):
        """Standard PPO update (same as PPOAgent)."""
        if len(self.states) < 2:
            self._clear_ppo()
            return {}

        advantages = []
        returns = []
        gae = 0

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])

        states_t = torch.FloatTensor(np.array(self.states))
        actions_t = torch.LongTensor(self.actions)
        old_log_probs_t = torch.FloatTensor(self.log_probs)
        advantages_t = torch.FloatTensor(advantages)
        returns_t = torch.FloatTensor(returns)

        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(self.update_epochs):
            indices = torch.randperm(len(self.states))

            for start in range(0, len(indices), self.mini_batch_size):
                end = start + self.mini_batch_size
                if end > len(indices):
                    break

                batch_idx = indices[start:end]

                b_states = states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_log_probs = old_log_probs_t[batch_idx]
                b_advantages = advantages_t[batch_idx]
                b_returns = returns_t[batch_idx]

                new_probs, new_values = self.net(b_states)
                dist = Categorical(new_probs)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_values.squeeze(-1), b_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        self._clear_ppo()

        if n_updates == 0:
            return {}

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def _world_model_update(self):
        """Train world model on collected transitions."""
        if len(self.wm_states) < 256:
            return None

        # Sample a batch
        batch_size = min(256, len(self.wm_states))
        indices = random.sample(range(len(self.wm_states)), batch_size)

        states_t = torch.FloatTensor(np.array([self.wm_states[i] for i in indices]))
        actions_t = torch.LongTensor([self.wm_actions[i] for i in indices])
        next_states_t = torch.FloatTensor(np.array([self.wm_next_states[i] for i in indices]))
        rewards_t = torch.FloatTensor([self.wm_rewards_buf[i] for i in indices])
        dones_t = torch.FloatTensor([self.wm_dones_buf[i] for i in indices])

        # Forward pass
        pred_next, pred_reward, pred_done = self.world_model(states_t, actions_t)

        # Losses
        state_loss = F.mse_loss(pred_next, next_states_t)
        reward_loss = F.mse_loss(pred_reward, rewards_t)
        done_loss = F.binary_cross_entropy(pred_done, dones_t)

        total_loss = state_loss + 0.1 * reward_loss + 0.1 * done_loss

        self.wm_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.wm_optimizer.step()

        self.wm_train_steps += 1
        self.wm_loss_avg = 0.99 * self.wm_loss_avg + 0.01 * total_loss.item()

        # Enable planning once world model is reasonably trained
        if self.wm_train_steps > 100 and self.wm_loss_avg < 0.5:
            self.use_planning = True

        return total_loss.item()

    def _clear_ppo(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "net": self.net.state_dict(),
            "opt": self.optimizer.state_dict(),
            "wm": self.world_model.state_dict(),
            "wm_opt": self.wm_optimizer.state_dict(),
        }, path)

    def load(self, path):
        if os.path.exists(path):
            data = torch.load(path, map_location="cpu")
            self.net.load_state_dict(data["net"])
            self.optimizer.load_state_dict(data["opt"])
            if "wm" in data:
                self.world_model.load_state_dict(data["wm"])
            if "wm_opt" in data:
                self.wm_optimizer.load_state_dict(data["wm_opt"])


# ============================================================
# MAIN TRAINING VISUALIZATION — THINKER VERSION
# ============================================================

class TrainAndWatchThinker:
    def __init__(self):
        pygame.init()
        self.W, self.H = 1000, 750
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Thinker RL - PPO + World Model + Beam Search")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("consolas", 28, bold=True)
        self.font_med = pygame.font.SysFont("consolas", 20)
        self.font_small = pygame.font.SysFont("consolas", 14)
        self.font_tiny = pygame.font.SysFont("consolas", 11)

        # RL components
        self.phase = 1
        self.env = SokobanEnv(phase=self.phase)
        self.agent = ThinkerAgent(grid_size=self.env.grid_size)
        self.renderer = GameRenderer(game_area_size=550)

        # Metrics
        self.episode_rewards = deque(maxlen=500)
        self.episode_solved = deque(maxlen=500)
        self.episode_deadlocks = deque(maxlen=500)
        self.episode_steps_list = deque(maxlen=500)
        self.total_episodes = 0
        self.total_steps = 0
        self.steps_since_update = 0
        self.steps_per_update = 512
        self.last_losses = {}
        self.action_probs = [0.25, 0.25, 0.25, 0.25]

        # Learning curve
        self.curve_data = []
        self.phase_changes = []

        # State
        self.mode = "visual"
        self.paused = False
        self.obs = self.env.reset()
        self.prev_obs = self.obs.copy()
        self.episode_reward = 0
        self.episode_steps = 0

        # Phase up
        self.phase_up_text = ""
        self.phase_up_timer = 0

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_f:
                        self.mode = "fast"
                    elif event.key == pygame.K_s:
                        self.mode = "slow"
                    elif event.key == pygame.K_v:
                        self.mode = "visual"

            if not self.paused:
                if self.mode == "fast":
                    for _ in range(200):
                        self._train_one_step()
                elif self.mode == "slow":
                    self._train_one_step()
                    pygame.time.wait(200)
                else:
                    for _ in range(3):
                        self._train_one_step()

            self._render()
            self.clock.tick(60 if self.mode != "slow" else 10)

        self.agent.save(f"checkpoints/thinker_phase{self.phase}.pt")
        pygame.quit()

    def _train_one_step(self):
        """One step of Thinker RL training."""
        action, log_prob, value, probs = self.agent.select_action(self.obs)
        self.action_probs = probs.tolist()

        next_obs, reward, done, info = self.env.step(action)

        # Store for PPO
        self.agent.store(self.obs, action, reward, log_prob, value, float(done))
        # Store for world model
        self.agent.store_transition(self.obs, action, next_obs, reward, done)

        self.total_steps += 1
        self.steps_since_update += 1
        self.episode_reward += reward
        self.episode_steps += 1

        if done:
            self.episode_rewards.append(self.episode_reward)
            self.episode_solved.append(info.get("solved", False))
            self.episode_deadlocks.append(info.get("deadlock", False))
            self.episode_steps_list.append(self.episode_steps)
            self.total_episodes += 1
            self.episode_reward = 0
            self.episode_steps = 0
            self.obs = self.env.reset()

            if self.total_episodes % 20 == 0 and len(self.episode_solved) >= 20:
                rate = sum(list(self.episode_solved)[-100:]) / min(len(self.episode_solved), 100) * 100
                self.curve_data.append((self.total_episodes, rate))

            if len(self.episode_solved) >= 100:
                recent_rate = sum(list(self.episode_solved)[-100:]) / 100
                if recent_rate > 0.70 and self.phase < 3:
                    self._advance_phase()
        else:
            self.obs = next_obs

        if self.steps_since_update >= self.steps_per_update:
            self.last_losses = self.agent.update()
            self.steps_since_update = 0

    def _advance_phase(self):
        self.phase += 1
        self.phase_up_text = f"PHASE {self.phase}: {['', 'Baby Steps', 'Two Boxes!', 'Three Boxes!'][self.phase]}"
        self.phase_up_timer = 120
        self.phase_changes.append(self.total_episodes)

        self.env = SokobanEnv(phase=self.phase)
        self.obs = self.env.reset()
        self.episode_solved.clear()
        self.episode_rewards.clear()

    def _render(self):
        self.screen.fill((18, 18, 30))

        # LEFT: Game grid
        render_data = self.env.get_render_data()
        self.renderer.render_grid(self.screen, render_data, offset_x=15, offset_y=15)

        # Phase label
        phase_names = {1: "Phase 1: Baby Steps (1 box)", 2: "Phase 2: Two Boxes", 3: "Phase 3: Three Boxes"}
        phase_text = self.font_small.render(phase_names.get(self.phase, ""), True, (150, 160, 180))
        self.screen.blit(phase_text, (15, 580))

        # Thinker status
        planning_status = "PLANNING ACTIVE" if self.agent.use_planning else "LEARNING WORLD MODEL..."
        planning_color = (52, 211, 153) if self.agent.use_planning else (251, 191, 36)
        status_text = self.font_small.render(planning_status, True, planning_color)
        self.screen.blit(status_text, (15, 600))

        # RIGHT: Stats panel
        panel_x = 600
        pygame.draw.rect(self.screen, (25, 25, 40), pygame.Rect(panel_x, 0, 400, self.H))

        x = panel_x + 20
        y = 20

        # Title
        self._text("THINKER RL", x, y, self.font_large, (147, 130, 241))
        y += 35
        self._text(f"Episode: {self.total_episodes:,}", x, y, self.font_med, (220, 225, 240))
        y += 28
        self._text(f"Total Steps: {self.total_steps:,}", x, y, self.font_small, (130, 140, 160))
        y += 25

        # Solve rate
        if len(self.episode_solved) > 0:
            solve_rate = sum(self.episode_solved) / len(self.episode_solved) * 100
        else:
            solve_rate = 0

        color = (239, 68, 68) if solve_rate < 15 else \
                (251, 146, 60) if solve_rate < 30 else \
                (251, 191, 36) if solve_rate < 60 else (52, 211, 153)

        self._text("Solve Rate:", x, y, self.font_small, (130, 140, 160))
        y += 20
        self._text(f"{solve_rate:.1f}%", x, y, self.font_large, color)
        y += 40

        # Other stats
        avg_reward = sum(self.episode_rewards) / max(len(self.episode_rewards), 1)
        avg_steps = sum(self.episode_steps_list) / max(len(self.episode_steps_list), 1)
        deadlock_rate = sum(self.episode_deadlocks) / max(len(self.episode_deadlocks), 1) * 100

        self._text(f"Avg Reward: {avg_reward:.1f}", x, y, self.font_med, (180, 190, 210))
        y += 25
        self._text(f"Avg Steps: {avg_steps:.0f}", x, y, self.font_med, (180, 190, 210))
        y += 25
        dl_color = (52, 211, 153) if deadlock_rate < 10 else (251, 191, 36) if deadlock_rate < 30 else (239, 68, 68)
        self._text(f"Deadlock Rate: {deadlock_rate:.0f}%", x, y, self.font_med, dl_color)
        y += 30

        # World model status
        self._text("World Model:", x, y, self.font_small, (130, 140, 160))
        y += 18
        wm_loss = self.agent.wm_loss_avg
        wm_color = (52, 211, 153) if wm_loss < 0.3 else (251, 191, 36) if wm_loss < 1.0 else (239, 68, 68)
        self._text(f"  Loss: {wm_loss:.4f}", x, y, self.font_small, wm_color)
        y += 18
        self._text(f"  Train Steps: {self.agent.wm_train_steps}", x, y, self.font_small, (130, 140, 160))
        y += 18
        self._text(f"  Buffer: {len(self.agent.wm_states):,}/{self.agent.wm_buffer_size:,}", x, y, self.font_small, (130, 140, 160))
        y += 25

        # Learning curve graph
        graph_rect = pygame.Rect(panel_x + 15, y, 370, 160)
        pygame.draw.rect(self.screen, (30, 30, 48), graph_rect, border_radius=8)
        pygame.draw.rect(self.screen, (50, 55, 70), graph_rect, 1, border_radius=8)
        self._text("Solve Rate (rolling 100)", panel_x + 25, y + 5, self.font_tiny, (100, 110, 130))

        gy = graph_rect.y + 25
        gh = graph_rect.height - 35
        gx = graph_rect.x + 10
        gw = graph_rect.width - 20

        for pct in [25, 50, 75]:
            line_y = gy + gh - int(gh * pct / 100)
            pygame.draw.line(self.screen, (45, 45, 60), (gx, line_y), (gx + gw, line_y), 1)
            self._text(f"{pct}%", gx - 5, line_y - 6, self.font_tiny, (70, 80, 100))

        if len(self.curve_data) >= 2:
            max_ep = max(d[0] for d in self.curve_data)
            points = []
            for ep, rate in self.curve_data:
                px = gx + int(gw * ep / max(max_ep, 1))
                py = gy + gh - int(gh * min(rate, 100) / 100)
                points.append((px, py))
            if len(points) >= 2:
                pygame.draw.lines(self.screen, (147, 130, 241), False, points, 2)

        y = graph_rect.bottom + 15

        # Action probabilities
        self._text("Action Probabilities:", x, y, self.font_small, (100, 110, 130))
        y += 22
        labels = ["UP:", "DOWN:", "LEFT:", "RIGHT:"]
        for i, (label, prob) in enumerate(zip(labels, self.action_probs)):
            self._text(label, x, y, self.font_small, (130, 140, 160))
            bar_x = x + 90
            bar_w = 150
            bar_h = 14
            pygame.draw.rect(self.screen, (40, 40, 55), pygame.Rect(bar_x, y+2, bar_w, bar_h), border_radius=3)
            fill_w = int(bar_w * prob)
            if fill_w > 0:
                bar_color = (147, 130, 241) if self.agent.use_planning else (99, 102, 241)
                pygame.draw.rect(self.screen, bar_color, pygame.Rect(bar_x, y+2, fill_w, bar_h), border_radius=3)
            self._text(f"{prob:.2f}", bar_x + bar_w + 8, y, self.font_small, (180, 190, 210))
            y += 22

        y += 5

        # Loss info
        if self.last_losses:
            self._text(f"Policy Loss: {self.last_losses.get('policy_loss', 0):.4f}", x, y, self.font_tiny, (100, 110, 130))
            y += 16
            self._text(f"Value Loss: {self.last_losses.get('value_loss', 0):.4f}", x, y, self.font_tiny, (100, 110, 130))
            y += 16
            self._text(f"Entropy: {self.last_losses.get('entropy', 0):.4f}", x, y, self.font_tiny, (100, 110, 130))
            y += 16
            if "wm_loss" in self.last_losses:
                self._text(f"WM Loss: {self.last_losses['wm_loss']:.4f}", x, y, self.font_tiny, (130, 110, 180))

        # Mode
        mode_text = {"visual": "VISUAL", "fast": "FAST", "slow": "SLOW"}
        mode_color = {"visual": (52, 211, 153), "fast": (251, 191, 36), "slow": (147, 197, 253)}
        self._text(f"Mode: {mode_text[self.mode]}", x, self.H - 60, self.font_med, mode_color[self.mode])

        self._text("Space=Pause  F=Fast  S=Slow  V=Visual  ESC=Quit",
                   panel_x + 20, self.H - 25, self.font_tiny, (70, 80, 100))

        # Phase up notification
        if self.phase_up_timer > 0:
            self.phase_up_timer -= 1
            overlay = pygame.Surface((580, 100), pygame.SRCALPHA)
            overlay.fill((18, 18, 30, 200))
            self.screen.blit(overlay, (10, 250))
            text = self.font_large.render(f"PHASE UP: {self.phase_up_text}", True, (147, 130, 241))
            self.screen.blit(text, (60, 280))

        if self.paused:
            overlay = pygame.Surface((580, 580), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            self.screen.blit(overlay, (10, 10))
            text = self.font_large.render("PAUSED", True, (255, 255, 255))
            self.screen.blit(text, (220, 280))

        pygame.display.flip()

    def _text(self, text, x, y, font, color):
        surf = font.render(str(text), True, color)
        self.screen.blit(surf, (x, y))


if __name__ == "__main__":
    print("=" * 50)
    print("  Thinker RL Training — Sokoban")
    print("  PPO + World Model + Beam Search Planning")
    print("  The AI learns a model of the world, then")
    print("  uses it to plan ahead!")
    print("=" * 50)
    print()
    print("Generating training levels (this may take a moment)...")

    app = TrainAndWatchThinker()

    print("Training levels generated! Starting visualization...")
    print()
    print("Controls:")
    print("  Space = Pause/Resume")
    print("  F = Fast mode (train quickly)")
    print("  S = Slow mode (watch every step)")
    print("  V = Visual mode (normal)")
    print("  ESC = Quit and save")

    app.run()
