"""
Side-by-side comparison: PPO vs Thinker RL on Sokoban.

Both agents train on the SAME levels simultaneously.
Watch how the Thinker (PPO + World Model + Beam Search) learns faster
than vanilla PPO because it can plan ahead using its learned world model.
"""

import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import sys
from collections import deque
import time
import math

from TRAIN_AND_WATCH import SokobanEnv, SokobanNet, PPOAgent, GameRenderer
from TRAIN_AND_WATCH_THINKER import ThinkerAgent


class CompareTraining:
    def __init__(self):
        pygame.init()
        self.W, self.H = 1300, 750
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("PPO vs Thinker RL - Side by Side Comparison")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("consolas", 24, bold=True)
        self.font_med = pygame.font.SysFont("consolas", 18)
        self.font_small = pygame.font.SysFont("consolas", 13)
        self.font_tiny = pygame.font.SysFont("consolas", 11)

        # Shared levels — both agents train on the SAME levels
        self.phase = 1
        self.shared_env_ppo = SokobanEnv(phase=self.phase)
        # Create thinker env with same levels
        self.shared_env_thinker = SokobanEnv(phase=self.phase)
        self.shared_env_thinker.levels = self.shared_env_ppo.levels  # share levels!

        # PPO agent
        self.ppo_agent = PPOAgent(grid_size=self.shared_env_ppo.grid_size)

        # Thinker agent
        self.thinker_agent = ThinkerAgent(grid_size=self.shared_env_thinker.grid_size)

        # Renderers
        self.renderer_left = GameRenderer(game_area_size=350)
        self.renderer_right = GameRenderer(game_area_size=350)

        # PPO metrics
        self.ppo_rewards = deque(maxlen=500)
        self.ppo_solved = deque(maxlen=500)
        self.ppo_episodes = 0
        self.ppo_steps = 0
        self.ppo_steps_since_update = 0
        self.ppo_obs = self.shared_env_ppo.reset()
        self.ppo_ep_reward = 0
        self.ppo_ep_steps = 0
        self.ppo_curve = []
        self.ppo_probs = [0.25, 0.25, 0.25, 0.25]
        self.ppo_last_losses = {}

        # Thinker metrics
        self.thinker_rewards = deque(maxlen=500)
        self.thinker_solved = deque(maxlen=500)
        self.thinker_episodes = 0
        self.thinker_steps = 0
        self.thinker_steps_since_update = 0
        self.thinker_obs = self.shared_env_thinker.reset()
        self.thinker_ep_reward = 0
        self.thinker_ep_steps = 0
        self.thinker_curve = []
        self.thinker_probs = [0.25, 0.25, 0.25, 0.25]
        self.thinker_last_losses = {}

        self.steps_per_update = 512

        # State
        self.mode = "visual"
        self.paused = False

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
                    for _ in range(150):
                        self._train_step_ppo()
                        self._train_step_thinker()
                elif self.mode == "slow":
                    self._train_step_ppo()
                    self._train_step_thinker()
                    pygame.time.wait(200)
                else:
                    for _ in range(3):
                        self._train_step_ppo()
                        self._train_step_thinker()

            self._render()
            self.clock.tick(60 if self.mode != "slow" else 10)

        self.ppo_agent.save("checkpoints/compare_ppo.pt")
        self.thinker_agent.save("checkpoints/compare_thinker.pt")
        pygame.quit()

    def _train_step_ppo(self):
        """One PPO training step."""
        action, log_prob, value, probs = self.ppo_agent.select_action(self.ppo_obs)
        self.ppo_probs = probs.tolist()
        next_obs, reward, done, info = self.shared_env_ppo.step(action)
        self.ppo_agent.store(self.ppo_obs, action, reward, log_prob, value, float(done))
        self.ppo_steps += 1
        self.ppo_steps_since_update += 1
        self.ppo_ep_reward += reward
        self.ppo_ep_steps += 1

        if done:
            self.ppo_rewards.append(self.ppo_ep_reward)
            self.ppo_solved.append(info.get("solved", False))
            self.ppo_episodes += 1
            self.ppo_ep_reward = 0
            self.ppo_ep_steps = 0
            self.ppo_obs = self.shared_env_ppo.reset()

            if self.ppo_episodes % 20 == 0 and len(self.ppo_solved) >= 20:
                rate = sum(list(self.ppo_solved)[-100:]) / min(len(self.ppo_solved), 100) * 100
                self.ppo_curve.append((self.ppo_episodes, rate))
        else:
            self.ppo_obs = next_obs

        if self.ppo_steps_since_update >= self.steps_per_update:
            self.ppo_last_losses = self.ppo_agent.update()
            self.ppo_steps_since_update = 0

    def _train_step_thinker(self):
        """One Thinker training step."""
        action, log_prob, value, probs = self.thinker_agent.select_action(self.thinker_obs)
        self.thinker_probs = probs.tolist()
        next_obs, reward, done, info = self.shared_env_thinker.step(action)
        self.thinker_agent.store(self.thinker_obs, action, reward, log_prob, value, float(done))
        self.thinker_agent.store_transition(self.thinker_obs, action, next_obs, reward, done)
        self.thinker_steps += 1
        self.thinker_steps_since_update += 1
        self.thinker_ep_reward += reward
        self.thinker_ep_steps += 1

        if done:
            self.thinker_rewards.append(self.thinker_ep_reward)
            self.thinker_solved.append(info.get("solved", False))
            self.thinker_episodes += 1
            self.thinker_ep_reward = 0
            self.thinker_ep_steps = 0
            self.thinker_obs = self.shared_env_thinker.reset()

            if self.thinker_episodes % 20 == 0 and len(self.thinker_solved) >= 20:
                rate = sum(list(self.thinker_solved)[-100:]) / min(len(self.thinker_solved), 100) * 100
                self.thinker_curve.append((self.thinker_episodes, rate))
        else:
            self.thinker_obs = next_obs

        if self.thinker_steps_since_update >= self.steps_per_update:
            self.thinker_last_losses = self.thinker_agent.update()
            self.thinker_steps_since_update = 0

    def _render(self):
        self.screen.fill((18, 18, 30))

        half_w = self.W // 2

        # ======== LEFT HALF: PPO ========
        # Title
        self._text("VANILLA PPO", 30, 8, self.font_large, (99, 102, 241))

        # Grid
        ppo_render = self.shared_env_ppo.get_render_data()
        self.renderer_left.render_grid(self.screen, ppo_render, offset_x=10, offset_y=35)

        # Stats below grid
        sy = 400
        ppo_rate = sum(self.ppo_solved) / max(len(self.ppo_solved), 1) * 100
        rate_color = self._rate_color(ppo_rate)
        self._text(f"Episode: {self.ppo_episodes:,}", 20, sy, self.font_med, (220, 225, 240))
        self._text(f"Solve Rate: {ppo_rate:.1f}%", 20, sy + 25, self.font_med, rate_color)
        avg_r = sum(self.ppo_rewards) / max(len(self.ppo_rewards), 1)
        self._text(f"Avg Reward: {avg_r:.1f}", 20, sy + 50, self.font_small, (160, 170, 190))

        # Action probs
        self._draw_action_bars(20, sy + 75, self.ppo_probs, (99, 102, 241))

        # ======== RIGHT HALF: THINKER ========
        self._text("THINKER RL", half_w + 30, 8, self.font_large, (147, 130, 241))

        thinker_render = self.shared_env_thinker.get_render_data()
        self.renderer_right.render_grid(self.screen, thinker_render, offset_x=half_w + 10, offset_y=35)

        thinker_rate = sum(self.thinker_solved) / max(len(self.thinker_solved), 1) * 100
        rate_color = self._rate_color(thinker_rate)
        self._text(f"Episode: {self.thinker_episodes:,}", half_w + 20, sy, self.font_med, (220, 225, 240))
        self._text(f"Solve Rate: {thinker_rate:.1f}%", half_w + 20, sy + 25, self.font_med, rate_color)
        avg_r = sum(self.thinker_rewards) / max(len(self.thinker_rewards), 1)
        self._text(f"Avg Reward: {avg_r:.1f}", half_w + 20, sy + 50, self.font_small, (160, 170, 190))

        # Thinker status
        planning = "Planning: ON" if self.thinker_agent.use_planning else "Planning: OFF (learning WM)"
        pc = (52, 211, 153) if self.thinker_agent.use_planning else (251, 191, 36)
        self._text(planning, half_w + 20, sy + 70, self.font_small, pc)

        self._draw_action_bars(half_w + 20, sy + 95, self.thinker_probs, (147, 130, 241))

        # ======== BOTTOM: Comparison Graph ========
        graph_y = 570
        graph_rect = pygame.Rect(30, graph_y, self.W - 60, 140)
        pygame.draw.rect(self.screen, (30, 30, 48), graph_rect, border_radius=8)
        pygame.draw.rect(self.screen, (50, 55, 70), graph_rect, 1, border_radius=8)

        self._text("Solve Rate Comparison", 50, graph_y + 5, self.font_small, (100, 110, 130))

        gy = graph_rect.y + 25
        gh = graph_rect.height - 35
        gx = graph_rect.x + 40
        gw = graph_rect.width - 60

        for pct in [25, 50, 75]:
            line_y = gy + gh - int(gh * pct / 100)
            pygame.draw.line(self.screen, (45, 45, 60), (gx, line_y), (gx + gw, line_y), 1)
            self._text(f"{pct}%", gx - 35, line_y - 6, self.font_tiny, (70, 80, 100))

        # Plot both curves
        max_ep = max(
            max((d[0] for d in self.ppo_curve), default=1),
            max((d[0] for d in self.thinker_curve), default=1),
        )

        # PPO curve (blue)
        if len(self.ppo_curve) >= 2:
            points = [(gx + int(gw * ep / max_ep), gy + gh - int(gh * min(r, 100) / 100))
                      for ep, r in self.ppo_curve]
            pygame.draw.lines(self.screen, (99, 102, 241), False, points, 2)

        # Thinker curve (purple)
        if len(self.thinker_curve) >= 2:
            points = [(gx + int(gw * ep / max_ep), gy + gh - int(gh * min(r, 100) / 100))
                      for ep, r in self.thinker_curve]
            pygame.draw.lines(self.screen, (147, 130, 241), False, points, 2)

        # Legend
        lx = gx + gw - 200
        ly = graph_y + 5
        pygame.draw.line(self.screen, (99, 102, 241), (lx, ly + 6), (lx + 20, ly + 6), 2)
        self._text("PPO", lx + 25, ly, self.font_tiny, (99, 102, 241))
        pygame.draw.line(self.screen, (147, 130, 241), (lx + 70, ly + 6), (lx + 90, ly + 6), 2)
        self._text("Thinker", lx + 95, ly, self.font_tiny, (147, 130, 241))

        # Divider
        pygame.draw.line(self.screen, (50, 55, 70), (half_w, 0), (half_w, graph_y - 5), 1)

        # Mode + controls
        mode_text = {"visual": "VISUAL", "fast": "FAST", "slow": "SLOW"}
        mode_color = {"visual": (52, 211, 153), "fast": (251, 191, 36), "slow": (147, 197, 253)}
        self._text(f"Mode: {mode_text[self.mode]}", 30, self.H - 25, self.font_small, mode_color[self.mode])
        self._text("Space=Pause  F=Fast  S=Slow  V=Visual  ESC=Quit",
                   self.W - 420, self.H - 25, self.font_tiny, (70, 80, 100))

        if self.paused:
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 100))
            self.screen.blit(overlay, (0, 0))
            text = self.font_large.render("PAUSED", True, (255, 255, 255))
            self.screen.blit(text, (self.W // 2 - 60, self.H // 2 - 15))

        pygame.display.flip()

    def _draw_action_bars(self, x, y, probs, color):
        labels = ["U:", "D:", "L:", "R:"]
        for i, (label, prob) in enumerate(zip(labels, probs)):
            self._text(label, x, y + i * 18, self.font_tiny, (100, 110, 130))
            bx = x + 20
            bw = 100
            bh = 12
            pygame.draw.rect(self.screen, (40, 40, 55), pygame.Rect(bx, y + i * 18 + 2, bw, bh), border_radius=2)
            fw = int(bw * prob)
            if fw > 0:
                pygame.draw.rect(self.screen, color, pygame.Rect(bx, y + i * 18 + 2, fw, bh), border_radius=2)
            self._text(f"{prob:.2f}", bx + bw + 5, y + i * 18, self.font_tiny, (150, 160, 180))

    def _rate_color(self, rate):
        if rate < 15:
            return (239, 68, 68)
        elif rate < 30:
            return (251, 146, 60)
        elif rate < 60:
            return (251, 191, 36)
        return (52, 211, 153)

    def _text(self, text, x, y, font, color):
        surf = font.render(str(text), True, color)
        self.screen.blit(surf, (x, y))


if __name__ == "__main__":
    print("=" * 55)
    print("  PPO vs Thinker RL — Side by Side Comparison")
    print("  Both agents start from scratch on the same levels.")
    print("  Watch which one learns faster!")
    print("=" * 55)
    print()
    print("Generating training levels (this may take a moment)...")

    app = CompareTraining()

    print("Levels generated! Starting comparison...")
    print()
    print("Controls:")
    print("  Space = Pause/Resume")
    print("  F = Fast mode")
    print("  S = Slow mode")
    print("  V = Visual mode (normal)")
    print("  ESC = Quit and save")

    app.run()
