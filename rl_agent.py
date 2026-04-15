from __future__ import annotations

import random
from collections import defaultdict, deque


class RLAgent:
    def __init__(self, alpha: float = 0.18, gamma: float = 0.93, epsilon: float = 0.22):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = 0.03
        self.q = defaultdict(lambda: [0.0] * 7)
        self.recent_scores = deque(maxlen=24)
        self.recent_efficiency = deque(maxlen=24)
        self.recent_collisions = deque(maxlen=24)

    def choose_action(self, state, explore: bool = True, preferred_action: int | None = None) -> int:
        if explore and preferred_action is not None:
            return preferred_action
        if explore and random.random() < self.epsilon:
            return random.randrange(7)
        values = self.q[state]
        return max(range(len(values)), key=lambda i: values[i])

    def learn(self, state, action: int, reward: float, next_state, done: bool):
        current = self.q[state][action]
        target = reward if done else reward + self.gamma * max(self.q[next_state])
        self.q[state][action] = current + self.alpha * (target - current)

    def adapt_after_episode(self, episode: dict):
        score = float(episode.get("score", 0.0))
        efficiency = float(episode.get("path_efficiency", 0.0))
        collision = bool(episode.get("collision", False))
        reached_goal = bool(episode.get("reached_goal", False))

        self.recent_scores.append(score)
        self.recent_efficiency.append(efficiency)
        self.recent_collisions.append(1.0 if collision else 0.0)

        if collision:
            self.epsilon = min(0.4, self.epsilon * 1.05 + 0.01)
            self.alpha = min(0.3, self.alpha * 1.02)
        elif reached_goal:
            self.epsilon = max(self.min_epsilon, self.epsilon * 0.97)
            self.alpha = max(0.08, self.alpha * 0.995)
        else:
            mean_efficiency = sum(self.recent_efficiency) / len(self.recent_efficiency) if self.recent_efficiency else 0.0
            if mean_efficiency > 0.55:
                self.epsilon = max(self.min_epsilon, self.epsilon * 0.985)
            else:
                self.epsilon = min(0.35, self.epsilon * 1.01)

    def decay(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * 0.96)
