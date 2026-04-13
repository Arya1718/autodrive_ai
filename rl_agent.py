from __future__ import annotations

import random
from collections import defaultdict


class RLAgent:
    def __init__(self, alpha: float = 0.18, gamma: float = 0.93, epsilon: float = 0.22):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = 0.03
        self.q = defaultdict(lambda: [0.0] * 7)

    def choose_action(self, state, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randrange(7)
        values = self.q[state]
        return max(range(len(values)), key=lambda i: values[i])

    def learn(self, state, action: int, reward: float, next_state, done: bool):
        current = self.q[state][action]
        target = reward if done else reward + self.gamma * max(self.q[next_state])
        self.q[state][action] = current + self.alpha * (target - current)

    def decay(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * 0.96)
