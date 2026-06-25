from __future__ import annotations

import random
from collections import defaultdict, deque


class RLAgent:
    def __init__(self, alpha: float = 0.25, gamma: float = 0.97, epsilon: float = 0.20):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = 0.03
        self.q = defaultdict(lambda: [0.0] * 7)
        self.recent_scores = deque(maxlen=24)
        self.recent_efficiency = deque(maxlen=24)
        self.recent_collisions = deque(maxlen=24)
        # Teacher blending: starts high, decays as agent learns
        self.teacher_blend = 0.80
        self.min_teacher_blend = 0.15
        # Eligibility traces for faster credit assignment
        self.trace = defaultdict(lambda: [0.0] * 7)
        self.trace_decay = 0.85  # lambda for eligibility traces
        self.recent_visits = deque(maxlen=50)  # recent (state, action) pairs

    def choose_action(self, state, explore: bool = True, preferred_action: int | None = None) -> int:
        """Choose action with teacher blending instead of always following teacher."""
        if explore and preferred_action is not None:
            # Blend: follow teacher with probability teacher_blend, else use Q-values/explore
            if random.random() < self.teacher_blend:
                return preferred_action
            # Fall through to Q-based selection below
        if explore and random.random() < self.epsilon:
            return random.randrange(7)
        values = self.q[state]
        return max(range(len(values)), key=lambda i: values[i])

    def learn(self, state, action: int, reward: float, next_state, done: bool):
        """Q-learning with eligibility traces for faster reward propagation."""
        current = self.q[state][action]
        target = reward if done else reward + self.gamma * max(self.q[next_state])
        td_error = target - current

        # Update current state-action
        self.q[state][action] = current + self.alpha * td_error

        # Update eligibility trace for the current state-action
        self.trace[state][action] = 1.0
        self.recent_visits.append((state, action))

        # Propagate TD error backward through recent visits (eligibility traces)
        for s, a in self.recent_visits:
            if s == state and a == action:
                continue
            trace_val = self.trace[s][a]
            if trace_val > 0.01:
                self.q[s][a] += self.alpha * td_error * trace_val * 0.5
                self.trace[s][a] *= self.gamma * self.trace_decay

        # Decay all traces
        if done:
            self.trace.clear()
            self.recent_visits.clear()

    def adapt_after_episode(self, episode: dict):
        score = float(episode.get("score", 0.0))
        efficiency = float(episode.get("path_efficiency", 0.0))
        collision = bool(episode.get("collision", False))
        reached_goal = bool(episode.get("reached_goal", False))

        self.recent_scores.append(score)
        self.recent_efficiency.append(efficiency)
        self.recent_collisions.append(1.0 if collision else 0.0)

        if collision:
            # Mild increase in exploration, not too aggressive
            self.epsilon = min(0.30, self.epsilon * 1.02 + 0.005)
            self.alpha = min(0.3, self.alpha * 1.01)
        elif reached_goal:
            self.epsilon = max(self.min_epsilon, self.epsilon * 0.92)
            self.alpha = max(0.08, self.alpha * 0.99)
            # Decay teacher reliance faster on success
            self.teacher_blend = max(self.min_teacher_blend, self.teacher_blend * 0.88)
        else:
            mean_efficiency = sum(self.recent_efficiency) / len(self.recent_efficiency) if self.recent_efficiency else 0.0
            if mean_efficiency > 0.55:
                self.epsilon = max(self.min_epsilon, self.epsilon * 0.97)
            else:
                self.epsilon = min(0.28, self.epsilon * 1.005)

        # Gradually reduce teacher blend regardless of outcome
        self.teacher_blend = max(self.min_teacher_blend, self.teacher_blend * 0.97)

    def decay(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * 0.95)
        self.teacher_blend = max(self.min_teacher_blend, self.teacher_blend * 0.95)
