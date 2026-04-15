from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SimulationConfig:
    environment_name: str = "Urban Circuit"
    weather: str = "clear"
    track_complexity: str = "moderate"
    obstacle_count: int = 8
    car_speed: float = 2.8
    fog_alpha: float = 0.0
    description: str = "Standard urban circuit."


COMPLEXITY_FACTOR = {"simple": 0.85, "moderate": 1.0, "complex": 1.2}
WEATHER_FACTOR = {"clear": 1.0, "rainy": 1.15, "foggy": 1.2}
ACTIONS = [(-1, -0.2), (-1, 0.0), (-0.4, 0.1), (0.0, 0.2), (0.4, 0.1), (1.0, 0.0), (1.0, -0.2)]
WEATHER_SEQUENCE = ["clear", "foggy", "rainy"]
GOAL_RADIUS = 82


def normalize_track_complexity(value: str) -> str:
    lookup = {
        "low": "simple",
        "easy": "simple",
        "medium": "moderate",
        "normal": "moderate",
        "high": "complex",
        "hard": "complex",
    }
    normalized = (value or "").strip().lower()
    if normalized in {"simple", "moderate", "complex"}:
        return normalized
    return lookup.get(normalized, "moderate")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def angle_diff(target: float, current: float) -> float:
    diff = target - current
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff


def shift_weather(weather: str, direction: int) -> str:
    current = (weather or "clear").strip().lower()
    if current not in WEATHER_SEQUENCE:
        current = "clear"
    index = WEATHER_SEQUENCE.index(current)
    next_index = int(clamp(index + direction, 0, len(WEATHER_SEQUENCE) - 1))
    return WEATHER_SEQUENCE[next_index]


class Track:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.track_complexity = normalize_track_complexity(config.track_complexity)
        self.width = 900
        self.height = 560
        self.centerline = self._build_track()
        self.destination = self.centerline[-18]
        self.start = self.centerline[0]
        self.obstacles = self._build_obstacles()

    def _build_track(self) -> List[Tuple[float, float]]:
        pts = []
        base = {
            "simple": [(120, 310), (180, 150), (420, 110), (700, 150), (770, 300), (690, 430), (380, 450), (160, 410)],
            "moderate": [(100, 300), (170, 120), (330, 80), (540, 78), (735, 122), (812, 290), (728, 442), (520, 485), (300, 472), (152, 415)],
            "complex": [(92, 310), (150, 118), (280, 72), (430, 125), (550, 60), (720, 110), (805, 250), (740, 405), (590, 485), (400, 430), (275, 485), (130, 402)],
        }[self.track_complexity]
        for i in range(len(base)):
            x1, y1 = base[i]
            x2, y2 = base[(i + 1) % len(base)]
            for t in np.linspace(0, 1, 18, endpoint=False):
                pts.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
        return pts

    def _build_obstacles(self) -> List[Tuple[float, float, float]]:
        rnd = random.Random(100 + self.config.obstacle_count)
        obstacles = []
        for i in range(self.config.obstacle_count):
            px, py = self.centerline[(i * 11 + 22) % len(self.centerline)]
            nx, ny = self.centerline[(i * 11 + 23) % len(self.centerline)]
            dx, dy = nx - px, ny - py
            norm = math.hypot(dx, dy) or 1
            offset = rnd.choice([-16, -10, 10, 16])
            ox = px + (-dy / norm) * offset
            oy = py + (dx / norm) * offset
            obstacles.append((ox, oy, 12))
        return obstacles


def state_key(sensors: List[float], speed_ratio: float, goal_angle: float, heading_error: float, turn_bias: float, goal_distance: float) -> Tuple[int, ...]:
    sensor_bins = tuple(min(4, int(x * 5)) for x in sensors)
    speed_bin = min(4, int(speed_ratio * 5))
    angle_bin = min(6, max(0, int((goal_angle + math.pi) / (2 * math.pi) * 7)))
    heading_bin = min(6, max(0, int((heading_error + math.pi) / (2 * math.pi) * 7)))
    bias_bin = min(4, max(0, int((turn_bias + 1.0) * 2)))
    distance_bin = min(6, int(min(goal_distance, 220.0) / 220.0 * 7))
    return sensor_bins + (speed_bin, angle_bin, heading_bin, bias_bin, distance_bin)


def directional_teacher_action(env: SimEnv, obs: Dict) -> int:
    target = env.track.centerline[env.goal_index % len(env.track.centerline)]
    destination = env.track.destination
    weather_drag = WEATHER_FACTOR[env.cfg.weather]
    front = obs["sensors"][3]
    left_open = sum(obs["sensors"][:3]) / 3.0
    right_open = sum(obs["sensors"][4:]) / 3.0

    best_action = 0
    best_score = float("inf")

    for index, (steer, accel) in enumerate(ACTIONS):
        next_angle = env.angle + steer * 0.09
        next_speed = min(env.cfg.car_speed, max(0.4, env.speed + accel / weather_drag))
        next_x = env.x + math.cos(next_angle) * next_speed * 3.0
        next_y = env.y + math.sin(next_angle) * next_speed * 3.0

        target_distance = math.hypot(next_x - target[0], next_y - target[1])
        destination_distance = math.hypot(next_x - destination[0], next_y - destination[1])
        target_angle = math.atan2(target[1] - next_y, target[0] - next_x)
        alignment_error = abs(angle_diff(target_angle, next_angle))
        road_penalty = 0.0 if env._is_on_track(next_x, next_y) else 30.0
        obstacle_penalty = 0.0
        for ox, oy, rad in env.track.obstacles:
            if math.hypot(next_x - ox, next_y - oy) <= rad + 8:
                obstacle_penalty = 45.0
                break

        open_side_bias = (right_open - left_open) * 2.0
        sensor_bias = -8.0 if front < 0.3 else -2.0 if front < 0.5 else 0.0
        score = (
            target_distance * 0.65
            + destination_distance * 0.08
            + alignment_error * 12.0
            + road_penalty
            + obstacle_penalty
            - open_side_bias
            + sensor_bias
        )

        if score < best_score:
            best_score = score
            best_action = index

    return best_action


class SimEnv:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.track = Track(config)
        self.max_steps = 220
        self.reset()

    def reset(self):
        self.x, self.y = self.track.start
        nx, ny = self.track.centerline[1]
        self.angle = math.atan2(ny - self.y, nx - self.x)
        self.speed = 0.6
        self.step_count = 0
        self.goal_index = 8
        self.total_progress = 0.0
        self.collision = False
        self.reached_goal = False
        self.path = [(self.x, self.y)]
        return self.observe()

    def observe(self) -> Dict:
        sensors = self._sense()
        target = self.track.centerline[self.goal_index % len(self.track.centerline)]
        target_angle = math.atan2(target[1] - self.y, target[0] - self.x)
        diff = angle_diff(target_angle, self.angle)
        destination_angle = math.atan2(self.track.destination[1] - self.y, self.track.destination[0] - self.x)
        heading_error = angle_diff(destination_angle, self.angle)
        left_bias = sum(sensors[:3]) / 3.0
        right_bias = sum(sensors[4:]) / 3.0
        turn_bias = right_bias - left_bias
        return {
            "sensors": sensors,
            "speed_ratio": min(1.0, self.speed / max(1.0, self.cfg.car_speed)),
            "goal_angle": diff,
            "heading_error": heading_error,
            "turn_bias": turn_bias,
            "goal_distance": math.hypot(target[0] - self.x, target[1] - self.y),
        }

    def _sense(self) -> List[float]:
        readings = []
        for delta in [-1.2, -0.8, -0.35, 0.0, 0.35, 0.8, 1.2]:
            readings.append(self._cast_ray(self.angle + delta))
        return readings

    def _cast_ray(self, ang: float, max_dist: float = 105) -> float:
        for dist in np.linspace(6, max_dist, 22):
            rx = self.x + math.cos(ang) * dist
            ry = self.y + math.sin(ang) * dist
            if not self._is_on_track(rx, ry):
                return dist / max_dist
            for ox, oy, rad in self.track.obstacles:
                if math.hypot(rx - ox, ry - oy) <= rad:
                    return dist / max_dist
        return 1.0

    def _is_on_track(self, x: float, y: float) -> bool:
        margin = 45 * COMPLEXITY_FACTOR[normalize_track_complexity(self.cfg.track_complexity)]
        return any(math.hypot(x - px, y - py) < margin for px, py in self.track.centerline)

    def step(self, action_id: int):
        steer, accel = ACTIONS[action_id]
        weather_drag = WEATHER_FACTOR[self.cfg.weather]
        self.speed = min(self.cfg.car_speed, max(0.4, self.speed + accel / weather_drag))
        self.angle += steer * 0.09
        self.x += math.cos(self.angle) * self.speed * 3.0
        self.y += math.sin(self.angle) * self.speed * 3.0
        self.step_count += 1
        self.path.append((self.x, self.y))

        reward = -0.25
        if not self._is_on_track(self.x, self.y):
            reward -= 8
            self.collision = True
        for ox, oy, rad in self.track.obstacles:
            if math.hypot(self.x - ox, self.y - oy) <= rad + 8:
                reward -= 14
                self.collision = True
                break

        target = self.track.centerline[self.goal_index % len(self.track.centerline)]
        target_angle = math.atan2(target[1] - self.y, target[0] - self.x)
        goal_distance = math.hypot(self.x - target[0], self.y - target[1])
        alignment_error = abs(angle_diff(target_angle, self.angle))
        destination_angle = math.atan2(self.track.destination[1] - self.y, self.track.destination[0] - self.x)
        heading_error = abs(angle_diff(destination_angle, self.angle))
        left_bias = sum(self._sense()[:3]) / 3.0
        right_bias = sum(self._sense()[4:]) / 3.0
        turn_bias = right_bias - left_bias

        reward += max(0, 4.5 - goal_distance * 0.04)
        reward += max(0, 1.8 - alignment_error * 0.9)
        reward += max(0, 1.0 - heading_error * 0.4)
        reward += max(-0.4, min(0.4, turn_bias * 0.15))
        if goal_distance < 26:
            self.goal_index += 1
            self.total_progress += 1
            reward += 10

        if math.hypot(self.x - self.track.destination[0], self.y - self.track.destination[1]) < GOAL_RADIUS:
            self.reached_goal = True
            reward += 60

        done = self.collision or self.reached_goal or self.step_count >= self.max_steps
        return self.observe(), reward, done, {
            "collision": self.collision,
            "reached_goal": self.reached_goal,
            "path": list(self.path),
        }


class EpisodeStats(dict):
    pass


def adapt_config_for_episode(config: SimulationConfig, episode: dict) -> tuple[SimulationConfig, dict]:
    score = float(episode.get("score", 0.0))
    efficiency = float(episode.get("path_efficiency", 0.0))
    collision = bool(episode.get("collision", False))
    reached_goal = bool(episode.get("reached_goal", False))
    difficulty = float(episode.get("difficulty", 0.0))

    next_cfg = asdict(config)
    complexity_rank = ["simple", "moderate", "complex"]
    complexity_index = complexity_rank.index(normalize_track_complexity(config.track_complexity))

    if collision:
        next_cfg["obstacle_count"] = max(3, int(next_cfg["obstacle_count"] - 1))
        next_cfg["car_speed"] = round(clamp(next_cfg["car_speed"] - 0.18, 1.0, 4.5), 2)
        next_cfg["track_complexity"] = complexity_rank[max(0, complexity_index - 1)]
        next_cfg["weather"] = shift_weather(next_cfg["weather"], -1)
        next_cfg["fog_alpha"] = round(clamp(float(next_cfg.get("fog_alpha", 0.0)) - 0.05, 0.0, 0.45), 2)
        analysis = "The environment backed off after a collision, lowering pressure so the policy can recover and relearn control margins."
    elif reached_goal or score > 45:
        next_cfg["obstacle_count"] = min(18, int(next_cfg["obstacle_count"] + 1))
        next_cfg["car_speed"] = round(clamp(next_cfg["car_speed"] + 0.12, 1.0, 4.5), 2)
        if efficiency > 0.55 or reached_goal:
            next_cfg["track_complexity"] = complexity_rank[min(2, complexity_index + 1)]
        next_cfg["weather"] = shift_weather(next_cfg["weather"], 1 if efficiency > 0.45 else 0)
        next_cfg["fog_alpha"] = round(clamp(float(next_cfg.get("fog_alpha", 0.0)) + (0.04 if next_cfg["weather"] == "foggy" else 0.0), 0.0, 0.45), 2)
        analysis = "The environment sharpened the challenge because the agent handled the last episode cleanly and can absorb harder scenarios."
    else:
        if efficiency < 0.35:
            next_cfg["car_speed"] = round(clamp(next_cfg["car_speed"] - 0.08, 1.0, 4.5), 2)
            next_cfg["fog_alpha"] = round(clamp(float(next_cfg.get("fog_alpha", 0.0)) + 0.03, 0.0, 0.45), 2)
            analysis = "The route stayed mostly steady, but the director softened the visibility and speed envelope slightly because the policy showed uncertainty."
        else:
            next_cfg["obstacle_count"] = max(3, min(18, int(next_cfg["obstacle_count"])))
            analysis = "The environment held a near-neutral setting to keep the learning signal stable."

    next_cfg["track_complexity"] = normalize_track_complexity(next_cfg["track_complexity"])
    adaptation = {
        "score": round(score, 2),
        "difficulty": round(difficulty, 2),
        "efficiency": round(efficiency, 3),
        "collision": collision,
        "reached_goal": reached_goal,
        "analysis": analysis,
        "next_config": next_cfg,
    }
    return SimulationConfig(**next_cfg), adaptation


def run_episode(agent, config: SimulationConfig, learn: bool = True) -> EpisodeStats:
    env = SimEnv(config)
    obs = env.reset()
    total_reward = 0.0
    avg_speed = []
    last_sensors = obs["sensors"]

    for _ in range(env.max_steps):
        state = state_key(obs["sensors"], obs["speed_ratio"], obs["goal_angle"], obs["heading_error"], obs["turn_bias"], obs["goal_distance"])
        guidance = directional_teacher_action(env, obs) if learn else None
        action = agent.choose_action(state, explore=learn, preferred_action=guidance)
        next_obs, reward, done, info = env.step(action)
        if learn and info.get("collision") and not info.get("reached_goal"):
            done = False
        next_state = state_key(next_obs["sensors"], next_obs["speed_ratio"], next_obs["goal_angle"], next_obs["heading_error"], next_obs["turn_bias"], next_obs["goal_distance"])
        if learn:
            agent.learn(state, action, reward, next_state, done)
        obs = next_obs
        total_reward += reward
        avg_speed.append(env.speed)
        last_sensors = obs["sensors"]
        if done:
            break

    path_len = sum(math.hypot(env.path[i][0] - env.path[i - 1][0], env.path[i][1] - env.path[i - 1][1]) for i in range(1, len(env.path)))
    direct_dist = math.hypot(env.track.start[0] - env.track.destination[0], env.track.start[1] - env.track.destination[1])
    efficiency = round(min(1.0, direct_dist / max(path_len, 1.0)), 3)
    complexity = normalize_track_complexity(config.track_complexity)
    difficulty = round((config.obstacle_count * 0.45) + (config.car_speed * 1.5) + (2 if config.weather != "clear" else 0) + (2 if complexity == "complex" else 1 if complexity == "moderate" else 0), 2)

    return EpisodeStats(
        score=round(total_reward, 2),
        collision=env.collision,
        reached_goal=env.reached_goal,
        path_efficiency=efficiency,
        avg_speed=round(float(np.mean(avg_speed)) if avg_speed else 0.0, 2),
        steps=env.step_count,
        sensor_snapshot=[round(x, 3) for x in last_sensors],
        difficulty=difficulty,
        directional_snapshot={
            "goal_angle": round(float(obs["goal_angle"]), 3),
            "heading_error": round(float(obs["heading_error"]), 3),
            "turn_bias": round(float(obs["turn_bias"]), 3),
            "goal_distance": round(float(obs["goal_distance"]), 2),
        },
        track_points=env.track.centerline,
        obstacles=env.track.obstacles,
        destination=env.track.destination,
        config=asdict(config),
        path=env.path,
    )


def train_agent(agent, config: SimulationConfig, episodes: int = 20) -> List[EpisodeStats]:
    history = []
    for _ in range(episodes):
        episode = run_episode(agent, config, learn=True)
        if hasattr(agent, "adapt_after_episode"):
            agent.adapt_after_episode(episode)
        history.append(episode)
    agent.decay()
    return history


def train_until_goal(agent, config: SimulationConfig, max_episodes: int = 30) -> List[EpisodeStats]:
    history = []
    for _ in range(max_episodes):
        episode = run_episode(agent, config, learn=True)
        if hasattr(agent, "adapt_after_episode"):
            agent.adapt_after_episode(episode)
        history.append(episode)
        if episode.get("reached_goal"):
            break
    agent.decay()
    return history


def train_coevolution(agent, config: SimulationConfig, episodes: int = 20):
    history = []
    current_config = config
    last_adaptation = None

    for _ in range(episodes):
        episode = run_episode(agent, current_config, learn=True)
        if hasattr(agent, "adapt_after_episode"):
            agent.adapt_after_episode(episode)
        next_config, adaptation = adapt_config_for_episode(current_config, episode)
        episode["adaptation"] = adaptation
        episode["next_config"] = adaptation["next_config"]
        history.append(episode)
        last_adaptation = adaptation
        if episode.get("reached_goal"):
            current_config = SimulationConfig(**episode["config"])
            break
        current_config = next_config

    agent.decay()
    return history, current_config, last_adaptation
