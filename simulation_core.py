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


def state_key(sensors: List[float], speed_ratio: float, goal_angle: float) -> Tuple[int, ...]:
    sensor_bins = tuple(min(4, int(x * 5)) for x in sensors)
    speed_bin = min(4, int(speed_ratio * 5))
    angle_bin = min(6, max(0, int((goal_angle + math.pi) / (2 * math.pi) * 7)))
    return sensor_bins + (speed_bin, angle_bin)


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
        diff = target_angle - self.angle
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return {
            "sensors": sensors,
            "speed_ratio": min(1.0, self.speed / max(1.0, self.cfg.car_speed)),
            "goal_angle": diff,
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

        reward = -0.4
        if not self._is_on_track(self.x, self.y):
            reward -= 8
            self.collision = True
        for ox, oy, rad in self.track.obstacles:
            if math.hypot(self.x - ox, self.y - oy) <= rad + 8:
                reward -= 14
                self.collision = True
                break

        target = self.track.centerline[self.goal_index % len(self.track.centerline)]
        d = math.hypot(self.x - target[0], self.y - target[1])
        reward += max(0, 6 - d * 0.05)
        if d < 26:
            self.goal_index += 1
            self.total_progress += 1
            reward += 10

        if math.hypot(self.x - self.track.destination[0], self.y - self.track.destination[1]) < 34:
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


def run_episode(agent, config: SimulationConfig, learn: bool = True) -> EpisodeStats:
    env = SimEnv(config)
    obs = env.reset()
    total_reward = 0.0
    avg_speed = []
    last_sensors = obs["sensors"]

    for _ in range(env.max_steps):
        state = state_key(obs["sensors"], obs["speed_ratio"], obs["goal_angle"])
        action = agent.choose_action(state, explore=learn)
        next_obs, reward, done, info = env.step(action)
        next_state = state_key(next_obs["sensors"], next_obs["speed_ratio"], next_obs["goal_angle"])
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
        track_points=env.track.centerline,
        obstacles=env.track.obstacles,
        destination=env.track.destination,
        config=asdict(config),
        path=env.path,
    )


def train_agent(agent, config: SimulationConfig, episodes: int = 20) -> List[EpisodeStats]:
    history = []
    for _ in range(episodes):
        history.append(run_episode(agent, config, learn=True))
    agent.decay()
    return history
