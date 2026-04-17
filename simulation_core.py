from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple

import numpy as np


# ─── VALID VALUES ────────────────────────────────────────────────────────────
VALID_WEATHERS = {"clear", "rainy", "foggy"}
VALID_COMPLEXITIES = {"simple", "moderate", "complex"}

COMPLEXITY_FACTOR = {"simple": 0.85, "moderate": 1.0, "complex": 1.2}
WEATHER_FACTOR = {"clear": 1.0, "rainy": 1.15, "foggy": 1.2}
ACTIONS = [(-1, -0.2), (-1, 0.0), (-0.4, 0.1), (0.0, 0.2), (0.4, 0.1), (1.0, 0.0), (1.0, -0.2)]
WEATHER_SEQUENCE = ["clear", "foggy", "rainy"]
GOAL_RADIUS = 50


# ─── SANITIZERS ──────────────────────────────────────────────────────────────
def sanitize_weather(weather: str) -> str:
    """Clamp any weather string to a valid simulation value."""
    w = (weather or "clear").strip().lower()
    if w in VALID_WEATHERS:
        return w
    mapping = {
        "overcast": "foggy", "sunny": "clear", "rain": "rainy", "fog": "foggy",
        "stormy": "rainy", "cloudy": "foggy", "snowy": "foggy", "drizzle": "rainy",
        "hazy": "foggy", "misty": "foggy", "windy": "clear", "thunder": "rainy",
        "partly_cloudy": "clear", "partly cloudy": "clear",
    }
    return mapping.get(w, "clear")


def normalize_track_complexity(value: str) -> str:
    lookup = {
        "low": "simple",
        "easy": "simple",
        "medium": "moderate",
        "normal": "moderate",
        "high": "complex",
        "hard": "complex",
        "beginner": "simple",
        "advanced": "complex",
        "expert": "complex",
        "intermediate": "moderate",
    }
    normalized = (value or "").strip().lower()
    if normalized in VALID_COMPLEXITIES:
        return normalized
    return lookup.get(normalized, "moderate")


def safe_weather_factor(weather: str) -> float:
    """Get the drag factor for a weather, always returning a valid float."""
    return WEATHER_FACTOR.get(sanitize_weather(weather), 1.0)


def safe_complexity_factor(complexity: str) -> float:
    """Get a complexity factor, always returning a valid float."""
    return COMPLEXITY_FACTOR.get(normalize_track_complexity(complexity), 1.0)


# ─── CONFIG ──────────────────────────────────────────────────────────────────
@dataclass
class SimulationConfig:
    environment_name: str = "Urban Circuit"
    weather: str = "clear"
    track_complexity: str = "moderate"
    obstacle_count: int = 8
    car_speed: float = 2.8
    fog_alpha: float = 0.0
    terminate_on_collision: bool = True
    description: str = "Standard urban circuit."

    def __post_init__(self):
        """Auto-sanitize all fields so LLM-generated configs never crash."""
        self.weather = sanitize_weather(self.weather)
        self.track_complexity = normalize_track_complexity(self.track_complexity)
        self.obstacle_count = max(2, min(18, int(self.obstacle_count)))
        self.car_speed = round(max(1.0, min(4.5, float(self.car_speed))), 2)
        self.fog_alpha = round(max(0.0, min(0.45, float(self.fog_alpha))), 2)
        self.terminate_on_collision = bool(self.terminate_on_collision)


# ─── UTILITIES ───────────────────────────────────────────────────────────────
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
    current = sanitize_weather(weather)
    if current not in WEATHER_SEQUENCE:
        current = "clear"
    index = WEATHER_SEQUENCE.index(current)
    next_index = int(clamp(index + direction, 0, len(WEATHER_SEQUENCE) - 1))
    return WEATHER_SEQUENCE[next_index]


# ─── TRACK ───────────────────────────────────────────────────────────────────
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
        # Precompute spatial grid for O(1) on-track lookups
        self.margin = 45 * safe_complexity_factor(self.track_complexity)
        self._grid_cell = 20  # grid cell size in px
        self._track_grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
        self._build_grid()

    def _build_grid(self):
        """Build a spatial hash grid from centerline points for fast proximity checks."""
        cell = self._grid_cell
        # Expand grid coverage to margin distance around each centerline point
        expand = int(math.ceil(self.margin / cell)) + 1
        for px, py in self.centerline:
            gx, gy = int(px // cell), int(py // cell)
            for dx in range(-expand, expand + 1):
                for dy in range(-expand, expand + 1):
                    key = (gx + dx, gy + dy)
                    if key not in self._track_grid:
                        self._track_grid[key] = []
                    self._track_grid[key].append((px, py))
        # Deduplicate lists
        for key in self._track_grid:
            self._track_grid[key] = list(set(self._track_grid[key]))

    def is_on_track(self, x: float, y: float) -> bool:
        """O(1) amortized on-track check using spatial grid."""
        cell = self._grid_cell
        key = (int(x // cell), int(y // cell))
        candidates = self._track_grid.get(key)
        if not candidates:
            return False
        margin_sq = self.margin * self.margin
        return any((x - px) * (x - px) + (y - py) * (y - py) < margin_sq for px, py in candidates)

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
            # Push obstacles further from the centerline so the driving line stays clear
            offset = rnd.choice([-28, -22, 22, 28])
            ox = px + (-dy / norm) * offset
            oy = py + (dx / norm) * offset
            obstacles.append((ox, oy, 10))
        return obstacles


# ─── STATE KEY ───────────────────────────────────────────────────────────────
def state_key(sensors: List[float], speed_ratio: float, goal_angle: float, heading_error: float, turn_bias: float, goal_distance: float) -> Tuple[int, ...]:
    sensor_bins = tuple(min(4, int(x * 5)) for x in sensors)
    speed_bin = min(4, int(speed_ratio * 5))
    angle_bin = min(6, max(0, int((goal_angle + math.pi) / (2 * math.pi) * 7)))
    heading_bin = min(6, max(0, int((heading_error + math.pi) / (2 * math.pi) * 7)))
    bias_bin = min(4, max(0, int((turn_bias + 1.0) * 2)))
    distance_bin = min(6, int(min(goal_distance, 220.0) / 220.0 * 7))
    return sensor_bins + (speed_bin, angle_bin, heading_bin, bias_bin, distance_bin)


# ─── TEACHER ─────────────────────────────────────────────────────────────────
def directional_teacher_action(env: SimEnv, obs: Dict) -> int:
    """Improved teacher that looks ahead multiple waypoints for better path planning."""
    cl = env.track.centerline
    n_cl = len(cl)
    # Look ahead 3 waypoints for smoother path planning
    lookahead_indices = [
        env.goal_index % n_cl,
        (env.goal_index + 3) % n_cl,
        (env.goal_index + 6) % n_cl,
    ]
    targets = [cl[i] for i in lookahead_indices]
    destination = env.track.destination
    weather_drag = safe_weather_factor(env.cfg.weather)
    front = obs["sensors"][3]
    left_open = sum(obs["sensors"][:3]) / 3.0
    right_open = sum(obs["sensors"][4:]) / 3.0

    best_action = 0
    best_score = float("inf")

    for index, (steer, accel) in enumerate(ACTIONS):
        next_angle = env.angle + steer * 0.09
        next_speed = min(env.cfg.car_speed, max(0.4, env.speed + accel / weather_drag))
        next_x = env.x + math.cos(next_angle) * next_speed * 2.5
        next_y = env.y + math.sin(next_angle) * next_speed * 2.5

        # Multi-waypoint distance: weight nearest target most, then further ones
        target_distance = (
            math.hypot(next_x - targets[0][0], next_y - targets[0][1]) * 0.55
            + math.hypot(next_x - targets[1][0], next_y - targets[1][1]) * 0.30
            + math.hypot(next_x - targets[2][0], next_y - targets[2][1]) * 0.15
        )
        destination_distance = math.hypot(next_x - destination[0], next_y - destination[1])
        # Align toward nearest target
        target_angle = math.atan2(targets[0][1] - next_y, targets[0][0] - next_x)
        alignment_error = abs(angle_diff(target_angle, next_angle))
        road_penalty = 0.0 if env._is_on_track(next_x, next_y) else 25.0
        obstacle_penalty = 0.0
        for ox, oy, rad in env.track.obstacles:
            dist_to_obs = math.hypot(next_x - ox, next_y - oy)
            if dist_to_obs <= rad + 8:
                obstacle_penalty = 40.0
                break
            elif dist_to_obs <= rad + 18:
                obstacle_penalty += 8.0  # Soft penalty for being close

        open_side_bias = (right_open - left_open) * 1.5
        sensor_bias = -6.0 if front < 0.3 else -1.5 if front < 0.5 else 0.0
        # Favor actions that maintain some speed (avoid constant braking)
        speed_incentive = -1.5 if next_speed > 1.0 else 0.0
        score = (
            target_distance * 0.6
            + destination_distance * 0.06
            + alignment_error * 10.0
            + road_penalty
            + obstacle_penalty
            - open_side_bias
            + sensor_bias
            + speed_incentive
        )

        if score < best_score:
            best_score = score
            best_action = index

    return best_action


# ─── SIMULATION ENVIRONMENT ─────────────────────────────────────────────────
class SimEnv:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.track = Track(config)
        self.max_steps = 500
        self.reset()

    def reset(self):
        self.x, self.y = self.track.start
        nx, ny = self.track.centerline[1]
        self.angle = math.atan2(ny - self.y, nx - self.x)
        self.speed = 0.6
        self.step_count = 0
        self.goal_index = 3
        self.total_progress = 0.0
        self.collision = False
        self.collision_count = 0
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
        return self.track.is_on_track(x, y)

    def step(self, action_id: int):
        steer, accel = ACTIONS[action_id]
        weather_drag = safe_weather_factor(self.cfg.weather)
        self.speed = min(self.cfg.car_speed, max(0.4, self.speed + accel / weather_drag))
        self.angle += steer * 0.09
        self.x += math.cos(self.angle) * self.speed * 2.5
        self.y += math.sin(self.angle) * self.speed * 2.5
        self.step_count += 1
        self.path.append((self.x, self.y))

        reward = -0.15
        hit_this_step = False

        if not self._is_on_track(self.x, self.y):
            reward -= 5
            hit_this_step = True
            # Gentle nudge back — keep 50/50 to avoid teleporting backward
            nearest = min(self.track.centerline, key=lambda p: math.hypot(self.x - p[0], self.y - p[1]))
            self.x = self.x * 0.5 + nearest[0] * 0.5
            self.y = self.y * 0.5 + nearest[1] * 0.5
            self.speed *= 0.6

        for ox, oy, rad in self.track.obstacles:
            if math.hypot(self.x - ox, self.y - oy) <= rad + 5:
                # Diminishing penalty: first few collisions hurt more, later ones less
                collision_penalty = max(2.0, 8.0 - self.collision_count * 0.8)
                reward -= collision_penalty
                hit_this_step = True
                # Gentle push away from obstacle
                dx = self.x - ox
                dy = self.y - oy
                dist = math.hypot(dx, dy) or 1.0
                self.x += (dx / dist) * 10
                self.y += (dy / dist) * 10
                self.speed *= 0.55
                break

        if hit_this_step:
            self.collision_count += 1
            self.collision = True  # At least one collision occurred in episode

        target = self.track.centerline[self.goal_index % len(self.track.centerline)]
        target_angle = math.atan2(target[1] - self.y, target[0] - self.x)
        goal_distance = math.hypot(self.x - target[0], self.y - target[1])
        alignment_error = abs(angle_diff(target_angle, self.angle))
        destination_angle = math.atan2(self.track.destination[1] - self.y, self.track.destination[0] - self.x)
        heading_error = abs(angle_diff(destination_angle, self.angle))
        left_bias = sum(self._sense()[:3]) / 3.0
        right_bias = sum(self._sense()[4:]) / 3.0
        turn_bias = right_bias - left_bias

        # Progress reward: reward approaching waypoints
        reward += max(0, 5.0 - goal_distance * 0.035)
        reward += max(0, 2.0 - alignment_error * 0.8)
        reward += max(0, 1.2 - heading_error * 0.35)
        reward += max(-0.3, min(0.3, turn_bias * 0.12))
        # Speed maintenance bonus — reward moving, not stalling
        if self.speed > 0.8 and not hit_this_step:
            reward += 0.4
        if goal_distance < 38:
            self.goal_index += 1
            self.total_progress += 1
            reward += 12

        if math.hypot(self.x - self.track.destination[0], self.y - self.track.destination[1]) < GOAL_RADIUS:
            self.reached_goal = True
            reward += 80

        # Episode ends on goal, max steps, or collision when configured.
        done = self.reached_goal or (self.collision and self.cfg.terminate_on_collision) or self.step_count >= self.max_steps
        return self.observe(), reward, done, {
            "collision": self.collision,
            "collision_count": self.collision_count,
            "reached_goal": self.reached_goal,
            "path": list(self.path),
        }

    def step_with_reward_shaping(self, action_id: int, reward_modifiers: dict | None = None):
        """Step with optional LLM-driven reward modifiers applied on top of base rewards."""
        obs, base_reward, done, info = self.step(action_id)

        if not reward_modifiers:
            return obs, base_reward, done, info

        shaped_reward = base_reward

        # Survival bonus: reward for each step without collision
        if not self.collision:
            shaped_reward += float(reward_modifiers.get("survival_bonus", 0))

        # Speed bonus: reward for maintaining good speed
        speed_ratio = self.speed / max(1.0, self.cfg.car_speed)
        if speed_ratio > 0.4 and not self.collision:
            shaped_reward += float(reward_modifiers.get("speed_bonus", 0)) * speed_ratio

        # Progress bonus: extra reward when reaching waypoints (already got +10 base)
        target = self.track.centerline[self.goal_index % len(self.track.centerline)]
        if math.hypot(self.x - target[0], self.y - target[1]) < 30:
            shaped_reward += float(reward_modifiers.get("progress_bonus", 0))

        # Goal bonus: extra reward for reaching destination
        if self.reached_goal:
            shaped_reward += float(reward_modifiers.get("goal_bonus", 0))

        # Smooth steering bonus: reward for gentle steering
        if len(self.path) >= 3:
            p1 = self.path[-3]
            p2 = self.path[-2]
            p3 = self.path[-1]
            a1 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            a2 = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
            steer_change = abs(angle_diff(a2, a1))
            if steer_change < 0.15:  # smooth steering
                shaped_reward += float(reward_modifiers.get("smooth_steering_bonus", 0))

        return obs, shaped_reward, done, info


class EpisodeStats(dict):
    pass


# ─── ADAPTIVE CONFIG ────────────────────────────────────────────────────────
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

    # Sanitize before creating config — catches any invalid values
    next_cfg["track_complexity"] = normalize_track_complexity(next_cfg["track_complexity"])
    next_cfg["weather"] = sanitize_weather(next_cfg["weather"])
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


# ─── EPISODE RUNNER ──────────────────────────────────────────────────────────
def run_episode(agent, config: SimulationConfig, learn: bool = True, reward_modifiers: dict | None = None, action_hints: dict | None = None) -> EpisodeStats:
    """Run a single episode. Optionally apply LLM reward modifiers and action hints."""
    env = SimEnv(config)
    obs = env.reset()
    total_reward = 0.0
    avg_speed = []
    last_sensors = obs["sensors"]

    for _ in range(env.max_steps):
        state = state_key(obs["sensors"], obs["speed_ratio"], obs["goal_angle"], obs["heading_error"], obs["turn_bias"], obs["goal_distance"])
        guidance = directional_teacher_action(env, obs) if learn else None

        # Apply LLM action hints to bias the teacher guidance
        if learn and action_hints and guidance is not None:
            if action_hints.get("prefer_cautious") and obs["sensors"][3] < 0.5:
                # Bias toward safer (slower, wider steering) actions
                cautious_actions = [0, 1, 5, 6]  # strong steer + decel
                if guidance not in cautious_actions and random.random() < 0.3:
                    guidance = random.choice(cautious_actions)
            elif action_hints.get("prefer_aggressive") and obs["sensors"][3] > 0.6:
                # Bias toward faster, more direct actions
                aggressive_actions = [2, 3, 4]  # accelerate + mild steer
                if guidance not in aggressive_actions and random.random() < 0.25:
                    guidance = random.choice(aggressive_actions)

        action = agent.choose_action(state, explore=learn, preferred_action=guidance)

        # Use reward-shaped step if modifiers are provided
        if reward_modifiers:
            next_obs, reward, done, info = env.step_with_reward_shaping(action, reward_modifiers)
        else:
            next_obs, reward, done, info = env.step(action)



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
        collision_count=env.collision_count,
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
        reward_modifiers_used=reward_modifiers is not None,
        action_hints_used=action_hints is not None,
    )


# ─── TRAINING HELPERS ────────────────────────────────────────────────────────
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
