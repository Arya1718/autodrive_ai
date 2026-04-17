"""Quick validation test for the autodrive system."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from simulation_core import SimulationConfig, sanitize_weather, run_episode
from adaptive_ai import AdaptiveDirector
from rl_agent import RLAgent

print("=== Test 1: Weather Sanitization ===")
for w in ["clear", "rainy", "foggy", "overcast", "sunny", "stormy", "hazy"]:
    print(f"  '{w}' -> '{sanitize_weather(w)}'")

print("\n=== Test 2: Config Auto-Sanitization ===")
cfg = SimulationConfig(weather="overcast", track_complexity="hard", obstacle_count=25, car_speed=10.0)
print(f"  Weather: {cfg.weather} (expect foggy)")
print(f"  Complexity: {cfg.track_complexity} (expect complex)")
print(f"  Obstacles: {cfg.obstacle_count} (expect 18)")
print(f"  Speed: {cfg.car_speed} (expect 4.5)")

print("\n=== Test 3: Run Episode ===")
agent = RLAgent()
cfg = SimulationConfig(weather="clear", track_complexity="simple", obstacle_count=4, car_speed=2.5)
ep = run_episode(agent, cfg, learn=True)
print(f"  Score: {ep['score']}, Collision: {ep['collision']}, Goal: {ep['reached_goal']}, Steps: {ep['steps']}")
print(f"  Waypoint progress: {ep.get('directional_snapshot', {}).get('goal_distance', '?')}")

print("\n=== Test 4: Reward Modifiers ===")
mods = {"goal_bonus": 15, "progress_bonus": 2.5, "speed_bonus": 0.5, "survival_bonus": 1.0, "smooth_steering_bonus": 0.5}
ep2 = run_episode(agent, cfg, learn=True, reward_modifiers=mods)
print(f"  Shaped Score: {ep2['score']}, Collision: {ep2['collision']}, Goal: {ep2['reached_goal']}")

print("\n=== Test 5: Curriculum Plan ===")
director = AdaptiveDirector()
target_cfg = SimulationConfig(weather="clear", track_complexity="moderate", obstacle_count=8, car_speed=2.8)
plan = director.generate_curriculum_plan(target_cfg, total_episodes=15)
for i, phase in enumerate(plan.get("phases", []), 1):
    print(f"  Phase {i}: {phase['phase_name']} | {phase['episodes']} eps | obs={phase['obstacle_count']} | spd={phase['car_speed']} | {phase['track_complexity']} | {phase['weather']}")

print("\n=== Test 6: Standard vs Adaptive (15 eps each) ===")
agent_std = RLAgent()
agent_adp = RLAgent()

# Standard: fixed moderate config
std_cfg = SimulationConfig(weather="clear", track_complexity="moderate", obstacle_count=8, car_speed=2.8)
std_goals, std_crashes, std_scores = 0, 0, []
for i in range(15):
    ep = run_episode(agent_std, std_cfg, learn=True)
    agent_std.adapt_after_episode(ep)
    if ep["reached_goal"]: std_goals += 1
    if ep["collision"]: std_crashes += 1
    std_scores.append(ep["score"])
    if (i + 1) % 5 == 0:
        print(f"    Standard ep {i+1}: score={ep['score']:.1f}, goal={ep['reached_goal']}, steps={ep['steps']}, teacher_blend={agent_std.teacher_blend:.2f}")
agent_std.decay()

# Adaptive: curriculum (easy -> medium -> hard) with reward shaping
phases = plan["phases"]
adp_goals, adp_crashes, adp_scores = 0, 0, []
reward_mods = {"goal_bonus": 20, "progress_bonus": 3, "speed_bonus": 0.5, "survival_bonus": 1.5, "smooth_steering_bonus": 0.8}
hints = {"prefer_cautious": True, "prefer_aggressive": False}

ep_num = 0
for phase in phases:
    phase_cfg = SimulationConfig(
        weather=phase["weather"],
        track_complexity=phase["track_complexity"],
        obstacle_count=phase["obstacle_count"],
        car_speed=phase["car_speed"],
    )
    for _ in range(phase["episodes"]):
        ep = run_episode(agent_adp, phase_cfg, learn=True, reward_modifiers=reward_mods, action_hints=hints)
        agent_adp.adapt_after_episode(ep)
        if ep["reached_goal"]: adp_goals += 1
        if ep["collision"]: adp_crashes += 1
        adp_scores.append(ep["score"])
        ep_num += 1
        if ep_num % 5 == 0:
            print(f"    Adaptive ep {ep_num}: score={ep['score']:.1f}, goal={ep['reached_goal']}, steps={ep['steps']}, phase={phase['phase_name']}, teacher_blend={agent_adp.teacher_blend:.2f}")
agent_adp.decay()

std_avg = sum(std_scores) / len(std_scores) if std_scores else 0
adp_avg = sum(adp_scores) / len(adp_scores) if adp_scores else 0
print(f"\n  Standard RL:  {std_goals} goals, {std_crashes} crashes, avg_score={std_avg:.1f} ({len(std_scores)} eps)")
print(f"  Adaptive RL:  {adp_goals} goals, {adp_crashes} crashes, avg_score={adp_avg:.1f} ({len(adp_scores)} eps)")
adaptive_better = (adp_goals > std_goals) or (adp_crashes < std_crashes) or (adp_avg > std_avg)
print(f"  Adaptive outperforms: {adaptive_better}")

print("\nAll tests complete!")
