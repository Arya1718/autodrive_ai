import json, os, re
from groq import Groq

class GroqAgent:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

    def _chat(self, system: str, user: str, max_tokens=400, temp=0.7) -> str:
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                temperature=temp, max_tokens=max_tokens
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            return f"[API Error: {e}]"

    def _extract_json(self, text: str) -> dict | None:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        return None

    def generate_environment(self, description: str) -> dict:
        system = """You are an AI that generates self-driving car simulation configs.
Output ONLY valid JSON with these exact fields:
{
  "obstacle_count": <int 3-18>,
  "car_speed": <float 1.5-4.0>,
  "environment_name": <string>,
  "weather": <"clear"|"foggy"|"rainy">,
  "track_complexity": <"simple"|"moderate"|"complex">,
  "fog_alpha": <float 0.0-0.5>,
  "description": <string 1-2 sentences>
}
Output only the JSON object, nothing else."""
        result = self._chat(system, f"Generate config for: {description}", max_tokens=350, temp=0.7)
        cfg = self._extract_json(result)
        if cfg:
            cfg.setdefault("obstacle_count", 8)
            cfg.setdefault("car_speed", 2.5)
            cfg.setdefault("fog_alpha", 0.0)
            return cfg
        return {
            "obstacle_count": 8, "car_speed": 2.5, "environment_name": "Urban Circuit",
            "weather": "clear", "track_complexity": "moderate", "fog_alpha": 0.0,
            "description": "A standard urban circuit with mixed obstacles."
        }

    def analyze_failure(self, crash_data: dict) -> str:
        system = """You are an expert autonomous vehicle safety analyst.
Given crash telemetry, provide a focused 3-sentence causal analysis:
1. Primary cause  2. Contributing sensor/speed factor  3. Recommended fix.
Be technical and specific."""
        return self._chat(system, f"Crash telemetry: {json.dumps(crash_data)}", max_tokens=220, temp=0.4)

    def get_adaptive_config(self, perf: dict) -> dict:
        system = """You are an adaptive training system.
Output ONLY a JSON:
{"obstacle_count":<int>,"car_speed":<float>,"difficulty_label":<string>,"reasoning":<string>}"""
        result = self._chat(system, f"Performance data: {json.dumps(perf)}", max_tokens=150, temp=0.3)
        cfg = self._extract_json(result)
        if cfg:
            return cfg
        score = perf.get("score", 0)
        crashes = perf.get("crashes", 0)
        if crashes > 5:
            return {"obstacle_count": 5, "car_speed": 2.0, "difficulty_label": "Easy", "reasoning": "High crash rate — reducing obstacles."}
        elif score > 200:
            return {"obstacle_count": 12, "car_speed": 3.2, "difficulty_label": "Hard", "reasoning": "Strong performance — increasing challenge."}
        return {"obstacle_count": 8, "car_speed": 2.5, "difficulty_label": "Medium", "reasoning": "Balanced performance."}

    def explain_sensor_state(self, sensors: list, speed: float) -> str:
        s = ["L90","L50","L25","FWD","R25","R50","R90"]
        reading = ", ".join(f"{s[i]}={v:.2f}" for i, v in enumerate(sensors))
        system = "You are a self-driving car AI. In one sentence (max 20 words), explain what the sensor readings mean for the car's next action."
        return self._chat(system, f"Speed={speed:.1f} Sensors: {reading}", max_tokens=60, temp=0.6)
