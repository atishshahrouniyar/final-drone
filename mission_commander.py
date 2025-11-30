"""
Mission Commander – Gemini-guided search-and-rescue planner.

This module turns the latest sensor snapshot into a structured prompt for a
Gemini model (default: `models/gemini-2.0-flash-lite-001`), parses the JSON
reply, and emits a sub-goal that the DQN policy can chase. A lightweight
heuristic fallback keeps the drone moving even when the API is unavailable.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import google.generativeai as genai


@dataclass
class LLMDecision:
    """Single LLM (or heuristic) navigation update."""

    subgoal: List[float]
    instruction: str
    reason: str
    source: str  # "llm" or "heuristic"


class MissionCommanderLLM:
    """
    Thin wrapper around Google's Gemini API for high-level waypoint planning.

    The class expects a `GOOGLE_API_KEY` environment variable (from Google AI
    Studio). If the key is missing or a request fails, we fall back to a
    deterministic heuristic so the hybrid loop never stalls mid-mission.
    """

    def __init__(
        self,
        model_name: str = "models/gemini-2.0-flash-lite-001",
        arena_radius: float = 20.0,
        altitude_limits: Sequence[float] = (0.8, 2.0),
        max_step: float = 5.0,
        temperature: float = 0.2,
        mission_story: str = "Grid-search the forest clearing, prioritizing eastern quadrants.",
    ) -> None:
        self.model_name = self._normalize_model_name(model_name)
        self.arena_radius = arena_radius
        self.min_z, self.max_z = altitude_limits
        self.max_step = max_step
        self.temperature = temperature
        self.mission_story = mission_story

        token = os.environ.get("GOOGLE_API_KEY")
        if not token:
            self.client = None
            self.online = False
            print(
                "[MissionCommanderLLM] GOOGLE_API_KEY not set. "
                "Fallback heuristic guidance will be used."
            )
        else:
            genai.configure(api_key=token)
            try:
                self.client = genai.GenerativeModel(self.model_name)
                self.online = True
            except Exception as exc:  # noqa: BLE001
                print(f"[MissionCommanderLLM] Failed to init Gemini model: {exc}")
                self.client = None
                self.online = False

    def _normalize_model_name(self, requested: str) -> str:
        if not requested:
            return "gemini-1.5-flash"
        lower = requested.lower()
        if lower.startswith("models/"):
            return requested
        if lower.startswith("gemini"):
            return f"models/{requested}"
        print(
            f"[MissionCommanderLLM] '{requested}' is not a Gemini model id; "
            "defaulting to 'models/gemini-2.0-flash-lite-001'."
        )
        return "models/gemini-2.0-flash-lite-001"

    # --------------------------------------------------------------------- API
    def get_subgoal(
        self,
        lidar: Sequence[float],
        drone_pos: Sequence[float],
        sensor_report: str,
        step_idx: int,
    ) -> LLMDecision:
        """
        Ask the LLM (or fallback heuristic) for the next subgoal.

        Args:
            lidar: Normalised lidar readings (0..1 fractions).
            drone_pos: Current xyz position.
            sensor_report: Textual summary from onboard sensors/camera.
            step_idx: Simulation step (only used for logging / randomness).
        """
        if self.online and self.client is not None:
            try:
                sys_prompt = self._build_system_prompt()
                user_prompt = self._build_user_prompt(lidar, drone_pos, sensor_report)
                raw = self._call_llm(sys_prompt, user_prompt)
                decision = self._parse_response(raw, drone_pos)
                return decision
            except Exception as exc:  # noqa: BLE001
                print(f"[MissionCommanderLLM] LLM request failed: {exc}")

        # Heuristic fallback (no token or API failure)
        return self._heuristic_decision(lidar, drone_pos, step_idx)

    # ----------------------------------------------------------------- helpers
    def _build_system_prompt(self) -> str:
        return (
            "You are Mission Commander, a high-level navigator for a rescue drone. "
            "The drone already has a low-level controller; you only provide the next "
            "sub-goal. Follow these rules strictly:\n"
            f"- Keep the drone within ±{self.arena_radius} meters on X/Y.\n"
            f"- Keep altitude between {self.min_z:.1f} and {self.max_z:.1f} meters.\n"
            f"- Limit each hop to {self.max_step:.1f} meters from the current position.\n"
            "- If survivors are known, move toward the nearest before exploring.\n"
            '- Respond ONLY with JSON: {"instruction": "...", "subgoal": [x,y,z], "reason": "..."}\n'
            "No markdown, code fences, or additional prose."
        )

    def _build_user_prompt(
        self,
        lidar: Sequence[float],
        drone_pos: Sequence[float],
        sensor_report: str,
    ) -> str:
        lidar_preview = ", ".join(f"{float(v):.2f}" for v in lidar[:12])
        return (
            "Mission Commander, reply with strict JSON using keys "
            'instruction, subgoal, reason. No markdown.\n'
            f"Mission story / objective: {self.mission_story}\n"
            f"Current drone position (meters): {list(map(lambda v: round(float(v), 3), drone_pos))}\n"
            f"Sensor report: {sensor_report}\n"
            f"Lidar readings (first 12 of {len(lidar)}): [{lidar_preview} ...]\n"
            f"Full lidar list: {list(map(lambda v: round(float(v), 3), lidar))}\n"
            "Provide the JSON object now."
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        if self.client is None:
            raise RuntimeError("LLM client not initialised")
        merged_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self.client.generate_content(
            merged_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                top_p=0.9,
                max_output_tokens=220,
            ),
        )
        text_parts = []
        for part in response.parts or []:
            if hasattr(part, "text"):
                text_parts.append(part.text)
        return "\n".join(text_parts).strip()

    def _parse_response(self, raw: str, drone_pos: Sequence[float]) -> LLMDecision:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
            cleaned = cleaned.rstrip("`").strip()
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match:
            cleaned = match.group(0)
        try:
            payload: Dict[str, Any] = json.loads(cleaned)
        except json.JSONDecodeError as exc:  # noqa: BLE001
            raise ValueError(f"Invalid JSON from LLM: {cleaned}") from exc

        instruction = str(payload.get("instruction", "Move cautiously.")).strip()
        reason = str(payload.get("reason", "LLM guidance")).strip()
        subgoal = self._coerce_subgoal(payload.get("subgoal"), drone_pos)
        subgoal = self._clamp_subgoal(subgoal, drone_pos)
        return LLMDecision(subgoal=subgoal, instruction=instruction, reason=reason, source="llm")

    def _coerce_subgoal(
        self, candidate: Any, drone_pos: Sequence[float]
    ) -> List[float]:
        if (
            isinstance(candidate, (list, tuple))
            and len(candidate) == 3
            and all(isinstance(v, (int, float)) for v in candidate)
        ):
            return [float(candidate[0]), float(candidate[1]), float(candidate[2])]

        # Fallback: step forward along +X
        return [float(drone_pos[0]) + 1.0, float(drone_pos[1]), float(drone_pos[2])]

    def _clamp_subgoal(
        self, subgoal: Sequence[float], drone_pos: Sequence[float]
    ) -> List[float]:
        gx, gy, gz = map(float, subgoal)

        # Limit hop distance
        dx = gx - float(drone_pos[0])
        dy = gy - float(drone_pos[1])
        dz = gz - float(drone_pos[2])
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist > self.max_step > 0:
            scale = self.max_step / dist
            gx = float(drone_pos[0]) + dx * scale
            gy = float(drone_pos[1]) + dy * scale
            gz = float(drone_pos[2]) + dz * scale

        # Keep inside arena cylinder
        r = math.sqrt(gx * gx + gy * gy)
        if r > self.arena_radius:
            factor = self.arena_radius / max(r, 1e-6)
            gx *= factor
            gy *= factor

        gz = max(self.min_z, min(self.max_z, gz))
        return [gx, gy, gz]

    # ---------------------------------------------------------- heuristics
    def _heuristic_decision(
        self,
        lidar: Sequence[float],
        drone_pos: Sequence[float],
        step_idx: int,
    ) -> LLMDecision:
        """Simple rule-based navigator as a safety net."""
        drone_xy = (float(drone_pos[0]), float(drone_pos[1]))
        drone_z = float(drone_pos[2])

        # Pick a safe direction based on lidar gaps
        safest_idx = self._select_safest_ray(lidar)
        angle = 2 * math.pi * safest_idx / max(len(lidar), 1)
        step = self.max_step
        target = [
            drone_xy[0] + step * math.cos(angle),
            drone_xy[1] + step * math.sin(angle),
            drone_z,
        ]
        # Add a slow rotation over time to avoid stagnation
        jitter = (step_idx % 60) / 60.0 * math.pi / 6.0
        target[0] += math.cos(angle + jitter) * 0.5
        target[1] += math.sin(angle + jitter) * 0.5
        target = self._clamp_subgoal(target, drone_pos)
        return LLMDecision(
            subgoal=target,
            instruction="Explore the safest open corridor while scanning for survivors.",
            reason="Heuristic sweep",
            source="heuristic",
        )

    def _select_safest_ray(self, lidar: Sequence[float]) -> int:
        if not lidar:
            return 0
        max_value = max(lidar)
        candidates = [i for i, v in enumerate(lidar) if math.isclose(v, max_value, abs_tol=1e-2)]
        return random.choice(candidates)
