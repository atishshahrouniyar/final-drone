#!/usr/bin/env python3
"""
LLM-guided evaluation loop for the RandomPointNav DQN policy.

The DQN handles low-level control while a lightweight Gemini planner produces
sub-goals every few steps. This script is meant for search-and-rescue
rehearsals where we want interpretable, waypoint-level guidance layered on top
of the trained policy.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pybullet as p
from stable_baselines3 import DQN

from mission_commander import LLMDecision, MissionCommanderLLM
from train_dqn import RandomPointNavEnv


def _unwrap_reset(result):
    return result[0] if isinstance(result, tuple) else result


def _unwrap_step(result):
    if len(result) == 5:
        return result
    obs, reward, done, info = result
    return obs, reward, done, False, info


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.dist((float(a[0]), float(a[1]), float(a[2])), (float(b[0]), float(b[1]), float(b[2])))


def _sync_goal_marker(env: RandomPointNavEnv, goal: Sequence[float], current_pos: Sequence[float]) -> None:
    env.goal_pos = np.array(goal, dtype=np.float32)
    env.prev_dist = float(
        np.linalg.norm(env.goal_pos[:2] - np.array(current_pos[:2], dtype=np.float32))
    )
    p.resetBasePositionAndOrientation(
        env.goal_id,
        env.goal_pos.tolist(),
        [0, 0, 0, 1],
        physicsClientId=env.client,
    )


def _detect_human_pixels(image: np.ndarray) -> Tuple[bool, Tuple[int, int] | None, int]:
    """Simple color-threshold detection for hi-vis targets."""
    if image is None or image.size == 0:
        return False, None, 0
    rgb = image[:, :, :3]
    mask = (
        (rgb[:, :, 0] > 200)
        & (rgb[:, :, 1] > 80)
        & (rgb[:, :, 1] < 210)
        & (rgb[:, :, 2] < 140)
    )
    count = int(np.count_nonzero(mask))
    if count < 150:
        return False, None, count
    ys, xs = np.where(mask)
    center = (int(xs.mean()), int(ys.mean()))
    return True, center, count

@dataclass
class EpisodeSummary:
    reward: float
    steps: int
    status: str
    llm_calls: int
    heuristic_calls: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DQN policy with Gemini-guided waypoints.")
    parser.add_argument(
        "--model-path",
        default="download/runs/20251128_143337/dqn_random_point_nav_20251128_143337.zip",
        help="Path to the trained DQN model (.zip).",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=600, help="Max steps per episode.")
    parser.add_argument("--llm-interval", type=int, default=25, help="Steps between LLM refreshes.")
    parser.add_argument("--dist-threshold", type=float, default=0.6, help="Distance to sub-goal that triggers refresh.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run PyBullet in DIRECT mode (no GUI). Useful for batch evaluation.",
    )
    parser.add_argument(
        "--llm-model",
        default="models/gemini-2.0-flash-lite-001",
        help="Gemini model id used for waypoint planning (e.g., models/gemini-2.0-flash-lite-001).",
    )
    parser.add_argument(
        "--mission-story",
        default="Conduct a methodical search of the forest arena, sweeping each quadrant for survivors.",
        help="Natural-language brief that guides the LLM's search strategy.",
    )
    return parser.parse_args()


def run_episode(
    env: RandomPointNavEnv,
    model: DQN,
    commander: MissionCommanderLLM,
    episode_idx: int,
    max_steps: int,
    llm_interval: int,
    dist_threshold: float,
) -> EpisodeSummary:
    obs = _unwrap_reset(env.reset())
    episode_reward = 0.0
    steps = 0
    terminated = truncated = False
    current_subgoal: List[float] | None = None
    steps_since_llm = llm_interval  # force immediate request

    llm_calls = 0
    heuristic_calls = 0
    last_detection_report: str | None = None

    print(f"\n--- Episode {episode_idx} ---")

    while not (terminated or truncated) and steps < max_steps:
        pos, orn = p.getBasePositionAndOrientation(env.drone_id, physicsClientId=env.client)
        yaw = p.getEulerFromQuaternion(orn)[2]

        if hasattr(env.scene, "_update_camera"):
            try:
                env.scene._update_camera()
            except Exception:  # noqa: BLE001
                pass

        lidar_hits = env._get_lidar(pos, yaw)
        if (
            hasattr(env.scene, "_render_lidar_overlay")
            and getattr(env.scene, "show_lidar_overlay", False)
        ):
            try:
                env.scene._render_lidar_overlay(pos, yaw, lidar_hits)
            except Exception:  # noqa: BLE001
                pass
        sensor_report = (
            last_detection_report
            if last_detection_report
            else "No confirmed survivor detections yet; continue systematic sweep."
        )
        camera_frame = None
        if hasattr(env.scene, "_get_drone_camera_image"):
            try:
                camera_frame = env.scene._get_drone_camera_image()
            except Exception:  # noqa: BLE001
                camera_frame = None
        if camera_frame is not None:
            detected, center, pix_count = _detect_human_pixels(camera_frame)
            if detected:
                sensor_report = (
                    f"Camera spotted a high-visibility target near pixel {center} "
                    f"(mask count {pix_count}). Investigate this bearing."
                )
                last_detection_report = sensor_report
            elif last_detection_report:
                sensor_report = (
                    f"No new visual detection; last report: {last_detection_report}"
                )

        need_refresh = (
            current_subgoal is None
            or _distance(pos, current_subgoal) < dist_threshold
            or steps_since_llm >= llm_interval
        )

        if need_refresh:
            decision: LLMDecision = commander.get_subgoal(
                lidar_hits, pos, sensor_report, steps
            )
            current_subgoal = decision.subgoal
            steps_since_llm = 0
            if decision.source == "llm":
                llm_calls += 1
            else:
                heuristic_calls += 1

            print("\n============= NAV UPDATE =============")
            print(f"Source      : {decision.source.upper()}")
            print(f"Instruction : {decision.instruction}")
            print(f"Reason      : {decision.reason}")
            print(f"Subgoal xyz : {[round(v, 2) for v in current_subgoal]}")
            print("======================================\n")

            _sync_goal_marker(env, current_subgoal, pos)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = _unwrap_step(env.step(action))
        episode_reward += reward
        steps += 1
        steps_since_llm += 1
        time.sleep(0.01 if env.gui else 0.0)

    status = "SUCCESS" if terminated and not truncated else "TIMEOUT"
    print(f"Episode {episode_idx} -> Reward {episode_reward:.2f}, Steps {steps}, Status {status}")
    return EpisodeSummary(
        reward=episode_reward,
        steps=steps,
        status=status,
        llm_calls=llm_calls,
        heuristic_calls=heuristic_calls,
    )


def main() -> None:
    args = parse_args()
    print("=" * 72)
    print(f"Loading DQN checkpoint: {args.model_path}")
    print("=" * 72)

    try:
        model = DQN.load(args.model_path)
    except FileNotFoundError as exc:  # noqa: BLE001
        raise SystemExit(f"Model not found: {args.model_path}") from exc

    env = RandomPointNavEnv(gui=not args.headless)
    commander = MissionCommanderLLM(
        model_name=args.llm_model,
        arena_radius=getattr(env, "radius", 20.0),
        altitude_limits=(0.8, 2.0),
        max_step=5.0,
        mission_story=args.mission_story,
    )

    summaries: List[EpisodeSummary] = []
    try:
        for ep in range(1, args.episodes + 1):
            summaries.append(
                run_episode(
                    env=env,
                    model=model,
                    commander=commander,
                    episode_idx=ep,
                    max_steps=args.max_steps,
                    llm_interval=args.llm_interval,
                    dist_threshold=args.dist_threshold,
                )
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        env.close()

    if summaries:
        rewards = [s.reward for s in summaries]
        lengths = [s.steps for s in summaries]
        llm_calls = sum(s.llm_calls for s in summaries)
        heur_calls = sum(s.heuristic_calls for s in summaries)

        print("\n" + "=" * 72)
        print("Evaluation summary")
        print("=" * 72)
        print(f"Episodes run     : {len(summaries)}")
        print(f"Avg reward       : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Avg length       : {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        print(f"Best / Worst     : {max(rewards):.2f} / {min(rewards):.2f}")
        print(f"LLM decisions    : {llm_calls}")
        print(f"Heuristic backup : {heur_calls}")
        print("Statuses         : " + ", ".join(s.status for s in summaries))
        print("=" * 72)


if __name__ == "__main__":
    main()
