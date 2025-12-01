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
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
from stable_baselines3 import DQN

from mission_commander import LLMDecision, MissionCommanderLLM, WaypointPlan
from train_dqn import RandomPointNavEnv


def _unwrap_reset(result):
    return result[0] if isinstance(result, tuple) else result


def _unwrap_step(result):
    if len(result) == 5:
        return result
    obs, reward, done, info = result
    return obs, reward, done, False, info


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    """3D Euclidean distance."""
    return math.dist((float(a[0]), float(a[1]), float(a[2])), (float(b[0]), float(b[1]), float(b[2])))


def _distance_2d(a: Sequence[float], b: Sequence[float]) -> float:
    """2D horizontal distance, ignoring altitude."""
    return math.sqrt((float(a[0]) - float(b[0]))**2 + (float(a[1]) - float(b[1]))**2)


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


def _estimate_detection_goal(
    drone_pos: Sequence[float],
    drone_yaw: float,
    pixel: Tuple[int, int],
    camera_width: int,
    fov_deg: float,
    distance: float = 4.0,
) -> List[float]:
    """Project a camera detection into world space roughly in front of the drone."""
    if camera_width <= 0 or fov_deg <= 0:
        return [float(drone_pos[0]), float(drone_pos[1]), float(drone_pos[2])]
    offset_norm = ((pixel[0] / max(camera_width, 1)) - 0.5) * 2.0
    yaw_offset = math.radians(fov_deg) * 0.5 * offset_norm
    heading = drone_yaw + yaw_offset
    return [
        float(drone_pos[0]) + distance * math.cos(heading),
        float(drone_pos[1]) + distance * math.sin(heading),
        max(0.5, float(drone_pos[2])),
    ]


class PathVisualizer:
    """Draws the flown path inside PyBullet."""

    def __init__(self, client_id: int):
        self.client_id = client_id
        self.prev_pos: Tuple[float, float, float] | None = None
        self.line_handles: List[int] = []

    def update(self, pos: Sequence[float]) -> None:
        point = (float(pos[0]), float(pos[1]), float(pos[2]))
        if self.prev_pos is not None:
            try:
                handle = p.addUserDebugLine(
                    self.prev_pos,
                    point,
                    [0.1, 0.7, 1.0],
                    lineWidth=2.5,
                    lifeTime=0,
                    physicsClientId=self.client_id,
                )
                self.line_handles.append(handle)
            except Exception:
                pass
        self.prev_pos = point

    def clear(self) -> None:
        for handle in self.line_handles:
            try:
                p.removeUserDebugItem(handle, physicsClientId=self.client_id)
            except Exception:
                pass
        self.line_handles.clear()
        self.prev_pos = None


def _spawn_detection_marker(client_id: Optional[int], position: Sequence[float]) -> List[int]:
    """Draw a crosshair marker around a suspected survivor location."""
    if client_id is None:
        return []
    handles = []
    offsets = [
        ((0.5, 0, 0), (-0.5, 0, 0)),
        ((0, 0.5, 0), (0, -0.5, 0)),
        ((0, 0, 0.5), (0, 0, -0.5)),
    ]
    for start_off, end_off in offsets:
        start = [
            position[0] + start_off[0],
            position[1] + start_off[1],
            position[2] + start_off[2],
        ]
        end = [
            position[0] + end_off[0],
            position[1] + end_off[1],
            position[2] + end_off[2],
        ]
        try:
            handle = p.addUserDebugLine(
                start,
                end,
                [1.0, 0.2, 0.2],
                lineWidth=3.0,
                lifeTime=0,
                physicsClientId=client_id,
            )
            handles.append(handle)
        except Exception:
            pass
    return handles


def _clear_debug_items(client_id: Optional[int], handles: List[int]) -> None:
    if client_id is None:
        return
    for handle in handles:
        try:
            p.removeUserDebugItem(handle, physicsClientId=client_id)
        except Exception:
            pass


class SearchCoverageTracker:
    """Coarse coverage map plus path logger."""

    def __init__(self) -> None:
        self.quadrant_counts: dict[str, int] = {"NE": 0, "NW": 0, "SE": 0, "SW": 0}
        self.path_trace: List[Tuple[float, float, float]] = []

    def update(self, pos: Sequence[float]) -> None:
        x, y, z = map(float, pos[:3])
        quad = self._quadrant_label(x, y)
        self.quadrant_counts[quad] += 1
        self.path_trace.append((x, y, z))

    def summary(self) -> str:
        counts = ", ".join(f"{q}:{c}" for q, c in self.quadrant_counts.items())
        return f"Sector coverage -> {counts}."

    @staticmethod
    def _quadrant_label(x: float, y: float) -> str:
        if x >= 0 and y >= 0:
            return "NE"
        if x < 0 and y >= 0:
            return "NW"
        if x >= 0 and y < 0:
            return "SE"
        return "SW"

@dataclass
class EpisodeSummary:
    reward: float
    steps: int
    status: str
    llm_calls: int
    heuristic_calls: int
    path_trace: List[Tuple[float, float, float]]
    waypoints_reached: int = 0
    detection_attempts: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DQN policy with Gemini-guided waypoints.")
    parser.add_argument(
        "--model-path",
        default="runs/20251130_165117/checkpoints/dqn_random_point_nav_500k.zip",
        help="Path to the trained DQN model (.zip).",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional cap on steps per episode (0 = no limit).",
    )
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
        default="Search North-East quadrant for survivors.",
        help="Natural-language brief that guides the LLM's search strategy.",
    )
    parser.add_argument(
        "--plan-waypoints",
        type=int,
        default=20,
        help="Number of waypoints to request per batch from the planner.",
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
    plan_waypoints: int,
) -> EpisodeSummary:
    
    # Determine target sector from story (naive parsing for demo)
    target_sector = None
    story_upper = commander.mission_story.upper()
    if "NORTH-EAST" in story_upper or "NORTHEAST" in story_upper: target_sector = "NE"
    elif "NORTH-WEST" in story_upper or "NORTHWEST" in story_upper: target_sector = "NW"
    elif "SOUTH-EAST" in story_upper or "SOUTHEAST" in story_upper: target_sector = "SE"
    elif "SOUTH-WEST" in story_upper or "SOUTHWEST" in story_upper: target_sector = "SW"

    obs = _unwrap_reset(env.reset(options={"center_start": True, "target_sector": target_sector}))
    episode_reward = 0.0
    steps = 0
    terminated = truncated = False
    current_subgoal: List[float] | None = None

    llm_calls = 0
    heuristic_calls = 0
    waypoints_reached = 0
    detection_attempts = 0
    last_detection_report: str | None = None
    coverage_tracker = SearchCoverageTracker()
    client_id = getattr(env, "client", getattr(env.scene, "client", None))
    path_viz = PathVisualizer(client_id) if client_id is not None else None
    detection_goal: List[float] | None = None
    detection_markers: List[int] = []
    last_info: Dict[str, bool] = {}
    planned_queue: Deque[Tuple[List[float], str, int, int]] = deque()
    plan_batch_counter = 1

    def enqueue_plan(plan_obj: WaypointPlan) -> None:
        nonlocal plan_batch_counter, llm_calls, heuristic_calls, planned_queue
        if not plan_obj.waypoints:
            return
        label = f"{plan_obj.source.upper()}-{plan_batch_counter}"
        total = len(plan_obj.waypoints)
        print("\n============= WAYPOINT PLAN =============")
        print(f"Label       : {label}")
        print(f"Source      : {plan_obj.source.upper()}")
        print(f"Waypoints   : {total}")
        print(f"Instruction : {plan_obj.instruction}")
        print(f"Reason      : {plan_obj.reason}")
        for idx, wp in enumerate(plan_obj.waypoints, start=1):
            rounded = [round(v, 2) for v in wp]
            print(f"  - {idx:02d}/{total}: {rounded}")
        print("=========================================\n")
        for idx, wp in enumerate(plan_obj.waypoints, start=1):
            planned_queue.append((list(wp), label, idx, total))
        if plan_obj.source == "llm":
            llm_calls += 1
        # Heuristic tracking removed
        plan_batch_counter += 1

    print(f"\n--- Episode {episode_idx} ---")

    pos0, orn0 = p.getBasePositionAndOrientation(env.drone_id, physicsClientId=env.client)
    yaw0 = p.getEulerFromQuaternion(orn0)[2]
    initial_lidar = env._get_lidar(pos0, yaw0)
    initial_report = "Initial sweep: no detections yet; awaiting plan."
    try:
        initial_plan = commander.get_waypoint_plan(
            initial_lidar,
            pos0,
            initial_report,
            max_points=plan_waypoints,
            allow_llm=True,
        )
        enqueue_plan(initial_plan)
    except RuntimeError as e:
        print(f"Initial planning failed: {e}")

    max_steps_limit = max_steps if max_steps > 0 else None
    step_limit_reached = False

    while not (terminated or truncated) and (max_steps_limit is None or steps < max_steps_limit):
        pos, orn = p.getBasePositionAndOrientation(env.drone_id, physicsClientId=env.client)
        yaw = p.getEulerFromQuaternion(orn)[2]
        coverage_tracker.update(pos)
        if path_viz is not None:
            path_viz.update(pos)

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
        coverage_summary = coverage_tracker.summary()
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
                detection_attempts += 1
                detection_text = (
                    f"Camera spotted a high-visibility target near pixel {center} "
                    f"(mask count {pix_count}). Investigate this bearing."
                )
                sensor_report = detection_text
                last_detection_report = sensor_report
                camera_width = int(getattr(env.scene, "drone_camera_width", camera_frame.shape[1]))
                camera_fov = float(getattr(env.scene, "drone_camera_fov", 80.0))
                detection_goal = _estimate_detection_goal(pos, yaw, center, camera_width, camera_fov)
                _clear_debug_items(client_id, detection_markers)
                detection_markers = _spawn_detection_marker(client_id, detection_goal)
            elif last_detection_report:
                sensor_report = f"No new visual detection; last report: {last_detection_report}"

        sensor_report = f"{coverage_summary} {sensor_report}"

        # Check if we've reached current planned waypoint (even if chasing detection)
        if current_subgoal is not None and _distance_2d(pos, current_subgoal) < dist_threshold:
            print("Planned waypoint reached. Advancing to the next queued target.")
            waypoints_reached += 1
            current_subgoal = None
            # Pop next waypoint if available
            if planned_queue:
                waypoint, label, idx, total = planned_queue.popleft()
                current_subgoal = waypoint
                print(
                    f"[WaypointQueue:{label}] Heading to waypoint {idx}/{total}: "
                    f"{[round(v, 2) for v in current_subgoal]}"
                )
                _sync_goal_marker(env, current_subgoal, pos)

        if detection_goal is not None:
            if _distance(pos, detection_goal) < 0.5:
                print("Detection target reached. Scanning area for survivors...")
                detection_goal = None
                _clear_debug_items(client_id, detection_markers)
                detection_markers = []
            else:
                # Override current subgoal temporarily to chase detection
                temp_goal = detection_goal
                _sync_goal_marker(env, temp_goal, pos)
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = _unwrap_step(env.step(action))
                last_info = info or {}
                episode_reward += reward
                steps += 1
                time.sleep(0.01 if env.gui else 0.0)
                continue

        # Normal waypoint following (when no detection target)
        if current_subgoal is None:
            if not planned_queue:
                context = f"{coverage_summary} {sensor_report}"
                # Always allow LLM (heuristic fallback removed)
                try:
                    fallback_plan = commander.get_waypoint_plan(
                            lidar_hits,
                        pos,
                        context,
                        max_points=plan_waypoints,
                        allow_llm=True,
                    )
                    enqueue_plan(fallback_plan)
                except RuntimeError as e:
                    print(f"Planner failed: {e}. Using emergency waypoint.")
                    # Generate a simple forward waypoint as emergency fallback
                    emergency_wp = [
                        float(pos[0]) + 2.0 * math.cos(yaw),
                        float(pos[1]) + 2.0 * math.sin(yaw),
                        1.5
                    ]
                    # Clamp to arena bounds
                    r = math.sqrt(emergency_wp[0]**2 + emergency_wp[1]**2)
                    if r > 18.0:  # Stay within safe bounds
                        factor = 18.0 / max(r, 1e-6)
                        emergency_wp[0] *= factor
                        emergency_wp[1] *= factor
                    planned_queue.append((emergency_wp, "EMERGENCY", 1, 1))

            if planned_queue:
                waypoint, label, idx, total = planned_queue.popleft()
                current_subgoal = waypoint
                print(
                    f"[WaypointQueue:{label}] Heading to waypoint {idx}/{total}: "
                    f"{[round(v, 2) for v in current_subgoal]}"
                )
                _sync_goal_marker(env, current_subgoal, pos)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = _unwrap_step(env.step(action))
        last_info = info or {}
        episode_reward += reward
        steps += 1
        
        # Periodic status update
        if llm_interval > 0 and steps % llm_interval == 0 and current_subgoal is not None:
            dist_to_subgoal = _distance_2d(pos, current_subgoal)
            print(f"[Step {steps}] Position: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}], "
                  f"Distance to waypoint: {dist_to_subgoal:.2f}m, Reward: {episode_reward:.1f}")
        
        time.sleep(0.01 if env.gui else 0.0)

    if max_steps_limit is not None and steps >= max_steps_limit and not terminated and not truncated:
        step_limit_reached = True

    if step_limit_reached:
        status = "STEP_LIMIT"
    elif truncated:
        status = "TIMEOUT"
    elif terminated:
        if last_info.get("collision"):
            status = "COLLISION"
        elif last_info.get("human_found"):
            status = "HUMAN_FOUND"
        elif last_info.get("goal_reached"):
            status = "GOAL_REACHED"
        else:
            status = "SUCCESS"
    else:
        status = "RUNNING"
    print(f"Episode {episode_idx} -> Reward {episode_reward:.2f}, Steps {steps}, Status {status}")
    return EpisodeSummary(
        reward=episode_reward,
        steps=steps,
        status=status,
        llm_calls=llm_calls,
        heuristic_calls=heuristic_calls,
        path_trace=coverage_tracker.path_trace,
        waypoints_reached=waypoints_reached,
        detection_attempts=detection_attempts,
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
                    plan_waypoints=args.plan_waypoints,
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
        total_waypoints = sum(s.waypoints_reached for s in summaries)
        total_detections = sum(s.detection_attempts for s in summaries)

        print("\n" + "=" * 72)
        print("Evaluation summary")
        print("=" * 72)
        print(f"Episodes run     : {len(summaries)}")
        print(f"Avg reward       : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Avg length       : {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        print(f"Best / Worst     : {max(rewards):.2f} / {min(rewards):.2f}")
        print(f"LLM decisions    : {llm_calls}")
        print(f"Heuristic backup : {heur_calls}")
        print(f"Waypoints reached: {total_waypoints} (avg {total_waypoints/max(len(summaries),1):.1f} per episode)")
        print(f"Visual detections: {total_detections}")
        print("Statuses         : " + ", ".join(s.status for s in summaries))
        
        # Success rate tracking
        success_count = sum(1 for s in summaries if s.status == "HUMAN_FOUND")
        collision_count = sum(1 for s in summaries if s.status == "COLLISION")
        print(f"\nSuccess Rate     : {success_count}/{len(summaries)} ({100*success_count/max(len(summaries),1):.1f}%)")
        print(f"Collision Rate   : {collision_count}/{len(summaries)} ({100*collision_count/max(len(summaries),1):.1f}%)")
        
        print("=" * 72)


if __name__ == "__main__":
    main()
