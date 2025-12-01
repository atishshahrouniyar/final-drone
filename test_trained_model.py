#!/usr/bin/env python3
"""
Visualize the trained random-point navigation policy with PyBullet GUI.

Usage:
    python test_trained_model.py [model_path] [num_episodes] [max_steps]
Defaults:
    model_path      = "runs/20251130_172849/checkpoints/best_model.zip"
    num_episodes    = 5
    max_steps       = 2000
"""

import sys
import time
from typing import Tuple

import numpy as np
import pybullet as p
from stable_baselines3 import DQN
from gymnasium.wrappers import TimeLimit

from train_dqn import RandomPointNavEnv


def _detect_human_pixels(image: np.ndarray):
    """Simple color-threshold detection for hi-vis human props."""
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


def _unwrap_reset(obs_result):
    """Handle both Gymnasium (obs, info) and Gym (obs) reset signatures."""
    if isinstance(obs_result, tuple):
        return obs_result[0]
    return obs_result


def _unwrap_step(step_result: Tuple):
    """
    Normalize step outputs to (obs, reward, terminated, truncated, info)
    regardless of Gym/Gymnasium version.
    """
    if len(step_result) == 5:
        return step_result
    obs, reward, done, info = step_result
    return obs, reward, done, False, info


def test_model(
    model_path: str = "dqn_random_point_nav",
    num_episodes: int = 5,
    max_steps_per_episode: int = 400,
):
    print("=" * 70)
    print(f"Loading trained model from: {model_path}")
    print("=" * 70)

    try:
        model = DQN.load(model_path)
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print(f"✗ Model file not found: {model_path}.zip")
        return

    print("\nCreating GUI environment...")
    env = RandomPointNavEnv(gui=True)
    env = TimeLimit(env, max_episode_steps=2000)  # Match training setup
    episode_rewards = []
    episode_lengths = []
    
    # Get reference to the base environment for accessing scene
    base_env = env.unwrapped

    try:
        for episode in range(1, num_episodes + 1):
            obs = _unwrap_reset(env.reset())

            start = getattr(base_env, "start_pos", None)
            goal = getattr(base_env, "goal_pos", None)
            if start is not None and goal is not None:
                print(f"Start: ({start[0]:.2f}, {start[1]:.2f})  ->  Goal: ({goal[0]:.2f}, {goal[1]:.2f})")
            if hasattr(base_env.scene, "_update_camera"):
                base_env.scene._update_camera()

            ep_reward = 0.0
            steps = 0
            terminated = truncated = False
            human_spotted = False

            print(f"\n--- Episode {episode}/{num_episodes} ---")

            while not (terminated or truncated) and steps < max_steps_per_episode:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = _unwrap_step(env.step(action))

                if hasattr(base_env.scene, "_update_camera"):
                    base_env.scene._update_camera()
                if hasattr(base_env.scene, "client") and base_env.scene.client is not None:
                    try:
                        pos, orn = p.getBasePositionAndOrientation(base_env.scene.drone_id, physicsClientId=base_env.scene.client)
                        yaw = p.getEulerFromQuaternion(orn)[2]
                        lidar_hits = base_env._get_lidar(pos, yaw)
                        base_env.scene._render_lidar_overlay(pos, yaw, lidar_hits)
                    except Exception:
                        pass

                camera_frame = None
                if hasattr(base_env.scene, "_get_drone_camera_image"):
                    camera_frame = base_env.scene._get_drone_camera_image()
                detected, center, pix_count = _detect_human_pixels(camera_frame)
                if detected and not human_spotted:
                    print(f"  ✓ Human detected through camera at pixels {center} (mask={pix_count})")
                    human_spotted = True

                ep_reward += reward
                steps += 1
                time.sleep(0.01)

            episode_rewards.append(ep_reward)
            episode_lengths.append(steps)
            status = "SUCCESS" if terminated and not truncated else "TIMEOUT"
            print(f"Reward: {ep_reward:.2f} | Steps: {steps} | Status: {status}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        if episode_rewards:
            print("\n" + "=" * 70)
            print("Episode statistics")
            print("=" * 70)
            print(f"Episodes run: {len(episode_rewards)}")
            print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
            print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
            print(f"Best reward: {np.max(episode_rewards):.2f}")
            print(f"Worst reward: {np.min(episode_rewards):.2f}")

        print("\nClosing environment...")
        env.close()
        print("Done.")


if __name__ == "__main__":
    # Default to latest best model from training
    model_path = sys.argv[1] if len(sys.argv) > 1 else "runs/20251130_172849/checkpoints/best_model.zip"
    num_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 2000  # Match training limit

    test_model(model_path=model_path, num_episodes=num_eps, max_steps_per_episode=max_steps)

