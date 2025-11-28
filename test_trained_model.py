#!/usr/bin/env python3
"""
Visualize the trained random-point navigation policy with PyBullet GUI.

Usage:
    python test_trained_model.py [model_path] [num_episodes] [max_steps]
Defaults:
    model_path      = "dqn_random_point_nav"
    num_episodes    = 5
    max_steps       = 400
"""

import sys
import time
from typing import Tuple

import numpy as np
from stable_baselines3 import DQN

from train_dqn import RandomPointNavEnv, USE_GYMNASIUM_API


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
    except Exception as exc:
        print(f"✗ Failed to load model: {exc}")
        return

    print("\nCreating GUI environment...")
    env = RandomPointNavEnv(gui=True)
    episode_rewards = []
    episode_lengths = []

    try:
        for episode in range(1, num_episodes + 1):
            obs = _unwrap_reset(env.reset())

            start = getattr(env, "start_pos", None)
            goal = getattr(env, "goal_pos", None)
            if start is not None and goal is not None:
                print(f"Start: ({start[0]:.2f}, {start[1]:.2f})  ->  Goal: ({goal[0]:.2f}, {goal[1]:.2f})")
            if hasattr(env.scene, "_update_camera"):
                env.scene._update_camera()

            ep_reward = 0.0
            steps = 0
            terminated = truncated = False

            print(f"\n--- Episode {episode}/{num_episodes} ---")

            while not (terminated or truncated) and steps < max_steps_per_episode:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = _unwrap_step(env.step(action))

                if hasattr(env.scene, "_update_camera"):
                    env.scene._update_camera()

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
    model_path = sys.argv[1] if len(sys.argv) > 1 else "dqn_random_point_nav_20251128_081136"
    num_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 4000

    test_model(model_path=model_path, num_episodes=num_eps, max_steps_per_episode=max_steps)

