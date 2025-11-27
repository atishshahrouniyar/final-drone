"""
Train a DQN agent to move a drone between two random points
inside a 40x40 PyBullet environment while avoiding obstacles.
"""

import math
import random
import time
from typing import Tuple

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from enhanced_navigation_env import EnhancedForestWithObstacles


# --------------------------- configuration ------------------------------------
TOTAL_TIMESTEPS = 5_000_000
LEARNING_RATE = 2e-4
BUFFER_SIZE = 200_000
BATCH_SIZE = 128
GAMMA = 0.99
TARGET_UPDATE_INTERVAL = 500
ACTION_REPEAT = 4
ARENA_RADIUS = 20.0  # 40 x 40 meters
GOAL_MIN_DIST = 3.0
GOAL_MAX_DIST = 5.0
SUCCESS_THRESHOLD = 0.5
PROXIMITY_THRESHOLD = 1.0
PROXIMITY_PENALTY = -0.5
COLLISION_PENALTY = -5.0
SUCCESS_REWARD = 20.0


class TrainingLogger(BaseCallback):
    """Lightweight logger for monitoring learning speed."""

    def __init__(self, check_freq: int = 5000):
        super().__init__()
        self.check_freq = check_freq
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            elapsed = max(time.time() - self.start_time, 1e-6)
            fps = self.num_timesteps / elapsed
            self.logger.record("custom/fps", fps)
        return True


class RandomPointNavEnv(gym.Env):
    """Single-drone navigation environment with random start/goal pairs."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, gui: bool = False):
        super().__init__()
        self.gui = gui

        # Build PyBullet scene (40x40 forest arena already configured)
        self.scene = EnhancedForestWithObstacles(gui=gui, radius=ARENA_RADIUS)
        self.client = self.scene.client
        self.drone_id = self.scene.drone_id
        self.radius = self.scene.radius

        # Create a goal marker
        goal_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.4, rgbaColor=[1.0, 0.1, 0.1, 0.8], physicsClientId=self.client
        )
        self.goal_id = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=goal_vis, basePosition=[0, 0, 1.0], physicsClientId=self.client
        )

        # Action space: hover, ±X, ±Y, ±Z
        self.action_space = spaces.Discrete(7)
        self.action_map = {
            0: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            1: np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            2: np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            3: np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            4: np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float32),
            5: np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
            6: np.array([0.0, 0.0, -1.0, 1.0], dtype=np.float32),
        }
        self.linear_speed = 3.0

        # Observation space: lidar(36) + goal distance + goal angle
        self.lidar_rays = 36
        self.lidar_range = 8.0
        low = np.array([0.0] * self.lidar_rays + [0.0, -np.pi], dtype=np.float32)
        high = np.array([1.0] * self.lidar_rays + [1.0, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.goal_pos = np.zeros(3, dtype=np.float32)
        self.prev_dist = None

    # ------------------------------------------------------------------ helpers
    def _sample_valid_point(self, margin: float = 2.0) -> Tuple[float, float]:
        for _ in range(100):
            r = random.uniform(margin, self.radius - margin)
            ang = random.uniform(0, 2 * math.pi)
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            if self.scene._valid(x, y, min_sep=0.5):
                return x, y
        return 0.0, 0.0

    def _sample_goal(self, start_xy: np.ndarray) -> np.ndarray:
        for _ in range(100):
            dist = random.uniform(GOAL_MIN_DIST, GOAL_MAX_DIST)
            ang = random.uniform(0, 2 * math.pi)
            gx = start_xy[0] + dist * math.cos(ang)
            gy = start_xy[1] + dist * math.sin(ang)
            if math.hypot(gx, gy) <= self.radius - 1.0 and self.scene._valid(gx, gy, min_sep=0.5):
                return np.array([gx, gy, 1.0], dtype=np.float32)
        return np.array([start_xy[0], start_xy[1], 1.0], dtype=np.float32)

    # ---------------------------------------------------------------------- Gym
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        sx, sy = self._sample_valid_point()
        start = np.array([sx, sy, 1.0], dtype=np.float32)
        self.goal_pos = self._sample_goal(start[:2])

        p.resetBasePositionAndOrientation(self.drone_id, start.tolist(), [0, 0, 0, 1], physicsClientId=self.client)
        p.resetBaseVelocity(self.drone_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        p.resetBasePositionAndOrientation(self.goal_id, self.goal_pos.tolist(), [0, 0, 0, 1], physicsClientId=self.client)

        self.prev_dist = np.linalg.norm(self.goal_pos[:2] - start[:2])
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        action_vec = self.action_map[int(action)]
        dx, dy, dz, throttle = action_vec

        pos, orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        yaw = p.getEulerFromQuaternion(orn)[2]

        vx = (dx * math.cos(yaw) - dy * math.sin(yaw)) * self.linear_speed * throttle
        vy = (dx * math.sin(yaw) + dy * math.cos(yaw)) * self.linear_speed * throttle
        vz = dz * self.linear_speed * throttle

        p.resetBaseVelocity(self.drone_id, [vx, vy, vz], [0, 0, 0], physicsClientId=self.client)
        for _ in range(ACTION_REPEAT):
            p.stepSimulation(physicsClientId=self.client)

        new_pos, _ = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        new_pos = np.array(new_pos)
        dist_to_goal = np.linalg.norm(self.goal_pos[:2] - new_pos[:2])

        lidar = self._get_lidar(new_pos, yaw)
        reward = 0.0

        # Progress reward
        if self.prev_dist is not None:
            reward += self.prev_dist - dist_to_goal
        self.prev_dist = dist_to_goal

        # Proximity penalty
        reward += self._proximity_penalty(lidar)

        terminated = False
        truncated = False

        # Collision penalty
        if p.getContactPoints(bodyA=self.drone_id, physicsClientId=self.client):
            reward += COLLISION_PENALTY
            terminated = True

        # Success reward
        if dist_to_goal < SUCCESS_THRESHOLD:
            reward += SUCCESS_REWARD
            terminated = True

        obs = np.concatenate(
            [lidar, np.array([min(dist_to_goal / GOAL_MAX_DIST, 1.0),
                              ((math.atan2(self.goal_pos[1] - new_pos[1],
                                           self.goal_pos[0] - new_pos[0]) - yaw + math.pi) % (2 * math.pi) - math.pi)],
                             dtype=np.float32)]
        )

        return obs, reward, terminated, truncated, {}

    # ---------------------------------------------------------------- utilities
    def _get_lidar(self, pos, yaw):
        ray_froms, ray_tos = [], []
        for i in range(self.lidar_rays):
            ang = yaw + i * (2 * math.pi / self.lidar_rays)
            ray_froms.append([pos[0], pos[1], pos[2]])
            ray_tos.append([pos[0] + self.lidar_range * math.cos(ang),
                            pos[1] + self.lidar_range * math.sin(ang),
                            pos[2]])

        results = p.rayTestBatch(ray_froms, ray_tos, physicsClientId=self.client)
        lidar = []
        for res in results:
            hit_fraction = res[2]
            lidar.append(1.0 if hit_fraction < 0 else hit_fraction)
        return np.array(lidar, dtype=np.float32)

    def _proximity_penalty(self, lidar_hits: np.ndarray) -> float:
        min_norm = float(np.min(lidar_hits))
        min_dist = min_norm * self.lidar_range
        if min_dist < PROXIMITY_THRESHOLD:
            ratio = (PROXIMITY_THRESHOLD - min_dist) / PROXIMITY_THRESHOLD
            return PROXIMITY_PENALTY * ratio
        return 0.0

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        yaw = p.getEulerFromQuaternion(orn)[2]
        lidar = self._get_lidar(pos, yaw)
        dist = np.linalg.norm(self.goal_pos[:2] - np.array(pos[:2]))
        rel_angle = (math.atan2(self.goal_pos[1] - pos[1], self.goal_pos[0] - pos[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
        goal_dist_norm = min(dist / GOAL_MAX_DIST, 1.0)
        return np.concatenate([lidar, np.array([goal_dist_norm, rel_angle], dtype=np.float32)])

    def close(self):
        try:
            p.disconnect(physicsClientId=self.client)
        except Exception:
            pass


def train():
    env = RandomPointNavEnv(gui=False)
    callback = TrainingLogger()

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=5_000,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        train_freq=4,
        gradient_steps=1,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        device="auto",
    )

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    finally:
        model.save("dqn_random_point_nav")
        env.close()


if __name__ == "__main__":
    train()

