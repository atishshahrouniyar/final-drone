"""
Train a DQN agent to move a drone between two random points
inside a 40x40 PyBullet environment while avoiding obstacles.
"""

import csv
import json
import math
import os
from datetime import datetime
import time
from typing import Tuple

from packaging import version

import numpy as np
import pybullet as p
from stable_baselines3 import DQN, __version__ as SB3_VERSION
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

from enhanced_navigation_env import EnhancedForestWithObstacles


# --------------------------- configuration ------------------------------------
TOTAL_TIMESTEPS = 1_000_000
LEARNING_RATE = 2e-4
BUFFER_SIZE = 200_000
BATCH_SIZE = 128
GAMMA = 0.99
TARGET_UPDATE_INTERVAL = 500
ACTION_REPEAT = 4
ARENA_RADIUS = 20.0  # 40 x 40 meters
GOAL_MIN_DIST = 3.0
GOAL_MAX_DIST = 5.0
SUCCESS_THRESHOLD = 0.3
PROXIMITY_THRESHOLD = 0.1
PROXIMITY_PENALTY = -1.0  # Increased from -0.5 for stronger obstacle avoidance
COLLISION_PENALTY = -100.0  # Increased from -40.0 for much harsher penalty
SUCCESS_REWARD = 20.0
HUMAN_FOUND_REWARD = SUCCESS_REWARD
HUMAN_DETECTION_RADIUS = 0.6
MIN_SAFE_ALTITUDE = 0.6
GROUND_CLEARANCE_PENALTY = -8.0
RUNS_ROOT = "./runs"
CHECKPOINT_MILESTONES = [100_000, 250_000, 500_000, 750_000, 1_000_000]  # More frequent checkpoints
EVAL_FREQUENCY = 100_000  # Reduced from 50_000 (evaluate less often)
EVAL_EPISODES = 3  # Reduced from 5 (faster evaluation)

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit


class TrainingLogger(BaseCallback):
    """Comprehensive logger for monitoring learning and collecting metrics."""

    def __init__(self, check_freq: int = 5000, log_dir: str = "./logs"):
        super().__init__()
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.start_time = time.time()
        self.csv_path = os.path.join(log_dir, "training_metrics.csv")
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize CSV file with headers
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestep', 'episode', 'fps', 'elapsed_time',
                'ep_len_mean', 'ep_rew_mean', 
                'train/loss', 'train/learning_rate',
                'train/n_updates', 'rollout/exploration_rate',
                'time/total_timesteps'
            ])

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            elapsed = max(time.time() - self.start_time, 1e-6)
            fps = self.num_timesteps / elapsed
            
            # Log to tensorboard
            self.logger.record("custom/fps", fps)
            self.logger.record("custom/elapsed_time", elapsed)
            
            # Collect metrics from logger
            metrics = {
                'timestep': self.num_timesteps,
                'episode': self.locals.get('episode_num', 0),
                'fps': fps,
                'elapsed_time': elapsed,
                'ep_len_mean': self.locals.get('ep_len_mean', 0),
                'ep_rew_mean': self.locals.get('ep_rew_mean', 0),
                'train/loss': self.model.logger.name_to_value.get('train/loss', 0),
                'train/learning_rate': self.model.logger.name_to_value.get('train/learning_rate', 0),
                'train/n_updates': self.model.logger.name_to_value.get('train/n_updates', 0),
                'rollout/exploration_rate': self.model.logger.name_to_value.get('rollout/exploration_rate', 0),
                'time/total_timesteps': self.num_timesteps,
            }
            
            # Save to CSV
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    metrics['timestep'], metrics['episode'], metrics['fps'], 
                    metrics['elapsed_time'], metrics['ep_len_mean'], metrics['ep_rew_mean'],
                    metrics['train/loss'], metrics['train/learning_rate'], 
                    metrics['train/n_updates'], metrics['rollout/exploration_rate'],
                    metrics['time/total_timesteps']
                ])
            
            # Print progress
            print(f"\n[Training Progress] Timestep: {self.num_timesteps:,} | "
                  f"FPS: {fps:.1f} | Reward: {metrics['ep_rew_mean']:.2f} | "
                  f"Loss: {metrics['train/loss']:.4f}")
        
        return True


class MilestoneCheckpoint(BaseCallback):
    """Save checkpoints at predefined timesteps."""

    def __init__(self, milestones, save_dir, prefix="dqn_random_point_nav"):
        super().__init__()
        self.milestones = sorted(milestones)
        self.save_dir = save_dir
        self.prefix = prefix
        self.saved = set()
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        for milestone in self.milestones:
            if milestone in self.saved:
                continue
            if self.num_timesteps >= milestone:
                path = os.path.join(self.save_dir, f"{self.prefix}_{milestone // 1000}k")
                self.model.save(path)
                print(f"[Checkpoint] Saved model at {milestone:,} steps -> {path}.zip")
                self.saved.add(milestone)
        return True


class DetailedMetricsCallback(BaseCallback):
    """Collect detailed training metrics including Q-values, TD error, etc."""
    
    def __init__(self, log_freq: int = 1000, log_dir: str = "./logs"):
        super().__init__()
        self.log_freq = log_freq
        self.log_dir = log_dir
        self.metrics_path = os.path.join(log_dir, "detailed_metrics.csv")
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize CSV
        with open(self.metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestep', 'q_value_mean', 'q_value_std', 'q_value_max', 'q_value_min',
                'td_error_mean', 'td_error_std', 'buffer_size', 'exploration_rate',
                'learning_rate', 'loss'
            ])
    
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            try:
                # Access DQN internals
                replay_buffer = self.model.replay_buffer
                
                # Sample batch to compute Q-value statistics
                if replay_buffer.size() > self.model.batch_size:
                    replay_data = replay_buffer.sample(self.model.batch_size)
                    
                    with self.model.policy.q_net.eval():
                        q_values = self.model.policy.q_net(replay_data.observations)
                        q_value_mean = float(q_values.mean().detach().cpu().numpy())
                        q_value_std = float(q_values.std().detach().cpu().numpy())
                        q_value_max = float(q_values.max().detach().cpu().numpy())
                        q_value_min = float(q_values.min().detach().cpu().numpy())
                    
                    # Compute TD errors
                    with self.model.policy.q_net_target.eval():
                        next_q_values = self.model.policy.q_net_target(replay_data.next_observations)
                        max_next_q_values = next_q_values.max(dim=1)[0]
                        target_q_values = replay_data.rewards.flatten() + (1 - replay_data.dones.flatten()) * self.model.gamma * max_next_q_values
                        
                        current_q_values = q_values.gather(1, replay_data.actions.long()).squeeze()
                        td_errors = (target_q_values - current_q_values).detach().cpu().numpy()
                        td_error_mean = float(np.mean(np.abs(td_errors)))
                        td_error_std = float(np.std(td_errors))
                else:
                    q_value_mean = q_value_std = q_value_max = q_value_min = 0.0
                    td_error_mean = td_error_std = 0.0
                
                buffer_size = replay_buffer.size()
                exploration_rate = self.model.exploration_rate
                learning_rate = self.model.learning_rate
                loss = self.model.logger.name_to_value.get('train/loss', 0.0)
                
                # Log to tensorboard
                self.logger.record("metrics/q_value_mean", q_value_mean)
                self.logger.record("metrics/q_value_std", q_value_std)
                self.logger.record("metrics/td_error_mean", td_error_mean)
                self.logger.record("metrics/buffer_size", buffer_size)
                
                # Save to CSV
                with open(self.metrics_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.num_timesteps, q_value_mean, q_value_std, 
                        q_value_max, q_value_min, td_error_mean, td_error_std,
                        buffer_size, exploration_rate, learning_rate, loss
                    ])
                    
            except Exception as e:
                print(f"[DetailedMetrics] Warning: Could not collect metrics: {e}")
        
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
        self.lidar_range = 3.0
        low = np.array([0.0] * self.lidar_rays + [0.0, -np.pi], dtype=np.float32)
        high = np.array([1.0] * self.lidar_rays + [1.0, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.goal_pos = np.zeros(3, dtype=np.float32)
        self.prev_dist = None
        self.start_pos = None
        self._rng = np.random.default_rng()
        self.start_pos = None

    # ------------------------------------------------------------------ helpers
    def _sample_valid_point(self, margin: float = 2.0) -> Tuple[float, float]:
        for _ in range(100):
            r = self._rng.uniform(margin, self.radius - margin)
            ang = self._rng.uniform(0, 2 * math.pi)
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            if self.scene._valid(x, y, min_sep=0.5):
                return x, y
        return 0.0, 0.0

    def _sample_goal(self, start_xy: np.ndarray) -> np.ndarray:
        for _ in range(100):
            dist = self._rng.uniform(GOAL_MIN_DIST, GOAL_MAX_DIST)
            ang = self._rng.uniform(0, 2 * math.pi)
            gx = start_xy[0] + dist * math.cos(ang)
            gy = start_xy[1] + dist * math.sin(ang)
            if math.hypot(gx, gy) <= self.radius - 1.0 and self.scene._valid(gx, gy, min_sep=0.5):
                return np.array([gx, gy, 1.0], dtype=np.float32)
        return np.array([start_xy[0], start_xy[1], 1.0], dtype=np.float32)

    # ---------------------------------------------------------------------- Gym
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(int(self.np_random.integers(1 << 63)))
        
        # Extract target sector from options if present
        target_sector = None
        if options and "target_sector" in options:
            target_sector = options["target_sector"]

        if hasattr(self.scene, "reposition_humans"):
            self.scene.reposition_humans(target_sector)

        # Force start at (0,0) if requested (for search missions)
        if options and options.get("center_start"):
            sx, sy = 0.0, 0.0
        else:
            sx, sy = self._sample_valid_point()
            
        start = np.array([sx, sy, 1.0], dtype=np.float32)
        self.start_pos = start.copy()
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
        
        # Force altitude to 1.5m to match deployment behavior (with small tolerance)
        if abs(new_pos[2] - 1.5) > 0.2:
            p.resetBasePositionAndOrientation(
                self.drone_id, 
                [new_pos[0], new_pos[1], 1.5], 
                [0, 0, 0, 1], 
                physicsClientId=self.client
            )
            new_pos[2] = 1.5
        
        dist_to_goal = np.linalg.norm(self.goal_pos[:2] - new_pos[:2])

        lidar = self._get_lidar(new_pos, yaw)
        reward = 0.0

        # Progress reward
        prev_dist = self.prev_dist
        if prev_dist is not None:
            reward += prev_dist - dist_to_goal
        self.prev_dist = dist_to_goal

        # Proximity penalty
        reward += self._proximity_penalty(lidar)
        
        # Bonus for maintaining safe distance from obstacles (encourage collision-free flying)
        min_norm = float(np.min(lidar))
        min_lidar_dist = min_norm * self.lidar_range
        if min_lidar_dist > 1.0:  # Safe distance maintained
            reward += 0.1  # Small bonus for safe flying

        # Strongly discourage skimming the ground (common failure mode)
        if new_pos[2] < MIN_SAFE_ALTITUDE:
            depth = MIN_SAFE_ALTITUDE - new_pos[2]
            reward += GROUND_CLEARANCE_PENALTY * depth

        terminated = False
        truncated = False
        info = {"collision": False, "goal_reached": False, "human_found": False}

        # Collision penalty
        if p.getContactPoints(bodyA=self.drone_id, physicsClientId=self.client):
            reward += COLLISION_PENALTY
            terminated = True
            info["collision"] = True

        # Human detection (considered mission success)
        if not info["collision"] and getattr(self.scene, "human_positions", None):
            for human_pos in self.scene.human_positions:
                if np.linalg.norm(new_pos[:2] - np.array(human_pos[:2])) < HUMAN_DETECTION_RADIUS:
                    reward += HUMAN_FOUND_REWARD
                    terminated = True
                    info["human_found"] = True
                    break

        # Success reward (goal marker) - resample a fresh waypoint instead of ending episode
        if (
            not info["collision"]
            and not info["human_found"]
            and dist_to_goal < SUCCESS_THRESHOLD
            and (prev_dist is None or prev_dist >= SUCCESS_THRESHOLD)
        ):
            reward += SUCCESS_REWARD
            info["goal_reached"] = True
            self.goal_pos = self._sample_goal(new_pos[:2])
            self.prev_dist = np.linalg.norm(self.goal_pos[:2] - new_pos[:2])
            p.resetBasePositionAndOrientation(
                self.goal_id, self.goal_pos.tolist(), [0, 0, 0, 1], physicsClientId=self.client
            )

        obs = np.concatenate(
            [lidar, np.array([min(dist_to_goal / GOAL_MAX_DIST, 1.0),
                              (((math.atan2(self.goal_pos[1] - new_pos[1],
                                           self.goal_pos[0] - new_pos[0]) - yaw + math.pi) % (2 * math.pi) - math.pi) / math.pi)],
                             dtype=np.float32)]
        )
        return obs, reward, terminated, truncated, info

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
        rel_angle_norm = rel_angle / math.pi  # Normalize to [-1, 1]
        return np.concatenate([lidar, np.array([goal_dist_norm, rel_angle_norm], dtype=np.float32)])

    def close(self):
        try:
            p.disconnect(physicsClientId=self.client)
        except Exception:
            pass


def train():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_ROOT, timestamp)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    eval_log_dir = os.path.join(run_dir, "eval_logs")
    metrics_dir = os.path.join(run_dir, "metrics")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save training configuration metadata
    config = {
        'timestamp': timestamp,
        'total_timesteps': TOTAL_TIMESTEPS,
        'learning_rate': LEARNING_RATE,
        'buffer_size': BUFFER_SIZE,
        'batch_size': BATCH_SIZE,
        'gamma': GAMMA,
        'target_update_interval': TARGET_UPDATE_INTERVAL,
        'action_repeat': ACTION_REPEAT,
        'arena_radius': ARENA_RADIUS,
        'collision_penalty': COLLISION_PENALTY,
        'success_reward': SUCCESS_REWARD,
        'proximity_penalty': PROXIMITY_PENALTY,
        'proximity_threshold': PROXIMITY_THRESHOLD,
        'ground_clearance_penalty': GROUND_CLEARANCE_PENALTY,
        'min_safe_altitude': MIN_SAFE_ALTITUDE,
        'eval_frequency': EVAL_FREQUENCY,
        'eval_episodes': EVAL_EPISODES,
        'checkpoint_milestones': CHECKPOINT_MILESTONES,
        'max_episode_steps': 2000,
        'exploration_fraction': 0.4,
        'exploration_final_eps': 0.02,
        'gradient_steps': 2,
    }
    
    with open(os.path.join(run_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Training Configuration Saved: {run_dir}/training_config.json")
    print(f"{'='*70}\n")

    env = RandomPointNavEnv(gui=False)
    env = TimeLimit(env, max_episode_steps=2000)  # Prevent infinite episodes
    env = Monitor(env)
    
    eval_env = RandomPointNavEnv(gui=False)
    eval_env = TimeLimit(eval_env, max_episode_steps=2000)
    eval_env = Monitor(eval_env)

    checkpoint_cb = MilestoneCheckpoint(CHECKPOINT_MILESTONES, checkpoint_dir)
    training_logger_cb = TrainingLogger(check_freq=5000, log_dir=metrics_dir)
    detailed_metrics_cb = DetailedMetricsCallback(log_freq=1000, log_dir=metrics_dir)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=eval_log_dir,
        eval_freq=EVAL_FREQUENCY,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        verbose=1,
    )
    callback = CallbackList([training_logger_cb, detailed_metrics_cb, checkpoint_cb, eval_cb])

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=5_000,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        exploration_fraction=0.4,  # Increased from 0.3 (explore longer)
        exploration_final_eps=0.02,  # Decreased from 0.05 (more exploitation)
        train_freq=4,
        gradient_steps=2,  # Increased from 1 (more learning per step)
        verbose=1,
        tensorboard_log=tensorboard_dir,
        device="auto",
    )

    try:
        print(f"\n{'='*70}")
        print(f"Starting Training: {TOTAL_TIMESTEPS:,} timesteps")
        print(f"Logs: {run_dir}")
        print(f"{'='*70}\n")
        
        training_start = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
        training_duration = time.time() - training_start
        
        # Save final training summary
        summary = {
            'training_completed': True,
            'total_timesteps': TOTAL_TIMESTEPS,
            'training_duration_seconds': training_duration,
            'training_duration_hours': training_duration / 3600,
            'final_exploration_rate': float(model.exploration_rate),
            'buffer_size_final': model.replay_buffer.size(),
            'timestamp_end': datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        
        with open(os.path.join(run_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"Duration: {training_duration/3600:.2f} hours")
        print(f"Final exploration rate: {model.exploration_rate:.4f}")
        print(f"Summary saved: {run_dir}/training_summary.json")
        print(f"{'='*70}\n")
        
    finally:
        model_path = os.path.join(run_dir, f"dqn_random_point_nav_{timestamp}")
        model.save(model_path)
        print(f"Saved final model to {model_path}.zip")
        env.close()
        eval_env.close()


if __name__ == "__main__":
    train()

