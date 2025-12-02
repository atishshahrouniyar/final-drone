"""
Train a DQN agent to move a drone between two random points
inside a 40x40 PyBullet environment while avoiding obstacles.
"""

import math
import os
from datetime import datetime
import time
from typing import Tuple
import json
import csv

from packaging import version

import numpy as np
import pybullet as p
import torch
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
PROXIMITY_PENALTY = -0.5
COLLISION_PENALTY = -15.0
SUCCESS_REWARD = 20.0
RUNS_ROOT = "./runs"
CHECKPOINT_MILESTONES = [500_000]
EVAL_FREQUENCY = 50_000
EVAL_EPISODES = 5
MIN_ALTITUDE = 1.0
MAX_ALTITUDE = 3.0

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit


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
    """
    Logs comprehensive training metrics including Q-values, TD error, etc.
    """
    def __init__(self, log_freq: int = 1000, save_dir: str = "./metrics"):
        super().__init__()
        self.log_freq = log_freq
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # CSV file for detailed metrics
        self.csv_path = os.path.join(self.save_dir, "detailed_metrics.csv")
        self.csv_file = None
        self.csv_writer = None
        
        # Metrics storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_collisions = []
        
    def _on_training_start(self) -> None:
        """Initialize CSV file with headers."""
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestep',
            'q_value_mean',
            'q_value_std',
            'q_value_max',
            'q_value_min',
            'td_error_mean',
            'td_error_std',
            'loss',
            'exploration_rate',
            'learning_rate',
            'episode_reward_mean',
            'episode_length_mean',
            'success_rate',
            'collision_rate',
        ])
        self.csv_file.flush()
        
    def _on_step(self) -> bool:
        # Collect episode statistics from info
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                ep_info = info['episode']
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
                
            # Track success and collision rates
            if 'is_success' in info:
                self.episode_successes.append(1 if info['is_success'] else 0)
            if 'is_collision' in info:
                self.episode_collisions.append(1 if info['is_collision'] else 0)
        
        # Log detailed metrics at specified frequency
        if self.n_calls % self.log_freq == 0:
            metrics = self._collect_metrics()
            if metrics and self.csv_writer:
                self.csv_writer.writerow([
                    self.num_timesteps,
                    metrics.get('q_value_mean', 0),
                    metrics.get('q_value_std', 0),
                    metrics.get('q_value_max', 0),
                    metrics.get('q_value_min', 0),
                    metrics.get('td_error_mean', 0),
                    metrics.get('td_error_std', 0),
                    metrics.get('loss', 0),
                    metrics.get('exploration_rate', 0),
                    metrics.get('learning_rate', 0),
                    metrics.get('episode_reward_mean', 0),
                    metrics.get('episode_length_mean', 0),
                    metrics.get('success_rate', 0),
                    metrics.get('collision_rate', 0),
                ])
                self.csv_file.flush()
                
                # Log to tensorboard as well
                for key, value in metrics.items():
                    self.logger.record(f"detailed/{key}", value)
        
        return True
    
    def _collect_metrics(self):
        """Collect Q-values, TD errors, and other metrics from the model."""
        try:
            metrics = {}
            
            # Get replay buffer
            if hasattr(self.model, 'replay_buffer') and self.model.replay_buffer.size() > 0:
                # Sample from replay buffer to estimate Q-values
                replay_data = self.model.replay_buffer.sample(min(256, self.model.replay_buffer.size()))
                
                with torch.no_grad():
                    # Get Q-values for current states
                    q_values = self.model.q_net(replay_data.observations)
                    q_values_np = q_values.cpu().numpy()
                    
                    metrics['q_value_mean'] = float(np.mean(q_values_np))
                    metrics['q_value_std'] = float(np.std(q_values_np))
                    metrics['q_value_max'] = float(np.max(q_values_np))
                    metrics['q_value_min'] = float(np.min(q_values_np))
                    
                    # Compute TD error
                    next_q_values = self.model.q_net_target(replay_data.next_observations)
                    next_q_values, _ = next_q_values.max(dim=1)
                    target_q_values = replay_data.rewards.flatten() + (1 - replay_data.dones.flatten()) * self.model.gamma * next_q_values
                    
                    current_q_values = q_values.gather(1, replay_data.actions.long()).squeeze()
                    td_errors = torch.abs(target_q_values - current_q_values).cpu().numpy()
                    
                    metrics['td_error_mean'] = float(np.mean(td_errors))
                    metrics['td_error_std'] = float(np.std(td_errors))
            
            # Get training loss (from logger)
            if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
                metrics['loss'] = self.model.logger.name_to_value.get('train/loss', 0)
            
            # Get exploration rate
            metrics['exploration_rate'] = self.model.exploration_rate
            
            # Get learning rate
            if hasattr(self.model, 'lr_schedule'):
                metrics['learning_rate'] = self.model.lr_schedule(self.model._current_progress_remaining)
            
            # Episode statistics
            if self.episode_rewards:
                metrics['episode_reward_mean'] = np.mean(self.episode_rewards[-100:])
            if self.episode_lengths:
                metrics['episode_length_mean'] = np.mean(self.episode_lengths[-100:])
            if self.episode_successes:
                metrics['success_rate'] = np.mean(self.episode_successes[-100:])
            if self.episode_collisions:
                metrics['collision_rate'] = np.mean(self.episode_collisions[-100:])
            
            # Buffer statistics
            if hasattr(self.model, 'replay_buffer'):
                buffer_size = self.model.replay_buffer.size()
                buffer_capacity = self.model.replay_buffer.buffer_size
                metrics['buffer_size'] = buffer_size
                metrics['sample_reuse_ratio'] = self.num_timesteps / max(buffer_size, 1)
                self.logger.record("detailed/buffer_size", buffer_size)
                self.logger.record("detailed/buffer_capacity", buffer_capacity)
                self.logger.record("detailed/sample_reuse_ratio", metrics['sample_reuse_ratio'])
            
            return metrics
            
        except Exception as e:
            print(f"[DetailedMetrics] Warning: Could not collect metrics: {e}")
            return {}
    
    def _on_training_end(self) -> None:
        """Close CSV file."""
        if self.csv_file:
            self.csv_file.close()
            print(f"[DetailedMetrics] Saved metrics to {self.csv_path}")


class TrainingConfigSaver(BaseCallback):
    """Save training configuration and hyperparameters."""
    
    def __init__(self, save_path: str):
        super().__init__()
        self.save_path = save_path
    
    def _on_training_start(self) -> None:
        """Save configuration at start of training."""
        config = {
            'timestamp': datetime.now().isoformat(),
            'total_timesteps': TOTAL_TIMESTEPS,
            'learning_rate': LEARNING_RATE,
            'buffer_size': BUFFER_SIZE,
            'batch_size': BATCH_SIZE,
            'gamma': GAMMA,
            'target_update_interval': TARGET_UPDATE_INTERVAL,
            'action_repeat': ACTION_REPEAT,
            'arena_radius': ARENA_RADIUS,
            'goal_min_dist': GOAL_MIN_DIST,
            'goal_max_dist': GOAL_MAX_DIST,
            'success_threshold': SUCCESS_THRESHOLD,
            'proximity_threshold': PROXIMITY_THRESHOLD,
            'proximity_penalty': PROXIMITY_PENALTY,
            'collision_penalty': COLLISION_PENALTY,
            'success_reward': SUCCESS_REWARD,
            'min_altitude': MIN_ALTITUDE,
            'max_altitude': MAX_ALTITUDE,
            'eval_frequency': EVAL_FREQUENCY,
            'eval_episodes': EVAL_EPISODES,
            'max_episode_steps': 2000,
            'sb3_version': SB3_VERSION,
        }
        
        # Save model hyperparameters
        if hasattr(self.model, 'learning_rate'):
            config['model_learning_rate'] = float(self.model.learning_rate)
        if hasattr(self.model, 'exploration_fraction'):
            config['exploration_fraction'] = float(self.model.exploration_fraction)
        if hasattr(self.model, 'exploration_initial_eps'):
            config['exploration_initial_eps'] = float(self.model.exploration_initial_eps)
        if hasattr(self.model, 'exploration_final_eps'):
            config['exploration_final_eps'] = float(self.model.exploration_final_eps)
        
        with open(self.save_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[Config] Saved training configuration to {self.save_path}")
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

        # Observation space: lidar(36) + goal distance + goal angle + altitude delta
        self.lidar_rays = 36
        self.lidar_range = 6.0
        low = np.array([0.0] * self.lidar_rays + [0.0, -np.pi, -1.0], dtype=np.float32)
        high = np.array([1.0] * self.lidar_rays + [1.0, np.pi, 1.0], dtype=np.float32)
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

    def _sample_goal(self, start_xyz: np.ndarray) -> np.ndarray:
        for _ in range(100):
            dist = self._rng.uniform(GOAL_MIN_DIST, GOAL_MAX_DIST)
            ang = self._rng.uniform(0, 2 * math.pi)
            gx = start_xyz[0] + dist * math.cos(ang)
            gy = start_xyz[1] + dist * math.sin(ang)
            if math.hypot(gx, gy) <= self.radius - 1.0 and self.scene._valid(gx, gy, min_sep=0.5):
                gz = float(self._rng.uniform(MIN_ALTITUDE, MAX_ALTITUDE))
                return np.array([gx, gy, gz], dtype=np.float32)
        return np.array(
            [start_xyz[0], start_xyz[1], float(self._rng.uniform(MIN_ALTITUDE, MAX_ALTITUDE))], dtype=np.float32
        )

    # ---------------------------------------------------------------------- Gym
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(int(self.np_random.integers(1 << 63)))
        if hasattr(self.scene, "reposition_humans"):
            self.scene.reposition_humans()

        sx, sy = self._sample_valid_point()
        sz = float(self._rng.uniform(MIN_ALTITUDE, MAX_ALTITUDE))
        start = np.array([sx, sy, sz], dtype=np.float32)
        self.start_pos = start.copy()
        self.goal_pos = self._sample_goal(start)

        p.resetBasePositionAndOrientation(self.drone_id, start.tolist(), [0, 0, 0, 1], physicsClientId=self.client)
        p.resetBaseVelocity(self.drone_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        p.resetBasePositionAndOrientation(self.goal_id, self.goal_pos.tolist(), [0, 0, 0, 1], physicsClientId=self.client)

        self.prev_dist = np.linalg.norm(self.goal_pos - start)
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

        new_pos, orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        new_pos = np.array(new_pos)
        if new_pos[2] < MIN_ALTITUDE or new_pos[2] > MAX_ALTITUDE:
            clamped = new_pos.copy()
            clamped[2] = np.clip(clamped[2], MIN_ALTITUDE, MAX_ALTITUDE)
            p.resetBasePositionAndOrientation(
                self.drone_id, clamped.tolist(), orn, physicsClientId=self.client
            )
            new_pos = clamped
        dist_to_goal = np.linalg.norm(self.goal_pos - new_pos)

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
        
        # Track episode outcomes separately
        is_collision = False
        is_success = False

        # Collision penalty
        if p.getContactPoints(bodyA=self.drone_id, physicsClientId=self.client):
            reward += COLLISION_PENALTY
            terminated = True
            is_collision = True

        # Success reward
        if dist_to_goal < SUCCESS_THRESHOLD:
            reward += SUCCESS_REWARD
            terminated = True
            is_success = True

        goal_dist_norm = min(dist_to_goal / GOAL_MAX_DIST, 1.0)
        rel_angle = (math.atan2(self.goal_pos[1] - new_pos[1], self.goal_pos[0] - new_pos[0]) - yaw + math.pi) % (
            2 * math.pi
        ) - math.pi
        altitude_delta = np.clip((self.goal_pos[2] - new_pos[2]) / (MAX_ALTITUDE - MIN_ALTITUDE), -1.0, 1.0)
        obs = np.concatenate([lidar, np.array([goal_dist_norm, rel_angle, altitude_delta], dtype=np.float32)])
        
        # Add episode outcome info
        info = {
            'is_success': is_success,
            'is_collision': is_collision,
        }
        
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
        dist = np.linalg.norm(self.goal_pos - np.array(pos))
        rel_angle = (math.atan2(self.goal_pos[1] - pos[1], self.goal_pos[0] - pos[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
        goal_dist_norm = min(dist / GOAL_MAX_DIST, 1.0)
        altitude_delta = np.clip((self.goal_pos[2] - pos[2]) / (MAX_ALTITUDE - MIN_ALTITUDE), -1.0, 1.0)
        return np.concatenate([lidar, np.array([goal_dist_norm, rel_angle, altitude_delta], dtype=np.float32)])

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

    print("=" * 70)
    print("Training Configuration Saved: {}".format(
        os.path.join(run_dir, "training_config.json")
    ))
    print("=" * 70)

    # Create environments with TimeLimit wrapper to prevent infinite episodes
    # 2000 steps = ~33 seconds of simulation time (sufficient for 40x40m arena)
    env = RandomPointNavEnv(gui=False)
    env = TimeLimit(env, max_episode_steps=2000)
    env = Monitor(env)
    
    eval_env = RandomPointNavEnv(gui=False)
    eval_env = TimeLimit(eval_env, max_episode_steps=2000)
    eval_env = Monitor(eval_env)

    # Create comprehensive callbacks
    config_saver = TrainingConfigSaver(os.path.join(run_dir, "training_config.json"))
    checkpoint_cb = MilestoneCheckpoint(CHECKPOINT_MILESTONES, checkpoint_dir)
    detailed_metrics_cb = DetailedMetricsCallback(log_freq=1000, save_dir=metrics_dir)
    training_logger_cb = TrainingLogger(check_freq=5000)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=eval_log_dir,
        eval_freq=EVAL_FREQUENCY,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
    )
    
    callback = CallbackList([
        config_saver,
        training_logger_cb,
        detailed_metrics_cb,
        checkpoint_cb,
        eval_cb
    ])
    
    print("=" * 70)
    print("Starting Training: {:,} timesteps".format(TOTAL_TIMESTEPS))
    print("Logs: {}".format(run_dir))
    print("=" * 70)

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
        tensorboard_log=tensorboard_dir,
        device="auto",
    )

    start_time = time.time()
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    finally:
        # Save final model
        model_path = os.path.join(run_dir, f"dqn_random_point_nav_{timestamp}")
        model.save(model_path)
        print(f"\n[Training Complete] Saved model to {model_path}.zip")
        
        # Create training summary
        training_time = time.time() - start_time
        summary = {
            'training_completed': datetime.now().isoformat(),
            'total_training_time_seconds': training_time,
            'total_training_time_hours': training_time / 3600,
            'total_timesteps': TOTAL_TIMESTEPS,
            'final_exploration_rate': float(model.exploration_rate),
            'final_buffer_size': model.replay_buffer.size() if hasattr(model, 'replay_buffer') else 0,
        }
        
        # Add final episode statistics if available
        if hasattr(detailed_metrics_cb, 'episode_rewards') and detailed_metrics_cb.episode_rewards:
            recent_rewards = detailed_metrics_cb.episode_rewards[-100:]
            recent_lengths = detailed_metrics_cb.episode_lengths[-100:]
            recent_successes = detailed_metrics_cb.episode_successes[-100:]
            recent_collisions = detailed_metrics_cb.episode_collisions[-100:]
            
            summary['final_metrics'] = {
                'episode_reward_mean': float(np.mean(recent_rewards)),
                'episode_reward_std': float(np.std(recent_rewards)),
                'episode_length_mean': float(np.mean(recent_lengths)),
                'episode_length_std': float(np.std(recent_lengths)),
                'success_rate': float(np.mean(recent_successes)),
                'collision_rate': float(np.mean(recent_collisions)),
                'total_episodes': len(detailed_metrics_cb.episode_rewards),
            }
        
        # Save summary
        summary_path = os.path.join(run_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "=" * 70)
        print("Training Summary")
        print("=" * 70)
        print(f"Total time: {training_time/3600:.2f} hours")
        print(f"Final exploration rate: {summary['final_exploration_rate']:.4f}")
        if 'final_metrics' in summary:
            fm = summary['final_metrics']
            print(f"Final episode reward: {fm['episode_reward_mean']:.2f} ± {fm['episode_reward_std']:.2f}")
            print(f"Final episode length: {fm['episode_length_mean']:.1f} ± {fm['episode_length_std']:.1f}")
            print(f"Final success rate: {fm['success_rate']*100:.1f}%")
            print(f"Final collision rate: {fm['collision_rate']*100:.1f}%")
            print(f"Total episodes: {fm['total_episodes']}")
        print(f"\nSummary saved to: {summary_path}")
        print("=" * 70)
        
        # Close environments
        env.close()
        eval_env.close()


if __name__ == "__main__":
    train()

