#!/usr/bin/env python3
"""
Test and visualize trained DQN model with GUI enabled.

Usage:
    python test_trained_model.py [model_path]
    
If model_path is not provided, defaults to "dqn_detailed_model"
"""

import sys
import time
import numpy as np
from stable_baselines3 import DQN
from train_dqn import DroneNavGymEnv

def test_model(model_path="dqn_detailed_model", num_episodes=10, max_steps_per_episode=1000):
    """
    Load and test a trained DQN model with GUI visualization.
    
    Args:
        model_path: Path to the saved model (without .zip extension)
        num_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
    """
    print("=" * 60)
    print("Loading trained DQN model...")
    print("=" * 60)
    
    try:
        # Load the trained model
        model = DQN.load(model_path)
        print(f"✓ Model loaded successfully from: {model_path}")
    except FileNotFoundError:
        print(f"✗ Error: Model file not found: {model_path}")
        print(f"  Looking for: {model_path}.zip")
        print("\nAvailable files in current directory:")
        import os
        for f in os.listdir('.'):
            if 'dqn' in f.lower() or 'model' in f.lower():
                print(f"  - {f}")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    print("\n" + "=" * 60)
    print("Creating environment with GUI enabled...")
    print("=" * 60)
    
    # Create environment with GUI enabled
    env = DroneNavGymEnv(gui=True)
    
    print("\n" + "=" * 60)
    print(f"Running {num_episodes} episodes...")
    print("Press Ctrl+C to stop early")
    print("=" * 60)
    
    episode_rewards = []
    episode_lengths = []
    
    try:
        for episode in range(num_episodes):
            obs, info = env.reset()
            
            # Update camera to follow the drone at episode start
            if hasattr(env, 'scene') and hasattr(env.scene, '_update_camera'):
                env.scene._update_camera()
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            while not done and episode_length < max_steps_per_episode:
                # Get action from the trained model (deterministic)
                action, _ = model.predict(obs, deterministic=True)
                
                # Step the environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Update camera to follow the drone
                if hasattr(env, 'scene') and hasattr(env.scene, '_update_camera'):
                    env.scene._update_camera()
                
                episode_reward += reward
                episode_length += 1
                
                # Small delay for visualization
                time.sleep(0.01)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            status = "SUCCESS" if episode_length < max_steps_per_episode else "TIMEOUT"
            print(f"  Reward: {episode_reward:.2f}, Steps: {episode_length}, Status: {status}")
            
            # Wait a bit between episodes
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    
    finally:
        # Print statistics
        if episode_rewards:
            print("\n" + "=" * 60)
            print("Episode Statistics:")
            print("=" * 60)
            print(f"Total Episodes: {len(episode_rewards)}")
            print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
            print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
            print(f"Best Reward: {np.max(episode_rewards):.2f}")
            print(f"Worst Reward: {np.min(episode_rewards):.2f}")
            print("=" * 60)
        
        print("\nClosing environment...")
        env.close()
        print("Done!")

if __name__ == "__main__":
    # Get model path from command line argument or use default
    model_path = sys.argv[1] if len(sys.argv) > 1 else "dqn_detailed_model"
    
    # Get number of episodes from command line or use default
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    test_model(model_path=model_path, num_episodes=num_episodes)

