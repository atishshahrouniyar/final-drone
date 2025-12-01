#!/usr/bin/env python3
"""
Extract actual rewards from TensorBoard logs and create corrected visualizations.
This fixes the issue where training_metrics.csv shows 0 rewards due to logging timing.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def extract_from_tensorboard(event_file):
    """Extract reward data from TensorBoard event file."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        data = {}
        
        # Extract all available scalar metrics
        for tag in ea.Tags()['scalars']:
            scalars = ea.Scalars(tag)
            data[tag] = {
                'steps': [s.step for s in scalars],
                'values': [s.value for s in scalars]
            }
        
        return data
        
    except ImportError:
        print("Error: tensorboard package not installed.")
        print("Install with: pip install tensorboard")
        return None
    except Exception as e:
        print(f"Error reading TensorBoard logs: {e}")
        return None


def plot_corrected_learning_curve(data, save_path):
    """Plot the corrected learning curve from TensorBoard data."""
    if 'rollout/ep_rew_mean' not in data:
        print("Warning: No reward data found in TensorBoard logs")
        return
    
    steps = np.array(data['rollout/ep_rew_mean']['steps'])
    rewards = np.array(data['rollout/ep_rew_mean']['values'])
    
    plt.figure(figsize=(14, 7))
    
    # Plot 1: Raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(steps, rewards, linewidth=2, alpha=0.7, label='Episode Reward')
    plt.axhline(y=np.max(rewards), color='r', linestyle='--', alpha=0.5, 
                label=f'Peak: {np.max(rewards):.1f}')
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Average Episode Reward', fontsize=12)
    plt.title('Training Learning Curve (Actual Data)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Smoothed rewards
    plt.subplot(1, 2, 2)
    window = min(20, len(rewards) // 10)
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        smooth_steps = steps[:len(smoothed)]
        plt.plot(smooth_steps, smoothed, linewidth=2.5, color='green', label=f'Smoothed (window={window})')
    plt.plot(steps, rewards, linewidth=1, alpha=0.3, color='blue', label='Raw')
    plt.axhline(y=np.max(rewards), color='r', linestyle='--', alpha=0.5, 
                label=f'Peak: {np.max(rewards):.1f}')
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Average Episode Reward', fontsize=12)
    plt.title('Smoothed Learning Curve', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved corrected learning curve: {save_path}")
    plt.close()
    
    # Print statistics
    print(f"\n{'='*60}")
    print("ACTUAL TRAINING STATISTICS")
    print(f"{'='*60}")
    print(f"Peak reward:        {np.max(rewards):.2f}")
    print(f"Final reward:       {rewards[-1]:.2f}")
    print(f"Mean reward:        {np.mean(rewards):.2f}")
    print(f"Std reward:         {np.std(rewards):.2f}")
    print(f"Total episodes:     {len(rewards)}")
    print(f"Final timesteps:    {steps[-1]:,}")
    
    # Find peak
    peak_idx = np.argmax(rewards)
    print(f"\nPeak at step:       {steps[peak_idx]:,}")
    print(f"Peak reward:        {rewards[peak_idx]:.2f}")
    print(f"{'='*60}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_tensorboard_rewards.py <run_directory>")
        print("Example: python extract_tensorboard_rewards.py runs/20251130_172849")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    
    # Find TensorBoard event file
    tb_dir = os.path.join(run_dir, 'tensorboard')
    event_file = None
    
    for root, dirs, files in os.walk(tb_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_file = os.path.join(root, file)
                break
        if event_file:
            break
    
    if not event_file:
        print(f"Error: No TensorBoard event files found in {tb_dir}")
        sys.exit(1)
    
    print(f"Reading TensorBoard data from: {event_file}")
    data = extract_from_tensorboard(os.path.dirname(event_file))
    
    if data is None:
        sys.exit(1)
    
    print(f"\nFound metrics: {list(data.keys())}")
    
    # Create plots directory
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate corrected learning curve
    save_path = os.path.join(plots_dir, 'learning_curve_corrected.png')
    plot_corrected_learning_curve(data, save_path)
    
    # Save raw data to CSV for reference
    if 'rollout/ep_rew_mean' in data:
        import csv
        csv_path = os.path.join(run_dir, 'actual_rewards.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestep', 'ep_rew_mean'])
            for step, reward in zip(data['rollout/ep_rew_mean']['steps'], 
                                   data['rollout/ep_rew_mean']['values']):
                writer.writerow([step, reward])
        print(f"✓ Saved actual rewards to: {csv_path}")


if __name__ == "__main__":
    main()

