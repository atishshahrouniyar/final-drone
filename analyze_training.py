#!/usr/bin/env python3
"""
Quick analysis script for training metrics.
Usage: python analyze_training.py runs/TIMESTAMP/
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_learning_curve(metrics_dir, save_dir):
    """Plot episode reward over time."""
    df = pd.read_csv(os.path.join(metrics_dir, 'training_metrics.csv'))
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestep'], df['ep_rew_mean'], linewidth=2)
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Average Episode Reward', fontsize=12)
    plt.title('Training Learning Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curve.png'), dpi=300)
    print(f"✓ Saved: {save_dir}/learning_curve.png")
    plt.close()


def plot_q_values(metrics_dir, save_dir):
    """Plot Q-value statistics."""
    df = pd.read_csv(os.path.join(metrics_dir, 'detailed_metrics.csv'))
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestep'], df['q_value_mean'], label='Mean Q-Value', linewidth=2)
    plt.fill_between(df['timestep'],
                     df['q_value_mean'] - df['q_value_std'],
                     df['q_value_mean'] + df['q_value_std'],
                     alpha=0.3, label='±1 Std Dev')
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Q-Value', fontsize=12)
    plt.title('Q-Value Statistics Over Training', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'q_values.png'), dpi=300)
    print(f"✓ Saved: {save_dir}/q_values.png")
    plt.close()


def plot_td_error(metrics_dir, save_dir):
    """Plot TD error over time."""
    df = pd.read_csv(os.path.join(metrics_dir, 'detailed_metrics.csv'))
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestep'], df['td_error_mean'], linewidth=2, color='red')
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Mean |TD Error|', fontsize=12)
    plt.title('TD Error Over Training', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'td_error.png'), dpi=300)
    print(f"✓ Saved: {save_dir}/td_error.png")
    plt.close()


def plot_exploration_rate(metrics_dir, save_dir):
    """Plot exploration rate decay."""
    df = pd.read_csv(os.path.join(metrics_dir, 'training_metrics.csv'))
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestep'], df['rollout/exploration_rate'], linewidth=2, color='green')
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Exploration Rate (ε)', fontsize=12)
    plt.title('Exploration-Exploitation Schedule', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exploration_rate.png'), dpi=300)
    print(f"✓ Saved: {save_dir}/exploration_rate.png")
    plt.close()


def plot_loss(metrics_dir, save_dir):
    """Plot training loss."""
    df = pd.read_csv(os.path.join(metrics_dir, 'training_metrics.csv'))
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestep'], df['train/loss'], linewidth=2, color='orange')
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('DQN Loss Over Training', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_loss.png'), dpi=300)
    print(f"✓ Saved: {save_dir}/training_loss.png")
    plt.close()


def plot_training_vs_eval(run_dir, save_dir):
    """Compare training and evaluation performance."""
    train_df = pd.read_csv(os.path.join(run_dir, 'metrics', 'training_metrics.csv'))
    eval_data = np.load(os.path.join(run_dir, 'eval_logs', 'evaluations.npz'))
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_df['timestep'], train_df['ep_rew_mean'], 
             label='Training', alpha=0.6, linewidth=2)
    plt.plot(eval_data['timesteps'], eval_data['results'].mean(axis=1),
             label='Evaluation', marker='o', linewidth=2, markersize=8)
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Training vs Evaluation Performance', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_vs_eval.png'), dpi=300)
    print(f"✓ Saved: {save_dir}/train_vs_eval.png")
    plt.close()


def generate_summary_stats(run_dir):
    """Print summary statistics."""
    with open(os.path.join(run_dir, 'training_config.json'), 'r') as f:
        config = json.load(f)
    
    with open(os.path.join(run_dir, 'training_summary.json'), 'r') as f:
        summary = json.load(f)
    
    train_df = pd.read_csv(os.path.join(run_dir, 'metrics', 'training_metrics.csv'))
    detailed_df = pd.read_csv(os.path.join(run_dir, 'metrics', 'detailed_metrics.csv'))
    
    # Calculate sample reuse ratio
    total_updates = train_df['train/n_updates'].iloc[-1]
    buffer_size = config['buffer_size']
    batch_size = config['batch_size']
    sample_reuse_ratio = (total_updates * batch_size) / buffer_size
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY STATISTICS")
    print("="*70)
    print(f"\nRun ID: {config['timestamp']}")
    print(f"Duration: {summary['training_duration_hours']:.2f} hours")
    print(f"\n--- Performance ---")
    print(f"Final Average Reward: {train_df['ep_rew_mean'].iloc[-1]:.2f}")
    print(f"Best Average Reward: {train_df['ep_rew_mean'].max():.2f}")
    print(f"Final Episode Length: {train_df['ep_len_mean'].iloc[-1]:.1f}")
    print(f"\n--- Learning ---")
    print(f"Total Updates: {int(total_updates):,}")
    print(f"Final Loss: {train_df['train/loss'].iloc[-1]:.6f}")
    print(f"Final Exploration Rate: {summary['final_exploration_rate']:.4f}")
    print(f"Sample Reuse Ratio: {sample_reuse_ratio:.2f}")
    print(f"\n--- Q-Values ---")
    print(f"Final Mean Q-Value: {detailed_df['q_value_mean'].iloc[-1]:.2f}")
    print(f"Final Q-Value Std: {detailed_df['q_value_std'].iloc[-1]:.2f}")
    print(f"Max Q-Value (ever): {detailed_df['q_value_max'].max():.2f}")
    print(f"\n--- TD Error ---")
    print(f"Final TD Error: {detailed_df['td_error_mean'].iloc[-1]:.4f}")
    print(f"Min TD Error: {detailed_df['td_error_mean'].min():.4f}")
    print(f"\n--- Configuration ---")
    print(f"Collision Penalty: {config['collision_penalty']}")
    print(f"Success Reward: {config['success_reward']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Buffer Size: {config['buffer_size']:,}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze training metrics')
    parser.add_argument('run_dir', help='Path to training run directory (e.g., runs/20231201_120000/)')
    args = parser.parse_args()
    
    run_dir = args.run_dir
    if not os.path.exists(run_dir):
        print(f"Error: Directory not found: {run_dir}")
        sys.exit(1)
    
    metrics_dir = os.path.join(run_dir, 'metrics')
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"\nAnalyzing training run: {run_dir}")
    print(f"Generating plots in: {plots_dir}\n")
    
    # Generate all plots
    plot_learning_curve(metrics_dir, plots_dir)
    plot_q_values(metrics_dir, plots_dir)
    plot_td_error(metrics_dir, plots_dir)
    plot_exploration_rate(metrics_dir, plots_dir)
    plot_loss(metrics_dir, plots_dir)
    plot_training_vs_eval(run_dir, plots_dir)
    
    # Print summary statistics
    generate_summary_stats(run_dir)
    
    print(f"✓ All plots saved to: {plots_dir}\n")


if __name__ == "__main__":
    main()

