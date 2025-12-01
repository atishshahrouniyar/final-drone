# Training Logs and Metrics Documentation

## Directory Structure

After training, you'll find the following structure in `./runs/[timestamp]/`:

```
runs/
└── YYYYMMDD_HHMMSS/
    ├── training_config.json       # Complete training configuration
    ├── training_summary.json      # Final training statistics
    ├── checkpoints/
    │   ├── dqn_random_point_nav_100k.zip
    │   ├── dqn_random_point_nav_250k.zip
    │   ├── dqn_random_point_nav_500k.zip
    │   ├── dqn_random_point_nav_750k.zip
    │   ├── dqn_random_point_nav_1000k.zip
    │   └── best_model.zip         # Best model based on evaluation
    ├── metrics/
    │   ├── training_metrics.csv   # High-level training metrics
    │   └── detailed_metrics.csv   # Detailed Q-values, TD errors
    ├── tensorboard/
    │   └── DQN_X/                 # TensorBoard event files
    └── eval_logs/
        ├── evaluations.npz        # Evaluation results
        └── monitor.csv            # Episode statistics
```

---

## Metrics Files

### 1. `training_config.json`
**Purpose**: Complete snapshot of training hyperparameters

**Contents**:
- Learning rate, batch size, buffer size
- Reward structure (collision penalty, success reward, etc.)
- Environment parameters (arena size, altitude limits)
- Evaluation frequency and checkpoint milestones

**Use**: Reproduce exact training configuration, compare different runs

---

### 2. `training_metrics.csv`
**Purpose**: High-level training progress

**Columns**:
- `timestep`: Current training timestep
- `episode`: Episode number
- `fps`: Frames per second (training speed)
- `elapsed_time`: Total training time (seconds)
- `ep_len_mean`: Average episode length
- `ep_rew_mean`: Average episode reward
- `train/loss`: DQN loss value
- `train/learning_rate`: Current learning rate
- `train/n_updates`: Number of gradient updates
- `rollout/exploration_rate`: Current epsilon value
- `time/total_timesteps`: Cumulative timesteps

**Frequency**: Every 5,000 timesteps

**Use**: 
- Plot learning curves (reward over time)
- Analyze training speed
- Monitor loss convergence
- Track exploration-exploitation balance

---

### 3. `detailed_metrics.csv`
**Purpose**: Detailed RL algorithm metrics

**Columns**:
- `timestep`: Current training timestep
- `q_value_mean`: Average Q-value across sampled batch
- `q_value_std`: Standard deviation of Q-values
- `q_value_max`: Maximum Q-value
- `q_value_min`: Minimum Q-value
- `td_error_mean`: Average TD error magnitude
- `td_error_std`: Standard deviation of TD errors
- `buffer_size`: Current replay buffer size
- `exploration_rate`: Epsilon value
- `learning_rate`: Current learning rate
- `loss`: Training loss

**Frequency**: Every 1,000 timesteps

**Use**:
- Analyze Q-value distribution and stability
- Monitor TD error (indicator of learning progress)
- Track replay buffer utilization
- Diagnose training issues (exploding Q-values, etc.)

---

### 4. `training_summary.json`
**Purpose**: Final training statistics

**Contents**:
- Total training duration (seconds and hours)
- Final exploration rate
- Final buffer size
- Completion timestamp

**Use**: Quick reference for training run statistics

---

### 5. `eval_logs/evaluations.npz`
**Purpose**: Evaluation episode results

**Contents** (NumPy arrays):
- `timesteps`: When evaluations occurred
- `results`: Episode rewards
- `ep_lengths`: Episode lengths

**Frequency**: Every 100,000 timesteps

**Use**: 
- Plot evaluation performance over time
- Compare with training performance
- Assess generalization

---

### 6. `eval_logs/monitor.csv`
**Purpose**: Per-episode statistics during evaluation

**Columns**:
- `r`: Episode reward
- `l`: Episode length
- `t`: Wall clock time

**Use**: Detailed episode-by-episode analysis

---

## TensorBoard Logs

### Viewing TensorBoard
```bash
tensorboard --logdir ./runs/[timestamp]/tensorboard/
```

### Available Metrics

**Scalars**:
- `rollout/ep_len_mean`: Episode length
- `rollout/ep_rew_mean`: Episode reward
- `rollout/exploration_rate`: Epsilon
- `time/fps`: Training speed
- `train/learning_rate`: Learning rate
- `train/loss`: DQN loss
- `train/n_updates`: Update count
- `custom/fps`: Custom FPS metric
- `custom/elapsed_time`: Training duration
- `metrics/q_value_mean`: Q-value statistics
- `metrics/q_value_std`: Q-value variance
- `metrics/td_error_mean`: TD error
- `metrics/buffer_size`: Replay buffer size

---

## Plotting and Analysis Examples

### 1. Learning Curve (Reward over Time)
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('runs/TIMESTAMP/metrics/training_metrics.csv')
plt.plot(df['timestep'], df['ep_rew_mean'])
plt.xlabel('Timesteps')
plt.ylabel('Average Reward')
plt.title('Training Learning Curve')
plt.show()
```

### 2. Q-Value Evolution
```python
df = pd.read_csv('runs/TIMESTAMP/metrics/detailed_metrics.csv')
plt.plot(df['timestep'], df['q_value_mean'], label='Mean')
plt.fill_between(df['timestep'], 
                 df['q_value_mean'] - df['q_value_std'],
                 df['q_value_mean'] + df['q_value_std'],
                 alpha=0.3)
plt.xlabel('Timesteps')
plt.ylabel('Q-Value')
plt.legend()
plt.title('Q-Value Statistics Over Training')
plt.show()
```

### 3. TD Error Analysis
```python
df = pd.read_csv('runs/TIMESTAMP/metrics/detailed_metrics.csv')
plt.plot(df['timestep'], df['td_error_mean'])
plt.xlabel('Timesteps')
plt.ylabel('Mean |TD Error|')
plt.title('TD Error Over Training')
plt.show()
```

### 4. Exploration Rate Decay
```python
df = pd.read_csv('runs/TIMESTAMP/metrics/training_metrics.csv')
plt.plot(df['timestep'], df['rollout/exploration_rate'])
plt.xlabel('Timesteps')
plt.ylabel('Epsilon (Exploration Rate)')
plt.title('Exploration-Exploitation Schedule')
plt.show()
```

### 5. Training vs Evaluation Performance
```python
import numpy as np

# Training performance
train_df = pd.read_csv('runs/TIMESTAMP/metrics/training_metrics.csv')

# Evaluation performance
eval_data = np.load('runs/TIMESTAMP/eval_logs/evaluations.npz')

plt.plot(train_df['timestep'], train_df['ep_rew_mean'], 
         label='Training', alpha=0.6)
plt.plot(eval_data['timesteps'], eval_data['results'].mean(axis=1), 
         label='Evaluation', marker='o')
plt.xlabel('Timesteps')
plt.ylabel('Average Reward')
plt.legend()
plt.title('Training vs Evaluation Performance')
plt.show()
```

---

## Sample Reuse Ratio

**Definition**: Number of times each transition is sampled from replay buffer

**Calculation**:
```python
total_updates = df['train/n_updates'].iloc[-1]
buffer_size = df['buffer_size'].iloc[-1]
batch_size = 128  # From config

sample_reuse_ratio = (total_updates * batch_size) / buffer_size
```

**Typical Range**: 10-50 for good learning efficiency

---

## Key Performance Indicators

Monitor these for successful training:

1. **Reward Trend**: Should increase over time
2. **Episode Length**: May vary, but should stabilize
3. **Loss**: Should generally decrease (some fluctuation is normal)
4. **Q-Values**: Should increase and stabilize (not explode)
5. **TD Error**: Should decrease as learning progresses
6. **Exploration Rate**: Should decay from 1.0 to ~0.02
7. **FPS**: Should be stable (indicates no performance issues)

---

## Troubleshooting

### Issue: Reward not improving
- Check Q-values (exploding or stuck at 0?)
- Check TD error (very high?)
- Verify learning rate and exploration schedule
- Review loss values (NaN or infinity?)

### Issue: Training stuck
- Check FPS (should be ~100-1000)
- Verify buffer is filling (buffer_size increasing)
- Check for evaluation hanging (long gaps in metrics)

### Issue: Unstable learning
- Monitor Q-value std (too high = instability)
- Check TD error variance
- Consider reducing learning rate or increasing batch size

---

## Checkpoint Usage

Load a specific checkpoint:
```python
from stable_baselines3 import DQN

model = DQN.load("runs/TIMESTAMP/checkpoints/dqn_random_point_nav_500k")
```

Compare different checkpoints:
```python
for checkpoint in [100, 250, 500, 750, 1000]:
    model = DQN.load(f"runs/TIMESTAMP/checkpoints/dqn_random_point_nav_{checkpoint}k")
    # Evaluate...
```

---

## Report Recommendations

Include in your report:
1. Learning curve plot (reward vs timesteps)
2. Q-value evolution plot
3. Training configuration table
4. Final performance statistics
5. Sample reuse ratio calculation
6. Training duration and FPS
7. Comparison of checkpoints at different stages
8. Exploration schedule visualization
9. TD error convergence plot
10. Loss curve over training

