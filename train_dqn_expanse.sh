#!/bin/bash

# ========================================================================
# SLURM Batch Script for Expanse GPU-Shared DQN Training
# ========================================================================
#
# SBATCH Directives: These configure the job resources.
# ------------------------------------------------------------------------
# Job Name: Name for your job (e.g., in squeue)
#SBATCH --job-name="dqn_training"

# Output/Error Files: placed in a local slurm_logs folder next to this script.
#SBATCH --output="slurm_logs/dqn_training.%j.out"
#SBATCH --error="slurm_logs/dqn_training.%j.err"

# Partition: MUST be 'gpu-shared' for cost-effective 1-GPU runs.
#SBATCH --partition=gpu-shared

# Account: MANDATORY on Expanse. 
# To find your account, run: sacctmgr show assoc user=$USER
# Or check with: sbank balance
#SBATCH --account=wsu133

# Resource Request: 1 Node, 1 Task (Python script), 1 GPU.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1

# CPUs & Memory: Request 8 CPUs and 64GB of RAM.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Walltime: Max for gpu-shared is 48 hours.
#SBATCH --time=48:00:00

# Robustness: DO NOT requeue the job if the node fails.
#SBATCH --no-requeue

# ------------------------------------------------------------------------
# Job Script Logic: These commands run on the allocated compute node.
# ------------------------------------------------------------------------

echo "========================================================================"
echo "SLURM Job $SLURM_JOB_ID starting on $HOSTNAME"
echo "Allocated $SLURM_NNODES node(s), $SLURM_GPUS_ON_NODE GPU(s), and $SLURM_CPUS_PER_TASK CPUs."
echo "========================================================================"

# Quick check: Verify account and partition access
echo "Verifying account and partition access..."
sacctmgr show assoc user=$USER format=account,partition 2>/dev/null || echo "Note: Could not verify account (this is OK if job starts)"

# 1. Define I/O Paths (The "write-local, copy-on-exit" strategy)
# Use the $SLURM_TMPDIR variable. SLURM creates this directory
# automatically on the node's local NVMe and gives your job write permission.
export SCRATCH_DIR="$SLURM_TMPDIR"

# This is the *persistent* Lustre storage for FINAL results.
export FINAL_OUTPUT_DIR="/home/arouniyar/drones/lidar"

LOG_DIR="$SLURM_SUBMIT_DIR/slurm_logs"

# Create these directories
mkdir -p $SCRATCH_DIR
mkdir -p $FINAL_OUTPUT_DIR
mkdir -p "$LOG_DIR"

echo "Node-Local Scratch (for high-freq I/O): $SCRATCH_DIR"
echo "Persistent Lustre (for final results): $FINAL_OUTPUT_DIR"

# 2. Set up Software Environment
echo "Setting up software environment..."

module purge
source /etc/profile.d/modules.sh
module load gpu
module load slurm
module load anaconda3/2021.05/q4munrg

# Activate the pre-built Conda environment
source ~/.bashrc
conda activate /home/arouniyar/drones

echo "Conda environment activated:"
which python
python --version

echo "PyTorch CUDA check:"
python -c 'import torch; print(f"CUDA Available: {torch.cuda.is_available()}"); print(f"CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}")'

# Check for required packages
echo "Checking required packages..."
python -c "import stable_baselines3; import gymnasium; import pybullet; print('All packages available')" || {
    echo "Error: Missing required packages"
    exit 1
}

# 3. Change to project directory
# Adjust this path to where your training code is located on Expanse
cd /home/arouniyar/drones/lidar || {
    echo "Error: Could not change to project directory"
    echo "Please update the path in the script to match your Expanse directory structure"
    exit 1
}

# Verify Python script exists
if [ ! -f "train_dqn.py" ]; then
    echo "Error: train_dqn.py not found in current directory"
    echo "Current directory: $(pwd)"
    echo "Files in directory:"
    ls -la
    exit 1
fi

# Verify environment file exists
if [ ! -f "enhanced_navigation_env.py" ]; then
    echo "Error: enhanced_navigation_env.py not found in current directory"
    exit 1
fi

# 4. Set up working directory in scratch for fast I/O
# Copy necessary files to scratch (optional, for very fast I/O)
WORK_DIR="$SCRATCH_DIR/work"
mkdir -p $WORK_DIR

# Copy training script and environment to scratch
cp train_dqn.py $WORK_DIR/
cp enhanced_navigation_env.py $WORK_DIR/

# Change to scratch working directory
cd $WORK_DIR

# Set output directories to scratch for performance
export MODEL_OUTPUT_DIR="$SCRATCH_DIR/models"
export LOGS_OUTPUT_DIR="$SCRATCH_DIR/detailed_logs"
mkdir -p $MODEL_OUTPUT_DIR
mkdir -p $LOGS_OUTPUT_DIR

# 5. Run the Training
echo "========================================================================"
echo "Starting Python training script..."
echo "========================================================================"

# Use srun to launch the task with GPU allocation
# Note: The training script saves to current directory, so we'll be in scratch
srun python train_dqn.py

TRAINING_EXIT_CODE=$?

# 6. Data Offload (Critical Final Step)
echo "========================================================================"
echo "Training script exited with code: $TRAINING_EXIT_CODE"
echo "Copying all data from local NVMe ($SCRATCH_DIR) to persistent Lustre ($FINAL_OUTPUT_DIR)..."
echo "========================================================================"

# Create final output directories
mkdir -p $FINAL_OUTPUT_DIR/models
mkdir -p $FINAL_OUTPUT_DIR/detailed_logs

# Copy model files
if [ -f "$WORK_DIR/dqn_detailed_model.zip" ]; then
    echo "Copying model checkpoint..."
    cp "$WORK_DIR/dqn_detailed_model.zip" "$FINAL_OUTPUT_DIR/models/" || true
    cp "$WORK_DIR/dqn_detailed_model"* "$FINAL_OUTPUT_DIR/models/" 2>/dev/null || true
fi

# Copy TensorBoard logs
if [ -d "$WORK_DIR/detailed_logs" ]; then
    echo "Copying TensorBoard logs..."
    cp -r "$WORK_DIR/detailed_logs"/* "$FINAL_OUTPUT_DIR/detailed_logs/" 2>/dev/null || true
fi

# Also check scratch directories directly
if [ -d "$LOGS_OUTPUT_DIR" ]; then
    echo "Copying logs from scratch logs directory..."
    cp -r "$LOGS_OUTPUT_DIR"/* "$FINAL_OUTPUT_DIR/detailed_logs/" 2>/dev/null || true
fi

if [ -d "$MODEL_OUTPUT_DIR" ]; then
    echo "Copying models from scratch models directory..."
    cp -r "$MODEL_OUTPUT_DIR"/* "$FINAL_OUTPUT_DIR/models/" 2>/dev/null || true
fi

# Copy any other output files
if [ -f "$WORK_DIR/dqn_detailed_model.pkl" ]; then
    cp "$WORK_DIR/dqn_detailed_model.pkl" "$FINAL_OUTPUT_DIR/models/" 2>/dev/null || true
fi

echo "========================================================================"
echo "Data copy complete."
echo "Results saved to: $FINAL_OUTPUT_DIR"
echo "  - Models: $FINAL_OUTPUT_DIR/models/"
echo "  - Logs: $FINAL_OUTPUT_DIR/detailed_logs/"
echo "SLURM Job $SLURM_JOB_ID finished with exit code: $TRAINING_EXIT_CODE"
echo "========================================================================"

# Exit with the training script's exit code
exit $TRAINING_EXIT_CODE

