#!/bin/bash

# Basic SLURM directives -------------------------------------------------
#SBATCH --job-name="dqn_training"
#SBATCH --output="slurm_logs/dqn_training.%j.out"
#SBATCH --error="slurm_logs/dqn_training.%j.err"
#SBATCH --partition=gpu-shared
#SBATCH --account=wsu133
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

echo "SLURM job ${SLURM_JOB_ID:-N/A} running on ${HOSTNAME}"
echo "Working directory: ${SLURM_SUBMIT_DIR}"

# -----------------------------------------------------------------------
# 1) Load modules / activate conda
# -----------------------------------------------------------------------
module purge
module load gpu || true        # harmless if module absent
module load anaconda3/2021.05/q4munrg || true  # use default anaconda if available

source ~/.bashrc
conda activate /home/arouniyar/drones

which python
python --version

# -----------------------------------------------------------------------
# 2) Move to the project directory (where we submitted the job)
# -----------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR}"

# -----------------------------------------------------------------------
# 3) Run training
# -----------------------------------------------------------------------
echo "Starting training at $(date)"
srun python -u train_dqn.py
EXIT_CODE=$?
echo "Training finished with exit code ${EXIT_CODE} at $(date)"

exit ${EXIT_CODE}

