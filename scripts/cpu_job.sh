#!/bin/bash
#SBATCH --job-name=cpu_test
#SBATCH --output=logs/cpu_output_%j.log
#SBATCH --error=logs/cpu_error_%j.log
#SBATCH --time=00:05:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=========================================="
echo "Hello from $(hostname)"
echo "Current directory: $(pwd)"
echo "Date: $(date)"
echo "=========================================="

python3 --version
