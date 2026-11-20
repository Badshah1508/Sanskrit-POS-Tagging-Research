#!/bin/bash
#SBATCH --job-name=sanskrit_experiments
#SBATCH --output=logs/exp_%j.out
#SBATCH --error=logs/exp_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=shard:16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

cd /home/siba/P25DS023/Cross-Domain-23
mkdir -p logs outputs

echo "=========================================="
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

nvidia-smi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source /home/siba/anaconda3/etc/profile.d/conda.sh
conda activate muril_env

MODEL_ARG="${1:-all}"
echo "Running Sanskrit Cross-Domain Experiments for model: $MODEL_ARG"

python3 src/experiments/run_experiments.py --model "$MODEL_ARG"

echo "Experiments finished at: $(date)"
