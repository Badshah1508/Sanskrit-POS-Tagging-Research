#!/bin/bash
#SBATCH --job-name=sanskrit_remaining
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

echo "=========================================="
echo "STAGE 1: Running Sanskrit-BERT (M5)..."
echo "=========================================="
python3 src/experiments/run_experiments.py --model sanskrit_bert || echo "Sanskrit-BERT exited with code $?"

echo "=========================================="
echo "STAGE 2: Running XLM-RoBERTa (M2)..."
echo "=========================================="
python3 src/experiments/run_experiments.py --model xlm_roberta || echo "XLM-RoBERTa exited with code $?"

echo "=========================================="
echo "STAGE 3: Running IndicBERT-v2 (M3)..."
echo "=========================================="
python3 src/experiments/run_experiments.py --model indicbert_v2 || echo "IndicBERT-v2 exited with code $?"

echo "All remaining models completed at: $(date)"
