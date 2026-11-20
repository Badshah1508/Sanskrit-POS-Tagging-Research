#!/bin/bash
#SBATCH --job-name=sanskrit_pos_nb
#SBATCH --output=logs/notebook_run_%j.out
#SBATCH --error=logs/notebook_run_%j.err
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

source /home/siba/anaconda3/etc/profile.d/conda.sh
conda activate muril_env

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_NOTEBOOK="outputs/notebook_run_${SLURM_JOB_ID}_${TIMESTAMP}.ipynb"

echo "Executing Sanskrit POS Notebook headlessly via nbconvert..."
jupyter nbconvert \
    --to notebook \
    --execute notebooks/Sanskrit_CrossDomain_POS_Experiments_Optimized.ipynb \
    --output "$OUT_NOTEBOOK" \
    --ExecutePreprocessor.timeout=14400

echo "Finished at: $(date)"
echo "Executed notebook saved to: $OUT_NOTEBOOK"
