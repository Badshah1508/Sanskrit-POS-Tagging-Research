#!/bin/bash
#SBATCH --job-name=panchatantra_pos
#SBATCH --output=outputs/Panchatantra_output/logs/nb_exec_%j.out
#SBATCH --error=outputs/Panchatantra_output/logs/nb_exec_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=shard:24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

cd /home/siba/P25DS023/Cross-Domain-23
mkdir -p outputs/Panchatantra_output/logs

echo "=========================================="
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

nvidia-smi

source /home/siba/anaconda3/etc/profile.d/conda.sh
conda activate muril_env

export PYTHONUNBUFFERED=1

jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/Sanskrit_Panchatantra_CrossDomain_Experiments.ipynb

echo "Notebook execution finished at: $(date)"
