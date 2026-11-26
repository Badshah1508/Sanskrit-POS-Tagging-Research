#!/bin/bash
#SBATCH --job-name=test_nb
#SBATCH --output=logs/test_nb_%j.out
#SBATCH --error=logs/test_nb_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=shard:16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

cd /home/siba/P25DS023/Cross-Domain-23
mkdir -p logs outputs

source /home/siba/anaconda3/etc/profile.d/conda.sh
conda activate muril_env

# Execute the notebook headlessly and save the executed copy with outputs
jupyter nbconvert \
    --to notebook \
    --execute notebooks/test.ipynb \
    --output ../outputs/test_executed_${SLURM_JOB_ID}.ipynb \
    --ExecutePreprocessor.timeout=3600
