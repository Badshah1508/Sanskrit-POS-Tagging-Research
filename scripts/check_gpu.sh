#!/bin/bash
#SBATCH --job-name=aurora_gpu_check
#SBATCH --output=logs/gpu_check_%j.out
#SBATCH --error=logs/gpu_check_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=shard:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00

cd /home/siba/P25DS023/Cross-Domain-23
mkdir -p logs

echo "=========================================="
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

echo "=== NVIDIA-SMI on Compute Node ==="
nvidia-smi

echo "=== Activating Conda Environment ==="
source /home/siba/anaconda3/etc/profile.d/conda.sh
conda activate muril_env

echo "=== PyTorch CUDA H100 Tensor Test ==="
python3 -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Count:', torch.cuda.device_count())
    print('Device Name:', torch.cuda.get_device_name(0))
    print('Current Device Index:', torch.cuda.current_device())
    a = torch.randn(2048, 2048, device='cuda')
    b = torch.randn(2048, 2048, device='cuda')
    c = torch.matmul(a, b)
    print('Matrix Multiplication on NVIDIA H100 Successful! Output shape:', c.shape)
else:
    print('ERROR: CUDA not available inside GPU allocation!')
"
