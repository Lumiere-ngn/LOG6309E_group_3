#!/bin/bash
#SBATCH --job-name=log6309e_gpucheck
#SBATCH --account=def-hamdaqa_gpu
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --exclude=fc10512
#SBATCH --output=/scratch/marwan/log6309e/logs/gpucheck_%j.out
# Why torch sees no GPU on Fir while nvidia-smi does (jobs 63823719, 63824604, 63825606, 63826919, 2026-10-09).
cd /scratch/marwan/log6309e
module load StdEnv/2023 python/3.12 arrow
source .venv-gpu/bin/activate
echo "== batch step"
python tests/cuda_diag.py
echo "== srun step"
srun --ntasks=1 python tests/cuda_diag.py
