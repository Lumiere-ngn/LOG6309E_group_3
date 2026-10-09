#!/bin/bash
#SBATCH --job-name=log6309e_gpucheck
#SBATCH --account=def-hamdaqa_gpu
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --exclude=fc10512,fc10612
#SBATCH --output=/scratch/marwan/log6309e/logs/gpucheck_%j.out
# Torch 2.7.1 saw no GPU in job 63826919 (tests/cuda_ok.py, torch first) but saw it in job 63829776
# (tests/cuda_diag.py, which calls cuInit through ctypes before importing torch). Same venv and modules.
# This runs torch-first, then driver-first, then torch-first again, and names the node.
cd /scratch/marwan/log6309e
module load StdEnv/2023 python/3.12 arrow
source .venv-gpu/bin/activate
echo "node $(hostname)"
echo "== 1. torch first (cuda_ok.py)"; python tests/cuda_ok.py; echo "exit $?"
echo "== 2. driver first (cuda_diag.py)"; python tests/cuda_diag.py | tail -2
echo "== 3. torch first again"; python tests/cuda_ok.py; echo "exit $?"
