#!/bin/bash
#SBATCH --job-name=log6309e_step7
#SBATCH --account=def-hamdaqa_gpu
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=fc10512,fc10612
#SBATCH --output=/scratch/marwan/log6309e/logs/step7_%j.out
# Step 7: RF, MLP, Isolation Forest and autoencoder on HDFS over 10 resamples (src/extension.py).
# .venv-gpu holds torch 2.7.1 (CUDA 12.6); fc10612 shows no GPU to torch (cluster/steps_3_6.sbatch).
set -euo pipefail
cd /scratch/marwan/log6309e
module load StdEnv/2023 python/3.12 arrow
source .venv-gpu/bin/activate
echo "job $SLURM_JOB_ID on $(hostname) at $(date -u +%FT%TZ)"
python tests/cuda_ok.py || { echo "torch sees no GPU; stopping before any run"; exit 3; }
cd src
python extension.py --out ../results/ext
echo "done at $(date -u +%FT%TZ)"
