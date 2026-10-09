#!/bin/bash
# A second venv whose torch matches Fir's GPU driver (580, CUDA 13.0). torch 2.14.1+computecanada targets CUDA 13.2
# and saw no GPU with or without the cuda modules (jobs 63823719, 63824604, 63825606 on 2026-10-09).
# Usage: bash cluster/setup_env_gpu.sh /scratch/$USER/log6309e [torch version]
set -euo pipefail
ROOT=${1:?usage: setup_env_gpu.sh <repo root> [torch version]}
TORCH=${2:-2.7.1}
module load StdEnv/2023 python/3.12 arrow
python -m venv "$ROOT/.venv-gpu"
source "$ROOT/.venv-gpu/bin/activate"
pip install -q --upgrade pip
pip install -q --no-index numpy pandas scikit-learn scipy statsmodels tqdm "torch==$TORCH"
pip install -q drain3
python -c "import torch, sklearn; print('env ok', torch.__version__, 'CUDA', torch.version.cuda, 'sklearn', sklearn.__version__)"
