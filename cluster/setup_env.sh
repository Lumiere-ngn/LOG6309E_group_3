#!/bin/bash
# Build the Python environment on a Fir login node (compute nodes may have no internet).
# Usage: bash cluster/setup_env.sh /scratch/$USER/log6309e
set -euo pipefail
ROOT=${1:?usage: setup_env.sh <repo root>}
module load StdEnv/2023 python/3.12 cuda/12.6 arrow
python -m venv "$ROOT/.venv"
source "$ROOT/.venv/bin/activate"
pip install -q --upgrade pip
# pyarrow comes from the arrow module. Alliance wheelhouse first (builds against the cluster's libraries); drain3 is only on PyPI.
pip install -q --no-index numpy pandas scikit-learn scipy statsmodels tqdm torch
pip install -q drain3
python -c "import numpy, pandas, sklearn, statsmodels, torch, drain3; print('env ok', torch.__version__, sklearn.__version__)"
