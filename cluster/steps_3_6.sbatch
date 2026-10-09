#!/bin/bash
#SBATCH --job-name=log6309e_steps3_6
#SBATCH --account=def-hamdaqa_gpu
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=fc10512,fc10612
#SBATCH --output=/scratch/marwan/log6309e/logs/steps3_6_%j.out
# Steps 3 to 6 of Marwan-plan.md: Drain parse, MCV, collinearity, RF/MLP replication.
# Submit from /scratch (not /home): cd /scratch/marwan/log6309e && sbatch cluster/steps_3_6.sbatch
# SEEDS overrides the seed list, joined with "+" because sbatch --export splits on commas.
# OUT is the results folder (default results). REQUIRE_GPU=0 lets the MLP run on CPU.
set -euo pipefail
ROOT=/scratch/marwan/log6309e
SEEDS=${SEEDS:-0+1+2+3+4}
OUT=${OUT:-results}
cd "$ROOT"
module load StdEnv/2023 python/3.12 arrow
# VENV=.venv-gpu holds torch 2.7.1 (CUDA 12.6, cluster/setup_env_gpu.sh). .venv's 2.14.1 targets CUDA 13.2, which the
# nodes' driver 580 (CUDA 13.0) cannot run. Node fc10612 shows no GPU to torch at all (job 63873126); fc10616 works.
VENV=${VENV:-.venv-gpu}
source "$VENV/bin/activate"
echo "job $SLURM_JOB_ID on $(hostname) at $(date -u +%FT%TZ); SEEDS=$SEEDS; OUT=$OUT; VENV=$VENV"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if [ "${REQUIRE_GPU:-1}" = 1 ]; then
  if ! python tests/cuda_ok.py; then
    echo "retrying with cuda/12.6 loaded"
    module load cuda/12.6
    python tests/cuda_ok.py || { echo "torch sees no GPU; stopping before any run"; exit 3; }
  fi
fi

for ds in HDFS BGL; do
  if [ ! -f "data/parsed/$ds/$ds.log_structured.parquet" ]; then
    python src/parse_drain.py "$ds"
  fi
  if [ ! -f "data/features/$ds/mcv.parquet" ]; then
    python src/build_mcv.py "$ds"
  fi
done
python src/validate_parse.py
python src/replicate.py --seeds ${SEEDS//+/ } --out "$OUT"
pip freeze > "$OUT/pip_freeze.txt"
echo "done at $(date -u +%FT%TZ)"
