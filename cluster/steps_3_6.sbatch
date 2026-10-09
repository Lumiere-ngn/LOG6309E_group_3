#!/bin/bash
#SBATCH --job-name=log6309e_steps3_6
#SBATCH --account=def-hamdaqa_gpu
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=fc10512
#SBATCH --output=/scratch/marwan/log6309e/logs/steps3_6_%j.out
# Steps 3 to 6 of Marwan-plan.md: Drain parse, MCV, collinearity, RF/MLP replication.
# Submit from /scratch (not /home): cd /scratch/marwan/log6309e && sbatch cluster/steps_3_6.sbatch
# SEEDS overrides the seed list, joined with "+" because sbatch --export splits on commas.
set -euo pipefail
ROOT=/scratch/marwan/log6309e
SEEDS=${SEEDS:-0+1+2+3+4}
cd "$ROOT"
module load StdEnv/2023 python/3.12 cuda/12.6 arrow
source .venv/bin/activate
echo "job $SLURM_JOB_ID on $(hostname) at $(date -u +%FT%TZ); SEEDS=$SEEDS; git $(git rev-parse --short HEAD 2>/dev/null || echo none)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for ds in HDFS BGL; do
  if [ ! -f "data/parsed/$ds/$ds.log_structured.parquet" ]; then
    python src/parse_drain.py "$ds"
  fi
  python src/build_mcv.py "$ds"
done
python src/validate_parse.py
python src/replicate.py --seeds ${SEEDS//+/ }
pip freeze > results/pip_freeze.txt
echo "done at $(date -u +%FT%TZ)"
