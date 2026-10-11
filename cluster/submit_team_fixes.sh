#!/bin/bash
# Submit the 2026-10-11 team-fixes reruns on Fir: step 7 (GPU), the sanity checks (CPU), then rank and explain
# after step 7 succeeds. Run on a login node: bash cluster/submit_team_fixes.sh
set -euo pipefail
cd /scratch/marwan/log6309e
mkdir -p logs results/sanity
E=$(sbatch --parsable cluster/ext.sbatch)
S=$(sbatch --parsable cluster/sanity.sbatch)
R=$(sbatch --parsable --dependency=afterok:"$E" --account=def-hamdaqa_cpu --time=01:00:00 --cpus-per-task=8 \
  --mem=32G --output=logs/rank_explain_%j.out \
  --wrap="cd /scratch/marwan/log6309e && module load StdEnv/2023 python/3.12 arrow && source .venv/bin/activate && cd src && python rank.py && python explain.py")
echo "ext=$E sanity=$S rank_explain=$R"
squeue -u "$USER" -h -o "%i %j %T %R"
