#!/bin/bash
#SBATCH --job-name=log6309e_rq1_cpu
#SBATCH --account=def-hamdaqa_cpu
#SBATCH --array=0-4
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/marwan/log6309e/logs/rq1_cpu_%A_%a.out
# Step 6 on CPU, one array task per seed. Torch saw no GPU on Fir (gpucheck jobs, 2026-10-09). One HDFS MLP took
# 1,739 s on 16 threads (job 63821173), so a seed's two HDFS MLPs need about an hour; 2.5 h leaves margin.
# Needs data/features/{HDFS,BGL}/mcv.parquet (built by steps_3_6.sbatch). Merge afterwards with:
#   python src/replicate.py --merge results_cpu/seed_*/rq1_runs.csv --out results
set -euo pipefail
cd /scratch/marwan/log6309e
module load StdEnv/2023 python/3.12 arrow
source .venv/bin/activate
SEED=$SLURM_ARRAY_TASK_ID
echo "array $SLURM_ARRAY_JOB_ID task $SEED on $(hostname) at $(date -u +%FT%TZ)"
for ds in HDFS BGL; do
  test -f "data/features/$ds/mcv.parquet" || { echo "missing data/features/$ds/mcv.parquet"; exit 2; }
done
python src/replicate.py --seeds "$SEED" --out "results_cpu/seed_$SEED"
echo "done at $(date -u +%FT%TZ)"
