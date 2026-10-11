#!/bin/bash
#SBATCH --job-name=log6309e_sanity
#SBATCH --account=def-hamdaqa_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/marwan/log6309e/logs/sanity_%j.out
# Sanity checks (plan/team-fixes-2026-10-11.md, item 1) on CPU: src/sanity_checks.py writes results/sanity/.
# Time: the BGL checks take minutes. The HDFS unseen-session check fits one HDFS MLP per seed, and one HDFS MLP took
# 1,739 s on 16 threads (job 63821173), so five seeds need about 2.4 h; 4 h leaves margin.
# Needs data/features/{HDFS,BGL}/mcv.parquet (built by steps_3_6.sbatch).
# CHECKS (joined with "+") reruns a subset, e.g. CHECKS=bgl_baselines+hdfs_overlap sbatch cluster/sanity.sbatch
set -euo pipefail
cd /scratch/marwan/log6309e
module load StdEnv/2023 python/3.12 arrow
source .venv/bin/activate
echo "job $SLURM_JOB_ID on $(hostname) at $(date -u +%FT%TZ)"
for ds in HDFS BGL; do
  test -f "data/features/$ds/mcv.parquet" || { echo "missing data/features/$ds/mcv.parquet"; exit 2; }
done
CHECKS=${CHECKS:-bgl_baselines+bgl_chronological+hdfs_overlap+hdfs_unseen_f1}
OUTDIR=${OUTDIR:-results/sanity}
python src/sanity_checks.py --checks ${CHECKS//+/ } --out "$OUTDIR"
echo "done at $(date -u +%FT%TZ)"
