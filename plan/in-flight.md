# In flight (read this first)

## Team-fixes rerun on Fir, submitted 2026-10-11 15:5x EDT

- Jobs: 64409181 step 7 COMPLETED (13 min); 64409182 sanity RUNNING; 64409183 rank+explain FAILED (no matplotlib in .venv), resubmitted as 64423135.
  64409183 rank + explain (CPU, runs after 64409181 succeeds). Submitted by `cluster/submit_team_fixes.sh`.
- Logs: `/scratch/marwan/log6309e/logs/step7_64409181.out`, `.../logs/rank_explain_64409183.out`, sanity log per
  `cluster/sanity.sbatch`. Outputs: `results/ext/`, `results/sanity/` on Fir.
- Check: `ssh fir 'sacct -j 64409181,64409182,64409183 -X -o JobID,State,Elapsed'`.
- When done: `rsync -a fir:/scratch/marwan/log6309e/results/ext/ results/ext/` and the same for `results/sanity/`;
  check seed 1 autoencoder recall (was 0.64), the BGL template rule (team reported 0.976) and the chronological RF
  (team reported 0.698); update PROGRESS.md steps 7 to 9 and a sanity section per `plan/council-thresholds/decision.md`;
  commit, push; delete this block.
