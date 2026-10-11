# Team fixes before the report (2026-10-11)

Proposed by the team, scoped by Marwan and Claude on 2026-10-11 15:1x EDT.

1. Sanity checks, BGL only, plus one HDFS count (new `src/sanity_checks.py`, outputs `results/sanity/*.csv`).
   - BGL baselines: "predict all anomalous", and the best single-template rule (window anomalous if template X count > 0, X chosen on the training split only).
   - BGL chronological 80/20 split (windows ordered by start time), RF and MLP with the step 6 settings, 5 model seeds.
   - HDFS: the share of test sessions whose count vector also appears in the training split, per seed 0 to 4. Counts only, no F1 on unseen sessions.
   - Team's reported numbers to reproduce: BGL template rule F1 0.976 (shuffled split), RF 0.698 under the chronological split, HDFS 99.95% overlap.
2. Autoencoder: up to 100 epochs with early stopping (patience 5) on the reconstruction loss of the held-out slice's normal sessions, instead of a fixed 20 epochs. Check: seed 1 recall recovers from 0.64.
3. Thresholds for Isolation Forest and the Autoencoder: the score cut that maximises F1 on the labeled held-out training slice (the 10% already used for the 99th percentile). Never the test set. Both are described as "unsupervised scoring, semi-supervised threshold".
4. Add average precision (`average_precision_score`) to `src/extension.py`. `src/rank.py` ranks on AUC and AP as primary, F1 as secondary, and prints the session counts behind each gap. Rerun steps 7 to 9 once, after items 2 and 3.

Not done: the HDFS chronological split, F1 on the about 90 unseen HDFS sessions per seed.
