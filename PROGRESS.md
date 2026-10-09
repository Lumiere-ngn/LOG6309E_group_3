# Progress

One section per completed step of `Marwan-plan.md`. Each section lists what was run, where the outputs are,
the numbers, and every deviation from the plan or the paper. Paper: Wu, Li and Khomh, "On the Effectiveness of Log
Representation for Log-based Anomaly Detection", EMSE 2023 (arXiv 2308.08736v3), with code at
github.com/mooselab/suppmaterial-LogRepForAnomalyDetection, referred to below as the paper's code.

Branch: `feat/steps-4-6-features-replication`. Compute: Alliance cluster Fir, CPU nodes (16 cores, 48 GB per task).

## Step 3. Parse (Drain), completed 2026-10-05, rerun on Fir 2026-10-09

- Code: `src/parse_drain.py` (drain3, depth 4, similarity threshold 0.5). HDFS masks block IDs and IP:port; BGL uses
  the refined regexes from the README of the paper's code package.
- Data: Loghub HDFS_v1 and BGL from Zenodo record 8196385, fetched by `cluster/fetch_loghub.sh`.
- Run time on Fir: HDFS 11,175,629 lines in about 100 s, 48 events; BGL 4,747,963 lines in 81 s, 3,548 Drain clusters
  merged into 379 events.
- Check: `src/validate_parse.py` aligns the HDFS parse with Loghub's `Event_traces.csv`. All 575,061 blocks align.
  Grouping accuracy is 0.846 (9,451,218 of 11,175,629 line-block pairs); the loss comes from one of our events
  merging ground-truth events E8 and E11.

## Step 4. Group and count (MCV), completed 2026-10-09

- Code: `src/build_mcv.py {HDFS,BGL}`, output `data/features/<DATASET>/mcv.parquet` (one row per session, one count
  column per event, plus `label`).
- HDFS: one session per block ID; labels from `anomaly_label.csv`. 575,061 sessions, 16,838 anomalous (2.93%),
  48 event columns.
- BGL: 6-hour windows, each opened at a log line and closed 6 hours later; the next line opens the next window. A
  window is anomalous if any line in it carries an alert label. 719 sessions, 384 anomalous (53.4%), 379 event
  columns. The paper reports 718 sessions averaging 6,565 lines (Table 1, Section 4.3.2); this rule gives 719
  averaging 6,604.
- Deviation found and fixed: windows anchored on the clock (first timestamp plus multiples of 6 h) give 826
  sessions, because stretches with no logs still split busy periods. `tests/bgl_window_count.py` compares the rules.
  `--window-mode clock` keeps the old rule; its results are in `results/bgl_clock_windows/`.

## Step 5. Collinearity, completed 2026-10-09

- Code: `src/collinearity.py`, fitted on each seed's training split only. It drops constant columns,
  clusters features on Spearman correlation (average linkage on 1 - |rho|, cut at |rho| = 0.7) and keeps the
  highest-variance feature per cluster, then drops the feature with the largest VIF until every VIF is at most 5.
- Check: `tests/smoke_test.py` asserts both directions on synthetic data. A duplicated column, a scaled column and a
  constant are removed, and four independent columns are kept.
- Features removed per pass, over seeds 0 to 4 (`results/rq1_reduction.csv`):

| Dataset | Input | Constant | Correlated | VIF > 5 | Kept |
|---|---|---|---|---|---|
| HDFS | 48 | 1 to 2 | 14 to 17 | 3 to 5 | 27 to 29 |
| BGL | 379 | 7 to 19 | 209 to 220 | 30 to 34 | 118 to 123 |

- The BGL VIF pass takes 14 to 25 minutes per seed.

## Step 6. Replication runs, completed 2026-10-09

- Code: `src/replicate.py`, run as a 5-task CPU array (`cluster/rq1_cpu_array.sbatch`, one task per seed 0 to 4).
  `src/merge_results.py` combines the per-seed outputs.
- Splits: stratified, seeded; HDFS 70/30 (402,542 train, 172,519 test, the paper's exact sizes), BGL 80/20 (575 train,
  144 test; paper 575 and 143).
- Models, as in the paper's code: Random Forest with 10 trees and `max_features="sqrt"` (`models/traditional.py`);
  MLP with two sigmoid hidden layers of 200 units, Adam at lr 0.01, 2,000 full-batch steps, anomalous training rows
  repeated `int(n / n_anomalous) - 1` extra times (`models/MLP.py`). The final MLP is evaluated; no checkpoint is
  chosen on the test set.
- Outputs: `results/rq1_runs.csv` (40 runs), `results/rq1_summary.csv` (mean and sd over 5 seeds beside the paper),
  `results/logs/` (job logs).

Mean over 5 seeds, all features (sd of F1 in brackets):

| Dataset, model | Precision | Recall | F1 | AUC | Paper P / R / F1 |
|---|---|---|---|---|---|
| HDFS, RF | 0.998 | 0.998 | 0.998 (0.000) | 1.000 | 0.998 / 1.000 / 0.999 |
| HDFS, MLP | 0.996 | 0.999 | 0.997 (0.001) | 1.000 | 0.999 / 0.999 / 0.999 |
| BGL, RF | 0.958 | 0.808 | 0.873 (0.062) | 0.965 | 0.830 / 0.963 / 0.891 |
| BGL, MLP | 0.840 | 0.792 | 0.814 (0.026) | 0.892 | 0.958 / 0.840 / 0.895 |

With collinear features removed (step 5), F1 drops in every case:

| Dataset, model | F1 all features | F1 reduced | Features kept (mean) |
|---|---|---|---|
| HDFS, RF | 0.998 | 0.996 | 28.0 of 48 |
| HDFS, MLP | 0.997 | 0.987 | 28.0 of 48 |
| BGL, RF | 0.873 | 0.810 | 120.6 of 379 |
| BGL, MLP | 0.814 | 0.740 | 120.6 of 379 |

Findings for the report:

1. HDFS matches the paper within 0.002 F1.
2. The paper's RF precision and recall are swapped. `traditional.py` defines `metrics(y_pred, y_true)` and calls
   `metrics(y_test, y_)`. Our BGL RF precision (0.958) and recall (0.808) sit beside the paper's published recall
   (0.963) and precision (0.830).
3. Our BGL MLP F1 is 0.081 below the paper. `MLP.py` prints test-set F1 every 10 steps and keeps no validation set,
   so the published value may be the best of those printouts. This is inferred, since the code shows the printing,
   not which value was reported.
4. Removing correlated and redundant features does not help these models; on BGL it costs 0.06 to 0.07 F1.
5. Under the clock windowing (826 sessions), BGL RF reaches F1 0.919 and MLP 0.821
   (`results/bgl_clock_windows/rq1_summary.csv`), so the windowing choice moves BGL RF by 0.046 F1.

Deviations from the plan:

- The MLP ran on CPU. One HDFS MLP takes 1,739 s on 16 cores, so each seed is its own array task. Two causes kept
  the GPU unused, both since resolved (`tests/cuda_ok.py`, `tests/cuda_diag.py`):
  1. The default environment's torch 2.14.1 targets CUDA 13.2, and the GPU nodes' driver 580 supports CUDA 13.0.
     `cluster/setup_env_gpu.sh` builds `.venv-gpu` with torch 2.7.1 (CUDA 12.6), which the job script now uses.
  2. Node fc10612 shows no GPU to torch in any configuration (job 63873126), while torch 2.7.1 works on fc10616
     (job 63829776). The GPU scripts now exclude fc10612. These results are unchanged by either fix, since the
     MLP is the same on CPU and GPU.
- Five seeds instead of the paper's single run, so every number has a standard deviation.

## Step 7. Extension runs (HDFS, four models), completed 2026-10-09

- Code: `src/extension.py`, run as `cluster/ext.sbatch` on one H100 (job 63907609, 12 minutes for all 40 fits).
- Data: HDFS MCV, all 48 features (the better condition in step 6). 10 stratified 70/30 resamples, seeds 0 to 9,
  each with 402,542 training and 172,519 test sessions.
- Models:
  1. Random Forest and MLP, supervised, with the step 6 settings.
  2. Isolation Forest (scikit-learn, 100 trees), fitted without labels on 90% of the training split.
  3. Autoencoder (48-32-8-32-48, ReLU, log1p input, Adam at lr 1e-3, batch 4,096, 20 epochs), trained on the normal
     sessions of that 90% only; the anomaly score is the mean squared reconstruction error.
- Thresholds for the two unsupervised models: the 99th percentile of the scores of normal sessions in the remaining
  10% of the training split. The test set never sets a threshold. AUC uses the raw scores.
- Outputs: `results/ext/rq2_runs.csv` (40 rows), `results/ext/rq2_summary.csv`, `results/logs/step7_63907609.out`.

Mean over 10 resamples (sd of F1 in brackets):

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Random Forest | 0.998 | 0.999 | 0.998 (0.000) | 1.000 |
| MLP | 0.994 | 0.999 | 0.997 (0.003) | 1.000 |
| Autoencoder | 0.824 | 0.964 | 0.885 (0.088) | 0.999 |
| Isolation Forest | 0.757 | 0.611 | 0.676 (0.013) | 0.960 |

- The autoencoder ranks sessions almost perfectly (AUC 0.999) but its F1 varies from 0.674 (seed 1) to 0.967
  (seed 7), so its weakness is the threshold, not the score. Isolation Forest is stable but misses 39% of anomalies.

## Step 8. Ranking, completed 2026-10-09

- Code: `src/rank.py` with the non-parametric Scott-Knott ESD port in `analysis/` (from the ScottKnottESD R package:
  models sorted by median; a block is one group when its extreme pair differs by a negligible Cliff's delta, below
  0.147; otherwise it is cut where the Kruskal-Wallis H is largest). Group 1 is the best.
- Outputs: `results/ext/rq2_ranking.csv`, `results/ext/rq2_f1.png`, `results/ext/rq2_auc.png`.

| Model | F1 median | F1 group | AUC median | AUC group |
|---|---|---|---|---|
| Random Forest | 0.998 | 1 | 1.000 | 1 |
| MLP | 0.997 | 2 | 1.000 | 1 |
| Autoencoder | 0.907 | 3 | 1.000 | 2 |
| Isolation Forest | 0.681 | 4 | 0.961 | 3 |

- On F1 every model forms its own group. On AUC the two supervised models tie. Testing every pair instead of the
  extreme pair (the package documentation's rule) gives the same groups.
- As the plan predicted, the ranking separates supervised from unsupervised models more than it separates the models
  within each pair.

## Step 9. Explanation, completed 2026-10-09

- Code: `src/explain.py`. It refits the resample-0 Random Forest (same split and seed as step 7) and runs SHAP
  TreeExplainer (shap 0.52.0) on 4,000 test sessions, 2,000 anomalous and 2,000 normal, drawn at random.
- Outputs: `results/ext/shap_top10.csv` (event, template, mean |SHAP|, mean SHAP on anomalous and on normal
  sessions) and `results/ext/shap_summary.png`.

Top 5 templates by mean |SHAP| for the anomalous class:

| Event | Template | Mean SHAP, anomalous | Mean SHAP, normal |
|---|---|---|---|
| 2e68ccc3 | Unexpected error trying to delete block <*>. BlockInfo not found in volumeMap. | 0.273 | -0.008 |
| 46003790 | Received block <*> of size <*> from /<*> | 0.134 | -0.004 |
| bbb51b95 | Receiving block <*> src: /<*> dest: /<*> | 0.123 | -0.003 |
| 5d5de21c | BLOCK* NameSystem.addStoredBlock: blockMap updated: <*> is added to <*> size <*> | 0.120 | -0.004 |
| dba996ef | Deleting block <*> file <*> | 0.055 | 0.011 |

- The model uses two kinds of evidence. Error templates push toward "anomalous" when they appear (high
  counts, red points at positive SHAP for 2e68ccc3, c294d20f "Redundant addStoredBlock" and 0567184d "Receiving empty
  packet"). The normal write path pushes toward "anomalous" when it is incomplete: low counts of "Received block"
  (46003790) and "PacketResponder" (d38aa58d) carry positive SHAP. HDFS writes each block to three replicas by default, so
  fewer than three receipts suggests a replica never arrived (an interpretation from HDFS defaults, not measured here).
- The first run of this step filled the 4,000-session sample with anomalies, because the test split
  holds about 5,000, which left no normal sessions to compare against. The reported run uses the balanced sample above.

## Reproduce

```bash
# on a Fir login node
bash cluster/fetch_loghub.sh /scratch/$USER/log6309e
bash cluster/setup_env.sh /scratch/$USER/log6309e
cd /scratch/$USER/log6309e
sbatch --export=ALL,REQUIRE_GPU=0 cluster/steps_3_6.sbatch   # parse, MCV, validate (its replicate run can be cancelled)
sbatch cluster/rq1_cpu_array.sbatch                           # step 6, seeds 0-4, about 1 h each
sbatch cluster/ext.sbatch                                     # step 7 on GPU, about 12 min
python src/rank.py && python src/explain.py                   # steps 8 and 9, minutes
python src/replicate.py --merge results_cpu/seed_*/rq1_runs.csv --out results
```
