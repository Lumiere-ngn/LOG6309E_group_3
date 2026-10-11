# Progress

One section per completed step of `Marwan-plan.md`. Each section lists what was run, where the outputs are,
the numbers, and every deviation from the plan or the paper. Paper: Wu, Li and Khomh, "On the Effectiveness of Log
Representation for Log-based Anomaly Detection", EMSE 2023 (arXiv 2308.08736v3), with code at
github.com/mooselab/suppmaterial-LogRepForAnomalyDetection, referred to below as the paper's code.

Branch: `main`. Compute: Alliance cluster Fir, CPU nodes (16 cores, 48 GB per task).

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

## Step 6. Replication runs, completed 2026-10-09

- Code: `src/replicate.py`, run as a 5-task CPU array (`cluster/rq1_cpu_array.sbatch`, one task per seed 0 to 4).
  `python src/replicate.py --merge` combines the per-seed outputs.
- Splits: stratified, seeded; HDFS 70/30 (402,542 train, 172,519 test, the paper's exact sizes), BGL 80/20 (575 train,
  144 test; paper 575 and 143).
- Models, as in the paper's code: Random Forest with 10 trees and `max_features="sqrt"` (`models/traditional.py`);
  MLP with two sigmoid hidden layers of 200 units, Adam at lr 0.01, 2,000 full-batch steps, anomalous training rows
  repeated `int(n / n_anomalous) - 1` extra times (`models/MLP.py`). The final MLP is evaluated; no checkpoint is
  chosen on the test set.
- Outputs: `results/rq1_runs.csv` (20 runs), `results/rq1_summary.csv` (mean and sd over 5 seeds beside the paper),
  `results/logs/` (job logs).

Mean over 5 seeds (sd of F1 in brackets):

| Dataset, model | Precision | Recall | F1 | AUC | Paper P / R / F1 |
|---|---|---|---|---|---|
| HDFS, RF | 0.998 | 0.998 | 0.998 (0.000) | 1.000 | 0.998 / 1.000 / 0.999 |
| HDFS, MLP | 0.996 | 0.999 | 0.997 (0.001) | 1.000 | 0.999 / 0.999 / 0.999 |
| BGL, RF | 0.958 | 0.808 | 0.873 (0.062) | 0.965 | 0.830 / 0.963 / 0.891 |
| BGL, MLP | 0.840 | 0.792 | 0.814 (0.026) | 0.892 | 0.958 / 0.840 / 0.895 |

Findings for the report:

1. HDFS matches the paper within 0.002 F1.
2. The paper's RF precision and recall are swapped. `traditional.py` defines `metrics(y_pred, y_true)` and calls
   `metrics(y_test, y_)`. Our BGL RF precision (0.958) and recall (0.808) sit beside the paper's published recall
   (0.963) and precision (0.830).
3. Our BGL MLP F1 is 0.081 below the paper. `MLP.py` prints test-set F1 every 10 steps and keeps no validation set,
   so the published value may be the best of those printouts. This is inferred, since the code shows the printing,
   not which value was reported.

Deviations from the paper:

- The MLP ran on CPU, one seed per array task (one HDFS MLP takes 1,739 s on 16 cores). The MLP is the same on CPU
  and GPU. GPU jobs use `.venv-gpu` (torch 2.7.1, `cluster/setup_env_gpu.sh`) because the nodes' driver cannot run
  the default torch build.
- Five seeds instead of the paper's single run, so every number has a standard deviation.

## Step 7. Extension runs (HDFS, four models), rerun 2026-10-11

- Code: `src/extension.py`, run as `cluster/ext.sbatch` on one H100 (job 64409181, 13 minutes for all 40 fits). The
  first run (job 63907609, 2026-10-09) is superseded; its log stays in `results/logs/step7_63907609.out`.
- Data: HDFS MCV, all 48 features. 10 stratified 70/30 resamples, seeds 0 to 9, each with 402,542 training and
  172,519 test sessions (5,051 anomalous).
- Models:
  1. Random Forest and MLP, supervised, with the step 6 settings.
  2. Isolation Forest (scikit-learn, 100 trees), unsupervised scoring, fitted without labels on 90% of the training
     split. Its cut is label-calibrated (below).
  3. Autoencoder (48-32-8-32-48, ReLU, log1p input, Adam at lr 1e-3, batch 4,096), semi-supervised (trained on the
     normal sessions of that 90% only), with a label-calibrated cut. The anomaly score is the mean squared
     reconstruction error. Training runs up to 100 epochs and stops when the reconstruction loss of the held-out
     slice's normal sessions has not improved for 5 epochs; the weights of the best epoch are kept. The first run used
     a fixed 20 epochs.
- Two cuts for Isolation Forest and the autoencoder, both set on the labeled 10% held-out slice of the training
  split, never on the test set (council decision, `plan/council-thresholds/decision.md`, 2 to 1):
  1. Primary: the 99th percentile of the normal sessions' scores in the slice.
  2. Sensitivity: the score that maximises F1 on the slice (`*_f1max` columns).
- Metrics: precision, recall and F1 at the cut; AUC and average precision (AP) from the raw scores, so neither
  depends on a cut.
- Outputs: `results/ext/rq2_runs.csv` (40 rows), `results/ext/rq2_summary.csv`, `results/logs/step7_64409181.out`.

Mean over 10 resamples (sd in brackets), primary cut:

| Model | Precision | Recall | F1 | AUC | AP |
|---|---|---|---|---|---|
| Random Forest (supervised) | 0.998 | 0.999 | 0.998 (0.000) | 1.000 | 1.000 |
| MLP (supervised) | 0.994 | 0.999 | 0.997 (0.003) | 1.000 | 0.999 |
| Autoencoder (semi-supervised, 99th-percentile cut) | 0.817 | 1.000 | 0.898 (0.038) | 1.000 | 0.994 |
| Isolation Forest (unsupervised scoring, 99th-percentile cut) | 0.757 | 0.611 | 0.676 (0.013) | 0.960 | 0.706 |

Mean over the same 10 resamples, F1-max cut (sensitivity):

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Autoencoder | 0.978 | 0.997 | 0.987 (0.003) |
| Isolation Forest | 0.747 | 0.656 | 0.681 (0.016) |

- Early stopping fixed the autoencoder's worst seed. Seed 1 recall was 0.637 and F1 0.674 under 20 fixed epochs; it is
  now 1.000 and 0.850. Recall is 1.000 on all 10 resamples. The mean F1 rises from 0.885 (sd 0.088) to 0.898 (sd
  0.038). Training ran 64 to 100 epochs; seeds 1, 2 and 5 reached the 100-epoch cap.
- The autoencoder's score is near perfect (AUC 0.9999, AP 0.994), and the cut decides its F1. At the
  99th-percentile cut precision is 0.739 to 0.888 across resamples (F1 0.850 to 0.941). At the F1-max cut precision is
  0.971 to 0.988 (F1 0.983 to 0.992). The cut costs the autoencoder 0.089 F1 on average.
- The cut barely matters for Isolation Forest (F1 0.676 against 0.681), but the F1-max cut is unstable for it:
  precision ranges from 0.532 to 0.939 across resamples (sd 0.122) and recall from 0.570 to 0.883. Its weakness is
  the score (AP 0.706), not the cut.
- Isolation Forest results at the primary cut equal the first run's, since only the autoencoder and the metrics
  changed.

## Step 8. Ranking, rerun 2026-10-11

- Code: `src/rank.py` with the non-parametric Scott-Knott ESD port in `analysis/` (from the ScottKnottESD R package:
  models sorted by median; a block is one group when its extreme pair differs by a negligible Cliff's delta, below
  0.147; otherwise it is cut where the Kruskal-Wallis H is largest). Group 1 is the best.
- Primary metrics are AUC and AP, which need no cut. F1 at the primary cut is secondary. Jobs 64423135 (ranking) and
  64426407 (SHAP).
- Outputs: `results/ext/rq2_ranking.csv`, `results/ext/rq2_gaps.csv`, `results/ext/rq2_auc.png`,
  `results/ext/rq2_ap.png`, `results/ext/rq2_f1.png`.

Scott-Knott group (median over 10 resamples in brackets):

| Model | AUC (primary) | AP (primary) | F1 (secondary) |
|---|---|---|---|
| Random Forest | 1 (1.0000) | 1 (0.9999) | 1 (0.998) |
| MLP | 1 (1.0000) | 2 (0.9996) | 2 (0.997) |
| Autoencoder | 2 (0.9999) | 3 (0.9949) | 3 (0.898) |
| Isolation Forest | 3 (0.961) | 4 (0.708) | 4 (0.681) |

- Random Forest and MLP tie on AUC and separate on AP and F1. The order of the four models is the same under AP and
  F1. Testing every pair instead of the extreme pair (the package documentation's rule) gives the same groups for
  all three metrics.
- Session counts behind the gaps (`rq2_gaps.csv`). Errors are false positives plus false negatives per test set of
  172,519 sessions (5,051 anomalous), mean over 10 resamples.

| Pair | Median F1 gap | Mean errors, first model | Mean errors, second model | Difference in sessions |
|---|---|---|---|---|
| Random Forest, MLP | 0.0008 | 19.7 | 34.9 | 15.2 |
| Random Forest, Autoencoder | 0.0999 | 19.7 | 1,162.6 | 1,142.9 |
| Random Forest, Isolation Forest | 0.3171 | 19.7 | 2,955.6 | 2,935.9 |
| MLP, Autoencoder | 0.0991 | 34.9 | 1,162.6 | 1,127.7 |
| MLP, Isolation Forest | 0.3163 | 34.9 | 2,955.6 | 2,920.7 |
| Autoencoder, Isolation Forest | 0.2172 | 1,162.6 | 2,955.6 | 1,793.0 |

- The Random Forest and MLP gap of 0.0008 F1 is 15.2 sessions out of 172,519, so the split into two groups on AP and
  F1 rests on a small count. The other gaps are 1,100 to 2,900 sessions.
- As the plan predicted, the ranking separates supervised from unsupervised models more than it separates the models
  within each pair.

## Step 9. Explanation, rerun 2026-10-11

- Code: `src/explain.py`. It refits the resample-0 Random Forest (same split and seed as step 7) and runs SHAP
  TreeExplainer (shap 0.52.0) on 4,000 test sessions, 2,000 anomalous and 2,000 normal, drawn at random. The Random
  Forest did not change in the rerun, and the numbers below equal those of the 2026-10-09 run.
- Outputs: `results/ext/shap_top10.csv` (event, template, mean |SHAP|, mean SHAP on anomalous and on normal
  sessions), `results/ext/shap_summary.png`, `results/logs/explain_64426407.out`.

Top 5 templates by mean |SHAP|, for the anomalous class:

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

## Sanity checks (2026-10-11)

- Code: `src/sanity_checks.py`, run as `cluster/sanity.sbatch` (job 64409182, 2 h 10 min, almost all of it the five
  HDFS MLP refits). Outputs: `results/sanity/*.csv`, `results/logs/sanity_64409182.out`. The checks were proposed by
  the team and scoped in `plan/team-fixes-2026-10-11.md`.

### BGL baselines (step 6 split: stratified 80/20, seeds 0 to 4; 575 training and 144 test windows, 77 anomalous)

| Baseline | Precision | Recall | F1 (range over seeds) |
|---|---|---|---|
| Flag every window | 0.535 | 1.000 | 0.697 (identical in every seed) |
| Best single template: flag a window when its count is above 0 | 0.996 | 0.605 | 0.750 (0.661 to 0.806) |
| Random Forest (step 6) | 0.958 | 0.808 | 0.873 |
| MLP (step 6) | 0.840 | 0.792 | 0.814 |

- The template is chosen on the training windows only, as the template with the best F1 there. It is a31b789f,
  "instruction address: <*>" (51,227 lines in the parsed BGL log), in all five seeds.
- The team's earlier F1 of 0.976 for this rule did not reproduce. With the template chosen on training windows only,
  the F1 on the 144 test windows is 0.750 on average and never above 0.806 in five seeds. The cause of the
  difference was not tested. Choosing the template with the test windows in view is one candidate.
- The rule's precision is 0.996 and its recall 0.605. A window is anomalous when it holds an alert line, and alert
  lines have templates of their own, so one count column nearly restates the label. The council calls this label
  redundancy in the feature space (3 to 0). The word leakage names a train/test boundary failure and does not apply.
- The rule beats flagging every window by 0.053 F1 on average. Random Forest and MLP beat the rule by 0.123 and 0.064.

### BGL chronological split (windows ordered by start time, first 575 train, last 144 test; 70 anomalous in test)

| Model | Precision | Recall | F1 (sd over 5 model seeds) | AUC | AP |
|---|---|---|---|---|---|
| Random Forest | 0.911 | 0.569 | 0.698 (0.019) | 0.835 | 0.848 |
| MLP | 0.529 | 0.960 | 0.682 (0.012) | 0.756 | 0.727 |

- One split, so the standard deviations come from model seeds only. Against the shuffled split, Random Forest F1 falls
  from 0.873 to 0.698 and MLP F1 from 0.814 to 0.682. The team's 0.698 for Random Forest reproduces.
- Flagging every window scores F1 0.654 on this test set (70 of 144 windows anomalous; arithmetic from those counts,
  not a results file). The two models sit 0.044 and 0.028 above it. The MLP flags 96% of anomalous windows at a
  precision of 0.529, close to the 0.486 share of anomalous windows in the test set.

### HDFS overlap of count vectors (step 6 split: stratified 70/30, seeds 0 to 4; 172,519 test sessions each)

- Test sessions whose count vector also appears in the training split: 172,424 to 172,436 of 172,519 per seed, mean
  99.948% (99.945% to 99.952%). The team's 99.95% reproduces.
- Test sessions with a count vector absent from training: 83 to 95 per seed (447 session-seed pairs), 277 distinct
  sessions, 206 of them anomalous. Pooling seeds would count the same session more than once, so the distinct union
  is used (council decision, 3 to 0).

### HDFS F1 on unseen sessions (distinct union of 277 sessions, each counted once, 206 anomalous)

| Model | Precision | Recall | F1 | 95% bootstrap interval for F1 |
|---|---|---|---|---|
| Random Forest | 0.968 | 0.947 | 0.957 | 0.936 to 0.975 |
| MLP | 0.946 | 0.942 | 0.944 | 0.921 to 0.964 |

- The interval comes from 2,000 bootstrap resamples of the 277 sessions. Per seed, Random Forest F1 on 83 to 95 unseen
  sessions is 0.937 to 0.965 and MLP F1 is 0.906 to 0.958.
- Against the F1 of 0.998 and 0.997 on the full test sets (step 7), the models lose about 0.04 to 0.05 F1 on sessions
  whose count vector they have not seen. This is an unranked stress test, not re-ranked with Scott-Knott. The HDFS
  chronological split is not run.

## Reproduce

```bash
# on a Fir login node
bash cluster/fetch_loghub.sh /scratch/$USER/log6309e
bash cluster/setup_env.sh /scratch/$USER/log6309e
bash cluster/setup_env_gpu.sh /scratch/$USER/log6309e     # torch 2.7.1 for the GPU jobs (step 7)
cd /scratch/$USER/log6309e
sbatch --export=ALL,REQUIRE_GPU=0 cluster/steps_3_6.sbatch   # parse, MCV, validate (its replicate run can be cancelled)
sbatch cluster/rq1_cpu_array.sbatch                           # step 6, seeds 0-4, about 1 h each
sbatch cluster/ext.sbatch                                     # step 7 on GPU, about 12 min
python src/rank.py && python src/explain.py                   # steps 8 and 9, minutes
sbatch cluster/sanity.sbatch                                  # sanity checks, about 2 h (HDFS MLP refits)
python src/replicate.py --merge results_cpu/seed_*/rq1_runs.csv --out results
```
