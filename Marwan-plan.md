# Plan for the group assignment (Marwan)

LOG6309E Intelligent DevOps, group assignment "Log-based Anomaly Detection". We replicate RQ1 of Wu, Li and Khomh, "On the Effectiveness of Log Representation for Log-based Anomaly Detection", EMSE 2023 (arXiv 2308.08736, DOI 10.1007/s10664-023-10364-1), then extend it.

The report is IEEE format, 6 to 8 pages, on the Moodle template (Introduction, Approach, Results, Conclusion, Team Contribution).

## Decisions

| Handout item (weight) | Choice | Reason |
|---|---|---|
| Datasets, replication | HDFS and BGL | The paper reports full precision/recall/F1 tables for both (Tables 2 and 3), and both are on Loghub (HDFS_v1.zip 186.6 MB, BGL.zip 57.5 MB). |
| 1. Parsing (15%) | Drain, once per dataset | The paper used Drain. HDFS has 11.2M lines and BGL 4.7M. |
| 2. Representation (15%) | Message Count Vector (MCV) for every model | MCV ranks first overall in the paper (Table 6) and needs no embedding model. |
| 3. Supervised models (20%) | Random Forest (classical) and MLP (deep learning) | The paper classes MLP as deep learning (Sec 4.3.1) and pairs MCV with it. |
| 3a. Collinearity step | Spearman hierarchical clustering, then VIF | The paper does neither, so its numbers are our "without" condition. |
| Metrics | Precision, recall, F1, AUC | The paper reports no AUC, so the AUC column has no paper value to compare against. |
| Extension dataset | HDFS | It has 575k sessions with 2.9% anomalies, which suits unsupervised models and leaves large test sets across 10 resamples. |
| Ext. 1 (20%): classical, unsupervised | Isolation Forest (scikit-learn) on MCV, fitted on the training split without labels | It is the model loglizer wraps, available directly in scikit-learn. |
| Ext. 1: deep learning, semi-supervised | Autoencoder on MCV, trained on normal sessions only; anomaly score = reconstruction error | The autoencoder is one of deep-loglizer's model families (Chen et al. 2021). On MCV it is about 30 lines of PyTorch. |
| Ext. 2 (15%): ranking | Scott-Knott ESD on F1 and AUC, 4 models, 10 stratified 70/30 resamples | Marwan has a Python port of Scott-Knott ESD (`sk_esd.py` + `effect.py`) and will add it to this repo. |
| Ext. 3 (15%): explanation | SHAP TreeExplainer on the Random Forest | RF is expected to rank first (paper HDFS F1 0.999). Each feature is one log template, so the explanation names templates. |

The four models in the ranking: Random Forest, MLP, Isolation Forest, Autoencoder.

## Paper numbers to compare against (MCV column)

| Dataset | Model | P | R | F1 |
|---|---|---|---|---|
| HDFS | RF | 0.998 | 1.000 | 0.999 |
| HDFS | MLP | 0.999 | 0.999 | 0.999 |
| BGL | RF | 0.830 | 0.963 | 0.891 |
| BGL | MLP | 0.958 | 0.840 | 0.895 |

The paper's setup (Table 1, Sec 4.3.2):
- HDFS: sessions by block ID; 402,542 train and 172,519 test (70/30, shuffled).
- BGL: 6-hour fixed windows; 575 train and 143 test (80/20, shuffled).
- Both: one run per model, no fixed seeds, no validation set. We fix seeds and keep a validation split.

## Steps

1. **Environment (0.5 h).** Python 3.12 with uv: `uv pip install numpy pandas scikit-learn scipy statsmodels shap torch drain3`.
2. **Data (0.5 h).** Download from Loghub (github.com/logpai/loghub).
   - HDFS_v1.zip contains `anomaly_label.csv` (block labels) and `preprocessed/Event_traces.csv`. Use the traces only to cross-check our parse.
   - BGL.zip: the label is the first column of each line ("-" means normal).
3. **Parse (2 h including run time).** Run Drain on each raw log.
   - BGL: use the regexes in the README of the paper's code package (github.com/mooselab/suppmaterial-LogRepForAnomalyDetection).
   - HDFS: use the HDFS settings of the logpai/logparser Drain benchmark (mask block IDs and IP addresses).
   - Optional shortcut: LogLead (github.com/EvoTestOps/LogLead, PyPI 2.0.0) ships HDFS and BGL loaders, a Drain enhancer, Random Forest and Isolation Forest. Try it for 30 minutes; otherwise do these steps by hand.
4. **Group and count (1 h).**
   - HDFS: group lines by block ID (regex `blk_-?\d+`); labels from `anomaly_label.csv`.
   - BGL: 6-hour windows; a window is anomalous if any line in it is.
   - Build the MCV matrix, one column per template.
5. **Collinearity (1 h).**
   - Drop constant columns.
   - Cluster features on Spearman correlation (`scipy.cluster.hierarchy`), cut at |rho| = 0.7, keep one feature per cluster.
   - Drop features with VIF > 5 (`statsmodels variance_inflation_factor`) one at a time until none remains.
   - Report how many features each step removed.
6. **Replication runs (2 h).** RF and MLP (two hidden layers of 200 units, as in the paper's `models/MLP.py`) on each dataset, with and without step 5: 2 datasets x 2 models x 2 conditions = 8 runs. Report P, R, F1 and AUC beside the paper's numbers.
7. **Extension runs (3 h), HDFS only.**
   - Draw 10 stratified 70/30 resamples with seeds 0 to 9.
   - Fit all four models on each resample.
   - Isolation Forest sees the whole training split without labels. The autoencoder sees normal training sessions only.
   - Set each unsupervised threshold on a held-out slice of the training data (for example the 99th percentile of normal scores), never on the test set.
   - Save F1 and AUC per model per resample.
8. **Rank (0.5 h).** Run `sk_esd.py` on the 4 x 10 F1 matrix and on the AUC matrix; one box plot per metric.
9. **Explain (1 h).** Run SHAP TreeExplainer on the RF from resample 0. Produce a summary plot of the top 10 templates, list their text, and write one paragraph on why they mark an anomalous block.
10. **Report (6 h, shared).** Results are organized as RQ1 (replication: both datasets, with and without collinearity removal) and RQ2 (ranking of the four models). Close with the explanation of the best model and three take-home messages.

Total: about 18 hours of hands-on work, about 6 per person. HDFS parsing and the 40 extension fits run unattended.

## Split

| Person | Owns | Report sections |
|---|---|---|
| A | Steps 3 to 6 on HDFS | Approach (data, parsing, features) |
| B | Steps 3 to 6 on BGL, plus the shared collinearity code | Results RQ1, comparison with the paper |
| C | Steps 7 to 9 | Results RQ2, explanation, Conclusion |

All three write the Introduction, Team Contribution and the slides.

## Points to state in the report

1. On HDFS with MCV the paper's F1 is 0.999, so the replication should match it closely, and the ranking will separate supervised from unsupervised models more than it separates the models within each pair.
2. BGL has 143 test windows, so one misclassified window moves F1 by about 0.01 to 0.02. A small gap from the paper is expected.
3. HDFS has about 46 templates, so the VIF step may remove few features and change little. That result is reported as a finding.
4. Do not reuse the paper's CNN and LSTM scripts as they are: they keep the epoch with the best test F1 (`CNN.py:254-261`, `LSTM.py:227-236`).

## Sources

- Paper: arXiv 2308.08736v3, Tables 1 to 6, Sections 4.2, 4.3 and 5.1.2.
- Paper code: github.com/mooselab/suppmaterial-LogRepForAnomalyDetection.
- Loghub: github.com/logpai/loghub.
- loglizer: github.com/logpai/loglizer. deep-loglizer: github.com/logpai/deep-loglizer.
- LogLead: github.com/EvoTestOps/LogLead.
- Scott-Knott ESD: CRAN package ScottKnottESD.
