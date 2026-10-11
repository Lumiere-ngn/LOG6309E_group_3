# Council decision, 2026-10-11 (Claude, Space Bunny, Gemini 3.8 Flash; two rounds)

Brief: brief.md. Seats: seat-claude-r1.md, r1-muse.md / r2-muse.md (Space Bunny, private seat), r1-gemini.md / r2-gemini.md.

1. Thresholds (2 to 1). Report both cuts for Isolation Forest and the autoencoder. Primary: the 99th percentile of
   normal held-out scores. Sensitivity: the F1-max cut on the labeled held-out slice (`*_f1max` columns). The
   ranking uses threshold-free AUC and AP, so the cut does not decide it. Naming: Random Forest and MLP supervised;
   Isolation Forest "unsupervised scoring, label-calibrated cut"; autoencoder "semi-supervised (trained on normal
   sessions), label-calibrated cut". Dissent (Space Bunny): F1-max primary, since a 1% false-positive operating
   point at 2.93% prevalence caps Isolation Forest precision.
2. HDFS unseen sessions (3 to 0). Report precision, recall and F1 on the distinct union of test sessions whose count
   vector never appears in training, each session counted once, with a 95% bootstrap interval over sessions, beside
   the 99.95% overlap. An unranked stress test, not re-ranked with Scott-Knott. Pooling seeds would count the same
   session several times.
3. BGL (3 to 0). Call the single-template result "label redundancy in the feature space": a window is labeled
   anomalous when it holds an alert line, alert lines have their own templates, so one count column nearly restates
   the label. Not "leakage", which names a train/test boundary failure.

Out of scope, noted: Gemini's suggestion to use PyOD's AutoEncoder.
