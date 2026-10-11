I'll read that file now.
Reading the referenced files to ground the decisions in evidence.
Read. Here are the three decisions.

**1. Threshold — report both; F1-max primary, 99th percentile as sensitivity.**

A labeled held-out slice is threshold *calibration*, not supervised fitting — `fit_iforest`/`fit_autoencoder` never pass labels to the scorer. And because item 4 ranks on AUC and AP, both threshold-free, Gemini's "invalidates the Scott-Knott ranking" fails; the ranking is identical under either cut. The 99th percentile is the weaker default: it fixes FPR at 1% against 2.93% prevalence, which is why IForest caps at precision 0.757. Names: RF/MLP *supervised*; IForest *unsupervised scoring, label-calibrated cut*; AE *label-calibrated twice* — label-selected normals (extension.py:143) **and** the cut, on a slice shared with early stopping. "Unsupervised" is false for the AE.

**2. Unseen HDFS — report F1 on the unseen subset, for all four models.**

The overlap count documents the threat; F1 on unseen sessions tests whether the ranking survives it, and all four models face the same unseen rows, so the comparison stays fair. Two corrections to the framing: pooling10 seeds does **not** yield 900 distinct sessions — seeds resample the same corpus, so report the distinct union and its positive count (likely tens, not thousands), and bootstrap over sessions, not seeds. The subset is also non-random (selected by novelty, likely rarer and longer), so label it a stress test, report CIs, and do not re-run Scott-Knott on it.

**3. BGL — not label leakage; it is label redundancy in the feature space.**

The parser never reads the label column, so nothing leaks. What happens is that the window label is *defined* by "any line carries an alert", alert lines have their own templates, and MCV preserves template presence verbatim — so the feature space contains a near-deterministic surrogate for the target. Overlapping 6-hour windows then multiply the same alert across many rows, inflating effective sample size. The keyword rule's 0.976 therefore measures BGL's intrinsic separability, not model quality. Gemini's causal story is also wrong: the split hurts the *learned* models (RF 0.873 → 0.698 chronologically), while a keyword rule works even chronologically. The decisive check is the template rule under the split the team already has.

**Ranking of the three by evidence strength:** (3) is certain — mechanism is documented in PROGRESS.md step 4. (1) is well-evidenced — Gemini's central objection is refuted by item 4. (2) is the weakest evidence base and the most likely to be misread, since its n is unverified; do it, but treat the result as a stress test, not a ranking.

No files edited.
