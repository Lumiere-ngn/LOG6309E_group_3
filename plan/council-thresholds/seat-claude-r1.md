1. Report both, with the 99th percentile of normal scores as primary. It uses labels only to pick normal sessions, which is the standard semi-supervised setup, and keeps the comparison with Random Forest fair. F1-max on the labeled slice is a sensitivity row that shows how much of the autoencoder's F1 swing is the threshold. Call both "unsupervised scoring, semi-supervised threshold".

2. Pool the unseen sessions across the 10 seeds (about 900) and report F1 with its count and a bootstrap interval, beside the 99.95% overlap. Per seed, about 90 sessions is too few, but pooled it answers whether the models generalize beyond memorized vectors.

3. Target leakage through feature construction: the label is defined by alert lines and those lines' templates are features, so one column restates the label. Gemini's point that the parser never reads the label column is true but does not change this; the leak is in how the label and the features are both derived from the same lines.
