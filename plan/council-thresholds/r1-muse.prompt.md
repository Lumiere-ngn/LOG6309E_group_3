# Council: three open points in the LOG6309E extension plan

Context: course replication of Wu, Li and Khomh (EMSE 2023), HDFS and BGL message count vectors. The graded
extension compares four models on HDFS over 10 resamples (Random Forest, MLP, Isolation Forest, Autoencoder) and
ranks them with Scott-Knott ESD. Files: plan/team-fixes-2026-10-11.md, plan/reviews/gemini-team-fixes.md,
src/extension.py, PROGRESS.md.

Current threshold for the two unsupervised models: the 99th percentile of scores of the NORMAL sessions in a
held-out 10% of the training split (so labels are used to select normal sessions). Proposed: the cut that
maximises F1 on that labeled held-out slice (about 40k sessions, about 1,180 anomalies).

Decide each point, with one reason:
1. Threshold: keep the 99th percentile of normal scores, switch to F1-max on the labeled slice, or report both
   (primary plus sensitivity)? What must the report call each model?
2. HDFS unseen sessions: 99.95% of test count vectors also occur in training. Report F1 on the about 90 unseen
   test sessions per seed (pooled over 10 seeds, about 900), or report only the overlap count?
3. BGL: a single-template rule (window anomalous if template X occurs, X chosen on training) reaches F1 0.976
   against Random Forest 0.873. BGL labels a window anomalous when any line in it carries an alert label, and
   alert lines have their own templates. Is "label leakage through feature construction" the right description,
   or something else?

Rank on evidence and validity only. Answer each point in under 80 words.


Files to read: E:\PhD\courses-term1\LOG6309E_group_3\plan\team-fixes-2026-10-11.md E:\PhD\courses-term1\LOG6309E_group_3\plan\reviews\gemini-team-fixes.md E:\PhD\courses-term1\LOG6309E_group_3\src\extension.py E:\PhD\courses-term1\LOG6309E_group_3\PROGRESS.md