# Experiment log

Append dated entries; never replace prior results with a better-looking trial.

## 2026-09-21: audit and prospective baseline protocol

- Commit: d284767bf7f1f04b24f758211a1d8d70e82f52ed (main).
- Purpose: faithful ODH reproduction and durable Phase 0-7 research plan.
- Protocol: inspect paper and official code, document exclusions and output
  semantics, run preprocessing and smoke before deciding on 1500-epoch cost.
- Result: initial worktree has only untracked .DS_Store; author files preserved.
  macOS 26.6.2 arm64, Apple M4, 24 GiB memory. Existing fish environment's plain
  torch import aborts with duplicate OpenMP runtime initialization; do not mask
  this with KMP_DUPLICATE_LIB_OK for a scientific run.
- Decision: inspect isolated compatibility options before training.
- Next step: complete source/data/paper audit and minimal reproduction adapter.

## 2026-09-21: ODH smoke gate

- Commit: d284767bf7f1f04b24f758211a1d8d70e82f52ed.
- Purpose: verify the official QGeoGNN data path and AL-readiness primitives
  before any long run.
- Protocol: cached author ODH graph/descriptors; exclude row 4231 and RTv > 60;
  NumPy split seed 388; PyTorch seed 525; 90/5/5 fixed random-row split;
  batch 2048; two training epochs on CPU; no test-based selection.
- Result: 4,942 eligible rows (4,447/247/247); output `(8,3)` and h_graph
  `(8,128)`; finite loss and gradients; backward/optimizer step and label-free
  forward passed. Epoch wall times 16.47 s and 14.83 s. Estimated 1500-epoch
  CPU cost 6.68 h. Smoke metrics are not baseline metrics.
- Decision: stop before formal training under the resource rule. Baseline is not
  yet credible enough for AL Phase 2.
- Artifact: `artifacts/reproduction/odh_smoke_20260921_v4/`.

## 2026-09-22: formal ODH run started

- Command: `.venv-reproduction/bin/python scripts/reproduce_odh_baseline.py --mode train --device cpu --threads 4 --output artifacts/reproduction/odh_baseline_20260922`.
- Runtime: detached screen session `odh_baseline_20260922`; a companion
  `odh_baseline_watch` session checks every 60 seconds and resumes only from a
  saved checkpoint if the main session disappears.
- Status at launch check: epoch 36 completed; recent epochs approximately 10 s;
  no checkpoint boundary reached yet. Final metrics are intentionally pending.

## 2026-09-22: formal ODH reproduction completed

- Runtime: 12,825.8809 s (3.56 h) on CPU with four threads; all 1,500 epochs
  completed. The official-code-compatible final epoch 1,500 remains the formal
  checkpoint; best validation epoch 913 is diagnostic only.
- Test metrics: RMSE 4.1773667, MAE 2.6786728, R2 0.7473608, median relative
  error 0.1366186. Paper source-data recomputation: RMSE 3.9172548, MAE
  2.7434876, R2 0.7778434, median relative error 0.1584022.
- Artifact: `artifacts/reproduction/odh_baseline_20260922/`, including the
  machine-readable `paper_comparison.json`. A fresh position-by-position check
  found the same 247 paper/reproduction targets in the same order (maximum
  numerical difference 1.8310546892053026e-06) and prediction Pearson
  correlation 0.9611928985986683.
- Decision: sufficiently reproduced for Phase 2 gradient-space AL
  infrastructure. Stop after Phase 1 closure pending explicit Phase 2
  authorization. Test q10-q90 coverage is only 0.1862348, so interval width is
  not supported as calibrated uncertainty.
