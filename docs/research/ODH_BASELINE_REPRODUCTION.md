# ODH baseline reproduction

## Prospective protocol (recorded before training, 2026-09-21)

Base commit: `d284767bf7f1f04b24f758211a1d8d70e82f52ed`.
Scope: one ODH baseline, no active learning or model tuning.
Status: **COMPLETE**. The official-code-compatible formal run completed all
1,500 epochs on 2026-09-22.

Provisional official-code configuration, pending paper cross-check: ODH CSV;
exclude known conformer position 4231 and apply the official pre-split RTv > 60
domain restriction with a complete exclusion manifest. Target RTv = RT * Speed.
Three outputs q10 / MSE central / q90; five layers, 128 hidden dimensions,
sum pooling, zero dropout; Adam lr 0.001, weight decay 1e-5; batch 2048;
1500 epochs; no scheduler steps or early stopping. NumPy split seed 388,
PyTorch initialization seed 525. Fixed random-row 90/5/5 split, integer floors,
last test segment and explicitly logged unused rounding rows; no loader shuffle.
Use the final epoch checkpoint, never test-based selection. Extra validation
logging is diagnostic only. Retain all test rows in the predeclared domain.

Before formal training: finish preprocessing; verify graph finiteness, nonempty
graphs, paired batch alignment, output and h_graph shapes, finite loss/backward
and optimizer update; run 1-5 complete epochs and validation/test diagnostics.
Record timings and estimate full cost. If prohibitively slow, stop before the
long run and report smoke results only, with an executable full-run command.

Paper reference only (not our result): ODH Figure 3c MAE 2.74, median relative
error 15.8%, R2 0.778; attached source-data recomputation gives RMSE 3.9173,
MAE 2.7435, R2 0.7778, median relative error 15.8402%.

## Results

### Formal 1,500-epoch reproduction (completed)

Artifact: `artifacts/reproduction/odh_baseline_20260922/`.

- Command: `.venv-reproduction/bin/python scripts/reproduce_odh_baseline.py --mode train --device cpu --threads 4 --output artifacts/reproduction/odh_baseline_20260922`
- Eligible rows: 4,942; split sizes: train 4,447, validation 247, test 247.
- Runtime: 12,825.8809 s (3.56 h) on CPU with four threads; 1,500 epochs completed.
- Checkpoint: final epoch 1,500, as used by the public single-column training
  branch. The best validation epoch, 913, is recorded only as a diagnostic.

The comparison below was recomputed from the formal `predictions.csv` and the
author's `artifacts/reproduction/audit/paper_source_data_ODH.csv`; it is not a
transcription of a chat result.

| Test metric | Paper source-data | Formal reproduction |
| --- | ---: | ---: |
| RMSE | 3.9172548 | 4.1773667 |
| MAE | 2.7434876 | 2.6786728 |
| R2 | 0.7778434 | 0.7473608 |
| Median relative error | 0.1584022 | 0.1366186 |

Both files contain 247 test targets. A programmatic position-by-position check
confirmed that they are the same targets in the same order: all 247 positions
match under `rtol=1e-7, atol=1e-6`, with maximum numerical difference
1.8310546892053026e-06 from float serialization. The corresponding paper and
reproduction predictions have Pearson correlation 0.9611928985986683. The full
machine-readable result and comparison method are in
`artifacts/reproduction/odh_baseline_20260922/paper_comparison.json`.

The formal test q10-q90 coverage is only 0.1862348 (46/247). Despite no
quantile crossing, this is far below the nominal 80% coverage implied by q10
and q90. The current interval width is therefore **not calibrated uncertainty**
and must not be used as uncertainty acquisition evidence.

### Smoke gate (completed)

Artifact: `artifacts/reproduction/odh_smoke_20260921_v4/`.

- Command: `.venv-reproduction/bin/python scripts/reproduce_odh_baseline.py --mode smoke --device cpu --threads 4 --smoke-epochs 2 --output artifacts/reproduction/odh_smoke_20260921_v4`
- Eligible rows: 4,942; split sizes: train 4,447, validation 247, test 247.
- Output shape `(8, 3)`; `h_graph` shape `(8, 128)`; 139 parameter tensors received finite central-output gradients.
- Finite graph/features/loss, backward, optimizer update and label-free forward equivalence all passed.
- Epoch 1: train objective 228.5426, validation MSE 199.8030, 16.47 s.
- Epoch 2: train objective 120.9485, validation MSE 267.6448, 14.83 s.
- Smoke test runtime 44.1 s; measured estimate for 1500 epochs is 6.68 h CPU.

The two-epoch smoke metrics are not baseline metrics. Its cost estimate informed
execution planning only; the completed formal artifact above supersedes the
earlier deferred status.

## Limitations and deviations

The paper's early-stopping description is not implemented in the public
single-column branch. The official-code-compatible baseline therefore remains
the final 1,500th epoch. Epoch 913 is a validation diagnostic only: the test set
was not used to select a checkpoint, and neither that diagnostic nor proximity
to the paper result justifies checkpoint replacement or retraining. The
author's ODH graph/descriptors are externally cached because
they are absent from Git; cache provenance is recorded alongside hashes. The
current macOS arm64 machine cannot use the existing `fish` environment because
importing its torch stack aborts on duplicate OpenMP runtimes. A clean isolated
Python 3.11 environment is being used; its exact installed versions are saved
in `environment.json`.

The random-row split includes molecular overlap between train and test, so this
baseline establishes compatibility with the paper protocol rather than
leakage-resistant chemical generalization. The q10-q90 coverage limitation
above also rules out treating interval width as calibrated uncertainty.

## Reproduction judgement

The ODH baseline is **sufficiently reproduced for Phase 2 gradient-space AL
infrastructure**. The full official-code-compatible trajectory, final-epoch
metrics, fixed split, stable identities, central-gradient path, `h_graph`, and
label-free forward path are now recorded and audited. This is a readiness
judgement, not authorization to start Phase 2; active-learning implementation
still requires a new explicit instruction.
