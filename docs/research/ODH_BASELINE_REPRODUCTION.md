# ODH baseline reproduction

## Prospective protocol (recorded before training, 2026-09-21)

Base commit: `d284767bf7f1f04b24f758211a1d8d70e82f52ed`.
Scope: one ODH baseline, no active learning or model tuning.
Status: audit and smoke gate complete; formal 1500-epoch reproduction deferred
after measured resource estimate.

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

### Smoke gate (completed)

Artifact: `artifacts/reproduction/odh_smoke_20260921_v4/`.

- Command: `.venv-reproduction/bin/python scripts/reproduce_odh_baseline.py --mode smoke --device cpu --threads 4 --smoke-epochs 2 --output artifacts/reproduction/odh_smoke_20260921_v4`
- Eligible rows: 4,942; split sizes: train 4,447, validation 247, test 247.
- Output shape `(8, 3)`; `h_graph` shape `(8, 128)`; 139 parameter tensors received finite central-output gradients.
- Finite graph/features/loss, backward, optimizer update and label-free forward equivalence all passed.
- Epoch 1: train objective 228.5426, validation MSE 199.8030, 16.47 s.
- Epoch 2: train objective 120.9485, validation MSE 267.6448, 14.83 s.
- Smoke test runtime 44.1 s; measured estimate for 1500 epochs is 6.68 h CPU.

The two-epoch metrics are not baseline metrics. The estimate exceeds this
round's resource boundary, so the formal run was not started. There is no
`odh_baseline` result yet and no claim of numerical reproduction.

Reserved formal command (not executed):
`.venv-reproduction/bin/python scripts/reproduce_odh_baseline.py --mode train --device cpu --threads 4 --output artifacts/reproduction/odh_baseline_20260921`.

The active run is `artifacts/reproduction/odh_baseline_20260922/` in detached
screen session `odh_baseline_20260922`. `scripts/watch_odh_baseline.sh` runs in
`odh_baseline_watch` and can restart only from `checkpoint_final.pt` if the main
session disappears. It stops for manual review if the process fails before the
first checkpoint, rather than silently changing the protocol.

## Limitations and deviations

The paper's early-stopping description is not implemented in the public
single-column branch. The planned official-code-compatible run therefore uses
the final 1500th epoch and records the best validation epoch only as a
diagnostic. The author's ODH graph/descriptors are externally cached because
they are absent from Git; cache provenance is recorded alongside hashes. The
current macOS arm64 machine cannot use the existing `fish` environment because
importing its torch stack aborts on duplicate OpenMP runtimes. A clean isolated
Python 3.11 environment is being used; its exact installed versions are saved
in `environment.json`.

## Reproduction judgement

The baseline is **not yet sufficiently reproduced to enter AL Phase 2**. The
protocol and implementation path are auditable, the data/split manifest is
frozen, and AL readiness primitives (central gradient, `h_graph`, label-free
forward and stable IDs) pass smoke checks. The missing requirement is one full
official-code-compatible ODH training trajectory and its test metrics. Do not
proceed to AL while this remains unchecked.
