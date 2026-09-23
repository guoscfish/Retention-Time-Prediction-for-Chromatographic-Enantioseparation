# ODH fixed-L0 training and gradient diagnosis

Status: **COMPLETE — STOPPED** (2026-09-23). This is a separate validation-only diagnosis.
The original tiny-smoke study and its protocol remain unchanged.

The frozen [protocol](PROTOCOL.md) registered five seed-525 arms and four paired
repeats at seeds 1525/2525: exactly nine complete scratch fits. Fixed L0=356 and
validation=247. No test/U0 labels, no AL trajectories, no acquisition label reveal.
Raw/unit selections are same-state diagnostics only, using seed-525 checkpoints.

## Artifacts

- `protocol.json`, `partition.json`, `frozen_source.zip`: registered contract,
  unchanged identities and the exact diagnostic implementation.
- `tests.xml`, `tests.log`: 65 passing preregistration tests, including the original suite.
- `practical_arm_selection.json`: choice from B–E using only Part-A validation,
  recorded before additional seeds.
- `runtime/seed_*/{arm}/`: full curves, best/final checkpoints, initialization/
  checkpoint hashes, exact optimizer updates and sample presentations.
- `results/training_protocol_comparison.csv`: nine-fit table, best/final metrics.
- `results/optimizer_step_comparison.csv`: planned exposure of A–E.
- `results/seed_metric_summary.csv`, `paired_seed_comparison.csv`: paired seed metrics.
- `results/gradient_error_relevance.csv`: per-validation-row predictions,
  residuals, full gradient norms and 512D sketch norms.
- `results/gradient_error_relevance.json`, `gradient_error_correlations.csv`:
  Spearman/Pearson of both norms against absolute/squared error.
- `results/same_state_selection_comparison.csv`: raw/unit overlap, norm bias,
  diversity, nearest-L distances and MaxDet gains.
- `runtime/geometry_seed_525/`: label-free features, norms, selected IDs and traces.
- `results/figures/`: RMSE versus steps/epochs/presentations, seed comparison,
  norm/error scatter and same-state geometry.
- `label_access_audit.csv`: only original L0/validation requests are permitted.
- `decision.json`: limitations, likely causes and next-stage recommendation;
  never permission to execute another experiment.
- `completion_manifest.json`: added only after reporting and final integrity checks.

Checkpoint binaries remain local and Git-ignored; retain them for full verification.
The source snapshot and hashes are not substitutes for checkpoint/cache backups.

## Commands from repository root

Historical registration/execution order:

```bash
.conda-hplc-al/bin/python -B scripts/run_hplc_training_diagnosis.py prepare
.conda-hplc-al/bin/python -B scripts/run_hplc_training_diagnosis.py train
.conda-hplc-al/bin/python -B scripts/run_hplc_training_diagnosis.py geometry
.conda-hplc-al/bin/python -B scripts/report_hplc_training_diagnosis.py summarize
```

Do not start another writer while a stage is running. `prepare` requires the
preregistration test evidence already present. Partial fits restart from scratch
into a numbered attempt; completed records are hash-checked and reused.
After completion, use only read-only verification:

```bash
.conda-hplc-al/bin/python -B scripts/report_hplc_training_diagnosis.py verify
```

No command here runs a new AL trajectory. The maintained original AL CLI continues
to verify its completed results without overwriting them.

Read the [complete diagnosis](../../../docs/research/HPLC_AL_TRAINING_DIAGNOSIS.md)
and `decision.json` for results and limitations. The result supports H3 only as
a next-stage hypothesis; no subsequent experiment has been started.
