# QGeoGNN HPLC reproduction and external active-learning roadmap

## Scope and scientific question

Can gradient-space batch active learning provide transferable label-efficiency
gains across chromatographic prediction tasks?

This fork reproduces Xu et al., Nature Communications 14 (2023),
DOI: 10.1038/s41467-023-38853-3. The independent companion project is
https://github.com/guoscfish/column_chromatography_prediction. Its approximately
4k-sample 4 g flash-column QGeoGNN-V2 task motivates an external test on the
approximately 5k-sample ODH chiral HPLC task. Related model ancestry does not
make their experimental systems, data sources, or targets equivalent.

Current status (2026-09-22): Phase 0/1 audit and baseline, Phase 2 infrastructure,
and one seed × three B=32 acquisition rounds of Random, Gradient-LCMD and
Gradient-MaxDet are COMPLETE. The authorized work stopped after its development
report. The larger Phase 3 screen remains unauthorized. No additional
strategies, formal benchmark, compound/block or multi-column AL. Do not tune
against test labels, change architecture/descriptors/target, or remove difficult
test samples to improve metrics.

## Phases and gates

| Phase | Work | Entrance criterion | Stop / exit criterion |
| --- | --- | --- | --- |
| 0 | **COMPLETE** — repository, paper, data, model and protocol audit | User authorized reproduction | Provenance, target/output semantics, exclusions, splits, dependencies and discrepancies documented |
| 1 | **COMPLETE** — one official-seed ODH single-column baseline | Protocol frozen; preprocessing and full smoke pipeline passed | All 1,500 epochs and final-checkpoint assessment completed |
| 2 | **COMPLETE** — AL infrastructure | Explicit authorization received; baseline preserved | 42 tests PASS; fixed roles/firewall, duration/gradient audits and frozen protocol validated |
| 3 | **tiny development smoke COMPLETE** — 1 seed, B=32, 3 rounds; Random vs Gradient-LCMD vs Gradient-MaxDet | Phase 2 passed; tiny subset explicitly authorized | All 12 points globally frozen and reported; stop. Larger 2-seed/6–10-round screen requires new authorization |
| 4 | Formal ODH benchmark: 5 seeds, fixed L0, B=32, full sequential trajectory | Clear Phase 3 signal and frozen protocol | Per-seed and aggregate curve, label-efficiency and runtime results archived |
| 5 | Leakage-resistant chemical generalization | Baseline chemistry/duplicate audit and formal benchmark available | Justified molecule/enantiomer/scaffold grouping frozen and evaluated; do not choose an arbitrary grouping now |
| 6 | Experimental-unit / block acquisition | Pair/condition provenance and outcome correspondence verified | Evaluate an experiment defined by enantiomer pair + column + eluent proportion + flow, including paired outcome cost |
| 7 | Multi-column AL using All_column_charity.csv | Earlier stages support extension and new authorization | Reassess column descriptors, query unit, costs, domain imbalance and coverage for molecule x column x conditions |

## AL protocol

**Phase 2 COMPLETE.** Implementation and execution gates are tracked in
[the protocol](HPLC_AL_PROTOCOL.md) and [the consolidated audit](HPLC_AL_IMPLEMENTATION_AUDIT.md).
All 42 tests in the original execution gate passed;
duration and real full-pool gradient audits passed. Frozen scratch training:
max 300 epochs, patience 100, best fixed-validation central MSE checkpoint.
Phase 3 status: **tiny development smoke COMPLETE**, one seed × three acquisitions
only. MaxDet validation mean partial AULC improves 4.52% versus Random, while
diagnostic test AULC worsens 2.59%. No consistent across-split winner. See
`studies/active_learning/odh_gradient_al_smoke/CRITICAL_REVIEW.md`.
The larger screen is not started or authorized; a two-seed/six-round development
check is only conditionally recommended, not a formal benchmark.
After maintenance, the current suite passes 52 tests. The original 42-test
record remains frozen. See the [root README](../../README.md#本地检查) for
current read-only verification commands; the study README records historical
execution stages.

Keep fixed test, fixed validation, initial labelled L0 and candidate U0 identical
across methods. Use identical initialization and training procedures. Acquisition
must never read candidate or test truth. Query labels become available only
through the acquisition boundary. Evaluate test truth only after trajectory
freeze. Validation access, stopping and any representation normalization must be
specified before running. Keep stable source-row IDs and immutable manifests.

The completed smoke used only Random, Gradient-LCMD and Gradient-MaxDet. A future
Hybrid comparator would need scientific justification and separate scope approval.
CC concatenates gradients of V1/V2 central outputs; HPLC uses the actual
single RTv central output. Index 1 in the audited code is MSE-trained central
prediction, not an explicitly pinball-trained q50. Name it `central`, and verify
semantics before any gradient extraction.

The completed smoke reports RMSE, MAE, R2, L0-normalized RMSE, four-point partial
AULC, overlap and runtime under the frozen protocol. A future formal benchmark
may additionally report labels-to-target, N80/N90/N95 and label saving; these
were not measured in the smoke and are not current implementation requirements.
Freeze target thresholds, direction, normalization reference, interpolation and
unreached-target handling before trials. Judge complete learning curves, not
only endpoints. Never use CC outcome metrics to tune this task.

## Reproducibility principles

Separate paper-reported claims, observed official code, and our measured results.
Record commit, source/data hashes, preprocessing settings, dependency versions,
hardware, seeds, split membership, exclusions, commands and timings. Preserve the
author's model code; use a small adapter with explicit compatibility changes.
The root README is maintained as current navigation and retains the paper citation
and original resource links. Historical experiment code is preserved in the
hash-verified source snapshot; frozen result documents are not edited for navigation.
Use no test-based checkpoint selection; do not declare shortened smoke training
a successful reproduction.
All experiments append to EXPERIMENT_LOG.md and retain machine-readable artifacts.

## AL readiness checklist

- [x] Central-output parameter gradients verified on a real batch
  (`smoke_checks.json`: 139 parameter tensors with finite central gradients).
- [x] Graph latent `h_graph` verified; official forward returns `(output, h_graph)`
  with observed smoke shape `(8, 128)`.
- [x] Label-free graph inputs and candidate inference verified
  (`label_free_forward_equal: true`).
- [x] Stable CSV row identity and source index preserved through exclusions/splits
  in `split_manifest.json`, `sample_manifest.csv`, and `predictions.csv`.
- [x] Output semantics audited: q10 / MSE central / q90, with inference clamp.
- [x] Reliable 1,500-epoch final-checkpoint baseline and limitations reviewed
  before Phase 2. In particular, test q10-q90 coverage is 0.1862348 and does
  not support interval width as calibrated uncertainty.

Before further research, read the [combined baseline and original-paper audit](ODH_BASELINE_REPRODUCTION.md)
and the [AL audit](HPLC_AL_IMPLEMENTATION_AUDIT.md). Completed gates do not
authorize a larger experiment.
