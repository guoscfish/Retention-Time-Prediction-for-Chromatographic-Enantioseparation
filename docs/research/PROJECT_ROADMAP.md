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

Current authorization: Phase 0 audit, Phase 1 baseline, and planning only.
After the reproduction report, STOP and wait for a new user instruction.
Do not implement acquisition, AL infrastructure, Gradient-LCMD, Gradient-MaxDet,
Kernel-IVR, Hybrid, uncertainty filtering, gradient/latent fusion, adaptive
batches, compound OOD AL, or multi-column AL in this round. Do not tune against
test labels, change architecture/descriptors/target, or remove difficult test
samples to improve metrics.

## Phases and gates

| Phase | Work | Entrance criterion | Stop / exit criterion |
| --- | --- | --- | --- |
| 0 | Repository, paper, data, model and protocol audit | User authorizes reproduction | Provenance, target/output semantics, exclusions, splits, dependencies and discrepancies documented; unresolved issues explicit |
| 1 | One official-seed ODH single-column baseline | Protocol frozen; preprocessing and full smoke pipeline pass | Full run and artifacts assessed, or measured prohibitive resource cost documented with full command; no premature success claim |
| 2 | AL infrastructure only after new instruction | Credible baseline and explicit approval to proceed | Fixed test/validation/L0/U0 manifests, label firewall and trajectory freeze validated |
| 3 | ODH feasibility screen: 2 seeds, B=32, 6-10 rounds; Random vs Gradient-LCMD vs Gradient-MaxDet | Phase 2 checks pass and screen explicitly authorized | Assess complete curves for a signal; overlapping curves trigger representation/task diagnosis before adding strategies |
| 4 | Formal ODH benchmark: 5 seeds, fixed L0, B=32, full sequential trajectory | Clear Phase 3 signal and frozen protocol | Per-seed and aggregate curve, label-efficiency and runtime results archived |
| 5 | Leakage-resistant chemical generalization | Baseline chemistry/duplicate audit and formal benchmark available | Justified molecule/enantiomer/scaffold grouping frozen and evaluated; do not choose an arbitrary grouping now |
| 6 | Experimental-unit / block acquisition | Pair/condition provenance and outcome correspondence verified | Evaluate an experiment defined by enantiomer pair + column + eluent proportion + flow, including paired outcome cost |
| 7 | Multi-column AL using All_column_charity.csv | Earlier stages support extension and new authorization | Reassess column descriptors, query unit, costs, domain imbalance and coverage for molecule x column x conditions |

## Future protocol (not implemented)

Keep fixed test, fixed validation, initial labelled L0 and candidate U0 identical
across methods. Use identical initialization and training procedures. Acquisition
must never read candidate or test truth. Query labels become available only
through the acquisition boundary. Evaluate test truth only after trajectory
freeze. Validation access, stopping and any representation normalization must be
specified before running. Keep stable source-row IDs and immutable manifests.

Start with Random, Gradient-LCMD and Gradient-MaxDet; add an already established
Hybrid comparator only if scientifically necessary. Do not copy all CC methods.
CC concatenates gradients of V1/V2 central outputs; HPLC should use the actual
single RTv central output. Index 1 in the audited code is MSE-trained central
prediction, not an explicitly pinball-trained q50. Name it `central`, and verify
semantics before any gradient extraction.

Prediction metrics: RMSE, MAE, R2, explicitly defined normalized RMSE. AL metrics:
normalized AULC and per-seed AULC over an identical budget range, labels-to-target,
N80/N90/N95, fixed-budget endpoint, label saving, training/acquisition runtime.
Freeze target thresholds, direction, normalization reference, interpolation and
unreached-target handling before trials. Judge complete learning curves, not
only endpoints. Never use CC outcome metrics to tune this task.

## Reproducibility principles

Separate paper-reported claims, observed official code, and our measured results.
Record commit, source/data hashes, preprocessing settings, dependency versions,
hardware, seeds, split membership, exclusions, commands and timings. Keep the
author's code and README intact except a short navigation appendix. Prefer a
small adapter with explicit compatibility changes. Use no test-based checkpoint
selection; do not declare shortened smoke training a successful reproduction.
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

Read ORIGINAL_PAPER_AND_REPO_AUDIT.md and ODH_BASELINE_REPRODUCTION.md before
resuming. An unchecked item is not permission to implement AL.
