# Fixed-L0 training exposure and gradient diagnosis — preregistration

Authorized 2026-09-22. Freeze this file and protocol.json before the first fit.
This is validation-only development evidence, not an AL experiment or independent
validation. Do not use U0 truth, test truth, historical test metrics, or test-based
model/protocol selection. Do not change the existing tiny-smoke protocol.

## Fixed inputs and model

Copy the existing frozen tiny-smoke partition byte-equivalently in meaning;
L0=356, validation=247, U0=4091, test identities=247; no new sampling, queries or
split changes. Source hashes and full original-artifact hashes are bound.
Only original L0 and validation truth may be materialized. Graphs are label-free.
Candidate rows may be used only for label-free geometry. Test remains identity-only.

Original HPLC QGeoGNN: 5 layers, hidden 128, sum pooling, dropout 0,
q10/MSE-central/q90, complete original loss, Adam lr=0.001, weight_decay=1e-5,
no scheduler, no shuffle, no dropped partial batch. CPU, four Torch threads,
deterministic scratch initialization, fresh optimizer. No warm start or early
stopping; all arms complete their registered epochs. Record every epoch's
validation central MSE and metrics. Best strict validation central MSE, earlier
ties, selects the diagnostic checkpoint. Also retain final checkpoint/metrics.

## Parts A–C: nine registered fits

| Arm | Batch | Epochs | Steps/epoch | Total updates | Sample presentations |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 2048 | 300 | 1 | 300 | 106800 |
| B | 256 | 150 | 2 | 300 | 53400 |
| C | 128 | 100 | 3 | 300 | 35600 |
| D | 256 | 300 | 2 | 600 | 106800 |
| E | 128 | 300 | 3 | 900 | 106800 |

Part A: all five arms, initialization seed 525. Part B: reference A and the best
practical challenger from B–E at seeds 1525 and 2525, four additional fits.
Choose that challenger by smallest Part-A best validation RMSE; exact ties use
fewer total updates, fewer epochs, then arm ID. No post-hoc runtime cutoff.
If A is better than all challengers, still repeat the best challenger and report
that it did not improve A. Freeze the selection before either additional seed.
Report all nine fits, and paired three-seed means/sample std (ddof=1), direction
count, absolute and relative mean RMSE improvement. NRMSE uses the already-frozen
L0 population std 8.603760561288459 for every fit. No scientific winner claim.

A/B/C exactly match 300 updates, but not sample presentations. A/D/E match 300
epochs and presentations, but not updates. BatchNorm batch composition, final
partial batches (256+100; 128+128+100), gradient noise and best-of-epoch selection
opportunities also differ. Curves versus steps, epochs and presentations are
needed; these comparisons cannot isolate a pure causal update-count effect.

## Part D: norm/error relevance

Use seed 525 only, A and the selected challenger, to avoid selecting an
initialization for prettier geometry. Validation only: per-row official eval
central prediction, absolute/squared error, full-network gradient norm, and
512D CountSketch norm. Match original eval clamp, fixed BN, parameter order,
None-as-zero handling, sketch seed 4000110, float64 accumulation/float32 feature
storage. Verify model state unchanged and predictions agree with batched eval.
Spearman is primary, Pearson descriptive. Compute both norms against both error
forms. Constant-vector correlations are undefined, not zero. Squaring a
nonnegative absolute residual preserves ranks, so its Spearman is redundant.
These correlations reuse the checkpoint-selection validation cohort and are
not independent confirmation; no significance-based protocol/strategy choice.

## Part E: same-state geometry only

For each of the same two checkpoints, extract original raw 512D sketch features
on fixed L0+U0. Form unit features as raw / max(raw L2 norm, 1e-12). Exact zeros
stay zero; retain all rows and record zero/below-epsilon counts.
Run B=32 raw and unit LCMD and MaxDet on exactly the same current L0/U0. Reuse
existing pure acquisition kernels; do not commit a batch or reveal any labels.
These are diagnostic transformations, not new AL strategies or trajectories.

Save selections and traces. Compare overlap/Jaccard, median selected original
full/sketch norm percentiles, top-decile selection fraction, zero handling,
pairwise diversity and nearest-L distances evaluated in both raw and unit
coordinates for every selected batch. Percentile is empirical U0 CDF:
100 * count(U0 norm <= selected norm) / 4091. MaxDet conditions on current L
and retains its L-based scale in each geometry; gains are within-geometry
numerical diagnostics and not calibrated cross-geometry information gains.

## Evidence, interpretation and stop

Store per-fit curves, measured optimizer steps/sample presentations, best/final
metrics, initialization/checkpoint hashes and runtime (updates/sec includes the
fit's validation/checkpoint overhead). Partial failures retain numbered attempts;
restart from scratch only, never overwrite completed bound evidence.
Code/environment/protocol changes after freeze must fail closed. One writer only.

Rank plausible causes with explicit limitations: insufficient updates, minibatch/
BN effects, gradient magnitude versus residual relevance, representation geometry,
and fixed small-data generalization. Report directions and effect sizes without
inventing significance from three seeds or one validation cohort. Recommend H1,
H2 or H3 only as a next-stage hypothesis; a mixed or inconclusive result must be
stated as such. Do not execute that recommendation.

Stop after nine fits, validation norm/error audit, same-state selections, figures,
report and decision.json. No new AL trajectory, added seed/round, U0/test reveal,
lr/architecture/target change, removed hard samples, uncertainty/Hybrid/IVR/fusion,
or automatic change to the official AL training/acquisition protocol.
