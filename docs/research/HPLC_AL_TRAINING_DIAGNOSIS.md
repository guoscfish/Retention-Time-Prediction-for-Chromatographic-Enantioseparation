# HPLC AL training-exposure and gradient-geometry diagnosis

Date: 2026-09-23. Status: **COMPLETE — STOPPED AT THE AUTHORIZED END**.

This is validation-only development evidence on the frozen tiny-smoke L0 and
validation split. It did not run an AL trajectory, reveal U0 labels, access test
labels or metrics, change the model/loss/target/learning rate, or modify the
existing tiny-smoke protocol. The registered protocol and complete artifacts are
in [`odh_training_protocol_diagnosis`](../../studies/active_learning/odh_training_protocol_diagnosis/).

## Main finding

The current batch-2048 predictor has a **modest, reproducible training-protocol
weakness**, but the evidence does not support a simple “too few optimizer updates”
explanation. Batch 256 / 300 epochs (arm D) improved best validation RMSE over
the reference in all three initializations, from 7.7922 ± 0.2417 to
7.6457 ± 0.1873 (sample standard deviation), a mean absolute improvement of
0.1465 RTv or 1.88%. However, at seed 525, batch 256 already produced nearly all
of its benefit at the same 300-update budget (arm B); extending B from 300 to
600 updates improved best RMSE by only another 0.07%. Batch 128 was worse at
both 300 and 900 updates.

Gradient magnitude is also a material geometry problem. Its association with
validation error is weak (Spearman about 0.23–0.24), while raw LCMD and MaxDet
select overwhelmingly from the highest-norm candidate tail. Unit normalization
removes that tail bias and produces almost disjoint batches. This establishes
that raw magnitude strongly controls selection, but does **not** establish that
unit-normalized acquisition improves prediction: no selected labels were read
and no unit-normalized AL trajectory was run.

The evidence therefore supports **H3 as a next-stage hypothesis**: first freeze
a better-supported training protocol, then separately preregister a raw-versus-
unit gradient AL ablation. This report does not authorize or execute either step.

## Fixed design and exposure

All fits use the same 356 L0 rows and fixed 247-row validation cohort. The model
is the original HPLC QGeoGNN (5 layers, hidden 128, sum pooling, dropout 0,
q10/MSE-central/q90), complete original loss, Adam with lr 0.001 and weight decay
1e-5, no scheduler, deterministic scratch training and no warm start. Every arm
runs all registered epochs; best validation central MSE selects the diagnostic
checkpoint. Test identities remain behind the firewall.

| Arm | Batch | Epochs | Steps/epoch | Total updates | Sample presentations |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 2048 | 300 | 1 | 300 | 106,800 |
| B | 256 | 150 | 2 | 300 | 53,400 |
| C | 128 | 100 | 3 | 300 | 35,600 |
| D | 256 | 300 | 2 | 600 | 106,800 |
| E | 128 | 300 | 3 | 900 | 106,800 |

Because L0 contains only 356 rows, batch 2048 yields exactly one optimizer update
per epoch. Batch 256 yields two updates (256 + 100 rows) and batch 128 yields
three (128 + 128 + 100). A/B/C match 300 total updates but differ in sample
presentations, minibatch noise, BatchNorm statistics and checkpoint opportunities.
A/D/E match 300 epochs and 106,800 sample presentations but differ in batch
composition and update count. The design diagnoses their combined effects; it
cannot isolate one pure causal variable.

The registered optimizer-step figure is
[`validation_rmse_vs_optimizer_steps.png`](../../studies/active_learning/odh_training_protocol_diagnosis/results/figures/validation_rmse_vs_optimizer_steps.png).
Companion views versus epoch and sample presentations are retained so the same
curves are not interpreted on one convenient x-axis only.

## Nine-fit results

All metrics are from the best validation checkpoint. Final-epoch RMSE is shown
to make checkpoint selection visible. Runtime covers training, per-epoch
validation and checkpoint I/O; updates/s therefore is end-to-end diagnostic
throughput rather than optimizer-kernel speed.

| Arm | Seed | Updates | Best epoch | RMSE | MAE | R2 | NRMSE | Final RMSE | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 525 | 300 | 134 | 7.91873 | 5.34069 | 0.18949 | 0.92038 | 8.01704 | 170.89 |
| B | 525 | 300 | 109 | 7.75793 | 5.09808 | 0.22207 | 0.90169 | 7.85702 | 167.83 |
| C | 525 | 300 | 55 | 8.30566 | 5.47539 | 0.10834 | 0.96535 | 8.55343 | 104.09 |
| D | 525 | 600 | 233 | 7.75241 | 5.11690 | 0.22317 | 0.90105 | 7.94270 | 263.65 |
| E | 525 | 900 | 192 | 8.21919 | 5.44739 | 0.12681 | 0.95530 | 8.34997 | 180.61 |
| A | 1525 | 300 | 86 | 7.51350 | 5.17503 | 0.27032 | 0.87328 | 7.63309 | 283.01 |
| D | 1525 | 600 | 277 | 7.42939 | 4.99057 | 0.28656 | 0.86351 | 7.50497 | 586.82 |
| A | 2525 | 300 | 279 | 7.94443 | 5.36358 | 0.18422 | 0.92337 | 8.00107 | 519.62 |
| D | 2525 | 600 | 102 | 7.75533 | 5.22805 | 0.22259 | 0.90139 | 7.95178 | 559.80 |

The nine successful fits used 4,200 optimizer updates and 836,600 sample
presentations in 2,836.3 s (47.27 min). A host interruption stopped the first E
attempt after its logged epoch 125 (at least 74.1 s); it was retained and E was
restarted from scratch. The completed retry exactly reproduced the observed
prefix. B equals the first 150 epochs of D and C equals the first 100 epochs of E,
confirming deterministic continuation rather than independent noise.

Arm D was selected from B–E using seed 525 only, then frozen before seeds 1525
and 2525. It improved against A for 3/3 seeds:

| Seed | A RMSE | D RMSE | Absolute improvement | Relative improvement |
| ---: | ---: | ---: | ---: | ---: |
| 525 | 7.91873 | 7.75241 | 0.16632 | 2.10% |
| 1525 | 7.51350 | 7.42939 | 0.08411 | 1.12% |
| 2525 | 7.94443 | 7.75533 | 0.18910 | 2.38% |

Across seeds, A versus D mean ± sample SD is:

| Metric | A, batch 2048 | D, batch 256 |
| --- | ---: | ---: |
| RMSE | 7.79222 ± 0.24172 | 7.64571 ± 0.18734 |
| MAE | 5.29310 ± 0.10289 | 5.11184 ± 0.11882 |
| R2 | 0.21467 ± 0.04826 | 0.24411 ± 0.03677 |
| NRMSE | 0.90568 ± 0.02809 | 0.88865 ± 0.02177 |

The 3/3 direction is encouraging, but this is not a formal hyperparameter winner:
seed 525 selected D on the same validation cohort, only two further seeds repeat
the comparison, and initialization variation is larger than the mean RMSE effect.

## Batch effect versus update exposure

At seed 525, A→B improves RMSE by 2.03% at the same 300 updates, while A→C
worsens it by 4.89%. B→D adds 300 updates at batch 256 and improves the best
RMSE by only 0.07%; C→E adds 600 updates at batch 128 and improves by 1.04% but
remains worse than A. D's best occurs at 466, 554 and 204 updates for seeds
525, 1525 and 2525 respectively; additional exposure is therefore not uniformly
needed to beat A.

The most defensible interpretation is that batch 2048's one-update-per-epoch
regime is not ideal, and batch 256 is consistently preferable in this diagnosis.
The gain appears to come primarily from the minibatch/BatchNorm/optimization path,
with only weak seed-525 evidence that 600 updates are better than 300. Calling
the original model “severely under-trained because it has only 300 updates” would
overstate the evidence.

## Gradient norm versus validation error

Diagnostics use seed 525 checkpoints only, on 247 validation rows. The exact
official eval central prediction is differentiated; full gradients and 512D
CountSketch gradients use identical parameter order and clamp semantics. Model
and BatchNorm states remain unchanged. Spearman is the primary statistic.

| Arm | Norm | Spearman vs absolute error | Pearson vs absolute error | Pearson vs squared error |
| --- | --- | ---: | ---: | ---: |
| A | Full | 0.23092 | 0.24646 | 0.17861 |
| A | Sketch | 0.24140 | 0.25450 | 0.18317 |
| D | Full | 0.23164 | 0.23695 | 0.14151 |
| D | Sketch | 0.23765 | 0.24636 | 0.14932 |

Spearman against squared error is identical to absolute error because squaring
nonnegative residuals preserves rank. The correlation is positive but weak and
barely changes under the better training arm. Full and sketch results are close,
so this audit does not point to CountSketch as the main source of the weak
error relevance. These values reuse the validation cohort used for checkpoint
selection and must remain descriptive.

## Raw versus unit-gradient same-state selection

Each comparison uses the same seed-525 checkpoint and exactly the same L0/U0;
`phi_unit = phi_raw / max(||phi_raw||, 1e-12)`. There were no zero or below-epsilon
candidate norms. No selection was committed and no selected label was revealed.

| Arm | Selection | Median raw sketch-norm percentile | Top-decile fraction | Raw/unit overlap | Jaccard | Mean pairwise distance in unit space | Mean nearest-L distance in unit space |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | Raw LCMD | 96.87 | 84.38% | 0/32 | 0.000 | 1.069 | 0.614 |
| A | Unit LCMD | 29.31 | 6.25% | 0/32 | 0.000 | 1.174 | 0.771 |
| A | Raw MaxDet | 98.77 | 100.00% | 1/32 | 0.016 | 1.101 | 0.604 |
| A | Unit MaxDet | 15.42 | 9.38% | 1/32 | 0.016 | 1.223 | 0.780 |
| D | Raw LCMD | 97.65 | 78.12% | 0/32 | 0.000 | 1.180 | 0.704 |
| D | Unit LCMD | 40.58 | 3.12% | 0/32 | 0.000 | 1.234 | 0.812 |
| D | Raw MaxDet | 99.11 | 100.00% | 1/32 | 0.016 | 1.209 | 0.696 |
| D | Unit MaxDet | 13.00 | 6.25% | 1/32 | 0.016 | 1.282 | 0.875 |

Unit normalization sharply removes the extreme high-norm bias and increases
angular-space diversity and distance from current L. Raw and unit batches are
almost disjoint under both training checkpoints, so this is a large geometry
change rather than a cosmetic rescaling. Raw-coordinate distances shrink after
unit selection because low-norm rows are now eligible; comparing raw and unit
MaxDet gain magnitudes directly would be invalid because they use different
coordinate scales and conditioning matrices.

## Answers and cause ranking

A. Batch 2048 shows a modest optimization/training-path weakness, not decisive
severe underexposure. D improves validation RMSE for 3/3 seeds by 1.88% on mean.

B. At seed 525, the main gain already appears at batch 256 with the same 300
updates; more updates add very little. Batch 128 remains worse even at 900
updates. Batch composition/BN/noise is therefore at least as important as count.

C–D. D improves mean RMSE by 0.14651 RTv (1.88%), mean R2 by 0.02944, and the
direction is consistent for 3/3 seeds. The small sample and selected validation
cohort prohibit a formal optimality claim.

E. Gradient norm versus absolute validation error Spearman is 0.231–0.241.

F–G. Raw acquisition has an extreme high-norm bias. Unit normalization removes
that bias and changes LCMD/MaxDet batches almost completely, but predictive value
is unknown without a separately authorized label-bearing ablation.

Most likely contributors to the weak prior tiny-smoke AL result, ordered by the
strength and directness of this evidence:

1. **Raw gradient magnitude dominates selection geometry despite weak error
   relevance.** This is directly observed under both A and D checkpoints.
2. **The batch-2048 training path is modestly inferior to batch 256.** The paired
   validation direction is 3/3, although the mean effect is small.
3. **Initialization and small-cohort variance obscure small AL effects.** A's
   across-seed RMSE SD (0.242) exceeds the mean A→D improvement (0.147), and the
   original AL smoke used one seed.
4. **Limited L0/validation chemical coverage and fixed row split constrain
   generalization.** This remains plausible but was not isolated here.
5. **CountSketch distortion is a lower-priority hypothesis.** Full and sketch
   norm/error correlations are very similar; this does not prove distances are
   fully preserved, but it gives no current evidence that sketch norm is the
   leading issue.

## Integrity and limitations

The pre-run suite passed 65 tests. Exact optimizer steps and sample presentations
were measured and matched the preregistration. All successful fits bind their
initialization, checkpoint, input and curve hashes. Source/environment/partition
and all original tiny-smoke files are hash-bound. The label audit contains only
original L0/validation requests: zero U0 requests, zero test requests and zero
selection commits. The original tiny-smoke study still verifies independently.

This diagnosis does not estimate AL performance of arm D or unit features. It
does not disentangle minibatch noise from BatchNorm composition, and best-of-curve
validation selection favors protocols with more checkpoints. It uses one fixed
L0/validation cohort, three initializations for A/D, and only seed 525 for
gradient geometry. H3 is therefore a disciplined next experiment, not a result.

