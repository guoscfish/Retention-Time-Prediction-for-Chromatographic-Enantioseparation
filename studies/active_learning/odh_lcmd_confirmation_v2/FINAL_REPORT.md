# ODH Raw-Gradient LCMD Confirmation

Status: complete. This is a two-seed development confirmation of Random versus
raw-gradient LCMD, using the existing ODH row split and training protocol.

## Protocol

Seeds 1525 and 2525 use the same fixed L333/U0 partition. Within each seed,
Random and LCMD share the same initialization, L333 fit, validation predictions,
and test-input predictions. Seeds differ in model initialization and use
independent fixed Random permutations. Each strategy adds three batches of 32
rows, reaching L429. Every later model is trained from scratch with batch 256,
deterministic epoch shuffling, Adam, the original full loss, and validation
central MSE checkpoint selection. Raw-LCMD selects from 512-dimensional
CountSketch features of the current model's central RTv gradients.

The primary outcome is trapezoidal NRMSE AULC over labels 333–429, normalized
by the 96-label interval. Lower is better. All method trajectories, checkpoints,
selections, predictions, and pre-test label-access evidence were frozen before
the single test-truth reveal. The exact settings and hashes are in `protocol.json`
and `global_pre_test_freeze.json`.

## Results

| Seed | Random test AULC | LCMD test AULC | LCMD improvement | Random validation AULC | LCMD validation AULC |
|---:|---:|---:|---:|---:|---:|
| 1525 | 0.85597 | 0.82433 | 3.70% | 0.90172 | 0.88141 |
| 2525 | 0.85064 | 0.81493 | 4.20% | 0.88186 | 0.89407 |

LCMD has lower test AULC in both paired seeds. Validation favors LCMD for seed
1525 and Random for seed 2525. At individual budgets, the curves move in both
directions; for example, seed 2525 LCMD has its strongest test point at 397
labels and then worsens at 429. Full per-budget RMSE, MAE, R2, and NRMSE are in
`results/metrics.csv`; per-seed AULCs are in `results/aulc_by_seed.csv` and
`results/paired_aulc.csv`.

## Interpretation

The matching direction of the two test AULCs is useful confirmation that the
earlier LCMD result was not unique to seed 525. The validation disagreement and
non-monotone learning curves show that the benefit is not yet stable enough to
claim general superiority. There are only two new seeds, one fixed row split,
and the same historically exposed test cohort. These results are development
evidence, not an independent external validation or a significance test.

Decision: keep raw-gradient LCMD as a promising candidate. Before using it as a
default strategy, repeat the paired comparison on an untouched split or new
experimental cohort and assess molecule/enantiomer grouping. Do not tune
acquisition settings against these test results.
