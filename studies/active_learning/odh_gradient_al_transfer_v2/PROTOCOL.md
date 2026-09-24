# ODH HPLC gradient AL transfer v2

This is a new, independent transfer study. The historical `odh_gradient_al_smoke`
and `odh_training_protocol_diagnosis` directories and their interpretations are
immutable evidence and are not overwritten or reinterpreted. No long-run AL is
started by this protocol change.

## Training contract

Each scratch fit uses the original HPLC QGeoGNN, original q10 / central MSE /
q90 loss and central prediction semantics. It uses batch size 256, maximum 500
epochs, patience 100, Adam learning rate 0.001, weight decay 1e-5, a fresh Adam
state every round, no warm start and no scheduler. Checkpoint selection is based
only on validation central MSE. Before AL, a fixed L0 duration smoke is required.
If any preregistered initialization has best epoch within 100 epochs of epoch 500,
report that the duration protocol is insufficient and do not tune automatically.

Training uses one shared permutation per epoch:
`rng = np.random.default_rng(training_seed * 10000 + epoch)`, followed by
`epoch_ids = labeled_ids[perm]` and `epoch_truth = truth[perm]`. Graph G/H batches
are built from `epoch_ids`, and each target batch comes from the same slice of
`epoch_truth`. Every labeled ID occurs once per epoch; validation, test and U0
labels never enter this shuffle. This is the deterministic-each-epoch approach
used by the column-chromatography formal AL, adapted to separately stored HPLC
targets. A generic `DataLoader(shuffle=True)` is not permitted.

## Absolute-label matched budget

The historical engineering smoke L0=356 remains unchanged. This transfer study
uses L0=333 and B=32, with budgets 333, 365, 397 and 429, and only three
acquisitions. Column chromatography is matched by absolute labels (333/3330 =
10% outer train); HPLC is 333/4447 = 7.49% outer train. The percentages are not
matched.

## Acquisition and gradient semantics

The five registered methods are Random, Raw Gradient-LCMD, UnitNorm
Gradient-LCMD, Raw Gradient-MaxDet and UnitNorm Gradient-MaxDet. Raw gradient
means `g(x) = grad_theta f_central_RTv(x)` through the full network, represented
by a deterministic 512D CountSketch; raw methods use that sketch directly. Unit
methods use a **unit-normalized sketched gradient**:
`phi_unit = phi_raw / max(||phi_raw||_2, 1e-12)` row-wise after the sketch,
retaining zero rows. This is not a normalized full gradient. LCMD and MaxDet
kernels are unchanged.

Each round records raw norm, selected norm percentile, top-decile selected
fraction, effective rank, nearest-L distance, batch pairwise diversity and zero
gradient rows. Center/Width-LCMD, q10/q90 width selectors, Kernel-IVR, Hybrid,
ensemble uncertainty and latent fusion are excluded: Center/Width in column
chromatography corresponds to physical V1/V2 center/width, while single-target
HPLC has no isomorphic physical output. q10-q90 is not used to construct HPLC
Center/Width because its current empirical coverage is poor and cannot be treated
as reliable uncertainty or width.

## Label firewall and stopping rule

For all five methods and four budgets, checkpoint, validation prediction, test-X
prediction, selection and trajectory artifacts are frozen before one unified
test-truth reveal. Acquisition receives no U labels. After the preflight passes,
run exactly one seed × five methods × three acquisitions, then stop. Do not
automatically enter two-seed × six-round, five-seed or ten-round studies.

The final report must include the shuffle implementation, its difference from the
column-chromatography formal training protocol, duration audit, five-method
learning curves, validation/test RMSE MAE R2 NRMSE, partial AULC 333–429, paired
Raw-vs-Unit results, selected gradient norm percentiles, selection overlap and
runtime, followed by a decision on whether a two-seed × six-round stage is
worthwhile.
