# ODH IVR / Hybrid seed-525 development screen

Frozen source: HPLC branch `codex/odh-representation-coreset-v1`, commit
`3a4ee4589a1bbb159930d8f0e908afd90f98b7d8`. CC source at
`e50ae7e3719e3563c1c4b11f1d2b7ff180461388` was audited through its
Kernel-IVR protocol, `ivr.py`, `sequential_acquisition.py`, `uncertainty.py`,
`coverage.py`, and tests. The CC `ivr.py` mechanism transfers directly as a
scalar row-kernel surrogate. HPLC uses its single central RTv output gradient
(`output[:,1]`) rather than CC's two-endpoint concatenation.

## Fixed experimental contract

Only seed 525, L0=333, B=32, and budgets 333, 365, 397, 429, 461, 493.
Original outer-training L0 union U0 is the fixed IVR reference. Original
validation and test identities are unchanged. Random and Raw Gradient-LCMD
are read-only historical comparators; they are never fitted or selected anew.
The existing L333 checkpoint and serialized validation/test-X predictions are
shared exactly by all methods. The L333 gradient and unit latent banks are
reused only after checking checkpoint and row alignment. Later IVR and Hybrid
trajectories independently acquire 32 labels and scratch-train an evaluation
model at every budget. The prediction target is RTv = RT × Speed. Training
uses batch 256, deterministic epoch shuffle, Adam at 0.001 with weight decay
1e-5, original complete quantile loss, at most 500 epochs, validation-central
MSE checkpoint selection, and patience 100. Each fit has a fresh optimizer and
no scheduler or warm start. NRMSE uses L333 population SD 8.879240547556758.

## Kernel-IVR

At each acquisition, refresh full-network central RTv gradient CountSketch
512D from the current IVR evaluation checkpoint for all fixed reference rows.
Scale all rows by a **single** reference RMS gradient norm. Set prior precision
and noise variance to 1, without tuning. With X_L and X_R, compute
`C=(I+X_L.T@X_L)^-1`, `M=X_R.T@X_R/|R|`, and score each candidate
`(x.T@C@M@C@x)/(1+x.T@C@x)`. Select 32 greedily; after each pick apply
Sherman–Morrison and rescore the remaining U. Selector arithmetic is float64,
Cholesky initializes C, and exact ties follow current sorted U order. No
candidate labels or validation/test inputs enter IVR.

## Hybrid HPLC adaptation

The CC principle is three-member epistemic disagreement, Top-25% shortlist,
then latent k-center. HPLC adapts this to scalar `output[:,1]` sample std
(`ddof=0`) and unit-normalized `h_graph` cosine distance. This is an HPLC
adaptation, **not** a bitwise/direct reproduction of CC Hybrid. Member 0 is
the current trajectory's evaluation checkpoint. Extra members 1 and 2 use
the same L, training protocol, and deterministic initialization/training seed
`525000000 + 10000*round_index + member`. The formula is fixed for rounds
0–4. At L493 no extra ensemble members are trained. Rank current U by
uncertainty descending, exact ties by current U order. Keep
`max(32, ceil(0.25*|U|))`; choose 32 from this shortlist by greedy farthest
first from all current L unit latent rows, adding each pick as a center.
Distance ties follow shortlist order. Morgan features enter diagnostics only.

## Firewall, metrics, decision

All new selections, checkpoints, validation metrics, and test-X predictions
for both methods through L493 must be content-frozen before one new source
test-truth read. Test outcomes cannot change selection, stopping, or settings.
This is development evidence on an already exposed test cohort, not an
independent confirmatory validation. AULC is trapezoidal NRMSE divided by
interval width for 333–429, 429–493, and 333–493. Report every budget's RMSE,
MAE, R², NRMSE, overlaps, acquisition geometry, and compute cost. Stop at
L493. A candidate for multi-seed confirmation must improve clearly on Random
in full AULC and beat Raw Gradient-LCMD on full AULC, extension AULC, or
endpoint L493 NRMSE. No additional seeds, budget extension, parameter sweep,
or extra method is authorized here.
