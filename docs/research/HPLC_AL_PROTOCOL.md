# HPLC AL protocol — authorized tiny development smoke

Status (2026-09-22): **COMPLETE**. This document describes the executed scientific
contract. The machine-readable frozen protocol in the study remains authoritative
for its exact settings and hashes. Implementation findings and preflight history
are consolidated in [the AL audit](HPLC_AL_IMPLEMENTATION_AUDIT.md).

Authorization (2026-09-22): Phase 2 infrastructure, then only one AL seed (73),
three B=32 acquisitions per method, Random / Gradient-LCMD / Gradient-MaxDet.
Budgets: 356, 388, 420, 452. No formal benchmark or automatic expansion.

## Identity and label contract

Keep the baseline's 4,447 outer-train, 247 validation and 247 test identities.
Eligible count 4,942 includes isolated unused rounding row 4209; the four
experimental roles cover 4,941. Preserve all 30 exclusions. Source row position,
export/source index and stable `ODH_charity_0616:row:N` IDs are distinct fields.
L0: 356 rows sampled from sorted outer-train with NumPy default_rng(104802);
U0: 4,091. Initial roles never change; current L/U transitions are separate.
Validation labels cost 247 in addition to each active-label budget.

The input loader uses allowlisted feature columns and the baseline's hashed
graph/descriptor caches. Graphs have no y/RT/RTv. RestrictedLabelStore validates
method-local selected IDs before reading authorized target rows; every request
is synchronously audited. Acquisition functions receive only IDs/features.
Test labels are accessible only after all 12 model/budget records, checkpoints,
predictions and selections have passed the global freeze verification.

## Preregistered duration audit (before any smoke training)

Use only shared L0 and fixed validation. Stage 1: maximum 300 epochs,
patience 300, 1,200-second resource cap. If the best epoch is above 200 and the
resource cap was not reached, scratch stage 2: maximum 600 epochs, patience
600, 1,800-second cap. Require at least 100 observed epochs after the best
validation epoch; otherwise stop and record an insufficient duration audit.
Freeze the smoke maximum to the successful audited stage cap and patience 100.
This rule is a development cost/plateau check, not proof of global convergence.

Fixed in both audit and smoke: original 5-layer 128-dimensional sum-pooling
model, q10/MSE-central/q90 outputs, original complete training objective;
Adam lr .001, weight decay 1e-5, batch 2048, no shuffle, no scheduler, dropout 0.
CPU four Torch threads in `.conda-hplc-al`; no unsafe OpenMP override.
Scratch initialization seed 525 for every method and round; fresh Adam state.
Evaluate fixed validation every epoch and select minimum eval central RTv MSE.
Strict improvement, min_delta 0, ties retain earlier checkpoint. Patience counts
epochs since that strict best; fail on nonfinite loss/gradient/prediction.
Maximum epochs is a hard cap. Save and restore the complete best model state,
including BatchNorm buffers. Test never selects epochs or stopping settings.

**Final measured freeze (2026-09-22, before smoke): maximum_epochs=300,
patience=100.** Stage 1 completed 300 epochs in 316.108 s; best validation epoch
134, MSE 62.7063091 (RMSE 7.91873), followed by 166 epochs without a better best.
Stage 2 was not triggered. Original L0 population std=8.6037605613.
Full-pool gradient extraction took 66.764 s, with exact repeatability on 16 rows,
zero zero-norm rows and zero exact duplicate feature rows. All 42 tests passed.
The final machine-readable contract is
`studies/active_learning/odh_gradient_al_smoke/protocol.json`.

## Gradient and acquisition contract

Single-output `grad_theta(output[:,1])` from official eval forward, including
its [0,1e8] inference clamp. BatchNorm buffers stay fixed. Full-network ordered
trainable parameters, None gradients fill zeros in place; no artificial second
endpoint. One deterministic 512D CountSketch: seed 4000110, NumPy int64 buckets
and signed entries, float64 accumulation, float32 feature storage. Retain mapping,
parameter layout and feature hashes. Audit full/sketched norms, received/None/
zero gradients, finite values, exact duplicates, clamping and repeatability.
One row-gradient workspace avoids an N-by-P Jacobian.

Before smoke: repeat a 16-row extraction exactly and measure all 4,447 outer-train
inputs. Require full-pool time <=1,200 s, zero-norm fraction <=5%, exact duplicate
fraction <=50%. These permissive gates catch catastrophic failures; their PASS
does not establish useful geometry. Recheck zero/duplicate gates every acquisition.

LCMD retains companion nearest-current-L assignment, squared-distance cluster
mass, largest-mass cluster, farthest-point and immediate center updates.
Mask empty clusters in zero-mass ties to repair the audited companion edge case.
Ties follow canonical center/candidate order.

MaxDet: psi=phi/sqrt(mean_L ||phi||²), A=I+psi_L.T@psi_L;
greedy log(1+psi.T@inv(A)@psi), updating A each selection, float64 throughout.
Use only the pure companion kernel; no gate, uncertainty, fusion or latent branch.
Random uses one fixed U0 permutation, seed 73900220, successive disjoint 32-row
slices; never reseed per round. All selection IDs and numerical traces are saved.

## Freezing, recovery and reporting

Bind source, code, environment, partition, training and duration/test evidence
before smoke. Reuse only identical shared round-0 fits/features with explicit
logical versus actual costs. Incomplete fits are retained as numbered attempts
and restarted from scratch; complete artifacts must pass hashes before reuse.
Selections commit before method-local reveal. Round 3 performs no acquisition.
Freeze predictions before test reveal, globally across all methods and budgets.
The test source is revealed once; subsequent reporting uses its bound cache.

The recovery rules above describe execution before completion. After
`completion_manifest.json` exists, the maintained CLI's `run` and `report`
only verify existing artifacts; they do not regenerate the report or train.
Use the [root README commands](../../README.md#本地检查) for current checks.
Historical study README commands are retained as frozen execution evidence.
The original 42-test gate and 17 source files remain verifiable; the current
maintenance suite has 52 tests. Its historical-source fallback never authorizes
training changed code under the old protocol.

Report RMSE, MAE, R2 and NRMSE=RMSE/std(original L0 labels, ddof=0), a fixed
positive denominator. Partial AULC uses the four budget points with trapezoidal
integration over 356..452; mean partial AULC=raw area/96. Lower error is better.
Report validation and post-freeze diagnostic test separately. Overlap: each
32-row batch and cumulative 96 new rows, excluding shared L0; later pools differ.

All findings are **ENGINEERING / DEVELOPMENT EVIDENCE ONLY**. The historical
test was already exposed in baseline reproduction; this firewall does not make
it an independent external-validation cohort. One seed and 96 added labels
cannot identify a winner or support statistical significance. Stop after report.
