# ODH gradient AL tiny smoke protocol

Scientific protocol: [HPLC_AL_PROTOCOL.md](../../../docs/research/HPLC_AL_PROTOCOL.md).
This study is **ENGINEERING / DEVELOPMENT EVIDENCE ONLY**.

Exactly seed 73, Random / Gradient-LCMD / Gradient-MaxDet, three acquisitions
of 32 rows and four budget points 356/388/420/452. Existing 247-row validation
and 247-row test memberships stay fixed. One unused eligible rounding row stays
isolated. Validation label cost is additional to every active-label budget.

Execution gates, in order:

1. Prepare immutable identity manifest and prospective duration-audit rules.
2. Pass all Phase 2 tests, recording code/environment and JUnit hash.
3. Run bounded L0/validation-only duration audit and full-pool gradient check.
4. Freeze `protocol.json`, including measured choice of max epochs/patience.
5. Execute three trajectories with method-local label access and selection commit.
6. Validate global checkpoint/selection/prediction freeze across all 12 records.
7. Reveal test once, calculate diagnostic metrics, curves, partial AULC and overlap.
8. Stop. Larger experiments require a new explicit instruction.

`duration_audit/protocol.json` is the first prospective freeze; `protocol.json`
is the final smoke freeze. Do not edit either once its associated run begins.
The code is content-hashed; runtime refuses code/environment/partition drift.
All smoke fits start from the same seeded initialization and fresh optimizer.
Only identical round-0 computations may be shared, with cost reuse recorded.

Gradient: official eval MSE central (column 1), full network, deterministic
single-output 512D CountSketch. No q50, second endpoint, uncertainty or fusion.
LCMD follows the companion squared-distance TP geometry with empty-cluster fix.
MaxDet conditions on current L, updates after every selected row and uses float64.
Random consumes successive chunks of one fixed U0 permutation.

NRMSE denominator: original L0 population standard deviation, fixed across all
methods and rounds. Mean partial AULC: trapezoidal area over 356..452 divided
by 96. Lower error is better. Overlap excludes common L0.
