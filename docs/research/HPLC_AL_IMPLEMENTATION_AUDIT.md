# HPLC AL audit — preflight, implementation and maintenance

Current status (2026-09-22): Phase 2 and the authorized tiny smoke are COMPLETE.
The original execution gate passed **42 tests**. The first maintenance pass
reached 52 tests; the later fixed-L0 diagnosis added 13 focused checks, so the
maintained code now passes **65 tests**.
These are separate records; the original frozen test evidence is unchanged.

Read [the protocol](HPLC_AL_PROTOCOL.md) for scientific settings,
[remaining limitations](#remaining-scientific-limitations) for interpretation,
and [maintenance instructions](#maintenance-after-completion-2026-09-22) for
current verification commands.

Date: 2026-09-22. Scope: Phase 2 and the explicitly authorized one-seed,
three-acquisition development smoke. Existing uncommitted AL draft files were
reviewed and completed in place; original baseline model/adapter stay unchanged.
Companion project was read only. Exact source hashes and its current HEAD are in
`studies/active_learning/odh_gradient_al_smoke/companion_provenance.json`.

## Preflight review and resolution

The preflight review was performed before implementation and training on
2026-09-22. Its baseline was commit
`34fde37f76a8899f2b5a0b6b4d36dbe6d1f2a754`; the companion HEAD was
`e28daef7f1033e8f9ddef3fafc90a72e4308e95b`. The companion had uncommitted
changes and was inspected read-only, so actual reference-file hashes, not HEAD
alone, identify the evidence. The seven hashes and original synthetic probes
remain in [the preflight audit JSON](../../artifacts/reproduction/audit/hplc_al_preflight_20260922.json).

The original review correctly reported that implementation, pytest and duration
measurements were absent **at that time**. Those blockers were resolved below;
they are not current project status. Its separate document was consolidated here.

| Preflight finding | Implemented resolution / evidence |
| --- | --- |
| 4,942 eligible rows do not equal four-role membership | Preserve 4,941 experimental identities plus isolated row 4209; no split changes |
| Baseline loader/manifests contain truth | Allowlisted graph loading and method-local label access; baseline truth-bearing artifacts are not acquisition inputs |
| LCMD can select an empty zero-mass cluster | Mask empty clusters; duplicate/zero-distance regression tests |
| Eval clamp and unused parameters affect gradients | Frozen eval/BN semantics, ordered 818,091 parameter slots, single 512D sketch and real-pool audit |
| Training duration was not measured | Prospective bounded L0 audit, followed by final max=300/patience=100 freeze |
| Per-method test access would leak across the development run | All 12 records must pass the global barrier before one test-source reveal |
| NRMSE/AULC and label costs were ambiguous | L0 population-std normalization, four-point partial AULC, and 247 validation labels accounted separately in the protocol |
| Old environment lacked pytest and NumPy trapezoid API | Isolated AL environment with pytest; explicit trapezoidal formula; no blanket dependency upgrade |

The LCMD failure probe used L=`[[0], [1]]`, U=40 copies of `[1]`, B=32 and
raised `LCMD selected an empty largest cluster` in the inspected companion core.
This was not a failure of the corrected HPLC implementation. The pure MaxDet
probe (100×8 features, 10 labeled rows, B=32) matched a repeated direct solve
in selection order, with maximum gain difference `5.55e-16`.

## Critical findings and corrections before smoke

1. **Identity accounting:** 4,942 eligible includes one unused baseline rounding
   row, source position 4209. Four AL roles total 4,941; preserve the unused row
   and all exclusions separately. No validation/test membership changes.
2. **Label-bearing baseline loader:** cannot be reused. New graphs carry only
   allowlisted fields; labels are separate authorized arrays. Target I/O skips
   unauthorized CSV rows before materializing the RT/Speed frame. Acquisition
   kernels have no source, graph labels or RestrictedLabelStore capability.
   This is a logical application boundary, not an OS security sandbox; trusted
   CSV I/O necessarily reads underlying bytes. Hashing a source is not label use.
3. **LCMD zero-mass empty clusters:** companion argmax could choose an empty
   cluster and fail on duplicate features. HPLC masks empty cluster mass to
   negative infinity; ordinary geometry matches an independently recomputed
   nearest-center reference. Duplicate/zero-distance/all-candidate tests pass.
4. **Single-output semantics:** q10 / MSE central / q90, never q50. Use official
   eval central with inference clamp and frozen BN state; inspect zero norms.
   Preserve all 818,091 trainable parameter slots; unused parameters are zeros.
   No fake second endpoint, CC target scale or CC-specific model logic.
5. **MaxDet fidelity:** pure core extracted, no uncertainty imports or gate.
   Greedy selections and gains tested against repeated direct matrix solves.
   Save normalization and numerical audit, excluding variable wall time from
   immutable selection contracts so crash replay compares deterministically.
6. **Incomplete global barrier in draft:** previously only generic file hashes
   and method/budget names were checked. Now require protocol, partition, access
   audit, nested checkpoint/prediction/selection bindings; replay every L/U
   transition and every successful pre-test label access against its method/round.
   All initial states must match. Incomplete and tampered evidence is refused.
7. **Recovery:** atomic JSON/checkpoint/feature writes; complete fit and gradient
   caches bind input, protocol and content hashes. Partial fits remain in
   numbered attempt directories and restart from scratch. A real-model simulated
   interruption produces the same checkpoint state as an uninterrupted fit.
   An interrupted test reveal fails closed rather than silently reading twice.
8. **Test process compatibility:** nested pytest subprocess after loading Torch
   aborts with SIGABRT on this host, while direct pytest passes. The CLI gate now
   executes pytest in its disposable process, keeping JUnit and text evidence.
   No unsafe OpenMP environment override is used.

## Test coverage

`tests/hplc_al/` covers the 15 requested categories and additional failure paths:
immutable role counts/unions/baseline identities; denied U/test/unused access
before source open; method-local reveal after committed selection; poisoned
unrequested label rows; exact B=32, unique/valid/deterministic selections;
LCMD degeneracy and reference geometry; MaxDet direct solve and exact ties;
nested independent Random ordering; valid/invalid transitions; identical scratch
state; 512D finite repeatable real gradients; eval clamp zero-gradient semantics;
unchanged BN/model state; gradient and prediction batch organization consistency;
no target graph attributes; incomplete/tampered global freeze; nested artifact
omission; replayed access violations; interrupted fit recovery and cache drift;
known-answer metric and AULC definitions.

The machine-readable gate records test count, zero failures/errors/skips, code
hashes, environment and JUnit hash. It is required before duration auditing and
protocol freeze. This is engineering verification, not a scientific AL result.

## Remaining scientific limitations

- **Cross-task fraction denominator mismatch:** the inspected companion
  `sequential_protocol.py` fixes L0=333, outer train=3,330, full cohort=4,163.
  Thus its initial fraction is 10% of outer train, or about 8% of the whole
  cohort. HPLC's authorized L0=356 is 8.005% of outer train (4,447), or 7.204%
  of eligible (4,942). Preserve 356 as explicitly requested, but do not claim
  these are matched outer-training fractions. Cross-task efficiency comparisons
  must align denominators and account for validation-label costs.
- This fixed row test was already used in Phase 1. Post-freeze diagnostics are
  development evidence, not independent external validation. Row split also
  retains known molecular overlap, which this authorized scope does not change.
- Validation contains 247 labels in addition to active budgets 356–452; those
  labels and checkpoint-selection reuse must remain visible in efficiency claims.
- Training duration uses one L0/initialization and a bounded plateau audit;
  patience can miss later improvements and does not prove convergence.
- Fixed batch 2048 means one optimizer update per AL epoch versus three in the
  full-data baseline; epoch counts are not comparable computational exposure.
- Eval clamp can create zero gradients; CountSketch may distort distances and
  determinant geometry. Finite/noncollapsed features alone do not imply relevance
  to RTv residuals. Geometry audits and batch diversity are diagnostic only.
- One seed, three queries and a reused validation set cannot establish a winner.
  Later-round overlap compares different pools. Low overlap alone is not gain.
- Logical firewall and content hashes protect this controlled runner, not a
  malicious process with direct filesystem access. Concurrent writers to one
  study directory are unsupported; run one CLI stage at a time.
- Checkpoints remain local ignored binary artifacts. Durable remote reproduction
  requires separately retaining the hashed checkpoints and original graph cache.
- Shared-host runtime varies with other system activity. Per-round wall times
  measure development feasibility; they are not controlled algorithm speed tests.

## Execution evidence

**Phase 2 COMPLETE.** All 42 tests passed. L0 duration audit completed 300 epochs
in 316.108 s, best at epoch 134, validation MSE 62.7063091. Freeze smoke max=300,
patience=100. Full 4,447-row extraction: 66.764 s, 818,091 trainable scalars in
237 tensors; 139 receive gradient and 98 unused tensors remain zero/None. No
zero-norm, duplicate-feature or clamped-central rows. Sixteen-row repeat was exact.
CountSketch mapping hash:
`5d770cca44d6c30cb064d509650c5d732ffe6348f30db4cd9516f0c54281e6cb`.

**Tiny development smoke COMPLETE.** All 12 logical budget records and 109 bound
files passed global freeze before any test request. Exactly one subsequent test
source reveal. Complete-run and report replays add no fits or source test reads.
Ten unique fits all early-stopped below max 300; five unique gradient passes.
One later MaxDet representation has one clamped zero-gradient row (343), below
the frozen gate; no exact duplicate feature rows in any pass.

MaxDet improves validation mean partial AULC by 4.52% but worsens diagnostic
test mean partial AULC by 2.59% versus Random. LCMD worsens validation by 1.58%
and improves test by only 0.61%. Terminal test R2 remains 0.007–0.055.
There is no stable across-split superiority and no winner claim.
See [the critical review](../../studies/active_learning/odh_gradient_al_smoke/CRITICAL_REVIEW.md)
and [the full report](../../studies/active_learning/odh_gradient_al_smoke/DEVELOPMENT_REPORT.md)
for geometry, norm bias, residual sensitivity, all metrics and conditional next-stage
advice. Only a separately authorized two-seed/six-round development check is
conditionally recommended; no expansion was executed.

## Maintenance after completion (2026-09-22)

The current working tree has been formatted and cleaned after the frozen smoke
completed. Its source hashes therefore differ from those in the original
protocol. `frozen_source.zip` in the study preserves all 17 original bound files;
each member is verified against the unchanged protocol hash. This archive is
historical evidence, not permission to train modified code under the old protocol.

Use `.conda-hplc-al/bin/python -B scripts/run_hplc_al_smoke.py verify` for a
read-only check. Completed `run` and `report` now use the same check, preserving
reviewed conclusions. Mutating preparation, test-gate, duration-audit and protocol
freeze commands reject completed studies; use `.conda-hplc-al/bin/python -B -m pytest -q` for current
code checks. The suite now has 52 passing cases, including completion protection
and historical-source integrity. The subsequent training diagnosis brings the
current suite to 65 tests. The original experiment's 42-test gate record
and all 148 completion-bound artifacts remain unchanged.
