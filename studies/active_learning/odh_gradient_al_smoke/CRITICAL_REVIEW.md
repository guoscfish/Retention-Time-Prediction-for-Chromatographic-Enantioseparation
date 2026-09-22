# Critical review of the completed tiny smoke

**ENGINEERING / DEVELOPMENT EVIDENCE ONLY.** Phase 2 COMPLETE; Phase 3 is only
a completed one-seed, three-acquisition tiny development smoke. No winner,
statistical significance, or independent external-validation claim is supported.

## Main scientific finding

MaxDet shows a validation direction worth checking, but it does **not** show a
consistent gain across validation and diagnostic test curves. Relative to Random:

| Method | Validation mean AULC improvement | Validation endpoint improvement | Test mean AULC improvement | Test endpoint improvement |
| --- | ---: | ---: | ---: | ---: |
| Gradient-LCMD | -1.58% | -4.15% | +0.61% | +2.05% |
| Gradient-MaxDet | +4.52% | +5.62% | -2.59% | +2.44% |

Positive means lower error. AULC uses all four points, not a best-point subset.
The same relative changes hold for RMSE and the fixed-scale NRMSE.
Neither a favorable MaxDet validation curve nor a favorable test endpoint is
enough to claim transferable label efficiency. LCMD's tiny test AULC change is
also insufficient to establish a signal given the opposite validation direction.

Absolute test performance remains weak at budget 452: R2 is 0.0074 for Random,
0.0476 for LCMD and 0.0553 for MaxDet. These are small-data development fits,
not comparable reproductions of the full-data 1,500-epoch baseline. The ten
largest squared errors contribute 42.0%, 39.1% and 45.1% of terminal test SSE,
respectively. This helps explain sensitivity of a 247-row diagnostic curve;
it is not justification to remove difficult rows. All 247 rows remain in all
reported metrics. No test-based training or protocol adjustment was made.

## Engineering and stability checks

- 42 tests passed with no failures, errors or skips before training audit/smoke.
- All 12 logical model/budget records and 109 files passed the global freeze.
  Test requests before that barrier: zero. Successful post-freeze source reveal:
  exactly one. A complete `run` replay did not add fits; a `report` replay reused
  the bound test cache. Baseline validation/test membership remains unchanged.
- Ten unique scratch fits, all stopped by patience before the 300-epoch cap;
  best epochs 43–134, stopped epochs 143–234. Initial state hashes are identical.
  Last/best validation-MSE ratios are 1.025–1.118. There is no nonfinite training
  failure or evidence that the maximum cap truncated these fits. Early stopping
  still cannot exclude later improvements beyond the observed trajectory.
- Three acquisitions per method, each 32 unique previous-U rows. Each method
  obtains 96 new labels; terminal active count 452, remaining U count 3,995.
  No fourth acquisition was performed.
- Actual unique training time: 2,948.997 s (49.15 min); five unique full-pool
  gradient passes: 442.639 s (7.38 min). Smoke wall time from first label audit
  to global freeze: 3,408.026 s (56.80 min). Prior duration audit costs an
  additional 316.108 s training + 66.764 s full-pool extraction, plus small probes.
  The first fit is shared by all three methods; first features by LCMD/MaxDet.
  Shared-host timings are not controlled algorithm speed comparisons.

## Gradient geometry and batch diversity

All maps retain 818,091 trainable parameter slots in 237 tensors. There are
139 gradient-receiving tensors and 98 unused/None tensors, retained as zeros.
Every pass is finite, has zero exact duplicate feature rows and keeps model/BN
state unchanged. The original 16-row repeat is exact. Mapping hashes match.

One exception to entirely nonzero rows is explicit: at the MaxDet 420-label
checkpoint, source row 343 has a clamped central output and zero feature norm
(1/4,447 = 0.0225%). This is below the prospective 5% failure threshold and is
not global collapse. Nevertheless, a zero vector has no MaxDet marginal gain,
so inference clamping can make such candidates invisible to that acquisition.
The prescribed eval-clamped output semantics were preserved, not changed mid-run.

Centered feature participation rank is about 12.9–17.0, with the top eigenvalue
carrying 16.7–22.7% of centered variance. There is usable spread, but a 512D
sketch is not 512 independent information directions. These diagnostics alone
neither prove representation collapse nor prove predictive relevance.

| Method | Selected median pool-norm percentile, rounds 1/2/3 | Batch/pool mean pairwise squared-distance ratio, rounds 1/2/3 |
| --- | --- | --- |
| Gradient-LCMD | 96.87 / 96.19 / 96.41 | 3.54 / 2.81 / 2.74 |
| Gradient-MaxDet | 98.77 / 98.97 / 98.58 | 4.83 / 3.90 / 4.22 |

Selected batches are diverse in raw gradient space, and they strongly favor
large-norm candidates. A possible concern is that geometric coverage emphasizes
high-sensitivity inputs without equivalent residual-error benefit; the present
audit does not establish causality. MaxDet optimizes a conditioned determinant,
not raw pairwise distance, so the latter is only a descriptive diagnostic.
Do not respond to these findings by silently adding normalization, raw-output
gradients, uncertainty gates or new methods.

## Selection overlap

| Pair | Batch intersections (of 32), rounds 1/2/3 | Cumulative intersection (of 96) | Cumulative Jaccard |
| --- | --- | ---: | ---: |
| Random / LCMD | 2 / 0 / 1 | 6 | 0.03226 |
| Random / MaxDet | 0 / 0 / 0 | 5 | 0.02674 |
| LCMD / MaxDet | 7 / 3 / 2 | 26 | 0.15663 |

Shared L0 is excluded. Later methods have different remaining pools and models;
overlap is not a controlled same-pool comparison after round 1. Different selected
IDs establish trajectory differentiation, not label-efficiency gains.

## Protocol interpretation and next-stage recommendation

The requested fraction rationale needs correction: companion sequential L0 is
333/3,330=10% of outer train (333/4,163≈8% of all rows); HPLC is 356/4,447≈8%
of outer train. The requested HPLC budget was preserved. Cross-task claims must
use explicit, aligned denominators. Add the fixed 247 validation labels to active
budgets: actual observed non-test labels are 603/635/667/699.

The baseline row split has chemical overlap: 47 validation and 46 test rows
share canonical isomeric SMILES with outer train. The historical test was already
exposed in Phase 1. This run's global barrier protects workflow integrity; it
does not create an independent external-validation cohort.

**Recommendation: conditionally worth a separately authorized 2-seed × 6-round
development screen, not an immediate 10-round or formal benchmark expansion.**
The reason to continue is to test whether the validation-only MaxDet direction
survives more initial pools and budgets; the reason for caution is contradictory
test AULC, weak absolute test R2 and high-norm selection bias. Treat the next run
as a falsifiable development check, not confirmation of a winner. Freeze its
seeds, exact six rounds, unchanged three methods, training protocol, reporting
and all-trajectory barrier before execution. Do not tune max epochs/patience
using these test diagnostics. Preserve this completed smoke as an immutable
historical result in its own study directory.

If the directional signal does not repeat, inspect representation relevance,
checkpoint stability and batch composition before adding methods. No next-stage
run, extra seed, extra round or extra strategy was executed by this task.
