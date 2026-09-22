# ODH gradient AL tiny development smoke

**ENGINEERING / DEVELOPMENT EVIDENCE ONLY.** One seed, three acquisition rounds,
B=32. This is not a formal benchmark or independent external validation.

**Status: COMPLETE.** See [full metrics and curves](DEVELOPMENT_REPORT.md) and
[critical interpretation](CRITICAL_REVIEW.md). Phase 2 passes 42 tests; all
12 logical budget points are globally frozen, with exactly one subsequent test
source reveal. MaxDet has a validation-only direction; diagnostic test AULC
does not confirm it. No follow-on experiment has been started.

From repository root, use the isolated conda interpreter:

```sh
.conda-hplc-al/bin/python scripts/run_hplc_al_smoke.py prepare
.conda-hplc-al/bin/python scripts/run_hplc_al_smoke.py test
.conda-hplc-al/bin/python scripts/run_hplc_al_smoke.py audit
.conda-hplc-al/bin/python scripts/run_hplc_al_smoke.py freeze
.conda-hplc-al/bin/python scripts/run_hplc_al_smoke.py run
.conda-hplc-al/bin/python scripts/run_hplc_al_smoke.py report
```

Run one stage/process at a time. `run` supports recovery: completed records are
verified and reused; interrupted fits retain their attempts and restart from
scratch. No completed result may be silently overwritten. The test report can
be regenerated from the sealed truth cache without another source reveal.
Do not rerun `test` after final protocol freeze: its evidence hash is frozen too.

Key artifacts:

- `splits/partition.json`, `roles.csv`: immutable complete source identities.
- `phase2_test_gate.json`, `phase2_tests.xml`, `phase2_tests.log`: test evidence.
- `companion_provenance.json`: read-only reference file hashes.
- `duration_audit/`: prospective cap rules, validation-only curves and gradient audit.
- `protocol.json`: final smoke training, seeds, semantics and content hashes.
- `runtime/`: per-round checkpoint, prediction, selection, gradient and fit evidence.
- `label_access_audit.csv`: persisted method/round/purpose/ID access log.
- `global_pre_test_freeze.json`: all-method/all-budget test barrier.
- `results/`: diagnostic curves, metrics, partial AULC, overlap and runtime CSVs.
- `DEVELOPMENT_REPORT.md`, `decision.json`: findings and explicit stopping decision.

Checkpoint `.pt` files and the upstream graph cache are retained locally and
ignored by Git; hashes do not replace the binaries for long-term archival.
The historical baseline already exposed this test cohort. These results may
support a decision to continue development, but cannot establish a method winner.
