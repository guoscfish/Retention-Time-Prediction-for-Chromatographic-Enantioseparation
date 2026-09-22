# Experiment log

Append dated entries; never replace prior results with a better-looking trial.
Status statements below describe the time of each entry. For current status,
read the [roadmap](PROJECT_ROADMAP.md). Documentation relocation annotations
change navigation only, not historical results or decisions.

## 2026-09-21: audit and prospective baseline protocol

- Commit: d284767bf7f1f04b24f758211a1d8d70e82f52ed (main).
- Purpose: faithful ODH reproduction and durable Phase 0-7 research plan.
- Protocol: inspect paper and official code, document exclusions and output
  semantics, run preprocessing and smoke before deciding on 1500-epoch cost.
- Result: initial worktree has only untracked .DS_Store; author files preserved.
  macOS 26.6.2 arm64, Apple M4, 24 GiB memory. Existing fish environment's plain
  torch import aborts with duplicate OpenMP runtime initialization; do not mask
  this with KMP_DUPLICATE_LIB_OK for a scientific run.
- Decision: inspect isolated compatibility options before training.
- Next step: complete source/data/paper audit and minimal reproduction adapter.

## 2026-09-21: ODH smoke gate

- Commit: d284767bf7f1f04b24f758211a1d8d70e82f52ed.
- Purpose: verify the official QGeoGNN data path and AL-readiness primitives
  before any long run.
- Protocol: cached author ODH graph/descriptors; exclude row 4231 and RTv > 60;
  NumPy split seed 388; PyTorch seed 525; 90/5/5 fixed random-row split;
  batch 2048; two training epochs on CPU; no test-based selection.
- Result: 4,942 eligible rows (4,447/247/247); output `(8,3)` and h_graph
  `(8,128)`; finite loss and gradients; backward/optimizer step and label-free
  forward passed. Epoch wall times 16.47 s and 14.83 s. Estimated 1500-epoch
  CPU cost 6.68 h. Smoke metrics are not baseline metrics.
- Decision: stop before formal training under the resource rule. Baseline is not
  yet credible enough for AL Phase 2.
- Artifact: `artifacts/reproduction/odh_smoke_20260921_v4/`.

## 2026-09-22: formal ODH run started

- Command: `.venv-reproduction/bin/python scripts/reproduce_odh_baseline.py --mode train --device cpu --threads 4 --output artifacts/reproduction/odh_baseline_20260922`.
- Runtime: detached screen session `odh_baseline_20260922`; a companion
  `odh_baseline_watch` session checks every 60 seconds and resumes only from a
  saved checkpoint if the main session disappears.
- Status at launch check: epoch 36 completed; recent epochs approximately 10 s;
  no checkpoint boundary reached yet. Final metrics are intentionally pending.

## 2026-09-22: formal ODH reproduction completed

- Runtime: 12,825.8809 s (3.56 h) on CPU with four threads; all 1,500 epochs
  completed. The official-code-compatible final epoch 1,500 remains the formal
  checkpoint; best validation epoch 913 is diagnostic only.
- Test metrics: RMSE 4.1773667, MAE 2.6786728, R2 0.7473608, median relative
  error 0.1366186. Paper source-data recomputation: RMSE 3.9172548, MAE
  2.7434876, R2 0.7778434, median relative error 0.1584022.
- Artifact: `artifacts/reproduction/odh_baseline_20260922/`, including the
  machine-readable `paper_comparison.json`. A fresh position-by-position check
  found the same 247 paper/reproduction targets in the same order (maximum
  numerical difference 1.8310546892053026e-06) and prediction Pearson
  correlation 0.9611928985986683.
- Decision: sufficiently reproduced for Phase 2 gradient-space AL
  infrastructure. Stop after Phase 1 closure pending explicit Phase 2
  authorization. Test q10-q90 coverage is only 0.1862348, so interval width is
  not supported as calibrated uncertainty.

## 2026-09-22: HPLC AL pre-run review

- User requested a review before executing the proposed Phase 2 and tiny
  development smoke. No training, AL trajectory, or test-label evaluation was
  started in this review.
- Verified baseline split identities and source/data/cache hashes. The 4,942
  eligible rows include unused rounding row 4209; the four experimental roles
  must cover 4,941 rows while preserving that separate unused identity.
- Reproduced an empty-cluster failure in the companion LCMD core on duplicate,
  zero-distance synthetic features. Ordinary LCMD returned 32 unique,
  repeatable selections. The pure MaxDet core matched a direct-solve reference
  for all 32 selections (maximum gain difference 5.55e-16).
- Identified required label-free loading, eval/clamp gradient semantics,
  global prediction/test freeze, training-duration audit, metric definitions,
  recovery checks, and environment compatibility work before execution.
- Artifacts: the preflight review (now consolidated in
  [the AL audit](HPLC_AL_IMPLEMENTATION_AUDIT.md#preflight-review-and-resolution)) and
  `artifacts/reproduction/audit/hplc_al_preflight_20260922.json`.
- Status: pre-run review complete; Phase 2 implementation/tests and training
  protocol freeze remain pending. This is not a Phase 2 PASS.

## 2026-09-22: authorized Phase 2 implementation and duration audit

- Scope: one seed 73, three B=32 acquisitions for Random / Gradient-LCMD /
  Gradient-MaxDet only. Initial budget 356; terminal budget 452.
- Audited existing uncommitted draft; corrected LCMD empty-cluster edge case,
  strengthened global freeze with transition/access replay and artifact bindings,
  validated logical label firewall and single-output full-network CountSketch.
- Current Phase 2 gate: 42 tests passed in 3.12 s, no failures/errors/skips.
  Nested pytest subprocess after Torch import aborted on this host; use pytest
  in the gate CLI process, with full JUnit/transcript. No OpenMP bypass.
- Preregistered L0-only duration audit: 300-epoch cap, optionally 600 scratch
  epochs only if late best; require 100 post-best epochs. Validation only;
  max/patience freeze and gradient feasibility gate precede any smoke trajectory.
- Command: `.conda-hplc-al/bin/python scripts/run_hplc_al_smoke.py audit`.
- Study: `studies/active_learning/odh_gradient_al_smoke/`.
- Status: duration audit started; no test access or Phase 2 completion claim yet.

## 2026-09-22: Phase 2 COMPLETE; tiny development protocol frozen

- 300-epoch L0 audit: 316.108 s, validation-best epoch 134, central MSE
  62.7063091; 166 post-best epochs, so no second audit stage was needed.
- Frozen smoke training: max_epochs 300, patience 100, min_delta 0, strict best
  validation central MSE, scratch seed 525, original architecture/loss/Adam.
- Full 4,447-row gradient audit: 66.764 s, 512D, 818,091 parameters, 139 tensors
  receiving gradients and 98 unused/zero tensors. Zero collapsed/duplicate/clamped
  rows. Exact 16-row repeatability, unchanged model/BN state. All 42 tests PASS.
- NRMSE denominator frozen from L0 only: 8.603760561288459.
- Final protocol and code hashes frozen before smoke. Phase 2 COMPLETE;
  Phase 3 limited to tiny development smoke, not the larger feasibility screen.

## 2026-09-22: tiny development smoke completed and critically reviewed

- Completed exactly one seed (73), three B=32 acquisitions per method, budgets
  356/388/420/452, Random / Gradient-LCMD / Gradient-MaxDet. No extension.
- Ten unique scratch fits; all stop by patience, best epochs 43–134. Actual
  training 2,948.997 s; five unique gradient passes 442.639 s; pre-test smoke
  wall time 3,408.026 s (56.80 min). Duration audit cost is additional.
- Global freeze validates 12 model/budget entries and 109 files; zero prior
  test requests. Exactly one successful source test reveal after freeze. Complete
  run/report replay reused artifacts without new fits or source test reads.
- Validation RMSE curves: Random 7.9187/7.7873/7.6157/7.3826;
  LCMD 7.9187/7.9371/7.6773/7.6887;
  MaxDet 7.9187/7.4065/7.1615/6.9678.
- Diagnostic test RMSE curves: Random 8.1861/7.2936/7.9622/8.2801;
  LCMD 8.1861/7.5757/7.6217/8.1108;
  MaxDet 8.1861/7.7753/8.1903/8.0777.
- Validation/test mean partial RMSE AULC: Random 7.68459/7.82964;
  LCMD 7.80603/7.78192; MaxDet 7.33709/8.03251.
- MaxDet validation AULC improves 4.52%, but test AULC worsens 2.59%.
  LCMD validation worsens 1.58%, test improves only 0.61%. All terminal test
  R2 values are low (0.0074–0.0553). No stable cross-split superiority or winner.
- Cumulative new-label overlap: Random–LCMD 6/96, Random–MaxDet 5/96,
  LCMD–MaxDet 26/96. All methods select 96 unique new rows.
- Geometry: no exact duplicate features; centered participation rank 12.9–17.0.
  Selected median norm percentiles around 96–99%; batches are diverse but
  high-norm biased. MaxDet round 2 has one clamped zero-feature row (343),
  below the frozen gate. Original eval-clamp semantics were not changed.
- Critical protocol correction: companion 333/3330 is 10% of outer train;
  its roughly 8% uses all 4163 rows. HPLC 356/4447 is 8% of outer train.
  Preserve requested 356, but do not claim matching outer-training fractions.
- Artifacts: `DEVELOPMENT_REPORT.md`, `CRITICAL_REVIEW.md`, full RMSE/MAE/R2/
  NRMSE and AULC CSVs, runtime/selection/access audits, frozen predictions,
  local checkpoints, and `decision.json` in the study directory.
- Decision: **ENGINEERING / DEVELOPMENT EVIDENCE ONLY**. Conditionally worth
  a separately authorized two-seed/six-round development check to test the
  validation direction, not an immediate ten-round or formal expansion.
  No extra seeds/rounds/methods were run; stop at the authorized end condition.

## 2026-09-22: repository cleanup and readable maintenance code

- Removed superseded `odh_smoke_20260921`, `_v2`, and `_v3` directories;
  retained the documented successful `_v4` run and formal baseline. The v3/v4
  final checkpoint SHA-256 was identical:
  `a911d2e061f56e8f31279493a1c4798859eedbfa5d136ff1261cabad8e7e2f85`.
- Removed the obsolete one-run `watch_odh_baseline.sh`, `screenlog.0`, desktop
  metadata and source/tool caches. Total removed: 42 files, 9,360,135 bytes.
  Added focused ignore rules; retained runtime environments and scientific
  evidence (including model checkpoints, predictions and label-access logs).
- Replaced AL wildcard imports, removed unused imports, expanded dense statements,
  clarified transition variable names, and added scoped Ruff/pytest configuration.
  Rewrote the root README around current entry points and module responsibilities.
- Before refactoring, archived all 17 protocol-bound original source/test files
  in the study's `frozen_source.zip` (48,895 bytes), matching every original hash.
  Original protocol and completion manifests remain unchanged. Historical source
  verification reads this archive without extracting or executing its contents.
- Completed studies now reject mutating setup/audit/freeze stages. `run` and
  `report` only verify completed artifacts, preventing accidental overwriting of
  the reviewed report. `verify` is the explicit read-only CLI. Live training still
  strictly requires matching working-tree code; an archive cannot bypass it.
- Validation: 52 tests passed; Ruff lint and formatting passed; git diff whitespace
  check passed. All 16 top-level function/class ASTs in acquisition, data,
  gradient and training match the original source. Original model/adapter/baseline
  runner files are unchanged. Real `verify`, `report`, and `run` completed with
  all 148 completion-bound artifacts unchanged and 12 budget records verified.
- No scientific training run or additional source test-label access was performed.

## 2026-09-22: documentation consolidation and status correction

- Reduced research Markdown documents from seven to five. Merged the original
  paper/repository audit into `ODH_BASELINE_REPRODUCTION.md`; consolidated the
  preflight findings, provenance and synthetic probes into
  `HPLC_AL_IMPLEMENTATION_AUDIT.md`. Removed the two superseded standalone files.
  The preflight machine-readable evidence remains unchanged.
- Kept the roadmap, scientific protocol and chronological log separate because
  they answer different questions. README now provides one document navigation
  table; updated references to merged material.
- Corrected stale Phase 2 authorization/implementation wording and the roadmap's
  obsolete README-preservation rule. Distinguished the original 42-test execution
  gate from the current 52-test maintenance suite, historical commands from
  current read-only verification, and proposed future metrics from smoke results.
- Preserved all frozen study documents and machine-readable manifests. This was
  a documentation-only change; no training, new evaluation or protocol change.
- Validation: all 39 local Markdown links/anchors resolve; no references to
  removed document paths remain in current repository documentation or code.
  Read-only study verification passed for 148 bound artifacts and 12 budget
  records; the original source snapshot also verifies. Whitespace check passed.
