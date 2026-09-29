# Cleanup report — 2026-09-29

**READY_FOR_API_KEY.** Repository cleanup, direct Responses implementation, hierarchical
FullPool-LLM32 V2, tests and a full real-input simulated dry-run are complete. No new
scientific LLM acquisition, API request, acquisition-label reveal or scratch training
was performed. The user explicitly chose to stop at environment-key readiness.

## Navigation and final structure

See [CLEANUP_AUDIT](CLEANUP_AUDIT.md) (committed before edits), the exhaustive
[pre-cleanup inventory](cleanup_inventory.json), and [canonical study index](../../studies/active_learning/README.md).

```text
README.md                         researcher entry points + original paper README
code/                             original model (unchanged)
dataset/                          original data (unchanged)
artifacts/reproduction/            original benchmark + local graph cache
src/
  reproduction/                   original reproduction pipeline
  hplc_diagnosis/                  training diagnosis
  hplc_al/
    training.py, data.py, gradient.py, protocol.py  original numerical/label contracts
    ...                           retained distinct numerical study runners
    llm/
      catalog.py                  label-free chemical metadata
      responses_transport.py      explicit env-key Responses HTTP, one JSON parser
      full_pool.py                field allowlist, lossless cards, chunks, summary, coverage
      planner.py                  frozen Stage1/Stage2 prompts, strict validation, receipt replay
      memory.py                   trajectory-local hypotheses and frozen feedback
      runner.py                   baseline registration, Git seal, reveal, scratch fit
      reporting.py                diagnostics, validation metrics, label-AULC
      dry_run.py                  explicitly simulated engineering fixture
scripts/
  run_hplc_fullpool_v2.py          explicit prepare/dry-run/select/advance/report stages
  preflight_llm_responses.py      content-free preflight only
  ...                            reproduction + historical numerical entry points
studies/
  active_learning/
    README.md                     six primary research entry points
    odh_training_protocol_diagnosis/
    odh_lcmd_confirmation_v2/      canonical Random/LCMD comparator
    odh_ivr_hybrid_screen_v1/      retained methodological screen
    odh_free_llm32_scientist_v1/
      README.md, PROTOCOL.md, STATUS.md, INTERIM_REPORT.md
      results/                    original result bytes
      provenance/                 success execution, path/hash map, failed-attempt summary
    odh_free_llm32_fullpool_v2/
      README.md, PROTOCOL.md, STATUS.md
      protocol.json, protocol_freeze.json, test_gate.json, dry_run.json
      transport_preflight.json    READY_FOR_API_KEY; requests_sent=0
      runtime/.../round_0/         offline packet/preparation only, no selection
    odh_gradient_al_smoke/         retained original engineering study
    odh_gradient_al_transfer_v2/   retained training/gradient transition evidence
    odh_representation_coreset_v1/
    odh_representation_coreset_extension_v1/
    odh_lcmd_confirmation_v1/      archive-only; frozen referenced paths retained
  archive/llm_hybrid/              unique historical evidence, duplicates mapped once
    relocation_manifest.json
    README.md
    odh_llm_hybrid_v1,v2,v3,v4,v6,v7,v8/
docs/
  repository/                     audit, report, transport guide, verification evidence
  archive/LLM_HYBRID_LEGACY_HISTORY.md
  research/                       original research interpretation
```

## Removed, moved and retained

- Removed seven legacy active Python modules: llm_transport, llm_cli_transport,
  llm_catalog, llm_hybrid, free_llm_scientist, free_llm_runner, free_llm_report.
  Useful chemistry/geometry was extracted; paging budgets, duplicate parser,
  16+16 execution and CLI/login/fallback scientific paths were retired. Recover
  exact historical implementations from annotated tags.
- Removed the mutable V1 script and its 21 historical runtime-oriented tests;
  current coverage instead tests new transport/planner/label contracts plus exact
  historical result and artifact preservation. Old tests remain in Git/tag history.
- Moved all 95 successful V1 execution files (including local-only files) from
  transport_revision_2 into results/provenance. No scientific bytes were changed.
  Original report remains byte-identical and its historical paths resolve through
  relocation_manifest.json. New STATUS is the single current completion statement.
- Replaced failed root Responses attempt, conflicting EXECUTION_STATUS and duplicate
  failed runtime with a concise failure summary, commit/tag locators. Failure had
  zero responses/selections/acquisition reveals. Local ignored failed arrays were
  preserved under ignored `.cleanup-local-backup/`, not silently discarded.
- Moved seven hybrid directories out of active_learning. 147 original files map to
  139 retained byte-identical files; eight exact duplicate copies are deduplicated.
  Unique selection/response/fit provenance remains available. v8 is PARTIAL: its
  STOPPED_OR_COMPLETE marker is not evidence of a complete matched comparison.
- Retained the original model, datasets, reproduction, training, graph, gradient,
  label store, all numerical studies and their reports/manifests/checkpoints. These
  are distinct scientific protocols or frozen dependencies; speculative flattening
  would destroy useful context or break provenance. Large previously ignored arrays
  remain local-only; archive tags never purport to contain those bytes.

V1 is frozen development evidence: RMSE L333=8.043997, L365=8.052217,
L397=8.022899; mean fixed-NRMSE AULC333–397: Free-LLM32=0.905802,
Random=0.906640, raw-gradient LCMD=0.883564. No V2 metrics are fabricated.

## Responses transport and readiness

See [transport contract/configuration](RESPONSES_TRANSPORT.md). Registered provider:
third-party token4research; model gpt-5.5; effort high; environment variable
**TOKEN4RESEARCH_API_KEY**. Reads the selected provider from ~/.codex/config.toml;
uses only its env_key. The global config was inspected but not changed. It currently
needs the requested provider setup. The key was not copied from chat into tools,
files, hashes, logs, commits or configuration.

Every request explicitly includes model, reasoning.effort, input, tools=[],
store=false and max_output_tokens. Strict HTTP/completed/output/unique-JSON validation,
secret-free receipts, served-model/usage reporting, redirect denial, no retries after
ambiguous dispatch, no native tools, no session/history or implicit fallback.
Preflight has **not sent HTTP**: it reports missing process environment variable and
READY_FOR_API_KEY. Backend compatibility, real latency/cost and actual provider
context capacity remain unverified. Small preflight will not prove context capacity
or backend retention policy; store=false is a request contract.

## Full-pool V2 and verification

```text
all legal U → salted ordering → token-sized Stage1 chunks (0–16 nominees/chunk)
→ exact full-coverage audit → full-pool summary + nominees/rationales + local memory
→ Stage2 global arbitration ↔ any screened chunk ID directory / non-nominee expansion
→ 32 unique legal choices → immutable selection/predictions/receipts → Git/protocol seal
→ reveal32 → frozen premeasurement feedback → scratch retrain → validation + label-AULC
```

No quotas or numerical pre-screen. All4114 legal U candidates entered simulated
screening contexts across17 chunks (16×250 +114). Missing/duplicate counts0;
coverage1.0. The simulation produced272 nominees and exercised a non-nominee rescue
(31/32 chosen from nominees). This is an engineering fixture, not an LLM judgment.
Max estimated input214562 plus max output16000 fits the registered262144 local
ceiling. Estimation uses o200k_base plus20% margin and4096 framing reserve. Overflow
fails closed; no silent truncation. Future rounds with larger memory/longer responses
can still hit this bound and stop without freezing a selection.

The first real-pool engineering draft exposed Stage2 overflow under131072; before
any scientific response, lossless observed-record compression and a measured262144
ceiling /16-nominee cap were registered. No experiment revision directory or mixed
trajectory was created. Initial L333 prediction/coverage reuse is verified against
the identical baseline checkpoint and original hashes; no V1 acquisition or memory
is reused. Original scalar targets in the initial packet are authorized L333 only.

**155 tests passed; 0 failures, 0 errors, 0 skips** after the source-integrity re-review. [Log](verification/tests.log),
[JUnit](verification/tests.xml), and V2 test_gate bind the evidence to current code.
The new package/scripts/tests pass Ruff E4/E7/E9/F/I (script bootstrap E402 excluded)
and Python compilation. Tests cover response failures/tool calls/JSON ambiguity/key
handling, stable full coverage/missing/duplicate rejection, budget overflow, all-U
rescue with zero nominees, immutable receipt replay, ambiguous calls, local memory,
premeasurement error provenance, label/Git barriers, completed-fit reuse, no round2,
fixed metrics/AULC and byte-identical historical artifacts. Dry-run made0 real LLM
calls, revealed0 acquisition labels and read0 test labels.

## Branch/tag reconciliation

All old important tips are ancestors of this cleanup branch. The audit commit is
13481f1; implementation is on `refactor/odh-clean-responses-fullpool-v2`.
Pushed and remotely verified annotated tags:

| Tag suffix after archive/pre-cleanup- | Protected commit |
|---|---|
| main-2026-09-29 | 3c15d0a715942dacd510611fc6b20861614cf73d |
| representation-2026-09-29 | 3a4ee4589a1bbb159930d8f0e908afd90f98b7d8 |
| ivr-hybrid-2026-09-29 | 9155889eecd259b97072711b90d61d0206a0ce11 |
| free-llm32-2026-09-29 | 8c8d3d0b7b46e673c9a64df40c252841c63ca634 |

Final remote branch deletion receipt is recorded in `branch_cleanup.json` after
all required gates and the implementation push. No merge to main, no force push,
and no changes to upstream are authorized by or performed in this task.

Branch cleanup completed after implementation commit582279f was pushed. The three
local/origin experimental branches were removed after verifying their exact tips,
pushed annotated tags, ancestry and154 passing tests. Origin now contains only
`main` and `refactor/odh-clean-responses-fullpool-v2`. See the
[branch cleanup receipt](branch_cleanup.json). All four archive tags remain pushed.

Follow-up [pre-run review](PRE_RUN_REVIEW.md) fixed CSV drift detection before label
access and documented a reproducible Stage2 context-capacity limitation. The original
154-test branch-deletion receipt remains historical evidence; latest gate has155 tests.
