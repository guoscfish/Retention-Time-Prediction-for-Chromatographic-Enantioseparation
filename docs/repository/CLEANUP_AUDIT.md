# Repository cleanup audit — 2026-09-29

Completed BEFORE source edits or deletion. `git fetch --all --tags --prune` succeeded for origin and upstream. Starting checkout clean at 8c8d3d0b7b46e673c9a64df40c252841c63ca634. No AGENTS.md found in repository. Read-only AST/byte inspection; no U/test targets parsed.

## Branch ancestry

All named experimental tips form one linear chain; latest Free-LLM32 contains every older commit. `unique` below means commits absent from main, not absent from later branches. Full SHAs, merge-bases, commit subjects and both local/remote contains checks are in [inventory](cleanup_inventory.json).

| Branch | Tip | Unique vs main | Unique artifacts and containment | Safe deletion |
|---|---|---:|---|---|
| codex/odh-free-llm32-scientist-v1 | `8c8d3d0b7b46e673c9a64df40c252841c63ca634` | 12 | Free-LLM32 selections, receipts, feedback, metrics; all contained in 8c8d3d0 | Conditional: pushed annotated tag + cleanup descendant + tests |
| codex/odh-ivr-hybrid-screen-v1 | `9155889eecd259b97072711b90d61d0206a0ce11` | 5 | IVR, LCMD and hybrid artifacts; all contained in 8c8d3d0 | Conditional: pushed annotated tag + cleanup descendant + tests |
| codex/odh-representation-coreset-v1 | `3a4ee4589a1bbb159930d8f0e908afd90f98b7d8` | 2 | representation and extension artifacts; all contained in 8c8d3d0 | Conditional: pushed annotated tag + cleanup descendant + tests |
| main | `3c15d0a715942dacd510611fc6b20861614cf73d` | 0 | original model / reproduction / diagnosis / gradient; all contained in 8c8d3d0 | KEEP (main/upstream) |
| origin | `3c15d0a715942dacd510611fc6b20861614cf73d` | 0 | original model / reproduction / diagnosis / gradient; all contained in 8c8d3d0 | KEEP (main/upstream) |
| origin/codex/odh-free-llm32-scientist-v1 | `8c8d3d0b7b46e673c9a64df40c252841c63ca634` | 12 | Free-LLM32 selections, receipts, feedback, metrics; all contained in 8c8d3d0 | Conditional: pushed annotated tag + cleanup descendant + tests |
| origin/codex/odh-ivr-hybrid-screen-v1 | `9155889eecd259b97072711b90d61d0206a0ce11` | 5 | IVR, LCMD and hybrid artifacts; all contained in 8c8d3d0 | Conditional: pushed annotated tag + cleanup descendant + tests |
| origin/codex/odh-representation-coreset-v1 | `3a4ee4589a1bbb159930d8f0e908afd90f98b7d8` | 2 | representation and extension artifacts; all contained in 8c8d3d0 | Conditional: pushed annotated tag + cleanup descendant + tests |
| origin/main | `3c15d0a715942dacd510611fc6b20861614cf73d` | 0 | original model / reproduction / diagnosis / gradient; all contained in 8c8d3d0 | KEEP (main/upstream) |
| upstream | `d284767bf7f1f04b24f758211a1d8d70e82f52ed` | 0 | original model / reproduction / diagnosis / gradient; all contained in 8c8d3d0 | KEEP (main/upstream) |
| upstream/main | `d284767bf7f1f04b24f758211a1d8d70e82f52ed` | 0 | original model / reproduction / diagnosis / gradient; all contained in 8c8d3d0 | KEEP (main/upstream) |

## Study census

COMPLETE describes registered completed scope, never a universal scientific conclusion. Development evidence remains development evidence. No new test-cohort independence is claimed.

| Study | Classification | Tracked files | Selection files / fits |
|---|---|---:|---:|
| odh_free_llm32_scientist_v1 | COMPLETE / DEVELOPMENT | 111 | 4 / 2 |
| odh_gradient_al_smoke | COMPLETE / DEVELOPMENT | 139 | 9 / 11 |
| odh_gradient_al_transfer_v2 | COMPLETE / DEVELOPMENT | 283 | 16 / 19 |
| odh_ivr_hybrid_screen_v1 | COMPLETE / DEVELOPMENT | 248 | 10 / 20 |
| odh_lcmd_confirmation_v1 | PARTIAL / SUPERSEDED / ARCHIVE_ONLY | 10 | 1 / 1 |
| odh_lcmd_confirmation_v2 | COMPLETE / CANONICAL_BASELINE | 83 | 12 / 14 |
| odh_llm_hybrid_v1 | FAILED / SUPERSEDED / ARCHIVE_ONLY | 3 | 0 / 0 |
| odh_llm_hybrid_v2 | FAILED / SUPERSEDED / ARCHIVE_ONLY | 16 | 0 / 0 |
| odh_llm_hybrid_v3 | FAILED / SUPERSEDED / ARCHIVE_ONLY | 4 | 0 / 0 |
| odh_llm_hybrid_v4 | PARTIAL / SUPERSEDED / ARCHIVE_ONLY | 16 | 2 / 0 |
| odh_llm_hybrid_v6 | PARTIAL / SUPERSEDED / ARCHIVE_ONLY | 16 | 2 / 0 |
| odh_llm_hybrid_v7 | PARTIAL / SUPERSEDED / ARCHIVE_ONLY | 16 | 2 / 0 |
| odh_llm_hybrid_v8 | PARTIAL / SUPERSEDED / ARCHIVE_ONLY | 64 | 6 / 3 |
| odh_representation_coreset_extension_v1 | COMPLETE / DEVELOPMENT | 143 | 8 / 8 |
| odh_representation_coreset_v1 | COMPLETE / DEVELOPMENT | 190 | 12 / 13 |
| odh_training_protocol_diagnosis | COMPLETE / CANONICAL_BASELINE | 87 | 0 / 9 |

## Code and entry points

Importer inventory is exhaustive for static imports; absence of an importer alone does not prove a module dead. Standalone script and test imports are included below.

| Module | Current importers | Disposition |
|---|---|---|
| __init__.py | none (entry/support) | retain; distinct numerical/baseline contract |
| acquisition.py | src/hplc_al/free_llm_runner.py, src/hplc_al/lcmd_confirmation.py, src/hplc_al/llm_hybrid.py, src/hplc_al/representation_extension.py, src/hplc_al/runner.py, src/hplc_al/transfer_v2_runner.py, src/hplc_diagnosis/geometry.py, tests/hplc_al/test_acquisition_gradient.py | retain; distinct numerical/baseline contract |
| common.py | scripts/report_hplc_gradient_transfer_v2.py, scripts/report_hplc_ivr_hybrid_screen.py, scripts/report_hplc_representation_coreset.py, scripts/report_hplc_representation_extension.py, scripts/report_hplc_training_diagnosis.py, scripts/verify_hplc_transfer_v2_runtime.py, scripts/write_hplc_transfer_v2_report.py, src/hplc_al/coreset.py, src/hplc_al/data.py, src/hplc_al/free_llm_report.py, src/hplc_al/free_llm_runner.py, src/hplc_al/free_llm_scientist.py, src/hplc_al/gradient.py, src/hplc_al/ivr_hybrid_screen.py, src/hplc_al/latent.py, src/hplc_al/lcmd_confirmation.py, src/hplc_al/llm_catalog.py, src/hplc_al/llm_cli_transport.py, src/hplc_al/llm_hybrid.py, src/hplc_al/protocol.py, src/hplc_al/reporting.py, src/hplc_al/representation_extension.py, src/hplc_al/representation_runner.py, src/hplc_al/runner.py, src/hplc_al/training.py, src/hplc_al/transfer_v2_audit.py, src/hplc_al/transfer_v2_duration.py, src/hplc_al/transfer_v2_runner.py, src/hplc_diagnosis/geometry.py, src/hplc_diagnosis/runner.py, src/hplc_diagnosis/training.py, tests/hplc_al/test_acquisition_gradient.py, tests/hplc_al/test_completed_study.py, tests/hplc_al/test_free_llm_scientist.py, tests/hplc_al/test_ivr_hybrid_screen.py, tests/hplc_al/test_protocol.py, tests/hplc_al/test_representation_coreset.py, tests/hplc_al/test_representation_extension.py, tests/hplc_al/test_transfer_v2_protocol.py | retain; distinct numerical/baseline contract |
| coreset.py | src/hplc_al/representation_diagnostics.py, src/hplc_al/representation_extension.py, src/hplc_al/representation_runner.py, tests/hplc_al/test_representation_coreset.py | retain; distinct numerical/baseline contract |
| data.py | scripts/reproduce_odh_baseline.py, scripts/verify_hplc_transfer_v2_runtime.py, src/hplc_al/data.py, src/hplc_al/free_llm_runner.py, src/hplc_al/gradient.py, src/hplc_al/ivr_hybrid_screen.py, src/hplc_al/latent.py, src/hplc_al/lcmd_confirmation.py, src/hplc_al/llm_hybrid.py, src/hplc_al/representation_extension.py, src/hplc_al/representation_runner.py, src/hplc_al/runner.py, src/hplc_al/training.py, src/hplc_al/transfer_v2_duration.py, src/hplc_al/transfer_v2_runner.py, src/hplc_diagnosis/geometry.py, src/hplc_diagnosis/runner.py, src/hplc_diagnosis/training.py, tests/hplc_al/conftest.py, tests/hplc_al/test_acquisition_gradient.py, tests/hplc_al/test_protocol.py | retain; distinct numerical/baseline contract |
| deterministic_shuffle.py | scripts/verify_hplc_transfer_v2_runtime.py, src/hplc_al/preflight_v2.py, src/hplc_al/training.py, src/hplc_al/transfer_v2.py, tests/hplc_al/test_transfer_v2_protocol.py | retain; distinct numerical/baseline contract |
| ensemble.py | src/hplc_al/hybrid.py, src/hplc_al/ivr_hybrid_screen.py, tests/hplc_al/test_ivr_hybrid_screen.py | retain; distinct numerical/baseline contract |
| free_llm_report.py | scripts/run_hplc_free_llm32.py | legacy-only; archive via Git, replace with llm package |
| free_llm_runner.py | scripts/run_hplc_free_llm32.py, src/hplc_al/free_llm_report.py, tests/hplc_al/test_free_llm_scientist.py | legacy-only; archive via Git, replace with llm package |
| free_llm_scientist.py | src/hplc_al/free_llm_report.py, src/hplc_al/free_llm_runner.py, tests/hplc_al/test_free_llm_scientist.py | legacy-only; archive via Git, replace with llm package |
| gradient.py | src/hplc_al/free_llm_runner.py, src/hplc_al/lcmd_confirmation.py, src/hplc_al/llm_hybrid.py, src/hplc_al/representation_runner.py, src/hplc_al/runner.py, src/hplc_al/transfer_v2_runner.py, src/hplc_diagnosis/geometry.py, tests/hplc_al/test_acquisition_gradient.py, tests/hplc_al/test_training_diagnosis.py | retain; distinct numerical/baseline contract |
| hybrid.py | src/hplc_al/ivr_hybrid_screen.py, tests/hplc_al/test_ivr_hybrid_screen.py | retain; distinct numerical/baseline contract |
| ivr_hybrid_screen.py | scripts/report_hplc_ivr_hybrid_screen.py, scripts/run_hplc_ivr_hybrid_screen.py, tests/hplc_al/test_ivr_hybrid_screen.py | retain; distinct numerical/baseline contract |
| kernel_ivr.py | src/hplc_al/ivr_hybrid_screen.py, tests/hplc_al/test_ivr_hybrid_screen.py | retain; distinct numerical/baseline contract |
| latent.py | src/hplc_al/ivr_hybrid_screen.py, src/hplc_al/representation_runner.py, tests/hplc_al/test_representation_coreset.py | retain; distinct numerical/baseline contract |
| lcmd_confirmation.py | scripts/report_hplc_lcmd_confirmation.py, scripts/run_hplc_lcmd_confirmation.py, src/hplc_al/free_llm_runner.py | retain; distinct numerical/baseline contract |
| llm_catalog.py | src/hplc_al/free_llm_report.py, src/hplc_al/free_llm_runner.py, src/hplc_al/free_llm_scientist.py, src/hplc_al/llm_hybrid.py, tests/hplc_al/test_free_llm_scientist.py | extract reusable chemistry only; remove paging/16+16 budgets |
| llm_cli_transport.py | src/hplc_al/free_llm_runner.py, src/hplc_al/free_llm_scientist.py, tests/hplc_al/test_free_llm_scientist.py | legacy-only; archive via Git, replace with llm package |
| llm_hybrid.py | none (entry/support) | legacy-only; archive via Git, replace with llm package |
| llm_transport.py | src/hplc_al/llm_hybrid.py | legacy-only; archive via Git, replace with llm package |
| preflight_v2.py | scripts/run_hplc_gradient_transfer_v2_preflight.py, tests/hplc_al/test_transfer_v2_protocol.py | retain; distinct numerical/baseline contract |
| protocol.py | scripts/report_hplc_gradient_transfer_v2.py, scripts/report_hplc_ivr_hybrid_screen.py, scripts/report_hplc_representation_coreset.py, scripts/report_hplc_representation_extension.py, scripts/verify_hplc_transfer_v2_runtime.py, src/hplc_al/free_llm_report.py, src/hplc_al/free_llm_runner.py, src/hplc_al/ivr_hybrid_screen.py, src/hplc_al/lcmd_confirmation.py, src/hplc_al/llm_hybrid.py, src/hplc_al/reporting.py, src/hplc_al/representation_extension.py, src/hplc_al/representation_runner.py, src/hplc_al/runner.py, src/hplc_al/transfer_v2_duration.py, src/hplc_al/transfer_v2_runner.py, src/hplc_diagnosis/runner.py, tests/hplc_al/conftest.py, tests/hplc_al/test_acquisition_gradient.py, tests/hplc_al/test_free_llm_scientist.py, tests/hplc_al/test_ivr_hybrid_screen.py, tests/hplc_al/test_protocol.py, tests/hplc_al/test_recovery.py, tests/hplc_al/test_representation_coreset.py, tests/hplc_al/test_representation_extension.py, tests/hplc_al/test_training_diagnosis.py | retain; distinct numerical/baseline contract |
| reporting.py | scripts/run_hplc_al_smoke.py | retain; distinct numerical/baseline contract |
| representation_diagnostics.py | src/hplc_al/representation_extension.py, src/hplc_al/representation_runner.py | retain; distinct numerical/baseline contract |
| representation_extension.py | scripts/report_hplc_representation_extension.py, scripts/run_hplc_representation_extension.py, src/hplc_al/ivr_hybrid_screen.py, tests/hplc_al/test_representation_extension.py | retain; distinct numerical/baseline contract |
| representation_runner.py | scripts/report_hplc_representation_coreset.py, scripts/run_hplc_representation_coreset.py, src/hplc_al/ivr_hybrid_screen.py, src/hplc_al/representation_extension.py, tests/hplc_al/test_representation_coreset.py | retain; distinct numerical/baseline contract |
| runner.py | scripts/report_hplc_training_diagnosis.py, scripts/run_hplc_al_smoke.py, scripts/run_hplc_training_diagnosis.py, scripts/verify_hplc_transfer_v2_runtime.py, src/hplc_al/free_llm_runner.py, src/hplc_al/ivr_hybrid_screen.py, src/hplc_al/lcmd_confirmation.py, src/hplc_al/reporting.py, src/hplc_al/representation_extension.py, src/hplc_al/representation_runner.py, src/hplc_al/transfer_v2_duration.py, src/hplc_al/transfer_v2_runner.py, src/hplc_diagnosis/runner.py, tests/hplc_al/test_recovery.py, tests/hplc_al/test_training_diagnosis.py | retain; distinct numerical/baseline contract |
| training.py | scripts/report_hplc_training_diagnosis.py, src/hplc_al/free_llm_runner.py, src/hplc_al/ivr_hybrid_screen.py, src/hplc_al/lcmd_confirmation.py, src/hplc_al/llm_hybrid.py, src/hplc_al/representation_extension.py, src/hplc_al/representation_runner.py, src/hplc_al/runner.py, src/hplc_al/transfer_v2_duration.py, src/hplc_al/transfer_v2_runner.py, src/hplc_diagnosis/runner.py, src/hplc_diagnosis/training.py, tests/hplc_al/test_acquisition_gradient.py, tests/hplc_al/test_training_diagnosis.py | retain; distinct numerical/baseline contract |
| transfer_v2.py | scripts/report_hplc_gradient_transfer_v2.py, scripts/verify_hplc_transfer_v2_runtime.py, scripts/write_hplc_transfer_v2_report.py, src/hplc_al/lcmd_confirmation.py, src/hplc_al/llm_hybrid.py, src/hplc_al/preflight_v2.py, src/hplc_al/representation_runner.py, src/hplc_al/transfer_v2_audit.py, src/hplc_al/transfer_v2_duration.py, src/hplc_al/transfer_v2_runner.py, tests/hplc_al/test_transfer_v2_protocol.py | retain; distinct numerical/baseline contract |
| transfer_v2_audit.py | scripts/audit_hplc_gradient_transfer_v2.py, scripts/verify_hplc_transfer_v2_runtime.py | retain; distinct numerical/baseline contract |
| transfer_v2_duration.py | scripts/run_hplc_gradient_transfer_v2_duration.py | retain; distinct numerical/baseline contract |
| transfer_v2_runner.py | scripts/report_hplc_gradient_transfer_v2.py, scripts/run_hplc_gradient_transfer_v2.py, src/hplc_al/ivr_hybrid_screen.py, src/hplc_al/representation_runner.py, tests/hplc_al/test_representation_coreset.py, tests/hplc_al/test_transfer_v2_protocol.py | retain; distinct numerical/baseline contract |

Duplicated transports: Responses vs ephemeral Codex CLI. Responses hard-codes gpt-6-sol and OPENAI_API_KEY, reads auth.json, defaults provider; remove all these paths. Two JSON parsers accept overly permissive answers. Three LLM flows duplicate orchestration/reporting. Numerical runners represent different frozen protocols; merging those during cleanup would risk provenance and is not justified.

| Script | Classification |
|---|---|
| audit_hplc_gradient_transfer_v2.py | legacy: documented numerical study/reporter, keep for provenance and read-only inspection |
| audit_reproduction_data.py | active: reproduction/support |
| fetch_official_odh_cache.py | active: reproduction/support |
| report_hplc_gradient_transfer_v2.py | legacy: documented numerical study/reporter, keep for provenance and read-only inspection |
| report_hplc_ivr_hybrid_screen.py | legacy: documented numerical study/reporter, keep for provenance and read-only inspection |
| report_hplc_lcmd_confirmation.py | legacy: documented numerical study/reporter, keep for provenance and read-only inspection |
| report_hplc_representation_coreset.py | legacy: documented numerical study/reporter, keep for provenance and read-only inspection |
| report_hplc_representation_extension.py | legacy: documented numerical study/reporter, keep for provenance and read-only inspection |
| report_hplc_training_diagnosis.py | legacy: documented numerical study/reporter, keep for provenance and read-only inspection |
| reproduce_odh_baseline.py | active: reproduction/support |
| run_hplc_al_smoke.py | legacy: documented numerical study/reporter, keep for provenance and read-only inspection |
| run_hplc_free_llm32.py | legacy, remove live V1 mutation entry |
| run_hplc_gradient_transfer_v2.py | legacy: documented numerical study/reporter, keep for provenance and read-only inspection |
| run_hplc_gradient_transfer_v2_duration.py | legacy: documented numerical study/reporter, keep for provenance and read-only inspection |
| run_hplc_gradient_transfer_v2_preflight.py | legacy: documented numerical study/reporter, keep for provenance and read-only inspection |
| run_hplc_ivr_hybrid_screen.py | legacy: documented numerical study/reporter, keep for provenance and read-only inspection |
| run_hplc_lcmd_confirmation.py | legacy: documented numerical study/reporter, keep for provenance and read-only inspection |
| run_hplc_representation_coreset.py | legacy: documented numerical study/reporter, keep for provenance and read-only inspection |
| run_hplc_representation_extension.py | legacy: documented numerical study/reporter, keep for provenance and read-only inspection |
| run_hplc_training_diagnosis.py | legacy: documented numerical study/reporter, keep for provenance and read-only inspection |
| verify_hplc_transfer_v2_runtime.py | legacy: documented numerical study/reporter, keep for provenance and read-only inspection |
| write_hplc_transfer_v2_report.py | legacy: documented numerical study/reporter, keep for provenance and read-only inspection |

No proven dead standalone script beyond retired LLM entry; do not infer dead code solely from filenames.

## Integrity findings and planned treatment

- Free-LLM32 root EXECUTION_STATUS says blocked L333 while INTERIM_REPORT and successful revision final_audit prove L397. Replace current status; preserve original success bytes with relocation manifest. Root failed attempt has zero receipts, zero selection, zero acquisition reveal; retain summary and tag locator.
- Hybrid v1 has only packet/protocol/audit; v2/v3 have responses but no frozen selection. They are failed partial attempts, not completed scientific runs. v4/v6/v7 have unique request/selection provenance but no fit.json. Archive unique bytes; remove duplicate copies only.
- Hybrid v8 complete.json is stale as a claim of full intended comparison: chemical has 2 fits, masked 1 fit; a round2 chemical packet/receipts has no selection. Keep unique evidence outside active index, mark PARTIAL.
- LCMD v1 is superseded by v2 but has historical artifacts; keep without relocating because frozen references may bind paths.
- representation DIAGNOSTICS.md says no test truth accessed: this is a pre-AL snapshot, not current study-wide status. Index clarifies historical scope; do not rewrite frozen text.
- Gradient transfer interrupted_attempt_20260923 and successful study are separate provenance; retain, no claim that interrupted complete substage equals complete study.
- Successful Free-LLM32 transport_revision_2 is obsolete layout, not obsolete results: move all success artifacts byte-for-byte into results and provenance. Preserve original path/hash mapping; old embedded paths resolve through mapping or archive tag.
- Byte duplicates (including copied packets, checkpoints and split files) are fully enumerated in inventory. Shared baseline/runtime copies may be hash-bound by original protocols; retain except explicitly archived hybrid duplicates.
- Empty tracked files: studies/active_learning/odh_free_llm32_scientist_v1/transport_revision_2/round0_select.log. No evidence that missing result files are valid zeros.
- Accidental interpreter/desktop/temp runtime files tracked: none. Scientific runtime records, test logs, request intents and fit manifests are intentional evidence. Large ignored checkpoints/NPZ exist locally: preserve during moves, do not claim Git tags contain ignored files.

## Destructive-action gate

Audit complete. Next: annotated snapshots for main/representation/IVR/Free-LLM32, push and verify remote tag targets; branch from 8c8d3d0 as refactor/odh-clean-responses-fullpool-v2. Only delete origin experimental refs after tests and ancestry/tag checks. Never delete upstream/main or force-push.


## Post-cleanup disposition

The audit above is the original pre-edit assessment. Implementation and verification
are in [CLEANUP_REPORT.md](CLEANUP_REPORT.md). All three experimental local/origin
branches were safely deleted after exact-tip tag verification, ancestry checks,
implementation push and154 passing tests; [receipt](branch_cleanup.json).
Main and upstream remain unchanged. Unique scientific bytes remain in cleaned
history and mapped archives; no ambiguous branch was deleted.
