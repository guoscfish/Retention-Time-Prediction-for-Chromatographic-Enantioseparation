# Phase 1 operation

Branch: `codex/odh-free-llm32-scientist-v1`. Python: `.conda-hplc-al/bin/python`.

Frozen study uses seed1525 and only Free-LLM32. Code rejects acquisition/advance round2. No automatic method switching. Reused LCMD confirmation files are read-only.

1. Run the tests in `test_gate.json` and retain XML/log evidence. `protocol.json` freezes source hashes, prompt/schema/config, matching baseline inputs and L333 denominator.
2. `scripts/run_hplc_free_llm32.py dry-run` prepares label-free round0 predictions/geometry and validates a synthetic 32-point selection; no acquired labels or LLM calls.
3. `... select --round 0`. Freeze selection and full receipt/query chain.
4. Git commit the new study artifacts including `selection_seal.json`. Large `.npz`/`.pt` remain local with hashes in committed manifests.
5. `... advance --round 0`: requires seal to exist in HEAD before reveal; records true feedback, scratch-fits and validation metrics.
6. `... select --round 1`, commit the new selection seal, then `... advance --round 1`.
7. `... report`. Stop at L397. Any further acquisition requires a separate explicit decision/version.

Never delete request intents or change frozen prompt/code to force a failed selection to resume. An intent without a completed receipt means an ambiguous remote request and fails closed. Preserve it, record the failure/revision, and decide whether a new study version is required. A frozen selection cannot be replaced. Receipt replay must reconstruct the identical message/config hash. Completed fits are reused after hash verification; incomplete scratch fits preserve old attempts and create a new attempt.

No historical report function is called. Evaluation loads only validation prediction arrays and authorized validation labels. Diagnostics compare both a same-state shadow LCMD32 batch (not shown to LLM) and the historical independent LCMD trajectory at the same round. They have different meanings after round0.

Hash inventories include files containing old test results only as opaque bytes for immutability checks. No old test labels or scores are parsed, used as selector context, or used for this report.
