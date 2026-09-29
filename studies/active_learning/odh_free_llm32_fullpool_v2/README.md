# FullPool-LLM32 v2

New protocol, separate from historical Free-LLM32 v1. **No scientific acquisition has
been run.** Current readiness: [STATUS](STATUS.md). [Protocol](PROTOCOL.md) and
[registered machine contract](protocol.json) define the experiment.

```text
same Row split / seed1525 / L333 / QGeoGNN / fixed denominator
  → label-free cards for every legal U candidate
  → round-salted hash ordering → token-budgeted chunks (target250)
  → Stage1 identical screening prompt for every chunk; 0–16 nominees each
  → assert screened IDs == legal U IDs, missing=[], duplicates=0, coverage=1.0
  → deterministic full-pool summary + all nominees and rationales + local memory
  → Stage2 global arbitration; optional chunk ID directory / non-nominee expansion
  → exactly32 legal choices, no quotas → freeze requests/receipts/predictions/selection
  → Git + protocol seal → reveal32 → feedback with frozen prediction errors
  → scratch retrain → validation metrics → next round; stop at L397
```

Implementation: [`src/hplc_al/llm`](../../../src/hplc_al/llm/) (relative repository
location: `src/hplc_al/llm/`; transport, catalog, memory, planner, full_pool, runner,
reporting and explicit simulated dry-run have separate responsibilities).

Use `.conda-hplc-al/bin/python` with `requirements-hplc-al.txt`. Current delivered
commands are `scripts/run_hplc_fullpool_v2.py prepare` and `dry-run`.
`scripts/preflight_llm_responses.py` sends only `Return exactly {"status":"ok"}`
when the configured environment key is present. It never starts a scientific run.

Future explicitly authorized execution uses `select --round 0`, commits the
selection artifacts, then `advance --round 0`; similarly round1, then `report`.
Do not run those commands merely because preflight passed. Never modify V1.
A request intent without receipt is ambiguous and is not automatically resent.
A protocol bug after any scientific response requires a new v3 study.

The dry-run reuses local, hash-verified L333 arrays/checkpoints that were already
ignored in the historical study. A fresh clone must restore those original local
artifacts before running host preparation; tags preserve their manifests, not binaries.
