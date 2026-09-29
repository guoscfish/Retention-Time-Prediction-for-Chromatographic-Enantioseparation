# FullPool-LLM32 v2

Current registration: **token4research / gpt-6-astra / high**, seed1525,
six acquisitions of32: **L333 → L365 → L397 → L429 → L461 → L493 → L525**.
Four scientific screening responses are saved; no acquisition has completed. [STATUS](STATUS.md), [PROTOCOL](PROTOCOL.md),
and [machine contract](protocol.json) define readiness and the frozen experiment.

```text
same Row split / L333 checkpoint / QGeoGNN / fixed denominator
  → label-free cards for every legal U candidate
  → salted ordering → token-budgeted chunks (target300)
  → offline admission check reserves room for arbitration before paid screening
  → Stage1 identical prompt; 0–16 nominees/chunk; every candidate screened
  → audit exact U coverage=1.0, no missing/duplicate/illegal IDs
  → Stage2 all nominee cards/rationales + complete pool counts + local feedback
  ↔ optional chunk directory / non-nominee card expansion
  → exactly32 legal choices → freeze → local Git/protocol seal
  → reveal32 → frozen prediction-error feedback → scratch retrain
  → repeat six rounds → L525 report and stop
```

## Start / resume

```sh
.conda-hplc-al/bin/python scripts/run_hplc_fullpool_v2.py run
```

Run from the repository directory. The launcher automatically reads the user-designated
`~/.config/qgeognn-scientist/token4research.api-key` into this process's
`TOKEN4RESEARCH_API_KEY`. Existing environment credentials take precedence; an explicit
`--key-file /absolute/path` overrides them. No key is stored in scientific artifacts.
The command runs preflight first and then automatically seals, acquires and trains.
Rerun the same command to resume. Full details and preflight-only instructions:
[RESPONSES_TRANSPORT](../../../docs/repository/RESPONSES_TRANSPORT.md).

Offline verification: `prepare`, `dry-run`, `context-stress`.
Manual stages remain available: `select --round 0` through5, commit the selection
artifacts, then `advance --round 0` through5; finally `report`.
Never modify V1. The user-authorized execution-only amendment enables bounded
identical-request retries and terminal progress; see [execution amendment](execution_amendment.json).
Scientific transient requests retry in five-attempt bursts across restarts, then continue
with bounded exponential backoff until a response arrives or the operator interrupts the
process. Ambiguous attempts are recorded and retried identically. Successful responses
are reused; authentication/schema/semantic failures are not resampled.
Scientific protocol changes still require V3, not nested revisions.

Local L333 arrays/checkpoints are hash-verified historical computational reuse.
A fresh clone must restore the original ignored binary artifacts; Git stores their
manifests, not their bytes. This working machine already has those artifacts.
