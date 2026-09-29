# FullPool-LLM32 v2

Current registration: **token4research / gpt-6-astra / high**, seed1525,
six acquisitions of32: **L333 → L365 → L397 → L429 → L461 → L493 → L525**.
No scientific V2 acquisition has run. [STATUS](STATUS.md), [PROTOCOL](PROTOCOL.md),
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

Run from the repository directory. **Paste the key at the hidden Terminal prompt.**
It enters only this process's `TOKEN4RESEARCH_API_KEY`; do not paste it into config,
source files, Git, or chat. Existing environment credentials skip the prompt.
The command runs preflight first and then automatically seals, acquires and trains.
Rerun the same command to resume. Full details and preflight-only instructions:
[RESPONSES_TRANSPORT](../../../docs/repository/RESPONSES_TRANSPORT.md).

Offline verification: `prepare`, `dry-run`, `context-stress`.
Manual stages remain available: `select --round 0` through5, commit the selection
artifacts, then `advance --round 0` through5; finally `report`.
Never modify V1. An ambiguous scientific request is not automatically retried.
A protocol bug after any scientific response requires V3, not nested revisions.

Local L333 arrays/checkpoints are hash-verified historical computational reuse.
A fresh clone must restore the original ignored binary artifacts; Git stores their
manifests, not their bytes. This working machine already has those artifacts.
