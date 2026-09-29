# Pre-run review — 2026-09-29

**READY_FOR_SCIENTIFIC_RUN**. Real content-free preflight passed; no scientific API call, new acquired label or scratch fit was run.

This machine is configured for **third-party token4research / gpt-6-astra / high**,
using `https://token4research.cn/responses`. The launcher reads the user-designated local key file into process memory. Start with `.conda-hplc-al/bin/python scripts/run_hplc_fullpool_v2.py run`
without needing to paste the key again. See
[complete startup instructions](RESPONSES_TRANSPORT.md).

## Registered loop

L333 → screen all legal U → select32 → committed selection/protocol seal → reveal32
→ frozen premeasurement feedback → scratch train L365 → repeat to L397, L429, L461,
L493 and L525 → report and stop. Six acquisitions,192 newly acquired labels.
This is offline active learning from existing measurements. The LLM receives local
feedback; its weights are not trained. The predictor restarts from scratch each round.
The247 validation labels are trainer-only; the247 test labels remain inaccessible.

## Resolved findings

1. Source integrity is checked on every entry, including resumes, before label-store
   construction or reveal. Protocol/source/environment changes fail closed.
2. Long-context handling is now explicit and verified across six simulated rounds:
   target300 candidates/chunk, at most16 nominees/chunk, bounded concise prose,
   shared values and columnar records, exact grouped counts and checked references.
   Floats displayed to the LLM have six decimal places; host packets, frozen
   premeasurement predictions, feedback, training and metrics keep original precision.
   All observed measurements/errors and current hypotheses remain in memory; the
   complete recent batch narrative window is registered as two batches.
   Admission reserves arbitration capacity before paid Stage1 calls. No U candidate
   is filtered out. Oversized expansion pages return an explicit error and do not
   mark unseen cards visible. Actual requests are checked again before dispatch.
3. The registered model and local user configuration now match gpt-6-astra/high.
4. `run` automates preflight, local selection-seal commits, six fits and reporting.
   It resumes verified artifacts, refuses unrelated staged Git changes, handles an
   interruption between feedback write and seal, and blocks duplicate pipeline runs.
   Ambiguous API requests intentionally stop rather than risk duplicate dispatch.

## Capacity evidence

Normal real-input dry-run: 4114 legal candidates in 14 chunks,
coverage1.0; maximum input estimate 127926; zero scientific calls/reveals.

Six-round stress uses real initial metadata, synthetic acquired labels and unchanged
later predictions. It nominates large cards, approaches the screening-text cap, and
adds eight hypotheses each round with maximum field lengths and evidence-list sizes.
It tests serialization and planning, not model quality or actual provider behavior.

| Round | L before | Legal U | Hypotheses seen | Max input estimate | Admission estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 333 | 4114 | 0 | 149444 | 166950 |
| 2 | 365 | 4082 | 8 | 161353 | 179031 |
| 3 | 397 | 4050 | 16 | 173198 | 190943 |
| 4 | 429 | 4018 | 24 | 183571 | 201352 |
| 5 | 461 | 3986 | 32 | 193784 | 211519 |
| 6 | 493 | 3954 | 40 | 203784 | 221591 |

Each round has coverage1.0. Input estimates include a20% tokenizer margin and4096
framing tokens. Admission additionally reserves4096 relay tokens and32000 output
tokens within the262144 local ceiling. This is a tested fixture, not a guarantee for
every future model response; exceeding a registered bound stops without truncation.
[Machine-readable evidence](verification/context_stress_review.json).

## Remaining external verification

The actual content-free preflight now passes with the user-designated key file and
returns gpt-6-astra. The HTTP403 failure was reproduced as Cloudflare1010 for the
default Python client signature; the truthful application User-Agent reaches the API.
See [diagnosis](verification/transport_403_diagnosis.json). The small preflight still
does not establish maximum provider context capacity. Provider rejection stops the
run without fallback; local estimates remain estimates.

Final verification: **178 tests passed, 0 failures/errors/skips**; Ruff and compileall passed.
