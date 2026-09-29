# Pre-run review — 2026-09-29

**READY_FOR_API_KEY**, not a claim that only a key remains before scientific execution.
No API request, new acquisition, real acquired label or retraining was performed.

Current user-level config still selects gpt-6-astra without an explicit model_provider;
it does not yet select the registered token4research/gpt-5.5/high transport. The
TOKEN4RESEARCH_API_KEY variable is absent from the inspected process environment.
The key must be a token4research platform key. Do not put its value in config.toml;
`env_key` contains only the environment-variable name. This runner does not load .env.

## Scientific loop verified

L333 / initial fixed checkpoint → screen all 4114 legal candidates → global selection32
→ committed selection/protocol seal → reveal32 from the existing dataset
→ frozen premeasurement feedback → scratch train the predictor on L365
→ newly predicted pool + measured feedback + local hypotheses → screen/select32
→ seal/reveal → scratch train L397 → stop.

This is offline active learning on existing measured data, not instrument control.
The LLM's weights are not trained. Its next request gets explicit trajectory-local
feedback. The predictor is trained from scratch each round, not warm-started.
There are also247 fixed validation rows for checkpoint selection; those labels and
scores are never fed to the LLM. The247 test labels remain inaccessible in V2.

## Findings

1. **Fixed: resumed source integrity check occurred too late.** Previously, prepare's
   existing-protocol path verified source-code/reused files but not the original CSV.
   advance could reveal/write feedback before load_graphs eventually detected CSV
   drift. prepare now rechecks the original CSV SHA256 on every entry, before any
   label-store construction or reveal. A regression test checks both prepare and
   advance reject a changed temporary source without invoking state/label access.
   Only pre-science V2 engineering artifacts were refreshed; no real response exists,
   so no scientific protocol history was overwritten. V1 evidence remains unchanged.
2. **Open: realistic Stage2 context capacity.** The normal real-pool mock has214562
   estimated input tokens. A synthetic valid-schema Stage1 response with longer
   reasons/evidence text raises that to261244; with16000 output reserve this exceeds
   the262144 local ceiling. The host correctly stops, but may have already paid for
   Stage1 calls in a real run. Synthetic next-round feedback with short reasons fits
   at240740 input tokens, leaving little space for richer hypotheses and expansions.
   This is not an API/schema compatibility finding. Verify provider capacity and
   resolve the representation/output-budget issue before treating the protocol as
   ready for unattended scientific execution. Do not silently truncate or change the
   protocol after scientific responses. See [stress evidence](verification/context_stress_review.json).
3. **Unverified: third-party endpoint/model/limits.** No real preflight yet. The
   configured endpoint is base_url + /responses; gpt-5.5/high/tools=[]/store=false
   acceptance and exact provider context limits are not established by offline tests.
   A small content-free preflight tests compatibility, not full context capacity.

Latest verification:155 tests passed,0 failed/errors/skipped; Ruff and compileall
passed; real-pool engineering dry-run still covers4114/4114 IDs across17 chunks.
These results do not erase the open context-capacity finding.
