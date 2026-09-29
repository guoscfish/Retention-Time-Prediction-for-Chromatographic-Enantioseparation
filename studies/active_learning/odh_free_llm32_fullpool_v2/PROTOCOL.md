# Hierarchical full-pool LLM screening protocol v2

Scope: seed1525, Free-LLM32, L333→L365→L397→L429→L461→L493→L525, six acquisitions, batch32. No second
seed or test evaluation. This protocol is development research, not independent
external validation. No V2 scientific response, selection, label reveal or training
has occurred during implementation/dry-run.

The original Row split, L333 IDs/checkpoint, original QGeoGNN central MSE predictor,
RTv target, initialization and training configuration are verified against LCMD
confirmation v2 and V1's shared L333. Fixed NRMSE denominator is
8.879240547556758 (L333 population SD). Scratch Adam training uses batch256,
max500epochs, patience100 and the original validation checkpoint criterion.
The original predictor/trainer/graph/gradient source bytes are unchanged.

Candidate cards have a strict field allowlist: opaque ID, canonical isomeric SMILES,
CIP centers, scaffold, IPA, flow, center/q10/q90/width, coverage percentile,
MW/LogP/TPSA/HBD/HBA, identity/scaffold IDs and functional groups. No U truth,
validation/test label, post-retrain error or acquisition-label store reaches planning.
Only authorized L333 observations and this trajectory's subsequent measured records
enter scientific memory. Round0 has no fabricated premeasurement errors.

Every U candidate is screened once, without LCMD/IVR/coverage/shortlist gating.
Ordering is a round-salted stable hash of opaque candidate IDs, invariant to input
list order. Columnar records, shared values and bijective opaque group aliases remove
repetition. All card fields and chemical structures are retained. Floats displayed to
the LLM are rounded to six decimal places (absolute display error ≤0.0000005); all
host packets, selection seals, frozen premeasurement predictions, feedback, fitting
and metrics retain original precision. Initial observations still have no invented
errors. Target chunk size300; the frozen serialization estimate uses o200k_base,
20% margin plus4096 framing tokens, max output32000, local context ceiling262144.
Chunks shrink as needed; no minimum size forces overflow. These are explicit local
budget assumptions, not a claim about token4research's actual context limit. Provider
context rejection stops the run; no automatic truncation/model switch is allowed.

Stage1 has one identical frozen prompt for all chunks and may nominate0–16 items.
Prospective output limits bound reasons/evidence and whole screening replies (2048
estimated serialized input tokens each). These limits are explicit in frozen prompts;
overlong output is rejected, never silently shortened. Before any screening request,
an admission estimate reserves space for the largest eligible nominee cards, all
chunk replies, current memory, full counts, output and relay. No candidate is filtered.

Each nomination records reason, role, linked old hypothesis (or null), priority and
what evidence would be learned. Within-chunk priority is not global superiority.
The audit records all screened IDs, chunk sizes/count, duplicates, missing/illegal IDs
and nominees. Full coverage1.0 and exact equality to legal U are required before freeze.

Stage2 gets all nominees with full detailed cards and screening rationales, every
chunk summary, deterministic full-pool summary, observed records and trajectory-local
memory. It has at most8 responses, including the final selection. JSON directory
requests expose every ID in any screened chunk; `expand_candidate_ids` exposes full
cards for any legal U ID (up to32 per request; one directory chunk per request). Thus
non-nominees remain selectable. A page that cannot fit returns a bounded error asking
for fewer cards or a final selection; it never counts unseen cards as visible. Expansion messages
remain in the same explicit round transcript, with context checks before every call.
No per-chunk, scaffold, uncertainty, coverage, stereo or condition quota applies.

Full-pool summary includes prediction/width/coverage distributions, IPA×flow,
all scaffold/identity counts (grouped losslessly by count for transmission), stereo contrasts, prediction extremes and candidate coverage
of high-error *observed* scaffolds. Stereo contrasts mean same nonisomeric SMILES,
different isomeric SMILES; they are not necessarily strict enantiomer pairs. Condition
contrasts count differing IPA/flow within exact identity. These definitions are fixed.

Memory checks study+seed+method+round continuity and retains previous hypotheses,
support/contradiction/unresolved status, the two most recent full batches and high-error
acquisitions. ALL observed measurements/frozen feedback and ALL current hypothesis
states remain available. Earlier complete batch narratives remain in sealed host
artifacts. Shared observation, choice, original-hypothesis and identical-update values
use validated references; older differing updates in recent batches remain explicit.
Feedback preserves measured RTv, frozen premeasurement prediction, signed/absolute
error, chemical metadata, condition, selection reason, role and hypothesis link.
The signed error is prediction−measurement. Evidence references must be observed IDs.

Label barrier: host preparation → LLM screening/arbitration → immutable selection and
transitive artifact hashes → committed Git/protocol seal → RestrictedLabelStore commit
→ reveal32 → durable feedback → scratch fit. The store is a logical I/O barrier,
not OS isolation. Validation truth is available only to training/evaluation, never to
the planner; test labels remain inaccessible in V2. Completed fits and receipts are
verified and reused. An ambiguous request is never retried. Separate pipeline and stage process locks prevent duplicate continuous runs and
concurrent host mutation. The `run` entry point performs a content-free preflight,
automatically commits each selection before reveal, resumes verified artifacts and
stops after six rounds. It never pushes automatically. A crash between feedback
write and feedback seal is recoverable by deterministic validation/reconstruction. Protocol/source/environment drift stops execution.

Scientific reporting includes RMSE, MAE, R², fixed-denominator NRMSE, raw and mean
label-AULC with explicit interval, full-pool coverage, nominee rate, selected-from-
nominees rate, selected coverage/width distributions, scaffold diversity, stereo and
condition contrasts, calls, available input/output tokens, and screened counts each
round. Missing provider usage is reported as unknown, not zero.

Model registration: third-party token4research, gpt-6-astra, reasoning high.
Direct third-party Responses transport details: [RESPONSES_TRANSPORT.md](../../../docs/repository/RESPONSES_TRANSPORT.md).
Small preflight verifies request acceptance/schema/JSON/model if reported; it cannot
independently verify backend data-retention policy or maximum context capacity.
