# Execution status — blocked at L333

The requested L397 experiment is **not complete**. No Free-LLM32 performance or partial AULC can yet be reported.

Completed: remote fetch/pull and branch audit; AUDIT.md; isolated branch; Free-LLM32 implementation; 60 passing tests plus 18 post-freeze contract checks; protocol/prompt/schema/source hashes; successful real-input dry-run; 566 old artifacts unchanged.

Round0 was invoked after the freeze commit. It stopped locally with `No API key configured for the frozen Responses provider`, before construction/sending of an HTTP request. There is one preserved request intent, zero provider receipts, zero selections, zero acquisition reveals, zero new fits. The only scientific labels read so far are authorized L333 labels. No test labels were read.

Current Codex uses ChatGPT login and has no accessible OPENAI_API_KEY. A separate content-free Codex CLI connectivity preflight returned JSON with native tool calls=0, but prompt diagnostics show extra injected skills/permissions context. This transport was not admitted for scientific selections; no source/prompt/protocol was changed after freeze.

Recovery: securely provision a Responses API key in the local environment/provider configuration (never paste it into chat or commit it). Preserve request_00.json and all failure evidence. Register a transport/preflight revision or a new study execution version before restarting; the current fail-closed runner deliberately refuses to resend an existing intent with no receipt. Do not delete it to bypass the barrier.

## Matched historical validation references (read-only)

These are existing seed1525 results, not results from this unfinished Free-LLM32 experiment. The frozen L333 denominator is 8.879240547556758. No test-side result file was parsed.

| Method | L | RMSE | MAE | R2 | Fixed NRMSE |
|---|---:|---:|---:|---:|---:|
| random | 333 | 8.043997 | 5.287642 | 0.163640 | 0.905933 |
| random | 365 | 8.113203 | 5.082832 | 0.149187 | 0.913727 |
| random | 397 | 7.930685 | 4.954846 | 0.187037 | 0.893172 |
| raw_gradient_lcmd | 333 | 8.043997 | 5.287642 | 0.163640 | 0.905933 |
| raw_gradient_lcmd | 365 | 7.748384 | 5.070312 | 0.223982 | 0.872640 |
| raw_gradient_lcmd | 397 | 7.840739 | 4.818740 | 0.205372 | 0.883042 |

Do not create PHASE1_COMPLETE_L397 or claim an L397 INTERIM_REPORT until both real acquisitions and scratch fits have completed and their manifests verify. No round2, second seed, hybrid or masked run has started.
