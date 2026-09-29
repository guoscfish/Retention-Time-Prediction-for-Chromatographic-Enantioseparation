# Direct token4research Responses transport

The registered V2 provider is **token4research, a third party**, at
`https://token4research.cn/responses`. Model: **gpt-6-astra**, reasoning: **high**.
The platform key must be a **token4research key**.

The following nonsecret settings have been applied to this machine's
`/Users/fish/.codex/config.toml`, preserving unrelated settings:

```toml
model_provider = "token4research"
model = "gpt-6-astra"
review_model = "gpt-6-astra"
model_reasoning_effort = "high"

[model_providers.token4research]
name = "token4research"
base_url = "https://token4research.cn"
wire_api = "responses"
env_key = "TOKEN4RESEARCH_API_KEY"
requires_openai_auth = false
```

`env_key` is a variable name, not a place to paste the key. No model catalog file is
required by the scientific runner; no nonexistent catalog path has been configured.

## Start or resume six rounds

In Terminal, run:

```sh
cd /Users/fish/Documents/GitHub/Retention-Time-Prediction-for-Chromatographic-Enantioseparation
.conda-hplc-al/bin/python scripts/run_hplc_fullpool_v2.py run
```

The launcher now automatically reads your designated file:
`/Users/fish/.config/qgeognn-scientist/token4research.api-key`.
No repeated key entry is needed. The file is read only into this process's
`TOKEN4RESEARCH_API_KEY`; the key is never printed, copied to study files or hashed.
Existing environment credentials take precedence. `--key-file /absolute/path` explicitly
selects a different file and overrides an existing environment value. If neither the
environment nor the default file provides a key, `run` offers a hidden terminal prompt.
The HTTP transport itself still reads only its registered environment variable.
The runner does not read `.env` or `auth.json`.

This command first checks the committed implementation/protocol and performs a
content-free Responses preflight. On success it automatically runs six acquisitions,
32 each: **333 → 365 → 397 → 429 → 461 → 493 → 525**, then reports and stops.
Each selection is committed locally before any acquisition label is revealed.
The runner commits only this study's nonignored artifacts; it does not push.
Keep the terminal/process running. Run the same command after interruption: completed
fits and responses are verified/reused; an unfinished fit restarts from scratch.
The 2026-09-29 user-authorized execution amendment permits an identical retry of
an ambiguous intent, counting it toward a five-attempt lifetime limit. Already
validated responses are reused. Retry waits are 15/30/60/60 seconds; authentication,
JSON/schema, scientific validation and source-integrity failures stop immediately.
Terminal output includes chunk/arbitration progress, cache reuse, request attempts,
elapsed-time heartbeats every 15 seconds, Git seals and training epoch progress.
No seventh acquisition is allowed. Extending the budget after science starts requires
a separately registered study. This is offline label acquisition from the existing
dataset and scratch QGeoGNN training, not instrument control or LLM fine-tuning.

For a **preflight only**, the same designated file is loaded automatically:

```sh
.conda-hplc-al/bin/python scripts/preflight_llm_responses.py
```

That separate preflight script never launches an acquisition. Missing keys produce
`RESPONSES_API_KEY_NOT_CONFIGURED` / `READY_FOR_API_KEY` without sending requests.
Only `os.environ[env_key]` supplies credentials to HTTP; the launcher injects the user-designated file. No auth.json, ChatGPT login,
Codex CLI, implicit provider or model fallback exists. Credentials and Authorization
headers are never persisted or hashed into study artifacts.

Every scientific POST explicitly sends model, reasoning.effort, input, tools=[],
store=false and max_output_tokens=32000. There are no native tools, sessions, previous
response IDs or external conversation history. Success requires HTTP success,
status=completed, only reasoning/message outputs, and exactly one assistant text
containing one JSON object. Duplicate keys, NaN, code fences, extra JSON, native tool
calls and model mismatch are rejected. Missing served model or usage is unknown.
Receipts record response ID, model when returned, usage when available, request/answer
SHA256, provider, hostname and timestamp. Redirects and untrusted HTTP error bodies
are rejected; ambiguous requests never automatically retry.

A real content-free preflight now passes with the designated key and returns gpt-6-astra.
It validates this endpoint/key/model request, not maximum context capacity. Local
context estimates use o200k_base with a 20% margin, not a provider guarantee.
Provider rejection stops execution without a fallback or silent truncation.
`store=false` is sent; this repository cannot independently verify backend retention.

Interface reference: [Responses API](https://developers.openai.com/api/reference/typescript/resources/beta/subresources/responses/methods/create).
This defines the compatible interface, not a claim about token4research internals.

## HTTP 403 diagnosis and fix

The default Python-urllib client signature received Cloudflare403/1010 even without
a key. The truthful `User-Agent: QGeoGNN-Scientist/2.0` and `Accept: application/json`
reach the API: without a key it returns401/API_KEY_REQUIRED; with the designated
key the exact gpt-6-astra/high preflight completes successfully. The endpoint, model,
reasoning setting and Bearer authentication are unchanged. No browser impersonation,
provider switch, model fallback or automatic retry is used.

The transport classifies Cloudflare1010 into a safe fixed error code. Raw HTTP error
bodies and credentials are not persisted or printed. CLI transport failures now print
a concise diagnostic instead of a traceback. [Evidence](verification/transport_403_diagnosis.json).


## HTTP 524 recovery and terminal progress (2026-09-29)

Snapshot `75dd26d` preserves all four successful Responses receipts and the fifth
ambiguous request. No new acquisition labels had been revealed. The user's explicit
permission for V2 logging/retries is recorded separately in `execution_amendment.json`;
`protocol.json`, its seal, frozen prompts and original requests remain byte-identical.
The runner verifies the new implementation against `execution_test_gate.json` and
commits the amendment before allowing label access. The original test gate is retained.

Transient 408/429/500/502/503/504/520/522/524 and network/read failures retry, up to
five total attempts per scientific request across process restarts. The original
`screen_004` consumes attempt 1. Each dispatch has a durable `.attempt_NN.started.json`
and failed outcomes have `.failed.json`; unknown provider usage is not invented.
First validated response wins; no retries are used to improve a scientific answer.
An exhausted request remains stopped on relaunch. No provider/model fallback.
`results/execution_attempts.json` separates dispatch intents from usable receipts;
receipt token totals do not include unknown usage for failed/unreturned attempts.

A 524 means the gateway did not receive a timely origin response. The existing
600-second client timeout does not change the gateway limit. This amendment leaves
nonstreaming requests unchanged; retries improve recovery but cannot guarantee that
a persistently slow gateway will succeed. See [Cloudflare 524 documentation](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-524/).

Verification: 196 tests passed; frozen payload AST and prompts checked; four actual
receipts replayed offline without new requests; simulated full-pool audit matches the
original; content-free third-party preflight passed. No real acquisition was started
by the repair. Run the same `run` command to resume.
