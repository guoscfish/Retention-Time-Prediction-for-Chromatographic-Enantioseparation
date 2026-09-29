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

Paste the token4research key at the hidden prompt and press Return. Input is not
echoed, placed in shell history or written to disk; it supplies the current process's
`TOKEN4RESEARCH_API_KEY` environment variable. If that environment variable is already
set, the prompt is skipped. The runner does not read `.env` files.

This command first checks the committed implementation/protocol and performs a
content-free Responses preflight. On success it automatically runs six acquisitions,
32 each: **333 → 365 → 397 → 429 → 461 → 493 → 525**, then reports and stops.
Each selection is committed locally before any acquisition label is revealed.
The runner commits only this study's nonignored artifacts; it does not push.
Keep the terminal/process running. Run the same command after interruption: completed
fits and responses are verified/reused; an unfinished fit restarts from scratch.
An intent without a response receipt stops as ambiguous and is never blindly resent.
No seventh acquisition is allowed. Extending the budget after science starts requires
a separately registered study. This is offline label acquisition from the existing
dataset and scratch QGeoGNN training, not instrument control or LLM fine-tuning.

For a **preflight only**, inject the key into the shell without recording its value:

```sh
read -rs 'TOKEN4RESEARCH_API_KEY?token4research API key: '
printf '\n'
export TOKEN4RESEARCH_API_KEY
.conda-hplc-al/bin/python scripts/preflight_llm_responses.py
```

That separate preflight script never launches an acquisition. Missing keys produce
`RESPONSES_API_KEY_NOT_CONFIGURED` / `READY_FOR_API_KEY` without sending requests.
Only `os.environ[env_key]` supplies credentials to HTTP; no auth.json, ChatGPT login,
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

Offline checks cannot establish third-party model availability, reasoning acceptance
or actual context capacity. Preflight validates the small request after key entry;
local context estimates use o200k_base with a 20% margin, not a provider guarantee.
Provider rejection stops execution without a fallback or silent truncation.
`store=false` is sent; this repository cannot independently verify backend retention.

Interface reference: [Responses API](https://developers.openai.com/api/reference/typescript/resources/beta/subresources/responses/methods/create).
This defines the compatible interface, not a claim about token4research internals.
