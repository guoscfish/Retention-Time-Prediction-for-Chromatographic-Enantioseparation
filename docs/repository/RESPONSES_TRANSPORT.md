# Direct Responses transport

The registered V2 provider is **token4research, a third party**, at
`https://token4research.cn`; this is not an OpenAI official endpoint.

The runner reads only explicit selected settings from `~/.codex/config.toml`:

```toml
model_provider = "token4research"
model = "gpt-5.5"
review_model = "gpt-5.5"
model_reasoning_effort = "high"
model_catalog_json = "~/.codex/codex-models.json"

[model_providers.token4research]
name = "token4research"
base_url = "https://token4research.cn"
wire_api = "responses"
env_key = "TOKEN4RESEARCH_API_KEY"
requires_openai_auth = false

[features]
goals = true
```

This is a configuration example, not a change made to the user's global settings.
Only `os.environ[env_key]` supplies the credential. Missing environment credentials
produce `RESPONSES_API_KEY_NOT_CONFIGURED` and name the missing variable. No credential
file, ChatGPT login, Codex CLI, implicit provider or model fallback exists. Credentials
and Authorization headers are never persisted or hashed into study artifacts.
HTTP redirects are rejected; network failures omit untrusted remote error bodies.

Each scientific request POSTs to configured `base_url + /responses` with explicit
model, reasoning.effort, input, tools=[], store=false and max_output_tokens.
There are no native tools, session IDs, previous_response_id, stored conversation
history, web, shell, MCP or file tools. The JSON relay is host-controlled text.
Responses must have HTTP success, status=completed, only reasoning/message outputs,
exactly one assistant message/text containing one JSON object. Duplicate keys,
NaN, code fences, appended JSON, native tool calls and model mismatch are rejected.
A missing served-model field is recorded as unknown rather than fabricated.

Receipts retain response ID, served model if returned, usage counters when available,
SHA256 of the exact serialized request body and answer text, provider ID, hostname
and timestamp. Raw HTTP headers/error bodies are not saved. Requests are journaled
before dispatch; an intent without receipt is never retried automatically.

Run `.conda-hplc-al/bin/python scripts/preflight_llm_responses.py` after securely
configuring the environment. Only a content-free JSON request is sent. The output
`transport_preflight.json` records PASS/FAILED or READY_FOR_API_KEY without a secret.
PASS permits a future separately authorized scientific run; it does not start one.
The preflight does not establish maximum provider context capacity or independently
prove server-side retention behavior. `store=false` is explicitly sent and acceptance
is checked; provider internals are outside this repository's evidence.

Field semantics were checked against the [official Responses API reference](https://developers.openai.com/api/reference/typescript/resources/beta/subresources/responses/methods/create).
That reference defines the interface; only a token4research preflight can establish
compatibility for the configured endpoint. No endpoint compatibility claim is made here.
