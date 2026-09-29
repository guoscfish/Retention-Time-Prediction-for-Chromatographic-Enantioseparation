# READY_FOR_API_KEY

Implementation and engineering verification only. No scientific LLM calls, no
new acquisition labels, no scratch retraining, no new V2 performance results.

Required environment variable: `TOKEN4RESEARCH_API_KEY`. The user explicitly chose
to finish this change at READY_FOR_API_KEY; no API request will be sent in this task.
A value sent in chat is not treated as a configured process environment variable.

The current user-level Codex configuration has not yet been switched to the
registered token4research / gpt-5.5 / high configuration. Configure it before
running the separate content-free preflight. No credentials are stored in this repository.

Pre-run re-review:155 tests pass after adding source-CSV integrity checks on every
resume, before label access. A realistic Stage1-text stress test can still overflow
Stage2's local context ceiling; the host stops safely. See
[pre-run findings](../../../docs/repository/PRE_RUN_REVIEW.md). A successful small
API preflight alone does not resolve this capacity issue.
