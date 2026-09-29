# READY_FOR_API_KEY

Code/configuration and offline engineering verification only. No real API request,
scientific response, new acquisition label, scratch fit or V2 performance result.

This machine's `~/.codex/config.toml` is configured for third-party **token4research /
gpt-6-astra / high / Responses**. No credential is stored in the config or repository.

Run `.conda-hplc-al/bin/python scripts/run_hplc_fullpool_v2.py run` and paste the
platform key at its hidden Terminal prompt. This fills the current process's
`TOKEN4RESEARCH_API_KEY`, runs a content-free preflight, then starts/resumes six rounds
of32 to L525. The source CSV is checked before every label-store access path.
Each selection is committed before labels; no seventh acquisition is allowed.

See [pre-run review](../../../docs/repository/PRE_RUN_REVIEW.md) and
[transport/start instructions](../../../docs/repository/RESPONSES_TRANSPORT.md).
Actual provider/model compatibility remains unverified until key entry and preflight.
Later runtime progress is recorded in `execution_status.json`; completion results in
`results/summary.json`. The continuous runner replaces this readiness state with the last verified execution state.
