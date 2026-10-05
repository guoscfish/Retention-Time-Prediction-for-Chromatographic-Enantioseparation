# LLM hybrid legacy history

Current checkout update (2026-10-05): these incomplete experiments were deleted at the user's request. This document describes historical evidence; the former archive paths below no longer exist. Use the recorded Git tags to inspect history.

All versions originate in commit `9155889` and are protected by pushed annotated tags
`archive/pre-cleanup-ivr-hybrid-2026-09-29` and
`archive/pre-cleanup-free-llm32-2026-09-29`. The former contains the original paths;
the latter includes subsequent budget/catalog changes. Git history preserves executable
code and all tracked bytes. Local ignored checkpoints/NPZ are also retained after relocation.

| Version | Purpose | Evidence / finding | Why retired |
|---|---|---|---|
| v1 | Initial 16 LCMD + 16 LLM relay | Packet/protocol/access audit, no accepted selection | Incomplete transport attempt |
| v2 | Query relay iteration | Seven responses, no accepted batch | Did not complete selection; responses are debugging evidence |
| v3 | Relay iteration | One response, no accepted batch | Did not complete selection |
| v4 | Selection contract iteration | Unique accepted selection, no fit manifest | Incomplete trajectory; not a performance result |
| v6 | Selection contract iteration | Unique selection/relay provenance, no fit manifest | Incomplete trajectory |
| v7 | Selection contract iteration | Unique selection/relay provenance, no fit manifest | Incomplete trajectory |
| v8 | Chemical vs masked hybrid development run | Chemical two fits, masked one fit; additional chemical round2 request chain without selection | PARTIAL; `complete.json` says STOPPED_OR_COMPLETE with stop_round=1 and is not proof of a completed matched comparison |

No v5 directory exists at audited tip. Version-specific causal explanations beyond
these artifacts are not recoverable from commit messages, so none are invented.
Later versions do **not** prove that all earlier scientific evidence is redundant.
Therefore unique packets, responses, selections, access audits, training records and
ignored binaries were moved to [the archive](../../studies/archive/llm_hybrid/), not deleted.
Exactly identical bytes share one archived copy through
[the relocation manifest](../../studies/archive/llm_hybrid/relocation_manifest.json).
The manifest records every old path, destination and SHA256, including tracked status.

The old catalog paging budgets, 16+16 runner, duplicate parser/report and both old
transports were removed from active Python. Recover them from the tags to inspect
historical behavior. Do not execute historical runners against cleaned paths or
interpret their complete markers as current status. V2 is a new protocol.
