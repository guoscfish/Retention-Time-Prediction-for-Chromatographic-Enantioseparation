# Failed first Responses attempt — debugging history

The old transport looked for OPENAI_API_KEY and a local authentication-file fallback,
but no usable API credential was available. It stopped locally before HTTP dispatch.
There were **0 LLM responses, 0 selections, 0 acquisition label reveals, 0 new fits**.
Authorized L333 observations were used to prepare a packet; no test labels were read.

Original freeze: e1077f2. Recorded failure: 3349a35. Recovery snapshot:
`archive/pre-cleanup-free-llm32-2026-09-29` (8c8d3d0), original study root.
Original failed request intent, dry run, audit, protocol and misleading root
EXECUTION_STATUS are recoverable there. Duplicate failed packets/runtime were removed
from the active tree; ignored failed arrays were retained in `.cleanup-local-backup/`.

The subsequent successful CLI execution is separate, preserved under
`successful_execution/`. V2 uses only explicit environment-key Responses HTTP;
there is no credential-file, CLI or provider fallback.
