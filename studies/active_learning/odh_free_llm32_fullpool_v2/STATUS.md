# READY_TO_RESUME_WITH_BOUNDED_RETRIES

Round 1/6, L333. Four valid screening responses are saved (1200/4114 candidates).
The fifth request (`screen_004`) stopped with user-reported HTTP 524 and no receipt.
No selection, acquisition label reveal or new training has occurred.

On 2026-09-29 the user explicitly authorized an execution-only amendment:
“允许 V2 仅修改日志和重试，复用已有结果”. See `execution_amendment.json`.
The original protocol, prompts, request contents and receipts remain unchanged.
Each scientific request has at most five attempts across restarts; the original
ambiguous fifth request counts as its first attempt. Validated receipts are reused.
Transient failures retry after 15/30/60/60 seconds. Terminal/validation failures stop.
Unknown failed-request usage/billing is not reported as zero.

196 tests passed. Real four-response replay and full-pool simulated selection passed.
Content-free token4research / gpt-6-astra / high preflight passed.
No scientific requests were sent during this repair. No process was left running.

Resume from the repository directory:

```sh
.conda-hplc-al/bin/python scripts/run_hplc_fullpool_v2.py run
```

Key auto-loads from `~/.config/qgeognn-scientist/token4research.api-key`.
The target remains six acquisitions of 32, ending at L525. This is the last
recorded checkpoint, not a claim that a training process is currently alive.
