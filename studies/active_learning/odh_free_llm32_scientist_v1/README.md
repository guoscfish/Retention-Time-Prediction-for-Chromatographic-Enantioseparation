# Free-LLM32 v1 — frozen development evidence

Current fact: **COMPLETE_PHASE1_L397**. Seed 1525; two acquisitions, 64 labels,
10 scientific LLM calls; zero test-label access. See [STATUS](STATUS.md).

- [Results](results/validation_metrics.csv), [partial AULC](results/partial_aulc.csv),
  and [original interim report](INTERIM_REPORT.md) preserve the existing numerical results.
- [Successful execution](provenance/successful_execution/) contains the former
  `transport_revision_2` artifacts, byte-for-byte, including original protocol,
  request receipts, feedback, seals, fit records and final audit.
- [Relocation manifest](provenance/relocation_manifest.json) maps original paths to
  current paths and SHA256. Embedded historical paths in reports/seals are historical
  locators, resolved via this map; they have not been rewritten to claim new execution.
- [Failed first transport](provenance/failed_transport_v1_summary.md) has no scientific result.

This study used gpt-6-sol through an isolated Codex CLI transport. It did not prove
100% candidate screening: round0 viewed 224 candidates, round1 184. These results are
single-seed exploratory validation evidence, not evidence for the Responses/full-pool
V2 protocol. V1 is closed; its former executable runner is available only in Git history.
