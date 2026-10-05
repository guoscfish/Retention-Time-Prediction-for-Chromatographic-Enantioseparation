# ODH research map

Start with the studies below. All AL results are development evidence on an existing
Row cohort; none should be relabeled as independent external validation.

| Research question | Canonical entry | Status / scope |
|---|---|---|
| Original model reproduction / benchmark | [Baseline reproduction](../../docs/research/ODH_BASELINE_REPRODUCTION.md), [artifacts](../../artifacts/reproduction/) | COMPLETE / CANONICAL_BASELINE; original model in `code/` |
| Why did early AL training underperform? | [Training diagnosis](odh_training_protocol_diagnosis/README.md) | COMPLETE; fixed-L0 validation diagnosis |
| Recommended numerical AL comparator | [Raw-gradient LCMD confirmation v2](odh_lcmd_confirmation_v2/) | COMPLETE / CANONICAL_BASELINE; Random vs raw-gradient LCMD, paired seeds; preserve registered scope |
| IVR/hybrid alternatives | [IVR/hybrid screen](odh_ivr_hybrid_screen_v1/REPORT.md) | COMPLETE / DEVELOPMENT; method comparison, not current winner by assertion |

Latest cross-study summary: [2026-10-05 results and comparison figures](../../artifacts/analysis/odh_results_20261005/REPORT.md). This report separates matched seed-1525 validation comparisons from historical numerical-method test results.

| Additional study | Status / scope |
|---|---|
| [Global full-pool LLM v3](odh_free_llm32_global_v3/) | COMPLETE_PHASE1_L525; six acquisitions, seed 1525, validation only; no test-label access |

Other retained methodological evidence:

| Study | Classification / reason retained |
|---|---|
| [gradient smoke](odh_gradient_al_smoke/README.md) | COMPLETE / DEVELOPMENT; original engineering evidence |
| [gradient transfer v2](odh_gradient_al_transfer_v2/REPORT.md) | COMPLETE / DEVELOPMENT; training/representation transition, including interrupted-attempt provenance |
| [representation coreset](odh_representation_coreset_v1/REPORT.md) | COMPLETE / DEVELOPMENT; representation ablation; pre-AL DIAGNOSTICS is a historical snapshot |
| [representation extension](odh_representation_coreset_extension_v1/) | COMPLETE / DEVELOPMENT; validation stop-rule extension |
| [LCMD confirmation v1](odh_lcmd_confirmation_v1/) | PARTIAL / SUPERSEDED / ARCHIVE_ONLY; frozen path references and unique fit artifacts retained |

Retired partial-visibility, incomplete hierarchical and legacy hybrid experiments were removed at the user's request on 2026-10-05. Only eight byte-preserved baseline dependency files remain at V1/V2 historical paths for the completed Global V3 seals and offline verification. See [cleanup inventory](../../docs/repository/cleanup_20261005.json). Historical audit documents describe their dates, not the current checkout.
