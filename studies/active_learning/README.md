# ODH research map

Start with these six studies. All AL results are development evidence on an existing
Row cohort; none should be relabeled as independent external validation.

| Research question | Canonical entry | Status / scope |
|---|---|---|
| Original model reproduction / benchmark | [Baseline reproduction](../../docs/research/ODH_BASELINE_REPRODUCTION.md), [artifacts](../../artifacts/reproduction/) | COMPLETE / CANONICAL_BASELINE; original model in `code/` |
| Why did early AL training underperform? | [Training diagnosis](odh_training_protocol_diagnosis/README.md) | COMPLETE; fixed-L0 validation diagnosis |
| Recommended numerical AL comparator | [Raw-gradient LCMD confirmation v2](odh_lcmd_confirmation_v2/) | COMPLETE / CANONICAL_BASELINE; Random vs raw-gradient LCMD, paired seeds; preserve registered scope |
| IVR/hybrid alternatives | [IVR/hybrid screen](odh_ivr_hybrid_screen_v1/REPORT.md) | COMPLETE / DEVELOPMENT; method comparison, not current winner by assertion |
| Free LLM batch planning | [Free-LLM32 v1](odh_free_llm32_scientist_v1/README.md) | COMPLETE_PHASE1_L397 / DEVELOPMENT; frozen historical CLI transport |
| Hierarchical full-pool LLM planning | [FullPool-LLM32 v2](odh_free_llm32_fullpool_v2/README.md) | PARTIAL screening; 4 saved responses, 0 acquisitions; bounded retry recovery ready, see STATUS.md |

Other retained methodological evidence:

| Study | Classification / reason retained |
|---|---|
| [gradient smoke](odh_gradient_al_smoke/README.md) | COMPLETE / DEVELOPMENT; original engineering evidence |
| [gradient transfer v2](odh_gradient_al_transfer_v2/REPORT.md) | COMPLETE / DEVELOPMENT; training/representation transition, including interrupted-attempt provenance |
| [representation coreset](odh_representation_coreset_v1/REPORT.md) | COMPLETE / DEVELOPMENT; representation ablation; pre-AL DIAGNOSTICS is a historical snapshot |
| [representation extension](odh_representation_coreset_extension_v1/) | COMPLETE / DEVELOPMENT; validation stop-rule extension |
| [LCMD confirmation v1](odh_lcmd_confirmation_v1/) | PARTIAL / SUPERSEDED / ARCHIVE_ONLY; frozen path references and unique fit artifacts retained |

Legacy LLM hybrid v1/v2/v3/v4/v6/v7/v8 moved out of this directory. Use
[legacy history](../../docs/archive/LLM_HYBRID_LEGACY_HISTORY.md) and
[archived evidence](../archive/llm_hybrid/); no need to guess a valid version.
The [repository audit](../../docs/repository/CLEANUP_AUDIT.md) lists all branches/studies
and the full static import inventory. Archived code is recovered from annotated tags.
