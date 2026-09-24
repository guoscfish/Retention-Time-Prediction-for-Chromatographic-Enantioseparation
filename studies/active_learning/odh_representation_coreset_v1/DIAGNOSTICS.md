# Pre-AL representation diagnostics

Gate: PASS. All diagnostics are label-free; no test truth has been accessed.

L333 latent–Morgan pair-distance Spearman: 0.135745 (50,000 sampled pairs, seed 525).
Latent norm–heavy atom count Spearman: 0.538453.

| Checkpoints | Representation | Row cosine | kNN@10 | kNN@50 | Distance rank correlation |
|---|---|---:|---:|---:|---:|
| 333-365 | gradient | 0.4069 | 0.3619 | 0.4488 | 0.5542 |
| 333-365 | latent | 0.8300 | 0.3955 | 0.5269 | 0.8243 |
| 365-397 | gradient | 0.4133 | 0.3783 | 0.4806 | 0.7365 |
| 365-397 | latent | 0.8293 | 0.4232 | 0.5686 | 0.8612 |
| 397-429 | gradient | 0.3834 | 0.3721 | 0.4720 | 0.6674 |
| 397-429 | latent | 0.8478 | 0.4445 | 0.5871 | 0.8687 |

Latent geometry is more stable on all three consecutive checkpoint comparisons by kNN@10, kNN@50 and distance rank correlation. This is along the historical LCMD trajectory and is not evidence that latent acquisition will improve prediction.

Morgan chirality: all 1,578 observed explicit tetrahedral mirror pairs are distinguished with includeChirality=True; none are distinguished without chirality. Unspecified/axial/non-tetrahedral stereochemistry is not inferred.

Historical LCMD picks show chemical novelty enrichment. Historical MaxDet does not consistently select more chemically novel rows than LCMD. Full per-round distributions and current-pool percentiles: diagnostics/historical_coverage.json.

The weak latent–Morgan correlation is descriptive. The task-aware latent includes geometry and experimental condition; it should not be interpreted as a pure molecular fingerprint.
