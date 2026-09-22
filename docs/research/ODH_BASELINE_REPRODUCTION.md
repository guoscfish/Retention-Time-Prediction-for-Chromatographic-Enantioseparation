# ODH baseline reproduction and original-paper audit

Current status (2026-09-22): baseline COMPLETE. Phase 2 infrastructure and the
one-seed/three-acquisition AL smoke also completed subsequently; see the
[roadmap](PROJECT_ROADMAP.md) for current scope.

This document combines the baseline protocol/results and the original paper/code
audit. Read [results](#results) for measured reproduction performance,
[limitations](#limitations-and-deviations) for interpretation, and
[the original audit](#original-paper-and-repository-audit) for source semantics.

## Prospective protocol (recorded before training, 2026-09-21)

The following records the pre-run plan. It is not a list of outstanding tasks.

Base commit: `d284767bf7f1f04b24f758211a1d8d70e82f52ed`.
Scope: one ODH baseline, no active learning or model tuning.
Status: **COMPLETE**. The official-code-compatible formal run completed all
1,500 epochs on 2026-09-22.

The pre-run official-code configuration, subsequently cross-checked below: ODH CSV;
exclude known conformer position 4231 and apply the official pre-split RTv > 60
domain restriction with a complete exclusion manifest. Target RTv = RT * Speed.
Three outputs q10 / MSE central / q90; five layers, 128 hidden dimensions,
sum pooling, zero dropout; Adam lr 0.001, weight decay 1e-5; batch 2048;
1500 epochs; no scheduler steps or early stopping. NumPy split seed 388,
PyTorch initialization seed 525. Fixed random-row 90/5/5 split, integer floors,
last test segment and explicitly logged unused rounding rows; no loader shuffle.
Use the final epoch checkpoint, never test-based selection. Extra validation
logging is diagnostic only. Retain all test rows in the predeclared domain.

Before formal training: finish preprocessing; verify graph finiteness, nonempty
graphs, paired batch alignment, output and h_graph shapes, finite loss/backward
and optimizer update; run 1-5 complete epochs and validation/test diagnostics.
Record timings and estimate full cost. If prohibitively slow, stop before the
long run and report smoke results only, with an executable full-run command.

Paper reference only (not our result): ODH Figure 3c MAE 2.74, median relative
error 15.8%, R2 0.778; attached source-data recomputation gives RMSE 3.9173,
MAE 2.7435, R2 0.7778, median relative error 15.8402%.

## Results

### Formal 1,500-epoch reproduction (completed)

Artifact: `artifacts/reproduction/odh_baseline_20260922/`.

- Historical execution command: `.venv-reproduction/bin/python scripts/reproduce_odh_baseline.py --mode train --device cpu --threads 4 --output artifacts/reproduction/odh_baseline_20260922`
- Eligible rows: 4,942; split sizes: train 4,447, validation 247, test 247.
- Runtime: 12,825.8809 s (3.56 h) on CPU with four threads; 1,500 epochs completed.
- Checkpoint: final epoch 1,500, as used by the public single-column training
  branch. The best validation epoch, 913, is recorded only as a diagnostic.

The comparison below was recomputed from the formal `predictions.csv` and the
author's `artifacts/reproduction/audit/paper_source_data_ODH.csv`; it is not a
transcription of a chat result.

| Test metric | Paper source-data | Formal reproduction |
| --- | ---: | ---: |
| RMSE | 3.9172548 | 4.1773667 |
| MAE | 2.7434876 | 2.6786728 |
| R2 | 0.7778434 | 0.7473608 |
| Median relative error | 0.1584022 | 0.1366186 |

Both files contain 247 test targets. A programmatic position-by-position check
confirmed that they are the same targets in the same order: all 247 positions
match under `rtol=1e-7, atol=1e-6`, with maximum numerical difference
1.8310546892053026e-06 from float serialization. The corresponding paper and
reproduction predictions have Pearson correlation 0.9611928985986683. The full
machine-readable result and comparison method are in
`artifacts/reproduction/odh_baseline_20260922/paper_comparison.json`.

The formal test q10-q90 coverage is only 0.1862348 (46/247). Despite no
quantile crossing, this is far below the nominal 80% coverage implied by q10
and q90. The current interval width is therefore **not calibrated uncertainty**
and must not be used as uncertainty acquisition evidence.

### Smoke gate (completed)

Artifact: `artifacts/reproduction/odh_smoke_20260921_v4/`.

- Historical execution command: `.venv-reproduction/bin/python scripts/reproduce_odh_baseline.py --mode smoke --device cpu --threads 4 --smoke-epochs 2 --output artifacts/reproduction/odh_smoke_20260921_v4`
- Eligible rows: 4,942; split sizes: train 4,447, validation 247, test 247.
- Output shape `(8, 3)`; `h_graph` shape `(8, 128)`; 139 parameter tensors received finite central-output gradients.
- Finite graph/features/loss, backward, optimizer update and label-free forward equivalence all passed.
- Epoch 1: train objective 228.5426, validation MSE 199.8030, 16.47 s.
- Epoch 2: train objective 120.9485, validation MSE 267.6448, 14.83 s.
- Smoke test runtime 44.1 s; measured estimate for 1500 epochs is 6.68 h CPU.

The two-epoch smoke metrics are not baseline metrics. Its cost estimate informed
execution planning only; the completed formal artifact above supersedes the
earlier deferred status.

## Limitations and deviations

The paper's early-stopping description is not implemented in the public
single-column branch. The official-code-compatible baseline therefore remains
the final 1,500th epoch. Epoch 913 is a validation diagnostic only: the test set
was not used to select a checkpoint, and neither that diagnostic nor proximity
to the paper result justifies checkpoint replacement or retraining. The
author's ODH graph/descriptors are externally cached because
they are absent from Git; cache provenance is recorded alongside hashes. The
current macOS arm64 machine cannot use the existing `fish` environment because
importing its torch stack aborts on duplicate OpenMP runtimes. A clean isolated
Python 3.11 environment is being used; its exact installed versions are saved
in `environment.json`.

The random-row split includes molecular overlap between train and test, so this
baseline establishes compatibility with the paper protocol rather than
leakage-resistant chemical generalization. The q10-q90 coverage limitation
above also rules out treating interval width as calibrated uncertainty.

## Reproduction judgement

The ODH baseline is **sufficiently reproduced for Phase 2 gradient-space AL
infrastructure**. The full official-code-compatible trajectory, final-epoch
metrics, fixed split, stable identities, central-gradient path, `h_graph`, and
label-free forward path are now recorded and audited. This is a readiness
judgement made at Phase 1 closure. Phase 2 was subsequently authorized and
completed; the [AL audit](HPLC_AL_IMPLEMENTATION_AUDIT.md) records its evidence.
The larger AL screen and formal benchmark have not been started.

## Original paper and repository audit

Merged from the original standalone audit on 2026-09-22. The observations below
retain their original scope; reproduction results are reported above.

Audit started 2026-09-21. Base commit:
`d284767bf7f1f04b24f758211a1d8d70e82f52ed`, branch `main`.
Origin is guoscfish's fork; upstream is woshixuhao's official repository.
At the initial audit, the worktree contained only an untracked `.DS_Store`;
that desktop metadata was subsequently removed during repository cleanup.

This document distinguishes paper-reported, official-code-observed and unresolved
claims.
The original scripts and pre-existing user files are not to be overwritten.

### Data audit

The committed CSVs have these row counts (including export index columns):

| File / column | Raw rows | Unique raw SMILES | Unique canonical isomeric SMILES |
| --- | ---: | ---: | ---: |
| `All_column_charity.csv` | 25,867 | 23,794 | 23,782 |
| ODH | 4,972 | 4,459 | 4,454 |
| ADH | 5,418 | 5,135 | 5,134 |
| IA | 1,849 | 1,724 | 1,724 |
| IC | 2,091 | 2,046 | 2,046 |

ODH has 4,972 raw rows, 4,971 after the checked `bad_ODH`/row-4231 conformer
exclusion, and 4,942 after the RTv > 60 exclusion. The audit found 46 test rows
whose canonical isomeric SMILES also occur in the random-split training rows.
This is consistent with the paper's interpolation-oriented random-row split and
is a reason for the later molecule/group-aware phase. Raw repeated structures
and enantiomer handling must not be conflated with an experimental duplicate.

CSV fields are `SMILES`, `RT`, `Speed`, `i-PrOH_proportion`, `Literature`, and
column/export identifiers. The target implemented by `Construct_dataset` is
`RTv = RT * Speed`; this follows the paper's chromatographic process equation.
The single-column model does not use column descriptors. It uses nine categorical
atom features, three categorical bond features, bond length plus eluent on Graph
G, and a bond-angle Graph H whose edge features append five Mordred values:
TPSA, RASA, RPSA, MDEC and MATS.

### Model and protocol audit

**Paper-reported:** Graph G plus bond-angle Graph H; 5 GINConv layers; sum
pooling; embedding dimension 128; batch size 2048; 1500 epochs; Adam with
learning rate 0.001; 90/5/5 random single-column split; validation loss used for
early stopping; RTv > 60 dropped. The paper describes three outputs as 90th
quantile, prediction, and 10th quantile (its figure ordering differs from the
code's columns).

**Official-code-observed:** `Single_column_prediction.py` defaults to
`MODEL='Test'`, target `IC`, and device index 1. For ODH it drops CSV index 4231,
then drops RTv > 60 during graph construction. It seeds NumPy with 388 and
PyTorch with 525, uses integer-floor 90/5/5 partitions and `test_mode='fixed'`
(the final test segment after shuffling). Loaders have `shuffle=False`. The
training branch hard-codes `range(1500)`, evaluates/saves every 100 epochs, but
does not load the best validation checkpoint, invoke early stopping, or call the
declared StepLR scheduler. `--epochs` is therefore not effective. The loss is
q10 pinball + central MSE + q90 pinball + two monotonicity penalties + a
`relu(2-pred)` deadtime penalty.

At training time `forward` returns `(output, h_graph)` without clamping. At
inference it clamps all three outputs to be nonnegative and returns the same
tuple. `test` uses output column 1 as the central prediction and columns 0/2 as
the two interval bounds. Therefore this audit calls column 1 **MSE central**;
it is not evidence of a separately trained q50. Graph-level `h_graph` has 128
components; the completed readiness check is reported in the smoke results above.

The implementation also creates an unused Mordred descriptor projection
(`NN_descriptor`) and unused condition encoder in the single-column model;
these do not affect the forward path. The code's `get_MMFF_atom_poses` currently
reads an existing conformer; the commented alternative would embed and optimize
conformers. The published cache therefore matters for bitwise provenance.

**Discrepancies / ambiguities:** The paper says validation loss is adapted for
early stopping, but the public single-column training branch has no early-stop
logic and always tests the 1500-epoch checkpoint. The paper's output ordering
and code's output ordering should not be silently reconciled. The repository
does not commit ODH graph caches or model checkpoints; the author-linked archive
is an external provenance input. Exact bad-conformer provenance for row 4231 is
not explained in the paper. We retain every predeclared RTv <= 60 test row.

The paper's Figure 3c reports ODH MAE 2.74, median relative error 15.8%, and
R2 0.778. The author's accompanying `source_data_ODH.csv` contains 247 points;
recomputed values are RMSE 3.9173, MAE 2.7435, R2 0.7778, median relative error
15.8402%, mean relative error 20.1255%. These are reference observations, not
our training result. The attached ODH CSV is byte-identical to the committed
`dataset/ODH_charity_0616.csv`.
