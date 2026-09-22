# Original paper and repository audit

Audit started 2026-09-21. Base commit:
`d284767bf7f1f04b24f758211a1d8d70e82f52ed`, branch `main`.
Origin is guoscfish's fork; upstream is woshixuhao's official repository.
Initial worktree: only untracked `.DS_Store`, preserved.

This document distinguishes paper-reported, official-code-observed and unresolved
claims.
The original scripts and pre-existing user files are not to be overwritten.

## Data audit

The committed CSVs have these row counts (including export index columns):

| File / column | Raw rows | Unique raw SMILES | Unique canonical isomeric SMILES |
| --- | ---: | ---: | ---: |
| `All_column_charity.csv` | 25,867 | 23,794 | 23,782 |
| ODH | 4,972 | 4,459 | 4,454 |
| ADH | 5,418 | 5,135 | 5,134 |
| IA | 1,849 | 1,724 | 1,724 |
| IC | 2,091 | 2,046 | 2,046 |

ODH has 5,393 rows after the checked `bad_ODH`/row-4231 exclusion and before
the RTv rule; 4,942 remain after RTv <= 60. The audit found 46 test rows whose
canonical isomeric SMILES also occur in the random-split training rows. This is
consistent with the paper's interpolation-oriented random-row split and is a
reason for the later molecule/group-aware phase. Raw repeated structures and
enantiomer handling must not be conflated with an experimental duplicate.

CSV fields are `SMILES`, `RT`, `Speed`, `i-PrOH_proportion`, `Literature`, and
column/export identifiers. The target implemented by `Construct_dataset` is
`RTv = RT * Speed`; this follows the paper's chromatographic process equation.
The single-column model does not use column descriptors. It uses nine categorical
atom features, three categorical bond features, bond length plus eluent on Graph
G, and a bond-angle Graph H whose edge features append five Mordred values:
TPSA, RASA, RPSA, MDEC and MATS.

## Model and protocol audit

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
components and is available for a future readiness check.

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
