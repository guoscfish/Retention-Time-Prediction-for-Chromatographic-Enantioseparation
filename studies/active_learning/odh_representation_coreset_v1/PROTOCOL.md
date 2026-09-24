# ODH representation coreset v1

本研究固定比较 parameter sensitivity、task-aware learned representation 与 chemical-space coverage。仅运行 initialization/training seed 525、4 方法、3 次 acquisition：333 → 365 → 397 → 429；完成后停止。

## Fixed training and roles

沿用 transfer_v2 的 ODH outer train / validation / test identities 和 L333。四方法共享同一个已冻结 L333 checkpoint，以及完全相同的 validation/test-X predictions。训练为 scratch、fresh Adam、batch 256、deterministic_each_epoch、maximum_epochs 500、patience 100、learning_rate 0.001、weight_decay 1e-5、无 scheduler、原始完整 HPLC loss、仅 validation central MSE 选择 checkpoint。

Raw Gradient-LCMD 的已冻结轨迹按训练配置、L/U IDs 和 checkpoint hash 验证后复用；这是相同实验条件的复用，不是新独立重复。其余三条方法从 L365 开始独立 scratch retrain。历史文件不作修改。

## Representations and selection

- `random`：seed 525900001 对排序 U0 一次 permutation；依次取前三个 32-row slice。
- `raw_gradient_lcmd`：full-network central output gradient 的固定 512D CountSketch；沿用 transfer_v2 LCMD 定义与冻结结果。
- `latent_coreset`：QGeoGNN pooling 后 128D h_graph。每次 acquisition 从该方法当前 checkpoint 提取，并单位化。距离固定为 `1 − cosine`。
- `morgan_coreset`：RDKit Morgan radius 2、2048 bits、includeChirality=True；距离为 `1 − Tanimoto`。Fingerprint bank 固定。

两种 coreset 均从 current L 作为已有 centers 开始，逐个选择距离已有 centers 最远的 U 样本，每批 32；并列时最小 sample ID 优先。未访问 U 的标签。指纹的手性开关只用于诊断，不运行第二条 AL 轨迹。

h_graph 是 learned structure/geometry/condition representation：H bond-angle 信息进入 G atom-bond message passing，还含键长、i-PrOH 及 TPSA/RASA/RPSA/MDEC/MATS，不是纯 molecular fingerprint。raw norm 使用 L2，unit = raw / max(norm, 1e-12)。提取验证 eval/no_grad、顺序 IDs、[N,128]、finite、duplicate/zero rows、状态 hash。

## Label-free diagnostics gate

所有训练前先完成：L333 latent/Morgan 距离关联（50,000 对，seed 525）；latent norm/size 关联；历史 Random/Raw-LCMD/Raw-MaxDet coverage；沿 Raw-LCMD 相邻 checkpoint 的稳定性比较。

历史 coverage 的 latent 统一用 L333 geometry，L/U 随历史轮次更新；不重跑历史 selectors。稳定性固定用历史 LCMD U429 随机抽取的 512 个 IDs。raw gradient 距离用欧氏距离，unit latent 用 cosine；同时比较逐行 cosine、gradient norm rank、kNN overlap@10/@50 和全部候选 pairwise-distance rank correlation。稳定性检验不使用标签，不用关联强弱决定是否继续；gate 只要求提取和数据完整性通过、所有诊断完成。

## Evaluation and stop

validation/test 报告 RMSE、MAE、R²、NRMSE。NRMSE 分母固定为 L333 标签 population SD；normalized AULC 为 NRMSE 对 budget 的梯形积分除以 96。报告各方法对 Random 的 AULC ratio 和方法间比较。

所有 16 个 method/budget records、checkpoints、predictions 与 selection/coverage artifacts 必须先冻结和校验，之后才统一 reveal test truth。历史 test 结果已为研究背景所知，故本 study 的 firewall 不意味着该数据集是从未使用过的外部 holdout。

training_seed 元数据修复只影响未来 audit；历史两个 seed 都为 525，数值和解释保持原状。单 seed 不作统计显著性结论。完成即 STOP，不自动扩大实验。

正式 validation 指标取自各轮 `round.json.validation_metrics`，统一揭盲后的 validation/test 指标取自 `results/metrics.csv`，两者均使用冻结 L333 SD。沿用 trainer 的 `fit.json` 内部诊断保留其 current-L SD 口径，不能用这些内部 NRMSE 拼接本 study 的学习曲线。
