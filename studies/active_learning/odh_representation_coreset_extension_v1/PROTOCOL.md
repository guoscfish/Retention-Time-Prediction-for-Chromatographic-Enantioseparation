# ODH representation coreset extension v1

本扩展接续已冻结的 `odh_representation_coreset_v1`：同一 ODH split、L333、seed 525、四方法、batch 32、模型与训练协议。四条轨迹各从自身 L429 checkpoint 和 labeled set 出发。原 study 保持不可变；新 study 只增加 L429 之后的选择与训练。

## 停止规则

先完成 L461 与 L493 四方法预测。此后每两个新预算点，检查其中任意一点的 validation NRMSE：如果 Latent-Coreset 或 Morgan-Coreset 比同预算 Raw Gradient-LCMD 至少低 **0.01（绝对差）**，就再做两个预算点；否则停止。最多新增 8 个预算点，至 L685。若上限处仍有积极信号，报告为“达到预定上限且信号仍在”，不能解释为没有信号。

停止决策仅依据 validation；所有 test-X 预测先冻结，扩展轨迹结束后才统一读取一次 test truth。此前的原 study 已揭示过同一 test 集，因此扩展结果属于开发集上的追加证据，不是独立外部验证。

## 轨迹与训练

Random 延续原 U0 的唯一冻结 permutation：L429→461 取 `[96:128]`，之后顺次取不重叠的 32 行。Raw Gradient-LCMD 在每次 acquisition 用当前 checkpoint 的 512D raw CountSketch；Latent-Coreset 用当前 checkpoint 的 unit h_graph，距离 `1 − cosine`；Morgan-Coreset 继续用固定、带手性的 radius 2 / 2048-bit Morgan 指纹，距离 `1 − Tanimoto`。两种 coreset 从当轮 current L 作为已有中心出发。

每次 acquisition 后独立 scratch retrain：batch 256、deterministic_each_epoch、max 500 epochs、patience 100、lr 0.001、weight decay 1e-5、fresh Adam、无 scheduler、原始完整 HPLC loss、validation-only checkpoint selection。正式 NRMSE 固定使用原 L333 population SD。每轮保存 gradient、latent、Morgan coverage 审计；新预算点的 RMSE、MAE、R²、NRMSE 全部记录。
