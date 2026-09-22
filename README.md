# Chromatographic Enantioseparation：保留时间预测

本仓库包含 QGeoGNN 原论文的代码和数据，以及 ODH 基线复现与主动学习扩展。
当前 ODH 基线和单种子、三轮主动学习开发实验均已完成。主动学习结果仅支持工程可行性检查，尚不能证明某种选样方法稳定优于随机选择。

## 从这里开始

- **阅读当前实现**：[主动学习入口](scripts/run_hplc_al_smoke.py) → [核心模块](src/hplc_al/) → [测试](tests/hplc_al/)。
- **查看实验结果**：[ODH 基线](docs/research/ODH_BASELINE_REPRODUCTION.md)、[主动学习报告](studies/active_learning/odh_gradient_al_smoke/DEVELOPMENT_REPORT.md)、[批判性检查](studies/active_learning/odh_gradient_al_smoke/CRITICAL_REVIEW.md)。

研究文档按用途保留五份，先读需要的那一份：

| 你想了解什么 | 文档 |
| --- | --- |
| 当前进度、已完成范围和后续阶段 | [项目路线图](docs/research/PROJECT_ROADMAP.md) |
| 原论文和代码差异、基线协议与实测结果 | [ODH 基线与原论文审计](docs/research/ODH_BASELINE_REPRODUCTION.md) |
| 主动学习的数据边界、训练和选样规则 | [主动学习协议](docs/research/HPLC_AL_PROTOCOL.md) |
| 运行前问题如何解决、测试覆盖与剩余限制 | [主动学习综合审计](docs/research/HPLC_AL_IMPLEMENTATION_AUDIT.md) |
| 每一步何时执行、当时的决策和证据 | [实验日志](docs/research/EXPERIMENT_LOG.md) |

日志中的状态描述对应记录当时；当前状态以路线图为准。原运行前审查已并入主动学习审计，原论文审计已并入基线报告。

## 目录用途

| 目录 | 内容与维护方式 |
| --- | --- |
| `src/hplc_al/` | 当前主动学习实现；日常开发主要修改这里 |
| `scripts/` | 数据审计、缓存获取、基线复现和主动学习命令入口 |
| `tests/hplc_al/` | 选样、梯度、数据边界和冻结结果保护的回归测试 |
| `src/reproduction/` | 原始模型的运行适配层 |
| `code/` | 原论文模型与预处理代码，作为复现依据保留 |
| `dataset/` | 原始色谱数据 |
| `application/`、`utils/` | 原作者应用与资源 |
| `docs/research/` | 协议、审计、研究决策与实验记录 |
| `artifacts/reproduction/` | 基线结果、有效的 v4 smoke 结果、数据审计和本地图缓存 |
| `studies/active_learning/odh_gradient_al_smoke/` | 已冻结的主动学习实验：报告、选样记录、预测、检查点和核验清单 |

主动学习模块建议按以下顺序阅读：

1. `data.py`：不含标签的分子图与批处理。
2. `protocol.py`：数据角色、标签访问边界和选样状态转换。
3. `training.py`：训练、验证与检查点。
4. `gradient.py`：模型梯度与 CountSketch 特征。
5. `acquisition.py`：Random、LCMD、MaxDet 选样。
6. `runner.py`：阶段门槛与实验流程；`reporting.py`：指标和报告。

`common.py` 集中管理路径、固定配置、哈希与文件操作。

## 本地检查

在仓库根目录，使用现有的主动学习环境：

```bash
# 检查当前代码，不写入已完成实验的测试记录
.conda-hplc-al/bin/python -B -m pytest -q

# 只读核验已完成实验及其历史源码
.conda-hplc-al/bin/python -B scripts/run_hplc_al_smoke.py verify
```

环境依赖见 [主动学习依赖](requirements-hplc-al.txt) 和 [基线复现依赖](requirements-reproduction.txt)。
`.conda-hplc-al/` 与 `.venv-reproduction/` 是两个现有运行环境，不属于实验结果或源码。
代码格式与静态检查规则集中在 [pyproject.toml](pyproject.toml)，范围限定为主动学习代码和测试。

实验目录中的 `protocol.json`、冻结清单及结果文件保留原样。
`frozen_source.zip` 保存运行实验时的 17 个源码与测试文件，逐项匹配原协议中的 SHA-256；当前源码整理后，通过该快照核验历史实现。
完成后的 `run` / `report` 仅核验已有结果，`prepare` / `test` / `audit` / `freeze` 会拒绝覆盖完成实验。实验目录 README 中的阶段命令是历史运行记录；日常检查使用上面的命令。

系统缓存、临时终端日志和过时 smoke 目录已清理；预测、检查点、审计记录和冻结清单是复核依据，保留在各自实验目录中。

## 原论文与原始资源

H. Xu, J. Lin, D. Zhang, F. Mo. *Retention time prediction for chromatographic enantioseparation by quantile geometry-enhanced graph neural network*. Nature Communications 14 (2023). [DOI: 10.1038/s41467-023-38853-3](https://doi.org/10.1038/s41467-023-38853-3).

- [原作者在线保留时间预测应用](https://huggingface.co/spaces/woshixuhao/Chromatographic_Enantioseparation)
- [原作者 TLC Rf 预测应用](https://huggingface.co/spaces/woshixuhao/Rf_prediction)
- [原作者完整程序下载](https://drive.google.com/file/d/1kzM2_pG-Ob_hNQ7ga2n6Ds2FXDeAppWx/view?usp=share_link)
- [原论文与仓库审计](docs/research/ODH_BASELINE_REPRODUCTION.md#original-paper-and-repository-audit)

原作者声明的环境为 Python 3.7、RDKit 2020.09.1.0、PyTorch 1.11.0、PyG 2.0.4、Mordred 1.2.0、pandas 1.3.5。当前本地复现使用独立环境，具体版本以依赖文件和实验环境记录为准。
