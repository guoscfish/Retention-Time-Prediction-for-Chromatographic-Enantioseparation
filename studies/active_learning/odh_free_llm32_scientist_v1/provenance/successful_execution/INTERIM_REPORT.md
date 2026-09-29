# Free-LLM32 阶段一报告：seed 1525，L333 → L397


本报告仅使用 validation；只完成两次 acquisition。所有科学机制描述均为 LLM 提出的假设，不是真实机制结论。


## 实现与边界

独立 `free_llm32_scientist`：pending=0，由模型决定全部32点，保留旧16+16实现。代码审计见 AUDIT.md。

输入包括 canonical isomeric SMILES、CIP、Murcko scaffold、官能团、理化描述符、IPA/flow、q10/center/q90/width和原始梯度空间最近已标记点距离百分位。center由MSE训练，不声称为q50；width不是校准epistemic uncertainty，coverage不是真误差。无任何选择配额。

代码确定性拼接本trajectory最近三轮原始批次、真实响应、测量前冻结预测及误差、最高误差观测、历史假设状态和未决问题。每轮新上下文；同轮relay保留完整已发送页。假设更新由同一个实验规划调用明确给出，不由摘要LLM生成。

selector只接收显式packet与allowlist查询。selection → SHA256 seal → Git commit → RestrictedLabelStore.commit_selection → reveal → feedback → scratch fit。逻辑标签边界不是OS隔离；使用单独冻结的Codex CLI transport revision 2、ephemeral新上下文；关闭技能、外部工具、历史规则，审计全部事件并强制native calls=0。原Responses认证失败版本保留；CLI不暴露实际served model快照版本，也不宣称验证其底层wire store标志。validation仅供训练checkpoint选择与独立评价。

冻结预算：{'query_budget': 24, 'view_budget': 480, 'call_budget': 28, 'page_size': 24}。模型 `gpt-6-sol`，effort `high`；provider由当前配置读取：`active OpenAI login through local Codex CLI`。具体served model/version与usage见每轮llm/turn*.json。

resume验证协议、代码、复用文件、请求哈希、selection/feedback/fit文件；已有selection不调用LLM，完成fit不重训；无回执但已有请求intent则fail closed。manifest逐步记录状态，complete须绑定两轮完成的manifest。

冻结L333 population SD = **8.879240547557**。训练器内动态分母NRMSE不用于本报告。基线split/L333/checkpoint/seed/training代码及配置/预算已逐项验证并冻结复用hash；Random32和Raw Gradient-LCMD32均未重训。


## Validation 性能

| Method | L | RMSE | MAE | R² | fixed NRMSE |

|---|---:|---:|---:|---:|---:|

| free_llm32_scientist | 333 | 8.043997 | 5.287642 | 0.163640 | 0.905933 |

| free_llm32_scientist | 365 | 8.052217 | 5.415389 | 0.161930 | 0.906859 |

| free_llm32_scientist | 397 | 8.022899 | 5.152679 | 0.168021 | 0.903557 |

| random | 333 | 8.043997 | 5.287642 | 0.163640 | 0.905933 |

| random | 365 | 8.113203 | 5.082832 | 0.149187 | 0.913727 |

| random | 397 | 7.930685 | 4.954846 | 0.187037 | 0.893172 |

| raw_gradient_lcmd | 333 | 8.043997 | 5.287642 | 0.163640 | 0.905933 |

| raw_gradient_lcmd | 365 | 7.748384 | 5.070312 | 0.223982 | 0.872640 |

| raw_gradient_lcmd | 397 | 7.840739 | 4.818740 | 0.205372 | 0.883042 |


**Exploratory partial AULC 333–397**：梯形积分，仅此区间；mean=area/64。不可与333–429/525完整区间混用。

| Method | NRMSE area | Mean NRMSE |

|---|---:|---:|

| free_llm32_scientist | 57.971317 | 0.905802 |

| random | 58.024942 | 0.906640 |

| raw_gradient_lcmd | 56.548087 | 0.883564 |


## 逐轮科学行为与选点诊断


### Round 0: L333 → L365

调用5次、query 24次、候选已浏览224点、选择32点。

**Batch strategy:** Use matched stereochemical and substitution contrasts around observed long-retention examples, plus condition and prediction-extreme probes. Favor interpretable contrasts over geometric coverage alone.

**Rationale:** This is round zero: the L333 responses inform the design, but there are no acquired premeasurement errors. Observed nitro diaryls and arylsulfonyl pyrrolidines have substantial retention, while the frozen model gives related candidates much lower centers. Matched pairs test whether those differences transfer across stereochemistry and substitution. Boronate flow contrasts and very low silicon-containing predictions probe distinct model behavior; high predicted centers are checked with enantiomer pairs. Neither q_width nor coverage is treated as calibrated error.

**Feedback interpretation (LLM):** No acquired-experiment feedback or prior hypotheses exist in round zero. Initial L333 responses are measured evidence, but provide no premeasurement signed errors. Their long-retention matched structures motivated targeted contrasts; directional claims remain hypotheses, not established mechanisms.

新hypotheses=8；previous updates=0。

- 新假设 `h1`：Sulfur-linked ester retention and enantiomer separation may change with aryl substitution in ways the frozen model does not transfer consistently. Alternative: The model's substitution and stereochemical differences transfer adequately; observed pyridyl retention is specific to that different ring.；预期误差符号=0；关联实验=['r358cbe3ca94e', 'raa8a2a5782ea', 'raedf65f32e8d', 'rd71ed92f0e1b', 're330f8c183c1', 'r0e1dfb89171c']；实际冻结signed errors=[9.3968, -1.6894, 8.9256, 9.0643, 10.5504, 1.0581]。这是方向性观测诊断，不自动证明机制。

- 新假设 `h2`：The frozen model may underpredict retention across related nitro diaryls, as suggested by long observed para-chloro and CF3 examples relative to centers of their candidate analogs. Alternative: The long responses are confined to the observed configurations or substituents; the candidate centers are appropriate.；预期误差符号=-1；关联实验=['ra67208785bd2', 'rc3147b13be2c', 'r6a6bf9e64c81', 'rc41a7b79f13d', 'r597fa3381226', 'r5db6e7674d48']；实际冻结signed errors=[-14.1049, -7.0465, 0.5042, -19.5669, -19.7265, -16.1255]。这是方向性观测诊断，不自动证明机制。

- 新假设 `h3`：Aryl position and configuration may interact strongly in sulfonyl pyrrolidine retention; the observed para-chloro S and ortho-chloro R responses differ. Alternative: The observed difference is explained by ordinary substitution effects already captured by the model.；预期误差符号=0；关联实验=['r5cfff95afb07', 'r7b1788cb96a1', 'r7cb3b65769c0', 'rdeea40c9bf86']；实际冻结signed errors=[-16.0501, -10.6542, -22.1971, -22.2054]。这是方向性观测诊断，不自动证明机制。

- 新假设 `h4`：The ester-containing ring family may exhibit substitution- and stereochemistry-dependent retention not represented by a single scaffold trend. Alternative: Its observed response differences reflect structures and recorded conditions that the current model already distinguishes.；预期误差符号=0；关联实验=['r94f483736449', 'rc6c7725e2474', 'rb366c0fa39f9', 'r4264ec388c7c', 'r6c7738f4f875']；实际冻结signed errors=[-7.2244, -9.7899, -7.9747, 6.4383, -14.5564]。这是方向性观测诊断，不自动证明机制。

- 新假设 `h5`：Very low boronate central predictions may reflect a representation blind spot, while RTv for an exact identity may remain similar across the two recorded flows. Alternative: The low centers are accurate and the small predicted flow difference is appropriate.；预期误差符号=0；关联实验=['r630e85ec162d', 'rb64de4da15c2', 'rc0f7f64c2909', 'rd5d96a5fd1e6']；实际冻结signed errors=[-1.1786, -0.4508, 1.3348, 1.265]。这是方向性观测诊断，不自动证明机制。

- 新假设 `h6`：The high predicted centers for the heterocyclic enantiomer pair may be a shared structural extrapolation rather than accurate retention. Alternative: Both high centers are accurate, with only the predicted modest stereochemical difference.；预期误差符号=0；关联实验=['rffbdd082b6a0', 'r8944df48a1b8']；实际冻结signed errors=[13.1093, 19.2109]。这是方向性观测诊断，不自动证明机制。

- 新假设 `h7`：The polar triazole diol pair may have a shared prediction error or an unmodeled enantiomeric difference. Alternative: The model captures both the absolute retention and small predicted enantiomer difference.；预期误差符号=0；关联实验=['re86bebb15fd8', 'ra67d23337bab']；实际冻结signed errors=[3.4088, 10.1643]。这是方向性观测诊断，不自动证明机制。

- 新假设 `h8`：Some prediction extremes may be underestimates: silicon-containing amides have near-zero centers, and a dibrominated cyclic alcohol has a substantially lower center than a related observed alcohol's response. Alternative: The silicon-containing compounds genuinely elute near their predicted centers, and the alcohol structural difference explains its lower center.；预期误差符号=-1；关联实验=['r416af6febd58', 'r1e04118171f1', 'rbb54e2582749']；实际冻结signed errors=[-3.4207, -3.9046, -21.763]。这是方向性观测诊断，不自动证明机制。

同状态LCMD32 overlap={'intersection': 2, 'jaccard': 0.03225806451612903}；历史独立LCMD轨迹同轮overlap={'intersection': 2, 'jaccard': 0.03225806451612903}（round1模型和候选池已分叉）。

对照pair数={'stereo_contrast': 11, 'same_scaffold_contrast': 50, 'exact_identity_condition_contrast': 1, 'near_identity_condition_contrast': 0}；参与点数={'stereo_contrast': 22, 'same_scaffold_contrast': 30, 'exact_identity_condition_contrast': 2, 'near_identity_condition_contrast': 0}。stereo contrast不自动等同严格enantiomer pair；pair可重叠，不相加当作32点分配。

高coverage(≥0.9)=21/32；高q-width(本U top10%)=14/32；prediction extremes(本U两端各10%)=21/32。

选点coverage百分位分布={'min': 0.09152237463458511, 'q25': 0.4627276815830897, 'median': 0.9612098043624915, 'q75': 0.9834157859230942, 'max': 0.9977512930065212}；q-width分布={'min': 7.04958963394165, 'q25': 17.515366077423096, 'median': 19.316720962524414, 'q75': 24.74617576599121, 'max': 81.62875366210938}。

不同isomeric identities=31，scaffolds(含空组)=10，空scaffold点=0。批内Morgan相似度分布={'min': 0.046153846153846156, 'q25': 0.09807604735883424, 'median': 0.1323529411764706, 'q75': 0.19402985074626866, 'max': 1.0}。

未决问题：['Do the observed long-retention nitro diaryl and sulfonyl pyrrolidine examples transfer across configuration and substitution?', 'Are the boronate and silicon-containing low central predictions genuine retention estimates or model failures?', 'How much do exact stereochemistry and the recorded flow change RTv within matched identities?', 'Do high predicted centers for the heterocyclic and triazole pairs represent real retention?']


### Round 1: L365 → L397

调用5次、query 24次、候选已浏览184点、选择32点。

**Batch strategy:** Use matched substitutions and enantiomer pairs to locate where the newly observed frozen-model errors transfer. Retain a few exact-identity condition and structurally distinct probes; do not select by coverage or q_width alone.

**Rationale:** Round-one feedback changes the emphasis: nitro diaryls and arylsulfonyl pyrrolidines show substantial underprediction, sulfur-linked esters show configuration- and substitution-dependent errors, and the high-center heterocyclic pair is overpredicted. The boronate low centers and flow contrast were comparatively accurate, so only a focused IPA contrast remains there. All signed errors cited are frozen premeasurement center minus measured RTv, not post-retraining errors. Matched measurements can distinguish transferable local corrections from stereochemical, substitution, and recorded-condition effects without treating any proposed recognition mechanism as established.

**Feedback interpretation (LLM):** The acquired observations identify opposite error directions in different local regions: marked negative errors in nitro diaryls, sulfonyl pyrrolidines and a dibromo alcohol, but positive errors in the high-center heterocyclic pair and several sulfur-linked esters. Enantiomer contrasts sometimes differ much more than frozen centers predict. Boronate feedback argues against spending this batch on another broad low-center search; its remaining useful question is a controlled IPA contrast. The batch therefore tests boundaries of the observed failures rather than assuming scaffold labels, gradient-sketch coverage, or q_width predict error.

新hypotheses=8；previous updates=8。

- 旧假设 `h1` → **supported**：R members have positive errors near 9–11 mL, whereas S errors range from about -1.7 to +9.1 mL; observed retention and enantiomer differences vary with substitution. This supports an error-pattern claim, not a specific recognition mechanism. Supporting=['r358cbe3ca94e', 'raa8a2a5782ea', 'raedf65f32e8d', 'rd71ed92f0e1b', 're330f8c183c1', 'r0e1dfb89171c']; contradicting=[]。

- 旧假设 `h2` → **supported**：Five of six acquired nitro diaryls have negative errors of about 7–20 mL, but the R para-fluoro analog is nearly exact. The family-wide directional claim needs this clear configuration-dependent qualification. Supporting=['ra67208785bd2', 'rc3147b13be2c', 'rc41a7b79f13d', 'r597fa3381226', 'r5db6e7674d48']; contradicting=['r6a6bf9e64c81']。

- 旧假设 `h3` → **weakened**：All four acquired pyrrolidines are underpredicted, and para versus ortho chloro differs, but the new matched data do not isolate a strong position-by-configuration interaction: para-fluoro errors are almost identical across enantiomers. Supporting=['r5cfff95afb07', 'r7b1788cb96a1', 'r7cb3b65769c0', 'rdeea40c9bf86']; contradicting=[]。

- 旧假设 `h4` → **supported**：Several acquired ester-containing ring examples are underpredicted, while the fluorinated example at different IPA and flow is overpredicted. Structural and condition differences prevent assigning one cause, but a single error trend is inadequate. Supporting=['r94f483736449', 'rc6c7725e2474', 'rb366c0fa39f9', 'r6c7738f4f875']; contradicting=['r4264ec388c7c']。

- 旧假设 `h5` → **weakened**：The exact-identity flow pair both measured 3.5 mL, supporting near-invariant RTv across those flows. Low boronate centers were close to measured values, weakening the proposed major blind spot. Supporting=['rc0f7f64c2909', 'rd5d96a5fd1e6']; contradicting=['r630e85ec162d', 'rb64de4da15c2']。

- 旧假设 `h6` → **supported**：Both high centers overpredicted measured RTv by 13.1 and 19.2 mL; the measured enantiomer gap also exceeded the frozen center gap. This establishes these errors, not their proposed structural cause. Supporting=['rffbdd082b6a0', 'r8944df48a1b8']; contradicting=[]。

- 旧假设 `h7` → **supported**：Both triazole diols were overpredicted, especially the R member, and their measured 7.3 mL enantiomer gap exceeded the roughly 0.5 mL center gap. Supporting=['re86bebb15fd8', 'ra67d23337bab']; contradicting=[]。

- 旧假设 `h8` → **supported**：Both silicon-containing centers underpredicted by about 3–4 mL, and the dibromo alcohol by 21.8 mL. The silicon discrepancy is modest compared with its output span, which is not calibrated epistemic uncertainty. Supporting=['r416af6febd58', 'r1e04118171f1', 'rbb54e2582749']; contradicting=[]。

- 新假设 `h9`：Sulfur-linked ester prediction errors depend on both configuration and which aryl ring is substituted; the measured R overprediction need not transfer uniformly to S. Alternative: The new substitutions have adequately predicted retention, and the measured differences are local to the tested compounds.；预期误差符号=0；关联实验=['rce6cd6331401', 'r5f3d645adffd', 'r3807ee109050', 'r553d5d71c7ce', 'r08f439af24d5', 'rb701437e5343']；实际冻结signed errors=[4.596, 7.6043, 3.7764, -5.5429, -1.7868, -6.6686]。这是方向性观测诊断，不自动证明机制。

- 新假设 `h10`：Nitro-diaryl underprediction transfers across some substitutions, but the near-zero R para-fluoro error indicates a configuration- and position-dependent boundary. Alternative: The measured negative errors are confined to those exact substitutions and configurations.；预期误差符号=0；关联实验=['rdd91211c250f', 'r29a8d4957353', 'r2cad56a44833', 'r3c9be3edf6e2', 'r8a4a70f02925', 'rcdca2a3008cc']；实际冻结signed errors=[-16.3005, -25.9724, -1.8442, 14.101, -3.1531, 1.3725]。这是方向性观测诊断，不自动证明机制。

- 新假设 `h11`：Aryl substitution within the sulfonyl pyrrolidines may produce broadly negative frozen errors, with the error size varying by substituent, position, or condition. Alternative: The underprediction is confined to the measured chloro and para-fluoro structures.；预期误差符号=-1；关联实验=['r3d924c6fb0ed', 'r5d08c2985ad8', 'rc2171317ddca', 'rbcd87f5003d1', 'r81c84a17d9c2', 'rb6dcc30cdc5c', 'r12a02897f3f7']；实际冻结signed errors=[-18.426, -20.1268, 6.1158, 4.0016, -21.1327, -3.0133, -6.3361]。这是方向性观测诊断，不自动证明机制。

- 新假设 `h12`：The ester-containing ring errors may change with aryl substitution and recorded conditions rather than admit one family-wide correction. Alternative: The new structures are well predicted, and prior error variation is explained by their already represented differences.；预期误差符号=0；关联实验=['r003e6977d8ff', 'r27528605a1f4']；实际冻结signed errors=[-0.8618, 2.5741]。这是方向性观测诊断，不自动证明机制。

- 新假设 `h13`：Boronate RTv may be sensitive to the recorded IPA change even though the measured exact-identity flow contrast was invariant. Alternative: The IPA contrast is negligible or already represented by the model.；预期误差符号=0；关联实验=['r35456a197b3b', 're0e698813d1a']；实际冻结signed errors=[-0.2883, 0.3058]。这是方向性观测诊断，不自动证明机制。

- 新假设 `h14`：The measured high-center isoxazoline overprediction may transfer to related heteroaryl substitutions, while its enantiomer gap may remain structure-dependent. Alternative: Replacing the heteroaryl ring removes the overprediction and the current centers are adequate.；预期误差符号=0；关联实验=['rd8ee1e150fa9', 'rfd3176f1b7dc', 'r48ab24c67ef1', 'r9c61bbab7ae1']；实际冻结signed errors=[10.101, 8.4829, 3.9348, 7.4817]。这是方向性观测诊断，不自动证明机制。

- 新假设 `h15`：The dibromo cyclic alcohol's large negative error may be substituent-specific rather than a universal cyclic-alcohol bias. Alternative: The frozen model underpredicts retention throughout these related cyclic alcohols.；预期误差符号=0；关联实验=['r31b886e9f456', 'r4e9a074e1a9c', 'ra27352464572']；实际冻结signed errors=[-13.2438, -9.6362, 1.7684]。这是方向性观测诊断，不自动证明机制。

- 新假设 `h16`：The observed triazole-diol enantiomer gap and positive errors may change when a hydroxymethyl group is replaced by a ketone. Alternative: The new ketone pair has the small enantiomer difference and retention predicted by the current model.；预期误差符号=0；关联实验=['r1211dbd23fb3', 'r797d51145a8f']；实际冻结signed errors=[-5.1941, 3.6133]。这是方向性观测诊断，不自动证明机制。

同状态LCMD32 overlap={'intersection': 0, 'jaccard': 0.0}；历史独立LCMD轨迹同轮overlap={'intersection': 0, 'jaccard': 0.0}（round1模型和候选池已分叉）。

对照pair数={'stereo_contrast': 12, 'same_scaffold_contrast': 58, 'exact_identity_condition_contrast': 1, 'near_identity_condition_contrast': 0}；参与点数={'stereo_contrast': 24, 'same_scaffold_contrast': 30, 'exact_identity_condition_contrast': 2, 'near_identity_condition_contrast': 0}。stereo contrast不自动等同严格enantiomer pair；pair可重叠，不相加当作32点分配。

高coverage(≥0.9)=5/32；高q-width(本U top10%)=12/32；prediction extremes(本U两端各10%)=20/32。

选点coverage百分位分布={'min': 0.13267371261524624, 'q25': 0.2742860355295705, 'median': 0.3241511131099618, 'q75': 0.7765347425230492, 'max': 0.9718911625815156}；q-width分布={'min': 15.091187477111816, 'q25': 16.70613431930542, 'median': 18.80739116668701, 'q75': 24.11796474456787, 'max': 32.6993408203125}。

不同isomeric identities=31，scaffolds(含空组)=9，空scaffold点=0。批内Morgan相似度分布={'min': 0.05405405405405406, 'q25': 0.1, 'median': 0.13636363636363635, 'q75': 0.20634920634920634, 'max': 1.0}。

未决问题：['Which nitro-diaryl substitutions and configurations share the underprediction, and why is R para-fluoro nearly exact?', 'How far does pyrrolidine underprediction transfer beyond the measured para-halogen and ortho-chloro examples?', 'Are sulfur-linked ester errors governed chiefly by configuration, substituent position, or an interaction?', 'Does the isoxazoline overprediction transfer after heteroaryl replacement?', 'Is the dibromo alcohol discrepancy specific to bromination?', 'How do recorded IPA changes affect RTv for an exact boronate identity?']


## 解释与下一步

阶段一只能提供单seed、两个批次的探索信号，不能证明LLM化学知识的因果增益。既有validation被用于checkpoint选择，且此cohort存在历史研究使用；不是独立外部验证。

Round1选择前可观察Round0实验反馈并修订假设；Round1实验回来后只记录实际证据，不启动Round2 LLM，因此没有第二批反馈后的LLM修订。必须保留这一闭环长度限制。


详细判断与闭环实例见下方人工审阅补充；所有32个理由、hypothesis links及完整响应见每轮selection.json与feedback.json。


候选后续（仅建议，不执行）：继续Free-LLM32到L429；用第二seed检验可重复性；以相同protocol单独运行16+16补充型策略。


保护校验：566个历史artifact保持原hash。新trajectory无test truth访问；无round2、第二seed、masked或hybrid运行。
