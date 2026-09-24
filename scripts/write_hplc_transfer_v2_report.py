"""Render the verified transfer-v2 results as a reviewable Markdown report."""
import csv
from pathlib import Path
from hplc_al.common import ROOT, read_json
from hplc_al.transfer_v2 import ACQUISITIONS, BUDGETS

p=ROOT/'studies/active_learning/odh_gradient_al_transfer_v2'
r=p/'results'
s=read_json(r/'summary.json')
names={'random':'Random','raw_gradient_lcmd':'Raw-LCMD','unit_gradient_lcmd':'Unit-LCMD','raw_gradient_maxdet':'Raw-MaxDet','unit_gradient_maxdet':'Unit-MaxDet'}
lines=['# HPLC transfer_v2：受限短程实验报告','',
'状态：**COMPLETE / STOP**。本实验为 initialization seed 525、5 methods、3 acquisitions；budget 333/365/397/429。未启动 2 seeds × 6 rounds 或其他长程实验。','',
'## Runtime 与 duration gate','',
'使用项目 `.conda-hplc-al/bin/python`，PyTorch 2.2.2、PyG 2.5.3、NumPy 1.23.5，CPU 4 threads，与历史 reproduction 的核心计算依赖一致。Python 小版本为 3.11.16（历史 baseline 为 3.11.14）；不声称解释器环境逐字节相同。','',
'协议 pytest：13 passed；HPLC 全套：78 passed。实际图 batch 检查覆盖 seeds 525/1525 各 500 epochs：same seed + epoch permutation 相同；相邻 epoch 不同；每个 L333 ID 恰好出现一次；图 ID 和 synthetic ID-coded truth 完全对齐；U0/validation/test ID 均未进入 L333 训练排列。每 epoch 恰有 2 steps，实际 batch sizes `[256,77]`。','',
'原始完整 HPLC loss = q10 pinball + central MSE + q90 pinball + 两个 quantile order penalty + dead-time penalty。训练固定 scratch / fresh Adam、batch 256、deterministic_each_epoch、lr 0.001、weight decay 1e-5、max epochs 500、patience 100；checkpoint 仅由 validation central MSE 选择。','']


def md_table(headers, rows):
    lines.append('| '+' | '.join(map(str, headers))+' |')
    lines.append('| '+' | '.join(['---']*len(headers))+' |')
    for row in rows:
        lines.append('| '+' | '.join(str(v) for v in row)+' |')
    lines.append('')


def f(v): return f'{float(v):.6f}'

rows=[]
for seed in [525,1525]:
    d=read_json(p/f'duration_smoke/seed_{seed}/duration_record.json')
    b=d['best_validation_metrics']; final=d['final_validation_metrics']
    rows.append([seed,d['best_epoch'],d['stopped_epoch'],d['best_optimizer_step'],*[f(b[k]) for k in ['rmse','mae','r2','nrmse']],f(d['seconds']),d['stopping_reason']])
md_table(['Seed','Best epoch','Stopped epoch','Best step','Best RMSE','MAE','R²','NRMSE','Runtime s','Stop'],rows)
rows=[]
for seed in [525,1525]:
    d=read_json(p/f'duration_smoke/seed_{seed}/duration_record.json')
    rows.append([seed,*[f(d['final_validation_metrics'][k]) for k in ['rmse','mae','r2','nrmse']]])
md_table(['Seed','Final RMSE','Final MAE','Final R²','Final NRMSE'],rows)
lines.extend(['两组均满足 patience，且 best epoch ≤400；training protocol 已冻结。逐 epoch batch sizes 与 optimizer-step 证据见各 duration_smoke/seed_*/attempt_000/training_curve.csv。','',
'## 实现审计与中断恢复','',
'`phi_raw = 512D CountSketch(full-network central gradient)`；`phi_unit = phi_raw / max(||phi_raw||₂, 1e-12)`。Unit 方法的准确名称是 **unit-normalized sketched gradient**。梯度提取为 eval 模式，中心输出列为 1，未使用 U truth。','',
'五方法在 L333 共享 duration seed 525 的 exact same predictor checkpoint、同一个 raw 512D gradient bank，Unit 仅由该 raw bank row-wise normalization 得到。首轮 acquisition 后，按各自 labeled ID sets scratch retrain。每个后续预算独立 fresh Adam，无 warm start。','',
'已有中断尝试保存在 interrupted_attempt_20260923。旧 AL 入口缺少 4-thread 环境固定，默认环境为 10 threads；实际旧 fit 线程数未记录。其 L333 checkpoint 与相同输入/初始化的 4-thread duration fit 不同，不能将旧 fit 混入本次配对比较。复用已验证的 duration checkpoint，重新进行统一环境下的后续 fits。旧 Random L365 后的 None gradient audit 引用错误已修复；acquisition selector、随机序列算法和 scientific hyperparameters 均未改变。','',
'冻结前已验证全部 20 个 method-budget 状态、checkpoint、validation/test-X predictions、selection 的文件哈希；test truth 仅在全局 freeze 后统一 reveal 一次。末预算无后续 acquisition，selection 为空，state 保留。预测文件按显式 IDs 合并存储 validation + test-X。','',
'## Learning curves','',
'![Validation and test learning curves](results/learning_curves.png)',''])
for split in ['validation','test']:
    lines.extend([f'### {split} central RMSE',''])
    rows=[]
    for method in ACQUISITIONS:
        rows.append([names[method]]+[f(next(x['rmse'] for x in s['curves'] if x['method']==method and x['split']==split and x['budget']==b)) for b in BUDGETS])
    md_table(['Method',*BUDGETS],rows)
lines.extend(['### 所有 budget 的完整指标','',f"NRMSE = RMSE / frozen L333 population SD，SD = {s['L0_population_std']:.9f}。所有方法和预算共用此分母。",''])
for split in ['validation','test']:
    lines.extend([f'#### {split}',''])
    md_table(['Method','Budget','RMSE','MAE','R²','NRMSE'],[[names[x['method']],x['budget'],*[f(x[k]) for k in ['rmse','mae','r2','nrmse']]] for x in s['curves'] if x['split']==split])
lines.extend(['## Normalized AULC 333–429','',
'使用梯形积分。budget-normalized AULC RMSE = ∫RMSE dB /96；normalized AULC NRMSE = ∫NRMSE dB /96。二者越低越好。另给出相对 Random 的 AULC 比值；1 表示与 Random 相同。',''])
md_table(['Method','Split','Mean RMSE AULC','Normalized NRMSE AULC','Ratio / Random'],[[names[x['method']],x['split'],f(x['budget_normalized_AULC_RMSE']),f(x['normalized_AULC_NRMSE']),f(x['AULC_ratio_to_random'])] for x in s['aulc']])
lines.extend(['## Paired comparisons','',
'差值为 A − B；RMSE/MAE/NRMSE 负值有利于 A，R² 正值有利于 A。仅一条 seed trajectory，不能进行跨 seed 稳定性或显著性判断。',''])
pairs=[('raw_gradient_lcmd','unit_gradient_lcmd'),('raw_gradient_maxdet','unit_gradient_maxdet')]+[(m,'random') for m in ACQUISITIONS if m!='random']
for split in ['validation','test']:
    lines.extend([f'### {split} paired RMSE / AULC',''])
    rows=[]
    for a,b in pairs:
        delta=[]
        for budget in BUDGETS:
            av=next(x['rmse'] for x in s['curves'] if x['method']==a and x['split']==split and x['budget']==budget)
            bv=next(x['rmse'] for x in s['curves'] if x['method']==b and x['split']==split and x['budget']==budget)
            delta.append(f(av-bv))
        av=next(x['normalized_AULC_NRMSE'] for x in s['aulc'] if x['method']==a and x['split']==split)
        bv=next(x['normalized_AULC_NRMSE'] for x in s['aulc'] if x['method']==b and x['split']==split)
        rows.append([names[a]+' − '+names[b],*delta,f(av-bv)])
    md_table(['A − B',*BUDGETS,'Δ normalized AULC'],rows)
lines.extend(['全部 RMSE/MAE/R²/NRMSE paired differences 见 [paired_differences.csv](results/paired_differences.csv)。','',
'## Acquisition geometry','',
'Norm percentile/top-decile 以当轮 U pool 为参照；CSV 也提供 outer-train 参照。有效秩采用中心化 selected feature matrix 的 singular-value entropy：`exp(-Σ p log p)`，`p = singular value / sum`；与 algebraic matrix rank 分开记录。距离在 raw、unit 空间分别计算。Random 的梯度只用于诊断，不影响 selection。',''])
g=list(csv.DictReader((r/'acquisition_geometry.csv').open()))
md_table(['Method','From B','Norm pct','Top 10% frac','Raw eff.rank','Unit eff.rank','Raw pair dist','Unit pair dist','Raw nearest-L','Unit nearest-L'],[[names[x['method']],x['from_budget'],*[f(x[k]) for k in ['selected_raw_norm_percentile_U','selected_top_decile_fraction_U','raw_effective_rank','unit_effective_rank','raw_pairwise_distance','unit_pairwise_distance','raw_nearest_L_distance','unit_nearest_L_distance']]] for x in g])
o=list(csv.DictReader((r/'selection_overlap.csv').open()))
md_table(['Pair','From B','Overlap /32','Jaccard','Cumulative acquired Jaccard'],[[names[x['method_A']]+' / '+names[x['method_B']],x['from_budget'],x['selected_overlap'],f(x['selected_jaccard']),f(x['cumulative_acquired_jaccard'])] for x in o if x['method_B']!='random'])
lines.extend(['Round 0 的 Raw/Unit 比较共享模型和候选集，因此可直接比较 normalization 对 geometry 的影响。后续轮模型、L/U 集均已分叉，几何差异同时包含 trajectory 的影响。','',
'## Compute',''])
md_table(['Quantity','Value'],[[k,f(v) if isinstance(v,float) else v] for k,v in s['compute'].items()])
c=list(csv.DictReader((r/'compute_by_method.csv').open()))
md_table(['Method','New training s','Gradient s excl shared','Selector s'],[[names[x['method']],f(x['new_training_seconds']),f(x['gradient_seconds_excluding_shared']),f(x['selector_seconds'])] for x in c])
lines.extend(['共享 L333 fit 的成本只计一次；其与 duration seed 525 为同一物理 fit。有效 AL 16 unique fits，另有 duration seed 1525；中断尝试的额外 fits/gradient pass 单独报告，不混入有效曲线。末预算无需 gradient extraction；Random rounds 1/2 两次 gradient passes 为 geometry diagnostics。','',
'所有 AL fit 的 best/stopped epochs、optimizer steps、actual batch sizes 和 stopping reason 见 [training_duration_all_fits.csv](results/training_duration_all_fits.csv)。','',
'## 解释边界','',
'本次是单 seed、三个 acquisition 的初步比较；预算间表现一致只能支持短程迹象，不能证明跨初始化或跨 split 稳定改善。历史 HPLC test 曾用于早期工程研究，本次 reveal firewall 防止当前 acquisition/checkpoint 读取 test truth，但该 test 不是从未使用过的新外部验证集。','',
'## 决策','',
f"修正后的 training protocol 在这个单 seed、三个 acquisition 的短程实验中出现有限改善迹象：Raw-LCMD 的 test normalized AULC 为 {next(x['normalized_AULC_NRMSE'] for x in s['aulc'] if x['method']=='raw_gradient_lcmd' and x['split']=='test'):.6f}，Random 为 {next(x['normalized_AULC_NRMSE'] for x in s['aulc'] if x['method']=='random' and x['split']=='test'):.6f}；但这不是跨 seed 的稳定性证据。",'',
'Unit normalization 没有改善 label efficiency；它改变了 acquisition geometry，但本次没有带来预测改善。Raw-LCMD 比 MaxDet 更值得进入下一阶段，MaxDet 两个变体都没有超过 Random。当前证据不足以直接授权 2 seeds × 6 rounds；若推进，应视为验证性复现。','',
'## Evidence index','',
'- [Runtime preflight](runtime_gate/runtime_preflight.json)','- [Protocol pytest](runtime_gate/test_transfer_v2_protocol_resume.log)','- [Full HPLC pytest](runtime_gate/test_hplc_al_resume.log)','- [Duration smoke](duration_smoke/summary.json)','- [Frozen training protocol](training_protocol_frozen.json)','- [Implementation audit](implementation_audit.json)','- [Global pre-test freeze](global_pre_test_freeze.json)','- [Single test reveal](test_reveal.json)','- [Result summary](results/summary.json)','- [Learning curve CSV](results/learning_curves.csv)','- [Geometry CSV](results/acquisition_geometry.csv)','- [Compute JSON](results/compute.json)',''])
(p/'REPORT.md').write_text('\n'.join(lines))
print(p/'REPORT.md')
