"""Read-only aggregation of sealed results; never trains or reads label stores."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
S = ROOT / 'studies/active_learning'
inputs = {}
def load(path):
    path = ROOT / path if not isinstance(path, Path) else path
    inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.suffix == '.csv':
        rows = list(csv.DictReader(path.open()))
        for row in rows:
            for key in ('budget', 'seed', 'rmse', 'mae', 'r2', 'nrmse'):
                if key in row: row[key] = float(row[key]) if key not in ('budget','seed') else int(row[key])
        return rows
    return json.loads(path.read_text())
def aulc(rows, end):
    rows = sorted((r for r in rows if r['budget'] <= end), key=lambda r:r['budget'])
    assert rows[0]['budget'] == 333 and rows[-1]['budget'] == end
    return float(np.trapz([r['nrmse'] for r in rows], [r['budget'] for r in rows]) / (end - 333))
def write_json(name, value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def mdtable(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

global_summary = load(S/'odh_free_llm32_global_v3/results/summary.json')
assert global_summary['status']=='COMPLETE_PHASE1_L525'
v3 = global_summary['validation']
confirmation = load(S/'odh_lcmd_confirmation_v2/results/metrics.csv')
curves={'Random':[r for r in confirmation if r['seed']==1525 and r['split']=='validation' and r['method']=='random'],
        'Raw-gradient LCMD':[r for r in confirmation if r['seed']==1525 and r['split']=='validation' and r['method']=='raw_gradient_lcmd'],
        'LLM v3 (full pool)':v3}
p = load(S/'odh_free_llm32_global_v3/protocol.json')
c = load(S/'odh_lcmd_confirmation_v2/protocol.json')
assert p['training']==c['training']['1525']
base_input=load(S/'odh_lcmd_confirmation_v2/shared/seed_1525/fit/input.json')
assert all(abs(rows[0]['rmse']-v3[0]['rmse'])<1e-12 for rows in curves.values())
partition=load(S/'odh_lcmd_confirmation_v2/splits/partition.json')
test_ids={r['sample_index'] for r in partition['rows'] if r['role']=='test'}
audit_rows=load(S/'odh_free_llm32_global_v3/label_access_audit.csv')
assert all(not set(map(int,r['ids'].split(';'))) & test_ids for r in audit_rows)
assert all(r['allowed']=='True' for r in audit_rows)
rounds=[]
for i in range(6):
    d=S/f'odh_free_llm32_global_v3/runtime/seed_1525/free_llm32_global/round_{i}'
    complete=load(d/'complete.json')
    for rel,digest in complete['files'].items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest
    fit=load(d/'fit/fit.json'); inp=load(d/'fit/input.json')
    assert inp['config']==base_input['config'] and inp['validation']==base_input['validation']
    audit=load(d/'llm/context_admission.json')
    rounds.append({'round':i+1,'budget':complete['budget'],'candidate_count':audit['candidate_count'],'epochs':fit['epochs_run'],'best_epoch':fit['best_epoch'],'training_seconds':fit['seconds'],**v3[i+1]})

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
colors={'Random':'#64748b','Raw-gradient LCMD':'#007f73','LLM v3 (full pool)':'#6853c8'}
fig,axes=plt.subplots(1,2,figsize=(13,4.7),gridspec_kw={'width_ratios':[1.65,1]})
for label, rows in curves.items():
    axes[0].plot([r['budget'] for r in rows],[r['rmse'] for r in rows],'-o',lw=2,ms=5,label=label,color=colors[label])
axes[0].axvspan(429,525,color='#eeeaf8',alpha=.65,zorder=0)
axes[0].text(477,8.18,'V3 only beyond L429\nNo matched numerical control',ha='center',fontsize=9,color='#66577e')
axes[0].set(title='Validation learning curves · seed 1525',xlabel='Labeled training rows',ylabel='RMSE (mL; lower is better)',xticks=[333,365,397,429,461,493,525],ylim=(7.65,8.3))
axes[0].legend(loc='lower left',fontsize=8.5);axes[0].grid(axis='y',alpha=.2)
common=[]
for label,rows in curves.items():common.append({'method':label,'end_budget':397,'mean_nrmse':aulc(rows,397)})
labels=[r['method'] for r in common]; vals=[r['mean_nrmse'] for r in common]
axes[1].scatter(vals,range(len(labels)),c=[colors[x] for x in labels],s=85);axes[1].set_yticks(range(len(labels)),labels);axes[1].set_ylim(-.6,len(labels)-.4);axes[1].invert_yaxis()
for i,v in enumerate(vals):axes[1].text(v+.001,i,f'{v:.5f}',va='center',fontsize=9)
axes[1].set(title='Common interval: L333–L397',xlabel='Mean NRMSE AULC (lower is better)',xlim=(.86,.93))
axes[1].grid(axis='x',alpha=.2)
fig.suptitle('LLM vs numerical active learning: matched development comparison',fontsize=14,fontweight='bold')
fig.text(.5,.015,'Same L333 checkpoint, validation rows and training configuration. One seed; no LLM test evaluation. Axes are zoomed.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.055,1,.94))
for ext in ('png','pdf'):fig.savefig(OUT/f'llm_vs_traditional.{ext}',dpi=190)
plt.close(fig)

# A fixed-seed numerical comparison, deduplicating reused Random/LCMD controls.
traditional={}
for study in ['odh_representation_coreset_extension_v1','odh_ivr_hybrid_screen_v1']:
    for r in load(S/study/'results/metrics.csv'):
        key=(r['method'],r['split'],r['budget'])
        if key in traditional: assert abs(traditional[key]['nrmse']-r['nrmse'])<1e-12
        traditional[key]=r
method_names={'random':'Random','raw_gradient_lcmd':'Raw-gradient LCMD','latent_coreset':'Latent coreset','morgan_coreset':'Morgan coreset','kernel_ivr':'Kernel IVR','hybrid_ensemble_latent':'Ensemble + latent hybrid'}
trad_summary=[]
for m in method_names:
    rows=[r for (method,split,b),r in traditional.items() if method==m and split=='test']
    trad_summary.append({'method':m,'name':method_names[m],'seed':525,'interval':[333,493],'test_mean_nrmse':aulc(rows,493),'endpoint_rmse':next(r['rmse'] for r in rows if r['budget']==493)})
trad_summary.sort(key=lambda r:r['test_mean_nrmse'])
transfer=load(S/'odh_gradient_al_transfer_v2/results/learning_curves.csv')
transfer_summary=[]
for m in sorted({r['method'] for r in transfer}):
    rows=[r for r in transfer if r['method']==m and r['split']=='test']
    transfer_summary.append({'method':m,'test_mean_nrmse':aulc(rows,429)})
transfer_summary.sort(key=lambda r:r['test_mean_nrmse'])
paired=load(S/'odh_lcmd_confirmation_v2/results/paired_aulc.csv')
fig,axes=plt.subplots(1,2,figsize=(12.5,4.7))
for i,r in enumerate(trad_summary):
    axes[0].scatter(r['test_mean_nrmse'],i,s=85,color='#007f73' if r['method']=='raw_gradient_lcmd' else '#8199b1')
    axes[0].text(r['test_mean_nrmse']+.001,i,f"{r['test_mean_nrmse']:.5f}",va='center')
axes[0].set(yticks=range(len(trad_summary)),yticklabels=[r['name'] for r in trad_summary],xlim=(.77,.9),xlabel='Test mean NRMSE AULC (lower is better)',title='Exploratory methods · seed 525 · L333–L493');axes[0].invert_yaxis()
for i,split in enumerate(['test','validation']):
    for j,seed in enumerate([1525,2525]):
        r=next(r for r in paired if r['seed']==seed and r['split']==split)
        y=i*3+j
        axes[1].plot([float(r['random']),float(r['raw_gradient_lcmd'])],[y,y],color='#aab2b8')
        axes[1].scatter(float(r['random']),y,color='#64748b',s=50,label='Random' if y==0 else None)
        axes[1].scatter(float(r['raw_gradient_lcmd']),y,color='#007f73',s=50,label='Raw-gradient LCMD' if y==0 else None)
axes[1].set(yticks=[0,1,3,4],yticklabels=['Test · 1525','Test · 2525','Validation · 1525','Validation · 2525'],xlabel='Mean NRMSE AULC (lower is better)',title='Paired confirmation · L333–L429',xlim=(.80,.92));axes[1].invert_yaxis();axes[1].legend(fontsize=9)
for ax in axes:ax.grid(axis='x',alpha=.2)
fig.suptitle('Traditional active learning: historical development evidence',fontsize=14,fontweight='bold')
fig.text(.5,.015,'Separate panels have different seeds / budget intervals. Existing row-split test cohort; not independent external validation. Axes are zoomed.',ha='center',fontsize=8.5)
fig.tight_layout(rect=(0,.05,1,.94))
for ext in ('png','pdf'):fig.savefig(OUT/f'traditional_methods.{ext}',dpi=190)
plt.close(fig)

common429=[{'method':label,'mean_nrmse':aulc(rows,429)} for label,rows in curves.items() if max(r['budget'] for r in rows)>=429]
write_json('comparison_data.json',{'global_rounds':rounds,'matched_333_397':common,'matched_333_429':common429,'traditional_seed525':trad_summary,'raw_unit_maxdet_seed525':transfer_summary,'paired_confirmation':paired,'six_round_artifact_verification':'PASS','matched_training_and_validation_rows':'PASS','llm_test_labels':0,'label_audit_events':len(audit_rows)})
# Detail export is generated from the same read-only sources used by the figures.
with (OUT/'matched_validation_metrics.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=['method','seed','budget','rmse','mae','r2','nrmse']);w.writeheader()
    for label,rows in curves.items():
        for r in rows:w.writerow({'method':label,'seed':1525,**{k:r[k] for k in ['budget','rmse','mae','r2','nrmse']}})

v3end=v3[-1];best=min(v3,key=lambda r:r['rmse'])
report=f'''# ODH 复现与主动学习结果汇总（2026-10-05）

## 当前进度

全量 LLM V3 已完成六轮：L333 → L525，共新增 192 条训练标签。日志完成时间为北京时间 2026-10-03 03:02；当前没有该流程的运行进程。已重新核验六轮 complete.json 所绑定的训练、指标及选择文件哈希，并核对相同训练设置及验证集行号。

最终验证 RMSE **{v3end['rmse']:.4f}**、MAE **{v3end['mae']:.4f}**、R² **{v3end['r2']:.4f}**。相对 L333 的 RMSE 降低 **{100*(1-v3end['rmse']/v3[0]['rmse']):.2f}%**。轨迹最低 RMSE 为 L{best['budget']} 的 **{best['rmse']:.4f}**；这是回顾性最佳点，不替代预定 L525 终点。LLM 流程未读取测试标签。

{mdtable(['轮次','标注数','全量候选数','验证 RMSE','验证 MAE','验证 R²'],[[r['round'],r['budget'],r['candidate_count'],f"{r['rmse']:.4f}",f"{r['mae']:.4f}",f"{r['r2']:.4f}"] for r in rounds])}

## 原模型复现

完整运行官方兼容训练 1,500 epochs，4,447 条训练样本，验证/测试各 247 条，使用最后 epoch 模型。论文数据重算与复现分别为：

| 测试指标 | 论文源数据重算 | 本项目正式复现 |
| --- | ---: | ---: |
| RMSE | 3.9173 | 4.1774 |
| MAE | 2.7435 | 2.6787 |
| R² | 0.7778 | 0.7474 |

正式复现使用全训练集、不同训练时长及划分设定，不能与 L333–L525 的主动学习验证误差直接排名。随机行划分存在分子交叠，属于论文协议兼容性复现，不代表对新分子的独立泛化。q10–q90 区间覆盖率仅 18.62%，不能把区间宽度当作已校准不确定性。

来源：../../../docs/research/ODH_BASELINE_REPRODUCTION.md。

## 传统主动学习

已完成 Random、raw/unit-gradient LCMD、raw/unit-gradient MaxDet、latent/Morgan coreset、kernel IVR、ensemble+latent hybrid 等探索。训练诊断另有固定 L0 的九次训练，不是九条 AL 轨迹。更改梯度归一化会显著改变选样，但现有结果不能证明归一化提高标签效率。

raw/unit-gradient 的早期配对探索均为 seed 525、L333–L429，历史测试集平均 NRMSE：

{mdtable(['方法','测试 NRMSE AULC'],[[r['method'],f"{r['test_mean_nrmse']:.5f}"] for r in transfer_summary])}

当前较有支持的传统比较基线是 **raw-gradient LCMD**。seed 525 的 L333–L493 历史测试集曲线均值如下（越低越好；重复使用的 Random/LCMD 数据只计一次）：

{mdtable(['方法','测试 NRMSE AULC','L493 测试 RMSE'],[[r['name'],f"{r['test_mean_nrmse']:.5f}",f"{r['endpoint_rmse']:.4f}"] for r in trad_summary])}

两种新增初始化种子 1525/2525 的配对确认只运行至 L429，LCMD 测试 AULC 比 Random 分别降低 **3.70% / 4.20%**；但验证集方向不一致（1525 有利于 LCMD，2525 有利于 Random）。不能据此声称普遍显著优越；所有结果仍来自同一个历史 Row cohort。

![传统主动学习](traditional_methods.png)

## LLM 主动学习尝试

| 版本 | 完成情况 | 可解释的证据 |
| --- | --- | --- |
| Global v3 | seed 1525，6 轮到 L525 | gpt-6-astra / Responses；每轮全部候选在一次请求中提交，统一选 32 条 |

V3 的分页只负责承载数据，无分块预筛选。按用户要求，单字段解释文字长度已由硬限制改为建议。另有历史假设更新缺失导致的一次第三轮回复拒收，重试后的完整回复通过校验；没有自动补写模型证据。每日额度阻塞造成的中断也保留了记录。

旧 partial-visibility、分块未完成和历史 hybrid 实验已按用户要求从当前版本移除，不进入本报告的性能比较。仅保留完整 V3 校验所必需的原始基线依赖文件。

## 传统与 LLM 的直接对比

下面使用 seed 1525、同一 L333 模型、相同训练设置与验证集。AULC 为预算区间内 NRMSE 的梯形积分除以区间长度，分母为固定 L333 目标标准差 8.879240547556758；数值越低越好。

共同区间 **L333–L397**（三种方法均有结果）：

{mdtable(['方法','验证 NRMSE AULC'],[[r['method'],f"{r['mean_nrmse']:.5f}"] for r in common])}

共同区间 **L333–L429**（三种方法均有结果）：

{mdtable(['方法','验证 NRMSE AULC'],[[r['method'],f"{r['mean_nrmse']:.5f}"] for r in common429])}

![LLM 与传统方法的匹配对比](llm_vs_traditional.png)

V3 在共同区间内的曲线均值优于 Random，但仍落后于 raw-gradient LCMD；这只是单种子开发集比较。L397 的单点上 V3 还略差于 Random，不能声称每轮都更好。V3 的 L461–L525 没有同种子、同预算的 Random/LCMD 对照，图中没有延长传统曲线或借用 seed 525 来拼接。V3 全区间 L333–L525 的均值 NRMSE 为 **{global_summary['label_aulc']['mean_nrmse']:.5f}**，不能与短区间的 AULC 直接排名。

## 下一步最有价值的比较

若继续评估收益，应先补齐 seed 1525 的 Random/LCMD 至 L525，并增加配对种子；最终确认应使用未用于开发的新划分或独立实验队列。当前汇总没有新增训练、调用 LLM 或打开新的测试标签。

## 文件与复核

- `comparison_data.json`：汇总指标和六轮数据。
- `matched_validation_metrics.csv`：主图的逐点数值。
- `llm_vs_traditional.png/.pdf`、`traditional_methods.png/.pdf`：可分享图片与矢量图。
- `source_manifest.json`：输入文件 SHA256。
- `build_report.py`：从已有指标重建本报告与图，运行环境为项目 `.conda-hplc-al`。
'''
(OUT/'REPORT.md').write_text(report)
# Include baseline and historical narratives as provenance too.
for rel in ['docs/research/ODH_BASELINE_REPRODUCTION.md','studies/active_learning/odh_free_llm32_scientist_v1/README.md']:
    inputs[rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
write_json('source_manifest.json',inputs)
print(json.dumps({'output':str(OUT),'common397':common,'common429':common429,'six_rounds_verified':True},indent=2))
