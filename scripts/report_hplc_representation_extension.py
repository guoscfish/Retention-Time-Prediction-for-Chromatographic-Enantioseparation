"""Render and seal the validation-gated HPLC continuation results."""
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/hplc-representation-extension-matplotlib')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hplc_al.common import atomic_json, metrics, read_json, sha, verify_files, write_once
from hplc_al.protocol import role_ids
from hplc_al.representation_extension import METHODS, PARENT, START_BUDGET, STUDY, STEP, verify_freeze


def _csv(path, rows):
    with path.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(study=STUDY):
    study=Path(study)
    if (study/'completion_manifest.json').exists():
        verify_files(study,read_json(study/'completion_manifest.json')['files'])
        verify_freeze(study)
        return
    freeze, records=verify_freeze(study)
    verify_files(study,read_json(study/'test_reveal.json')['files'])
    partition=read_json(study/'splits/partition.json')
    valid,test=role_ids(partition,'validation'),role_ids(partition,'test')
    with np.load(study/'test_truth.npz') as bank:
        if bank['ids'].tolist()!=test:
            raise RuntimeError('test truth alignment drift')
        truth=bank['truth'].copy()
    scale=read_json(PARENT/'protocol.json')['scale']
    output=study/'results'
    output.mkdir(exist_ok=True)
    old=list(csv.DictReader((PARENT/'results/metrics.csv').open()))
    rows=[]
    for method in METHODS:
        for prior in old:
            if prior['method']==method:
                rows.append(dict(method=method,budget=int(prior['budget']),split=prior['split'],
                    **{k:float(prior[k]) for k in ('rmse','mae','r2','nrmse')}))
        for r in range(4,freeze['last_round']+1):
            rec=records[f'{method}/{r}']
            with np.load(study/f'runtime/{method}/round_{r}/predictions.npz') as bank:
                if bank['ids'].tolist()!=valid+test:
                    raise RuntimeError('prediction row drift')
                test_metrics=metrics(bank['predictions'][len(valid):,1],truth,scale)
            for split,value in [('validation',rec['validation_metrics']),('test',test_metrics)]:
                rows.append(dict(method=method,budget=333+STEP*r,split=split,**value))
    _csv(output/'metrics.csv',rows)
    aulc=[]
    for split in ('validation','test'):
        full={}
        extension={}
        for method in METHODS:
            points=sorted((row for row in rows if row['method']==method and row['split']==split),key=lambda x:x['budget'])
            budgets=[p['budget'] for p in points]
            if budgets!=list(range(333,freeze['stop_budget']+1,STEP)):
                raise RuntimeError('incomplete metric grid')
            full[method]=float(np.trapz([p['nrmse'] for p in points],budgets)/(budgets[-1]-333))
            later=[p for p in points if p['budget']>=START_BUDGET]
            extension[method]=float(np.trapz([p['nrmse'] for p in later],[p['budget'] for p in later])/(budgets[-1]-START_BUDGET))
        for method in METHODS:
            aulc.append(dict(method=method,split=split,full_normalized_aulc=full[method],
                full_ratio_to_random=full[method]/full['random'],
                extension_normalized_aulc=extension[method],
                extension_ratio_to_random=extension[method]/extension['random']))
    _csv(output/'aulc.csv',aulc)
    decisions=[read_json(path) for path in sorted((study/'decisions').glob('after_*.json'))]
    decision_rows=[]
    for d in decisions:
        for detail in d['details']:
            decision_rows.append(dict(check_after=d['checked_after_budget'],budget=detail['budget'],
                positive=d['positive'],latent_gain_vs_lcmd=detail['gains_vs_lcmd']['latent_coreset'],
                morgan_gain_vs_lcmd=detail['gains_vs_lcmd']['morgan_coreset'],
                positive_methods=';'.join(detail['positive_methods'])))
    _csv(output/'validation_decisions.csv',decision_rows)
    geometry=[]
    for method in METHODS:
        for r in range(3,freeze['last_round']):
            a=read_json(study/f'runtime/{method}/round_{r}/coverage.json')
            geometry.append(dict(method=method,budget=333+STEP*r,
                gradient_norm_percentile=a['gradient_norm']['selected_pool_percentile']['mean'],
                latent_novelty_percentile=a['latent_nearest_L_cosine_distance']['selected_pool_percentile']['mean'],
                morgan_novelty_percentile=a['morgan_novelty']['selected_pool_percentile']['mean'],
                novel_scaffold_fraction=a['novel_scaffold_fraction'],
                heavy_atom_count=a['heavy_atom_count']['selected']['mean'],
                molecular_weight=a['molecular_weight']['selected']['mean']))
    _csv(output/'acquisition_geometry.csv',geometry)
    overlap=[]
    for r in range(3,freeze['last_round']):
        gradient=set(records[f'raw_gradient_lcmd/{r}']['selected'])
        for method in ('latent_coreset','morgan_coreset'):
            other=set(records[f'{method}/{r}']['selected'])
            overlap.append(dict(budget=333+STEP*r,comparison=f'raw_gradient_lcmd vs {method}',
                count=len(gradient&other),jaccard=len(gradient&other)/len(gradient|other),
                shared_ids=';'.join(map(str,sorted(gradient&other)))))
    _csv(output/'acquisition_overlap.csv',overlap)
    colors=dict(zip(METHODS,['#64748b','#dc2626','#2563eb','#059669']))
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,split in zip(axes,('validation','test')):
        for method in METHODS:
            points=sorted((r for r in rows if r['method']==method and r['split']==split),key=lambda r:r['budget'])
            ax.plot([r['budget'] for r in points],[r['nrmse'] for r in points],
                    'o-',color=colors[method],label=method)
        ax.axvline(START_BUDGET,color='#555',ls=':',lw=1)
        ax.set(xlabel='Labeled budget',ylabel='NRMSE (frozen L333 SD)',title=split.title())
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.savefig(output/'learning_curves.png',dpi=180)
    plt.close(fig)
    lines=['# ODH representation coreset: adaptive round extension','',
        '同一 seed 525、同四方法，从已冻结 L429 checkpoint 接续。先跑两个新增预算点；每两个新点以 validation NRMSE 检查 Latent/Morgan 是否至少一次比同预算 Raw Gradient-LCMD 低 0.01。阳性则再跑两个点，否则停止。最多新增 8 轮。','',
        f"**实际停止**：budget {freeze['stop_budget']}，原因 `{freeze['stop_reason']}`；新增 {freeze['last_round']-3} 个预算点。",'',
        '## Validation-only 决策轨迹','',
        '| 检查时预算 | 新预算点 | Latent 相对 LCMD 改善 | Morgan 相对 LCMD 改善 | 阳性方法 |',
        '|---:|---:|---:|---:|---|']
    for d in decision_rows:
        lines.append(f"| {d['check_after']} | {d['budget']} | {d['latent_gain_vs_lcmd']:.4f} | {d['morgan_gain_vs_lcmd']:.4f} | {d['positive_methods'] or 'none'} |")
    lines += ['', '正数表示 coreset 的 validation NRMSE 更低。任一新预算点达到 ≥0.01 就继续下一对预算点。没有用 test 指标参与停止决策。','',
        '## 学习曲线与 AULC','', '![Learning curves](results/learning_curves.png)','',
        'AULC 均为 NRMSE 随 labeled budget 的梯形积分除以对应区间宽度；越低越好。', '',
        '| 方法 | Validation 全程 | Test 全程 | Test 429→停止点 | Test 扩展段 / Random |',
        '|---|---:|---:|---:|---:|']
    by={(a['method'],a['split']):a for a in aulc}
    for method in METHODS:
        va,te=by[method,'validation'],by[method,'test']
        lines.append(f"| {method} | {va['full_normalized_aulc']:.6f} | {te['full_normalized_aulc']:.6f} | {te['extension_normalized_aulc']:.6f} | {te['extension_ratio_to_random']:.4f} |")
    lines += ['', '## 每预算点 test 结果','', '| 方法 | Budget | RMSE | MAE | R² | NRMSE |',
        '|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        for row in sorted((r for r in rows if r['method']==method and r['split']=='test'),key=lambda r:r['budget']):
            lines.append(f"| {method} | {row['budget']} | {row['rmse']:.4f} | {row['mae']:.4f} | {row['r2']:.4f} | {row['nrmse']:.4f} |")
    lines += ['', '## 选样覆盖与交集','',
        '每轮的梯度范数、latent/Morgan novelty 百分位、scaffold 新颖度、分子大小和分子量见 results/acquisition_geometry.csv；完整池/选中分布见各轮 coverage.json。新轮次的 LCMD 与两种 coreset 交集如下。','',
        '| Budget | 比较 | 共同 IDs | Jaccard |', '|---:|---|---:|---:|']
    for o in overlap:
        lines.append(f"| {o['budget']} | {o['comparison']} | {o['count']} | {o['jaccard']:.4f} |")
    best_extension=min(METHODS,key=lambda m:by[m,'test']['extension_normalized_aulc'])
    lines += ['', '## 解释与限制','',
        f"本次扩展段 test AULC 最低的方法是 **{best_extension}**。该排序只描述 seed 525 的轨迹。停止规则和 checkpoint 选择均使用 validation，因此继续轮次本身是适应 validation 的过程；本研究不据此作显著性结论。",'',
        '先前 study 的 test truth 已经揭示。本次扩展在新轮次全部冻结后统一评估 test，但它仍是同一开发数据集，不能作为独立外部验证。原 study 和旧历史 artifacts 保持不可变。', '',
        '如果在 685 达到上限且最后一个 validation 检查仍为阳性，结论应是“达到预定上限，信号仍在”，不能称为“没有积极信号而停止”。', '']
    (study/'REPORT.md').write_text('\n'.join(lines))
    summary=dict(status='COMPLETE_STOPPED',start_budget=START_BUDGET,stop_budget=freeze['stop_budget'],
                 stop_reason=freeze['stop_reason'],extra_rounds=freeze['last_round']-3,
                 methods=list(METHODS),seed=525,best_test_extension_method=best_extension,aulc=aulc)
    atomic_json(output/'summary.json',summary)
    files={str(p.relative_to(study)):sha(p) for p in study.rglob('*') if p.is_file() and p.suffix!='.log'}
    write_once(study/'completion_manifest.json',dict(**summary,files=files))
    print(summary)


if __name__=='__main__':
    report()
