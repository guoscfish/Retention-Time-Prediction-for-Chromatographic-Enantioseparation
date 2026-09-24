"""Read frozen results and render the bounded representation study report."""
import csv
import itertools
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/hplc-representation-matplotlib')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hplc_al.common import atomic_json, metrics, read_json, sha, verify_files, write_once
from hplc_al.representation_runner import STUDY, METHODS, BUDGETS, verify_freeze, tree_hashes
from hplc_al.protocol import role_ids


def csv_write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(study=STUDY):
    study = Path(study)
    if (study / 'completion_manifest.json').exists():
        verify_files(study, read_json(study / 'completion_manifest.json')['files'])
        verify_freeze(study)
        return
    records = verify_freeze(study)
    reveal = read_json(study / 'test_reveal.json')
    verify_files(study, reveal['files'])
    protocol = read_json(study / 'protocol.json')
    partition = read_json(study / 'splits/partition.json')
    valid, test = role_ids(partition, 'validation'), role_ids(partition, 'test')
    with np.load(study / 'test_truth.npz') as bank:
        assert bank['ids'].tolist() == test
        truth = bank['truth'].copy()
    result = study / 'results'
    result.mkdir(exist_ok=True)
    rows = []
    for method in METHODS:
        for r, budget in enumerate(BUDGETS):
            rec = records[f'{method}/{r}']
            with np.load(study / f'runtime/{method}/round_{r}/predictions.npz') as pred:
                assert pred['ids'].tolist() == valid + test
                test_metrics = metrics(pred['predictions'][len(valid):, 1], truth, protocol['scale'])
            for split, value in [('validation', rec['validation_metrics']), ('test', test_metrics)]:
                rows.append(dict(method=method, budget=budget, split=split, **value))
    aulc = []
    for split in ['validation', 'test']:
        values = {method: float(np.trapz([row['nrmse'] for row in rows if row['method'] == method and row['split'] == split], BUDGETS) / 96) for method in METHODS}
        for method, value in values.items():
            aulc.append(dict(method=method, split=split, normalized_aulc=value, ratio_to_random=value / values['random']))
    comparisons = []
    for split in ['validation', 'test']:
        values = {r['method']: r['normalized_aulc'] for r in aulc if r['split'] == split}
        for a, b in itertools.combinations(METHODS, 2):
            comparisons.append(dict(split=split, method_a=a, method_b=b,
                                    a_minus_b=values[a]-values[b], a_over_b=values[a]/values[b]))
    csv_write(result / 'metrics.csv', rows)
    csv_write(result / 'aulc.csv', aulc)
    csv_write(result / 'pairwise_comparisons.csv', comparisons)
    geometry, selected_rows = [], []
    for method in METHODS:
        for r in range(3):
            audit = read_json(study / f'runtime/{method}/round_{r}/coverage.json')
            for selected in audit['selected_rows']:
                selected_rows.append(dict(method=method, budget=BUDGETS[r], **selected))
            geometry.append(dict(method=method, budget=BUDGETS[r],
                gradient_norm_percentile=audit['gradient_norm']['selected_pool_percentile']['mean'],
                top_decile_gradient_fraction=audit['top_decile_gradient_fraction'],
                latent_nearest_L_cosine_distance=audit['latent_nearest_L_cosine_distance']['selected']['mean'],
                latent_novelty_percentile=audit['latent_nearest_L_cosine_distance']['selected_pool_percentile']['mean'],
                latent_pairwise_distance=audit['batch_pairwise_latent_cosine_distance']['mean'],
                latent_effective_rank=audit['latent_effective_rank'],
                morgan_max_tanimoto=audit['morgan_nearest_L_max_tanimoto']['selected']['mean'],
                morgan_novelty=audit['morgan_novelty']['selected']['mean'],
                morgan_novelty_percentile=audit['morgan_novelty']['selected_pool_percentile']['mean'],
                morgan_pairwise_distance=audit['batch_pairwise_morgan_tanimoto_distance']['mean'],
                novel_scaffold_fraction=audit['novel_scaffold_fraction'],
                scaffold_novelty_percentile=audit['scaffold_novelty']['selected_pool_percentile']['mean'],
                heavy_atom_count=audit['heavy_atom_count']['selected']['mean'],
                molecular_weight=audit['molecular_weight']['selected']['mean']))
    csv_write(result / 'acquisition_geometry.csv', geometry)
    csv_write(result / 'selected_rows.csv', selected_rows)
    overlap = read_json(study / 'acquisition_overlap.json')
    csv_write(result / 'acquisition_overlap.csv', [dict(comparison=k, count=v['count'], jaccard=v['jaccard'], selected_ids=';'.join(map(str, v['ids']))) for k, v in overlap.items()])
    relation = read_json(study / 'diagnostics/relation.json')
    stability = read_json(study / 'diagnostics/stability.json')
    history = read_json(study / 'diagnostics/historical_coverage.json')
    chirality = read_json(study / 'diagnostics/chirality.json')
    historical_rows = []
    for key, audit in history.items():
        method, r = key.split('/')
        historical_rows.append(dict(method=method, budget=BUDGETS[int(r)],
            morgan_novelty=audit['morgan_novelty']['selected']['mean'],
            morgan_novelty_percentile=audit['morgan_novelty']['selected_pool_percentile']['mean'],
            latent_novelty_percentile=audit['latent_nearest_L_cosine_distance']['selected_pool_percentile']['mean'],
            novel_scaffold_fraction=audit['novel_scaffold_fraction'],
            heavy_atom_count=audit['heavy_atom_count']['selected']['mean'],
            heavy_atom_percentile=audit['heavy_atom_count']['selected_pool_percentile']['mean']))
    csv_write(result / 'historical_coverage.csv', historical_rows)
    colors = dict(zip(METHODS, ['#64748b', '#dc2626', '#2563eb', '#059669']))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, split in zip(axes, ['validation', 'test']):
        for method in METHODS:
            ax.plot(BUDGETS, [r['nrmse'] for r in rows if r['method'] == method and r['split'] == split], 'o-', label=method, color=colors[method])
        ax.set(title=split.title(), xlabel='Labeled budget', ylabel='NRMSE (frozen L333 SD)')
        ax.set_xticks(BUDGETS)
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.savefig(result / 'learning_curves.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for ax, metric, title in zip(axes, ['knn_overlap_at_10', 'knn_overlap_at_50', 'pairwise_distance_rank_correlation'], ['kNN overlap@10', 'kNN overlap@50', 'Distance rank correlation']):
        for name, color in [('gradient', '#dc2626'), ('latent', '#2563eb')]:
            values = [v[name][metric]['mean'] if isinstance(v[name][metric], dict) else v[name][metric] for v in stability.values()]
            ax.plot(list(stability), values, 'o-', label=name, color=color)
        ax.set(title=title, ylim=(0, 1))
        ax.tick_params(axis='x', rotation=15)
        ax.grid(alpha=.2)
    axes[-1].legend()
    fig.savefig(result / 'stability.png', dpi=180)
    plt.close(fig)
    lines = [
        '# ODH representation coreset v1', '',
        '完成范围：seed 525；4 方法；333→365→397→429；3 次 acquisition。已停止，不扩 seed、budget 或方法。', '',
        '## 协议与审计', '',
        '- 当前 main 基线为 3c15d0a。全部历史 study 文件在开始和完成时逐项 SHA-256 对比。',
        '- 四方法共享同一 L333 IDs、checkpoint、validation 和 test-X predictions。Raw Gradient-LCMD 的冻结轨迹按相同 IDs、训练配置和 checkpoint 校验复用；未重新运行历史 selector。',
        '- 新 Random 使用 seed 525900001 对排序 U0 仅置换一次，之后依次取 [0:32]、[32:64]、[64:96]。不与历史 Random 混用。',
        '- 其余新增轮次使用 batch 256、deterministic_each_epoch、max 500 epochs、patience 100、lr .001、weight decay 1e-5、scratch/fresh Adam、无 scheduler、完整原始 loss 和仅 validation 选 checkpoint。',
        '- training_seed metadata 修复仅影响未来记录；历史 initialization_seed=training_seed=525，因此不改变或重新解释历史数值。',
        '- h_graph 为 G atom-bond 与 H bond-angle 信息融合、global_add_pool 后的 128D task-aware latent；还包含键长、i-PrOH 条件以及 TPSA/RASA/RPSA/MDEC/MATS 信息，不能称为纯结构指纹。',
        '- Latent 使用单位化后 1−cosine 距离；Morgan 使用 radius=2、2048 bits、includeChirality=True、1−Tanimoto。K-center 以 current L 为初始中心，精确并列时取最小 ID。',
        '- 各方法每轮使用自身当前 checkpoint 提取 latent 和 gradient；Morgan 固定。全部 16 个 method/budget 记录及预测先冻结，校验后仅统一读取一次 test truth。',
        f"- NRMSE 分母为冻结 L333 population SD={protocol['scale']:.10f}。normalized AULC=NRMSE 在 333–429 的梯形积分/96。", '',
        '## 表征诊断', '',
        f"L333 outer-train latent cosine distance 与 Morgan Tanimoto distance 的 Spearman={relation['latent_morgan_spearman']:.4f}（50,000 对有放回随机样本对，无自身配对，seed 525）。这是描述性关联，不是 task label 预测能力证明。", '',
        '| latent norm vs size | Spearman | Pearson |', '|---|---:|---:|']
    for name, value in relation['norm_size'].items():
        lines.append(f"| {name} | {value['spearman']:.4f} | {value['pearson']:.4f} |")
    lines += ['', '图 atom count 含缓存图的节点；RDKit atom count 不包含隐式 H，可能与 heavy atom count 相同。norm 与尺寸关联为使用 unit latent 的依据，但不说明它是唯一影响因素。', '',
        f"手性诊断：观察到 {chirality['observed_mirror_pairs']} 对显式四面体镜像分子；启用 chirality 可区分 {chirality['observed_distinguished_chiral']} 对，关闭时可区分 {chirality['observed_distinguished_achiral']} 对。未指定、轴手性与非四面体手性不作推断。完整镜像配对和诊断保存在 diagnostics/chirality.json。", '',
        '## Gradient 与 latent 稳定性', '',
        '固定候选为 Raw Gradient-LCMD 的 U429 中随机选取的 512 个 IDs，在四个 checkpoint 上保持完全一致。Gradient 的邻域/距离秩采用 raw CountSketch 欧氏距离，latent 采用 unit cosine；坐标逐行 cosine 单独报告。历史前 3 个 gradient bank 直接读取，仅为 L429 提取缺少的 bank。', '',
        '| checkpoint pair | representation | row cosine | norm-rank ρ | kNN@10 | kNN@50 | distance-rank ρ |',
        '|---|---|---:|---:|---:|---:|---:|']
    for pair, values in stability.items():
        for name, value in values.items():
            norm = f"{value['norm_rank_spearman']:.4f}" if 'norm_rank_spearman' in value else '—'
            lines.append(f"| {pair} | {name} | {value['row_cosine']['mean']:.4f} | {norm} | {value['knn_overlap_at_10']['mean']:.4f} | {value['knn_overlap_at_50']['mean']:.4f} | {value['pairwise_distance_rank_correlation']:.4f} |")
    lines += ['', '![Stability](results/stability.png)', '',
        '不同 scratch checkpoint 的坐标 cosine 也可能受表示基底变化影响；邻域和距离排序更直接衡量几何稳定性。本诊断沿 LCMD 轨迹开展，不能替代所有方法的稳定性比较。', '',
        '## 历史 acquisition 的 coverage', '',
        '只读取历史已冻结 batch，不重跑 selector。每轮以该方法 current L/U 计算 novelty；latent 始终使用共享 L333 checkpoint，以固定参考几何比较历史轨迹。百分位为 current U 的 midrank，重复分子同等处理。', '',
        '| method | budget | Morgan novelty percentile | latent novelty percentile | novel scaffold fraction | heavy atoms mean | heavy atom percentile |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in historical_rows:
        lines.append(f"| {row['method']} | {row['budget']} | {row['morgan_novelty_percentile']:.2f} | {row['latent_novelty_percentile']:.2f} | {row['novel_scaffold_fraction']:.3f} | {row['heavy_atom_count']:.2f} | {row['heavy_atom_percentile']:.2f} |")
    lines += ['', '## 预测结果', '', '![Learning curves](results/learning_curves.png)', '',
        '| method | validation AULC | validation / Random | test AULC | test / Random |', '|---|---:|---:|---:|---:|']
    values = {(row['method'], row['split']): row for row in aulc}
    for method in METHODS:
        va, te = values[method, 'validation'], values[method, 'test']
        lines.append(f"| {method} | {va['normalized_aulc']:.6f} | {va['ratio_to_random']:.4f} | {te['normalized_aulc']:.6f} | {te['ratio_to_random']:.4f} |")
    lines += ['', '| method | budget | split | RMSE | MAE | R² | NRMSE |', '|---|---:|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['method']} | {row['budget']} | {row['split']} | {row['rmse']:.4f} | {row['mae']:.4f} | {row['r2']:.4f} | {row['nrmse']:.4f} |")
    lines += ['', '## Acquisition geometry', '',
        '每行使用该方法自己的当前 checkpoint；同一 budget 的不同方法 latent 数值并不处于同一个坐标空间。Morgan geometry 可直接跨方法比较。全部选中 ID 的值、percentile、分位数和分布见 results/selected_rows.csv 与各轮 coverage.json。', '',
        '| method | budget | gradient norm pct | top-decile fraction | latent nearest-L | latent pairwise | latent effective rank | Morgan max Tanimoto | Morgan novelty | Morgan pairwise | novel scaffold | heavy atoms | mol weight |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in geometry:
        lines.append('| ' + ' | '.join([row['method'], str(row['budget'])] + [f"{row[k]:.4f}" for k in ['gradient_norm_percentile', 'top_decile_gradient_fraction', 'latent_nearest_L_cosine_distance', 'latent_pairwise_distance', 'latent_effective_rank', 'morgan_max_tanimoto', 'morgan_novelty', 'morgan_pairwise_distance', 'novel_scaffold_fraction', 'heavy_atom_count', 'molecular_weight']]) + ' |')
    lines += ['', '## LCMD 与两种 coreset 的 batch overlap', '', '| comparison | count | Jaccard | shared IDs |', '|---|---:|---:|---|']
    for key, value in overlap.items():
        if 'raw_gradient_lcmd vs' in key:
            lines.append(f"| {key} | {value['count']} | {value['jaccard']:.4f} | {', '.join(map(str, value['ids'])) or 'none'} |")
    test_aulc = {m: values[m, 'test']['normalized_aulc'] for m in METHODS}
    rank = sorted(test_aulc, key=test_aulc.get)
    grad, lat, morg, rnd = [test_aulc[m] for m in ['raw_gradient_lcmd', 'latent_coreset', 'morgan_coreset', 'random']]
    st_better = sum(v['latent']['knn_overlap_at_10']['mean'] > v['gradient']['knn_overlap_at_10']['mean'] for v in stability.values())
    hist_mean = {m: {k: float(np.mean([row[k] for row in historical_rows if row['method'] == m])) for k in ['morgan_novelty_percentile', 'latent_novelty_percentile']} for m in ['random', 'raw_gradient_lcmd', 'raw_gradient_maxdet']}
    lines += ['', '## 科学问题 Q1–Q7', '',
        f"**Q1**：L333 latent 与 Morgan 距离的秩相关为 {relation['latent_morgan_spearman']:.4f}。非零关联说明 learned geometry 与 chemical similarity 存在对应，但二者不等价；latent 还编码 geometry/condition，不能仅凭该关联断言其 task-aware 信息已经成熟。", '',
        f"**Q2**：沿 333→429 的三段相邻 checkpoint，latent 的 kNN@10 稳定性在 {st_better}/3 段高于 gradient。应结合上表 kNN@50 与 distance-rank 一起判断；这是从 L333 出发的跨训练稳定性证据，而不是单一 L333 checkpoint 的内在稳定性测量。", '',
        f"**Q3**：历史 LCMD 的平均 Morgan/latent novelty 百分位为 {hist_mean['raw_gradient_lcmd']['morgan_novelty_percentile']:.1f}/{hist_mean['raw_gradient_lcmd']['latent_novelty_percentile']:.1f}，Random 为 {hist_mean['random']['morgan_novelty_percentile']:.1f}/{hist_mean['random']['latent_novelty_percentile']:.1f}。MaxDet 为 {hist_mean['raw_gradient_maxdet']['morgan_novelty_percentile']:.1f}/{hist_mean['raw_gradient_maxdet']['latent_novelty_percentile']:.1f}；其 Morgan novelty {'更高，支持更偏向 chemical outliers 的描述' if hist_mean['raw_gradient_maxdet']['morgan_novelty_percentile'] > hist_mean['raw_gradient_lcmd']['morgan_novelty_percentile'] else '并未高于 LCMD，不支持它更极端偏向 chemical outliers 的说法'}。这些是选择几何的描述，不是收益的因果分解。", '',
        f"**Q4**：Latent/Gradient test AULC={lat/grad:.4f}；本 seed 上 Latent {'优于' if lat < grad else '未优于'} Gradient-LCMD。{'为 early-stage 使用 latent coverage 提供方向性证据' if lat < grad else '不支持在该 early stage 用 latent coverage 替代 gradient 的优势'}，尚不能推广到其他初始化。", '',
        f"**Q5**：Morgan/Latent test AULC={morg/lat:.4f}。{'Morgan 更好，与 model-independent coverage 更可靠、learned representation 尚未成熟的假设一致；但算法与几何差异也可能解释结果，不能确证机制。' if morg < lat else 'Morgan 未优于 Latent，本实验不支持该条件性推论。'}", '',
        f"**Q6**：Latent/Morgan test AULC={lat/morg:.4f}。{'Latent 更好，支持 QGeoGNN 已包含有用 task-aware 信息的方向性解释，即使最终 R² 仍有限；不能仅凭一条轨迹排除采样偶然性。' if lat < morg else 'Latent 未优于 Morgan，本实验不支持该条件性推论。'}", '',
        f"**Q7**：test AULC 排序为 {' < '.join(rank)}。{'Gradient-LCMD 最好，与 task sensitivity 含有 chemical coverage 之外信息的假设一致，但还需要机制消融验证。' if rank[0] == 'raw_gradient_lcmd' else 'Gradient-LCMD 并非最低，因此不能用本实验支持它仍然最好的前提。'}三个 active methods 相对 Random 的 AULC ratio 分别为 Gradient={grad/rnd:.4f}、Latent={lat/rnd:.4f}、Morgan={morg/rnd:.4f}。", '',
        '## 限制与下一阶段建议', '',
        '只有一个初始化/训练 seed，不计算显著性、不把小差异视为可靠胜负。已有 transfer_v2 的开发 test 结果在本研究提出前已被知晓，因此本次统一 reveal 维持的是本 study 的计算防火墙，不构成全新、未接触的外部验证集。分子重复、条件变化和 scaffold 定义也会影响 coverage；空 scaffold 视为同一个 acyclic 类。', '',
        '下一阶段建议优先 A：在更大 L0 下做 competence-matched representation comparison，检查 predictor 能力提高是否改变排序。若当前 Morgan 占优，这能检验 learned representation 是否随训练变成熟；若 latent 或 gradient 占优，则能检验收益是否随阶段持续。Ensemble uncertainty、chemical+uncertainty hybrid 和 stage-adaptive acquisition 仅列为后续候选，本轮不启动。', '',
        '所有完整指标、分布、交集和 pairwise AULC ratios 已写入 results/；冻结记录与历史不可变性核查见 completion_manifest.json。', '']
    (study / 'REPORT.md').write_text('\n'.join(lines))
    original = read_json(study / 'historical_artifacts.json')
    audit = {name: tree_hashes(study.parent / name) == hashes for name, hashes in original.items()}
    if not all(audit.values()):
        raise RuntimeError('historical artifacts changed')
    atomic_json(study / 'historical_integrity_final.json', audit)
    summary = dict(status='COMPLETE_STOPPED', methods=4, seeds=1, budgets=list(BUDGETS),
                   aulc=aulc, test_ranking=rank, history_unchanged=all(audit.values()))
    atomic_json(result / 'summary.json', summary)
    files = {str(p.relative_to(study)): sha(p) for p in study.rglob('*') if p.is_file() and p.suffix != '.log'}
    write_once(study / 'completion_manifest.json', dict(**summary, files=files))
    print(summary)


if __name__ == '__main__':
    report()
