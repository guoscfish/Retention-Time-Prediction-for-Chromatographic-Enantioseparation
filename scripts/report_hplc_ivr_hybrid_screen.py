"""Post-freeze report for the fixed HPLC IVR/Hybrid development screen."""

import csv
import os
import sys
from itertools import combinations
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/hplc-ivr-hybrid-mpl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hplc_al.common import atomic_json, metrics, read_json, sha, verify_files, write_once
from hplc_al.ivr_hybrid_screen import (BUDGETS, COMPARATORS, EXTENSION, METHODS,
                                        PARENT, SCALE, STUDY, source_round,
                                        verify_freeze)
from hplc_al.protocol import role_ids


def csv_write(path, rows):
    if not rows:
        raise ValueError('empty CSV')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aulc(points, start, stop):
    group = [p for p in points if start <= p['budget'] <= stop]
    if [p['budget'] for p in group] != list(range(start, stop+1, 32)):
        raise RuntimeError('incomplete AULC grid')
    return float(np.trapz([p['nrmse'] for p in group], [p['budget'] for p in group]) / (stop-start))


def report(study=STUDY):
    study = Path(study)
    frozen = verify_freeze(study)
    verify_files(study, read_json(study / 'test_reveal.json')['files'])
    partition = read_json(study / 'splits/partition.json')
    valid, test = role_ids(partition, 'validation'), role_ids(partition, 'test')
    with np.load(study / 'test_truth.npz') as bank:
        if bank['ids'].tolist() != test:
            raise RuntimeError('test truth row alignment drift')
        truth = bank['truth'].copy()
    output = study / 'results'
    output.mkdir(exist_ok=True)
    methods = COMPARATORS + METHODS
    historical = list(csv.DictReader((EXTENSION / 'results/metrics.csv').open()))
    rows = []
    for method in methods:
        for r, budget in enumerate(BUDGETS):
            if method in COMPARATORS:
                directory = source_round(method, r)
                record = read_json(directory / 'round.json')
            else:
                directory = study / f'runtime/{method}/round_{r}'
                record = read_json(directory / 'round.json')
            with np.load(directory / 'predictions.npz') as bank:
                if bank['ids'].tolist() != valid + test:
                    raise RuntimeError('prediction ID drift')
                predictions = bank['predictions'].copy()
            recomputed = metrics(predictions[len(valid):, 1], truth, SCALE)
            if method in COMPARATORS:
                original = next(row for row in historical if row['method'] == method and
                                int(row['budget']) == budget and row['split'] == 'test')
                if any(not np.isclose(float(original[k]), recomputed[k], atol=1e-8, rtol=1e-8)
                       for k in ('rmse', 'mae', 'r2', 'nrmse')):
                    raise RuntimeError('historical comparator test metric recomputation differs')
            for split, value in [('validation', record['validation_metrics']), ('test', recomputed)]:
                rows.append(dict(method=method, budget=budget, split=split,
                                 **{k:float(value[k]) for k in ('rmse', 'mae', 'r2', 'nrmse')}))
    csv_write(output / 'metrics.csv', rows)
    areas = []
    lookup = {}
    for split in ('validation', 'test'):
        for method in methods:
            points = sorted((row for row in rows if row['method'] == method and row['split'] == split),
                            key=lambda row: row['budget'])
            lookup[method, split] = {name: aulc(points, first, last) for name, first, last in
                                     [('early',333,429), ('extension',429,493), ('full',333,493)]}
        for method in methods:
            for segment, first, last in [('early',333,429), ('extension',429,493), ('full',333,493)]:
                value = lookup[method,split][segment]
                areas.append(dict(method=method, split=split, segment=segment,
                                  start_budget=first, end_budget=last,
                                  normalized_nrmse_aulc=value,
                                  ratio_to_random=value/lookup['random',split][segment],
                                  ratio_to_raw_gradient_lcmd=value/lookup['raw_gradient_lcmd',split][segment]))
    csv_write(output / 'aulc.csv', areas)
    geometry, overlaps, ensemble_rows = [], [], []
    for r, budget in enumerate(BUDGETS[:-1]):
        batches = {}
        for method in methods:
            selection_dir = (EXTENSION / f'runtime/{method}/round_3' if method in COMPARATORS and r == 3
                             else source_round(method, r) if method in COMPARATORS
                             else study / f'runtime/{method}/round_{r}')
            selection = read_json(selection_dir / 'selection.json')
            batches[method] = set(selection['selected'])
            if method in METHODS:
                g = selection['geometry']
                base = dict(method=method, budget=budget,
                            selected_gradient_norm_percentile='', ivr_score_percentile='',
                            predicted_variance_reduction='', relative_predicted_reduction='',
                            nearest_l_gradient_distance='', ensemble_uncertainty_percentile='',
                            shortlist_uncertainty_percentile='', shortlist_threshold='',
                            selected_uncertainty_mean='', shortlist_uncertainty_mean='',
                            selected_latent_nearest_l_distance='', selected_morgan_novelty='',
                            novel_scaffold_fraction='', top32_uncertainty_retained='',
                            top_shortlist_quartile_retained='')
                if method == 'kernel_ivr':
                    base.update(selected_gradient_norm_percentile=g['gradient_norm_percentile'],
                                ivr_score_percentile=g['initial_ivr_score_percentile'],
                                predicted_variance_reduction=g['ivr']['predicted_total_reduction'],
                                relative_predicted_reduction=g['ivr']['relative_predicted_reduction'],
                                nearest_l_gradient_distance=g['nearest_l_gradient_distance'])
                else:
                    with np.load(selection_dir / 'ensemble_predictions.npz') as bank:
                        current_u = bank['ids'].tolist()
                        std = bank['central'].astype(np.float64).std(axis=0, ddof=0)
                    order = np.argsort(-std, kind='stable')
                    ranks = np.empty(len(std), dtype=np.int64)
                    ranks[order] = np.arange(1, len(std)+1)
                    shortlist_ids = {current_u[i] for i in order[:g['hybrid']['shortlist_size']]}
                    for i, sample_id in enumerate(current_u):
                        ensemble_rows.append(dict(budget=budget, sample_id=sample_id,
                                                  uncertainty=std[i],
                                                  normalized_uncertainty=std[i]/SCALE,
                                                  rank=int(ranks[i]),
                                                  in_shortlist=sample_id in shortlist_ids,
                                                  selected=sample_id in batches[method]))
                    base.update(ensemble_uncertainty_percentile=g['uncertainty_percentile'],
                                shortlist_uncertainty_percentile=g['shortlist_uncertainty_percentile'],
                                shortlist_threshold=g['hybrid']['shortlist_threshold'],
                                selected_uncertainty_mean=g['selected_uncertainty_mean'],
                                shortlist_uncertainty_mean=g['shortlist_uncertainty_mean'],
                                selected_latent_nearest_l_distance=g['selected_latent_nearest_l_distance'],
                                selected_morgan_novelty=g['selected_morgan_novelty'],
                                novel_scaffold_fraction=g['novel_scaffold_fraction'],
                                top32_uncertainty_retained=g['top32_uncertainty_retained'],
                                top_shortlist_quartile_retained=g['top_shortlist_quartile_retained'])
                geometry.append(base)
        for first, second in combinations(methods, 2):
            shared = sorted(batches[first] & batches[second])
            overlaps.append(dict(budget=budget, first=first, second=second,
                                 count=len(shared), jaccard=len(shared)/len(batches[first]|batches[second]),
                                 shared_ids=';'.join(map(str,shared))))
    csv_write(output / 'acquisition_geometry.csv', geometry)
    csv_write(output / 'selection_overlap.csv', overlaps)
    csv_write(output / 'ensemble_uncertainty.csv', ensemble_rows)
    costs = []
    for method in METHODS:
        evaluation = [read_json(study / f'runtime/{method}/round_{r}/fit/fit.json') for r in range(1,6)]
        extras = ([read_json(study / f'runtime/{method}/round_{r}/member_{member}/fit/fit.json')
                   for r in range(5) for member in (1,2)] if method == 'hybrid_ensemble_latent' else [])
        selections = [read_json(study / f'runtime/{method}/round_{r}/selection.json') for r in range(5)]
        costs.append(dict(method=method, evaluation_fits_new=len(evaluation),
                          evaluation_fit_l333_reused=1, extra_ensemble_fits=len(extras),
                          total_new_fits=len(evaluation)+len(extras),
                          evaluation_epochs=sum(x['epochs_run'] for x in evaluation),
                          extra_ensemble_epochs=sum(x['epochs_run'] for x in extras),
                          total_new_epochs=sum(x['epochs_run'] for x in evaluation+extras),
                          evaluation_training_seconds=sum(x['seconds'] for x in evaluation),
                          extra_ensemble_training_seconds=sum(x['seconds'] for x in extras),
                          total_training_seconds=sum(x['seconds'] for x in evaluation+extras),
                          acquisition_stage_seconds=sum(x['selection_seconds'] for x in selections),
                          new_gradient_extractions=4 if method == 'kernel_ivr' else 0))
    csv_write(output / 'compute_cost.csv', costs)
    colors = dict(random='#64748b', raw_gradient_lcmd='#dc2626',
                  kernel_ivr='#2563eb', hybrid_ensemble_latent='#059669')
    fig, axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax, split in zip(axes, ('validation','test')):
        for method in methods:
            points = sorted((row for row in rows if row['method']==method and row['split']==split),
                            key=lambda row:row['budget'])
            ax.plot([p['budget'] for p in points], [p['nrmse'] for p in points],
                    marker='o', label=method, color=colors[method])
        ax.set(title=split.title(), xlabel='Labeled budget', ylabel='NRMSE (frozen L333 SD)',
               xticks=BUDGETS)
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=7, loc='best')
    fig.savefig(output / 'learning_curves.png', dpi=180)
    plt.close(fig)
    full = {m:lookup[m,'test']['full'] for m in methods}
    extension = {m:lookup[m,'test']['extension'] for m in methods}
    endpoint = {m:next(row['nrmse'] for row in rows if row['method']==m and row['split']=='test' and row['budget']==493)
                for m in methods}
    qualifying = [m for m in METHODS if full[m] < full['random'] and
                  (full[m] < full['raw_gradient_lcmd'] or extension[m] < extension['raw_gradient_lcmd']
                   or endpoint[m] < endpoint['raw_gradient_lcmd'])]
    recommended = min(qualifying, key=lambda m:full[m]) if qualifying else 'raw_gradient_lcmd'
    latent_context = next(row for row in historical if row['method']=='latent_coreset' and row['split']=='test' and int(row['budget'])==493)
    overlaps_lookup = {(row['budget'],row['first'],row['second']):row for row in overlaps}
    ivr_lcmd_overlap = [overlaps_lookup[b,'raw_gradient_lcmd','kernel_ivr']['count'] for b in BUDGETS[:-1]]
    hybrid_lcmd_overlap = [overlaps_lookup[b,'raw_gradient_lcmd','hybrid_ensemble_latent']['count'] for b in BUDGETS[:-1]]
    ivr_hybrid_overlap = [overlaps_lookup[b,'kernel_ivr','hybrid_ensemble_latent']['count'] for b in BUDGETS[:-1]]
    hybrid_geometry = [g for g in geometry if g['method']=='hybrid_ensemble_latent']
    mean_selected_uncertainty_percentile = float(np.mean([g['ensemble_uncertainty_percentile'] for g in hybrid_geometry]))
    mean_shortlist_uncertainty_percentile = float(np.mean([g['shortlist_uncertainty_percentile'] for g in hybrid_geometry]))
    top32_retained = sum(g['top32_uncertainty_retained'] for g in hybrid_geometry)
    latent_full = next(float(row['full_normalized_aulc']) for row in csv.DictReader((EXTENSION/'results/aulc.csv').open())
                       if row['method']=='latent_coreset' and row['split']=='test')
    lines = ['# ODH Kernel-IVR / Hybrid development screen', '',
             'Seed 525 only; fixed L333–L493 schedule; hard stop at L493. The same test cohort was exposed in historical work, so these results are exploratory development evidence.', '',
             '## Source transfer and comparability', '',
             'CC Kernel-IVR exact mechanism: fixed original outer-training reference, one global gradient RMS, unit prior/noise, scalar row-kernel IVR, Sherman–Morrison greedy conditioning after every pick. HPLC feature adaptation: its full-network central RTv `output[:,1]` 512D CountSketch instead of CC endpoint-scaled concatenation. Prior/noise, batch size, reference and numerical objective remained frozen.', '',
             'CC Hybrid principle: three-model epistemic uncertainty → Top-25% shortlist → latent k-center. HPLC adaptation: scalar central RTv std with `ddof=0`, unit `h_graph`, and cosine distance because raw HPLC latent norm correlates with molecule size. This is not a direct or bitwise CC Hybrid reproduction. K=3, 25%, B=32, seeds and training protocol remained frozen.', '',
             'Historical Random and Raw Gradient-LCMD artifacts were reused read-only. Their partition, L333 IDs, training contract, source checkpoint, validation/test-X prediction IDs, budget transitions, target, metric and NRMSE denominator passed the comparator audit. Their test metrics were independently recomputed only after the new global freeze. The two new methods share the exact L333 checkpoint and test-X prediction file contents; later states branch independently.', '',
             '## Primary AULC (test NRMSE; lower is better)', '',
             '| Method | Early 333–429 | Extension 429–493 | Full 333–493 | Full / Random | Full / LCMD |',
             '|---|---:|---:|---:|---:|---:|']
    for method in methods:
        lines.append(f"| {method} | {lookup[method,'test']['early']:.6f} | {extension[method]:.6f} | {full[method]:.6f} | {full[method]/full['random']:.4f} | {full[method]/full['raw_gradient_lcmd']:.4f} |")
    lines += ['', '## Per-budget test metrics', '',
              '| Method | Budget | RMSE | MAE | R² | NRMSE |', '|---|---:|---:|---:|---:|---:|']
    for method in methods:
        for point in (row for row in rows if row['method']==method and row['split']=='test'):
            lines.append(f"| {method} | {point['budget']} | {point['rmse']:.4f} | {point['mae']:.4f} | {point['r2']:.4f} | {point['nrmse']:.4f} |")
    lines += ['', '## Acquisition mechanism and cost', '',
              f"Batch overlap counts with Raw Gradient-LCMD at acquisition budgets {list(BUDGETS[:-1])}: IVR {ivr_lcmd_overlap}; Hybrid {hybrid_lcmd_overlap}. IVR–Hybrid overlap: {ivr_hybrid_overlap}. Detailed Jaccard and IDs are in `results/selection_overlap.csv`.", '',
              'IVR initial score distribution, integrated variance before/after, predicted reduction, denominator and condition diagnostics are in each IVR `selection.json`; the CSV summarizes norm and score percentiles. Hybrid shortlist thresholds, uncertainty percentiles, retained top uncertainty ranks, latent distance, Morgan novelty and scaffold fraction are in its `selection.json` and the CSV. Morgan is diagnostic only.', '',
              '| Method | New evaluation fits | Extra ensemble fits | New epochs | Training seconds |',
              '|---|---:|---:|---:|---:|']
    for cost in costs:
        lines.append(f"| {cost['method']} | {cost['evaluation_fits_new']} | {cost['extra_ensemble_fits']} | {cost['total_new_epochs']} | {cost['total_training_seconds']:.1f} |")
    lines += ['', '## Scientific interpretation', '',
              f"Q1. A scalar RTv makes the IVR surrogate directly aligned with the single evaluated output; that is a modeling simplification, not proof of superiority. IVR full AULC is {full['kernel_ivr']:.6f} versus Random {full['random']:.6f}, a {(1-full['kernel_ivr']/full['random'])*100:.2f}% descriptive gain. It does not beat LCMD.", '',
              f"Q2. IVR and LCMD share the gradient representation at L333 and refresh that same feature definition from their own checkpoint thereafter. Their full AULCs are {full['kernel_ivr']:.6f} and {full['raw_gradient_lcmd']:.6f}; first-batch overlap {ivr_lcmd_overlap[0]}/32 at the identical initial state, and only {sum(ivr_lcmd_overlap)}/160 picks overlap across all five rounds. The first difference is attributable to acquisition objective; later performance also includes trajectory-induced model changes. IVR is nearly tied on early AULC but loses on extension AULC.", '',
              f"Q3. Hybrid full AULC is {full['hybrid_ensemble_latent']:.6f}. Across five batches, selected uncertainty percentile averages {mean_selected_uncertainty_percentile:.2f} versus shortlist {mean_shortlist_uncertainty_percentile:.2f}; only {top32_retained}/160 selected samples retain a Top-32 uncertainty rank. Diversity preserves generally high disagreement but removes most extreme uncertainty picks. The resulting trajectory does not beat LCMD, so this screen does not show added label efficiency from ensemble disagreement beyond the gradient baseline.", '',
              f"Q4. Historical Latent-Coreset full AULC was {latent_full:.6f}, versus Hybrid {full['hybrid_ensemble_latent']:.6f}; its L493 test NRMSE was {float(latent_context['nrmse']):.6f}, versus Hybrid {endpoint['hybrid_ensemble_latent']:.6f}. Hybrid improves on this mechanism background within seed 525, but Latent-Coreset is not a primary comparator.", '',
              f"Q5. LCMD test NRMSE at L461/L493 is {[next(row['nrmse'] for row in rows if row['method']=='raw_gradient_lcmd' and row['split']=='test' and row['budget']==b) for b in (461,493)]}; its extension AULC is {extension['raw_gradient_lcmd']:.6f} versus Random {extension['random']:.6f}, IVR {extension['kernel_ivr']:.6f}, and Hybrid {extension['hybrid_ensemble_latent']:.6f}. Its late-budget advantage persists in this seed.", '',
              f"Q6. The fixed practical gate recommends **{recommended}** for the next confirmation stage. The minimum follow-up is Random versus Raw Gradient-LCMD" + (f" versus {recommended}" if recommended != 'raw_gradient_lcmd' else '') + ' on seeds 1525 and 2525. No such run was started.', '',
              f"No significance conclusion is available from one seed. Validation selected every checkpoint; IVR validation full AULC ({lookup['kernel_ivr','validation']['full']:.6f}) is slightly lower than LCMD ({lookup['raw_gradient_lcmd','validation']['full']:.6f}), while their test ordering reverses. Historical test exposure means this cohort is not an independent fresh external test, despite the new computational test-truth firewall. Unit prior/noise and global gradient RMS remain surrogate assumptions; no tuning was conducted.", '',
              '![Learning curves](results/learning_curves.png)', '']
    (study / 'REPORT.md').write_text('\n'.join(lines))
    summary = dict(status='COMPLETE_STOPPED_AT_493', seed=525, methods=list(methods),
                   test_full_aulc=full, test_extension_aulc=extension,
                   test_early_aulc={m:lookup[m,'test']['early'] for m in methods},
                   test_endpoint_493_nrmse=endpoint, recommended_confirmation=recommended,
                   ivr_lcmd_batch_overlap=ivr_lcmd_overlap,
                   hybrid_lcmd_batch_overlap=hybrid_lcmd_overlap,
                   ivr_hybrid_batch_overlap=ivr_hybrid_overlap,
                   comparator_audit_sha256=sha(study / 'comparator_audit.json'),
                   global_freeze_sha256=sha(study / 'global_pre_test_freeze.json'))
    atomic_json(output / 'summary.json', summary)
    files = {str(p.relative_to(study)):sha(p) for p in study.rglob('*')
             if p.is_file() and p.name not in ('label_access_audit.csv', 'completion_manifest.json')
             and p.suffix != '.log'}
    atomic_json(study / 'completion_manifest.json', dict(status='COMPLETE_STOPPED_AT_493', files=files))
    print(summary)


if __name__ == '__main__':
    report()
