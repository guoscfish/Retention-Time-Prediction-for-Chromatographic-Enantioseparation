"""Read-only evaluation of the globally frozen bounded transfer-v2 study."""
import csv
import itertools
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/hplc-transfer-v2-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hplc_al.common import ROOT, atomic_json, metrics, read_json, sha, stable_hash
from hplc_al.protocol import role_ids
from hplc_al.transfer_v2 import ACQUISITIONS, BUDGETS, unit_gradient
from hplc_al.transfer_v2_runner import verify_runtime_freeze

STUDY = ROOT / 'studies/active_learning/odh_gradient_al_transfer_v2'
NAMES = {'random':'Random', 'raw_gradient_lcmd':'Raw-LCMD', 'unit_gradient_lcmd':'Unit-LCMD', 'raw_gradient_maxdet':'Raw-MaxDet', 'unit_gradient_maxdet':'Unit-MaxDet'}


def table(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def geometry(raw, selected, labeled, pool):
    norm = np.linalg.norm(raw, axis=1)
    selected_norm = norm[selected]
    out = dict(selected_raw_norm_percentile_U=float(np.mean(norm[pool,None] <= selected_norm[None,:])*100),
               selected_top_decile_fraction_U=float(np.mean(selected_norm >= np.quantile(norm[pool],.9))),
               selected_raw_norm_percentile_outer=float(np.mean(norm[:,None] <= selected_norm[None,:])*100),
               selected_top_decile_fraction_outer=float(np.mean(selected_norm >= np.quantile(norm,.9))),
               selected_raw_norm_mean=float(selected_norm.mean()))
    for name,matrix in [('raw',raw),('unit',unit_gradient(raw))]:
        x=matrix[selected]
        centered=x-x.mean(axis=0,keepdims=True)
        singular=np.linalg.svd(centered,compute_uv=False)
        p=singular[singular>0]/singular.sum() if singular.sum()>0 else np.array([])
        effective=float(np.exp(-np.sum(p*np.log(p)))) if len(p) else 0.0
        pair=np.linalg.norm(x[:,None]-x[None,:],axis=-1)
        near=np.linalg.norm(x[:,None]-matrix[None,labeled,:],axis=-1).min(axis=1)
        out.update({f'{name}_effective_rank':effective,
                    f'{name}_algebraic_rank':int(np.linalg.matrix_rank(centered)),
                    f'{name}_pairwise_distance':float(pair[np.triu_indices(len(x),1)].mean()),
                    f'{name}_nearest_L_distance':float(near.mean())})
    return out


def main():
    records=verify_runtime_freeze(STUDY)
    reveal=read_json(STUDY/'test_reveal.json')
    assert reveal['test_truth_access_count']==1 and reveal['records']==20
    accesses=list(csv.DictReader((STUDY/'label_access_audit.csv').open()))
    final=[row for row in accesses if row['purpose']=='final_test']
    assert len(final)==1 and final[0]['allowed']=='True' and final[0]['test_frozen']=='True'
    assert accesses[-1]==final[0]
    partition=read_json(STUDY/'splits/partition.json')
    valid=role_ids(partition,'validation'); test=role_ids(partition,'test')
    l0=role_ids(partition,'l0'); u0=role_ids(partition,'u0'); outer=sorted(l0+u0)
    ix={row:i for i,row in enumerate(outer)}
    scale=read_json(STUDY/'duration_smoke/seed_525/duration_record.json')['L0_population_std']
    with np.load(STUDY/'test_truth.npz') as f:
        assert f['ids'].tolist()==test
        test_truth=f['truth'].copy()
    out=STUDY/'results'; out.mkdir(exist_ok=True)
    curves=[]; geometry_rows=[]; fit_paths=set(); banks=set(); selectors=[]; stopping=[]
    for method in ACQUISITIONS:
        for round_index,budget in enumerate(BUDGETS):
            r=records[f'{method}/{round_index}']
            directory=STUDY/f'runtime/{method}/round_{round_index}'
            with np.load(directory/'predictions.npz') as f:
                assert f['ids'].tolist()==valid+test
                pred=f['predictions'].copy()
            test_metric=metrics(pred[len(valid):,1],test_truth,scale)
            curves.extend([dict(method=method,budget=budget,split=split,**m) for split,m in [('validation',r['validation_metrics']),('test',test_metric)]])
            fit_path=next(name for name in r['files'] if name.endswith('/fit.json'))
            fit_paths.add(fit_path)
            fit=read_json(STUDY/fit_path)
            stopping.append(dict(method=method,budget=budget,best_epoch=fit['best_epoch'],stopped_epoch=fit['stopped_epoch'],best_optimizer_step=fit['best_optimizer_step'],stopping_reason=fit['stopping_reason'],steps_per_epoch=fit['steps_per_epoch'],actual_batch_sizes=str(fit['observed_batch_sizes']),seconds=fit['seconds'],reused_round_0=round_index==0))
            if round_index==3:
                continue
            bank=STUDY/'runtime/shared_round_0/raw_gradient_bank' if round_index==0 else directory/'raw_gradient_bank'
            banks.add(str(bank.relative_to(STUDY)))
            with np.load(bank/'features.npz') as f:
                assert f['ids'].tolist()==outer
                raw=f['features'].astype(np.float64)
            selected=[ix[i] for i in r['selected']]; labeled=[ix[i] for i in r['labeled_ids']]; pool=[ix[i] for i in r['unlabeled_ids']]
            audit=read_json(directory/'selection.json')['selection_audit']
            selectors.append(dict(method=method,round=round_index,seconds=audit.get('selector_seconds',0)))
            geometry_rows.append(dict(method=method,from_budget=budget,to_budget=budget+32,gradient_use='diagnostic_only' if method=='random' else 'acquisition',**geometry(raw,selected,labeled,pool)))
    aulc=[]
    for method in ACQUISITIONS:
        for split in ['validation','test']:
            points=[next(row for row in curves if row['method']==method and row['split']==split and row['budget']==b) for b in BUDGETS]
            mean_rmse=float(np.trapz([p['rmse'] for p in points], BUDGETS)/(BUDGETS[-1]-BUDGETS[0]))
            aulc.append(dict(method=method,split=split,integrated_rmse=mean_rmse*96,budget_normalized_AULC_RMSE=mean_rmse,normalized_AULC_NRMSE=mean_rmse/scale))
    for a in aulc:
        baseline=next(x for x in aulc if x['method']=='random' and x['split']==a['split'])
        a['AULC_ratio_to_random']=a['budget_normalized_AULC_RMSE']/baseline['budget_normalized_AULC_RMSE']
    pairs=[('raw_gradient_lcmd','unit_gradient_lcmd'),('raw_gradient_maxdet','unit_gradient_maxdet')]+[(m,'random') for m in ACQUISITIONS if m!='random']
    paired=[]; overlaps=[]
    for a,b in pairs:
        for split in ['validation','test']:
            for budget in BUDGETS:
                left=next(x for x in curves if x['method']==a and x['split']==split and x['budget']==budget)
                right=next(x for x in curves if x['method']==b and x['split']==split and x['budget']==budget)
                paired.append(dict(method_A=a,method_B=b,split=split,budget=budget,**{f'delta_{k}_A_minus_B':left[k]-right[k] for k in ['rmse','mae','r2','nrmse']}))
        for round_index in range(3):
            aa=set(records[f'{a}/{round_index}']['selected']); bb=set(records[f'{b}/{round_index}']['selected'])
            la=set(records[f'{a}/{round_index}']['labeled_ids'])|aa; lb=set(records[f'{b}/{round_index}']['labeled_ids'])|bb
            overlaps.append(dict(method_A=a,method_B=b,from_budget=BUDGETS[round_index],selected_overlap=len(aa&bb),selected_overlap_fraction=len(aa&bb)/32,selected_jaccard=len(aa&bb)/len(aa|bb),cumulative_acquired_overlap=len((la-set(l0))&(lb-set(l0))),cumulative_acquired_jaccard=len((la-set(l0))&(lb-set(l0)))/len((la|lb)-set(l0)),same_predictor_and_candidate_pool=round_index==0))
    fit_records=[read_json(STUDY/p) for p in sorted(fit_paths)]
    training_seconds=sum(r['seconds'] for r in fit_records)
    gradient_seconds=sum(read_json(STUDY/b/'audit.json')['seconds'] for b in banks)
    duration_extra=read_json(STUDY/'duration_smoke/seed_1525/fit.json')['seconds']
    archived_fits=[read_json(p) for p in (STUDY/'interrupted_attempt_20260923').glob('runtime/**/fit.json')]
    archived_banks=[read_json(p) for p in (STUDY/'interrupted_attempt_20260923').glob('runtime/**/raw_gradient_bank/audit.json')]
    compute=dict(valid_AL_unique_fits=len(fit_paths),AL_new_fits_this_continuation=len(fit_paths)-1,AL_round0_reused_duration_fit=True,valid_AL_unique_gradient_passes=len(banks),acquisition_gradient_passes=len(banks)-2,random_diagnostic_gradient_passes=2,training_seconds_including_reused_seed525=training_seconds,new_AL_training_seconds=training_seconds-read_json(STUDY/'duration_smoke/seed_525/fit.json')['seconds'],gradient_extraction_seconds=gradient_seconds,selector_seconds=sum(s['seconds'] for s in selectors),duration_seed1525_extra_seconds=duration_extra,valid_work_unique_fits_including_duration=len(fit_paths)+1,archived_unique_fits=len(archived_fits),archived_unique_gradient_passes=len(archived_banks),archived_training_seconds=sum(r['seconds'] for r in archived_fits),archived_gradient_seconds=sum(r['seconds'] for r in archived_banks),total_executed_unique_fits=len(fit_paths)+1+len(archived_fits),total_executed_unique_gradient_passes=len(banks)+len(archived_banks))
    per_method=[]
    for method in ACQUISITIONS:
        per_method.append(dict(method=method,new_training_seconds=sum(r['seconds'] for r in stopping if r['method']==method and r['budget']>333),gradient_seconds_excluding_shared=sum(read_json(STUDY/f'runtime/{method}/round_{i}/raw_gradient_bank/audit.json')['seconds'] for i in [1,2]),selector_seconds=sum(s['seconds'] for s in selectors if s['method']==method)))
    table(out/'learning_curves.csv',curves); table(out/'normalized_aulc.csv',aulc)
    table(out/'paired_differences.csv',paired); table(out/'acquisition_geometry.csv',geometry_rows)
    table(out/'selection_overlap.csv',overlaps); table(out/'training_duration_all_fits.csv',stopping)
    table(out/'compute_by_method.csv',per_method)
    atomic_json(out/'compute.json',compute)
    # Verify the label audit against the sealed selections, with no source truth access.
    for row in accesses:
        ids=set(map(int,row['ids'].split(';')))
        assert row['allowed']=='True'
        if row['purpose']=='fit':
            authorized=set(l0)
            if row['method'] in ACQUISITIONS:
                for round_index in range(int(row['round'])):
                    authorized.update(records[f"{row['method']}/{round_index}"]['selected'])
            assert ids <= authorized
        elif row['purpose']=='validation':
            assert ids == set(valid)
        else:
            assert row['purpose']=='final_test' and ids==set(test)
    colors=['#5f6b7a','#a44a20','#eb9549','#2264ac','#46a7a1']
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),constrained_layout=True)
    for ax,split in zip(axes,['validation','test']):
        for method,color in zip(ACQUISITIONS,colors):
            points=[next(r for r in curves if r['method']==method and r['split']==split and r['budget']==b) for b in BUDGETS]
            ax.plot(BUDGETS,[p['rmse'] for p in points],marker='o',label=NAMES[method],color=color,linewidth=1.8)
        ax.set(xlabel='Labeled budget',ylabel='Central RMSE (RTv)',title=f'{split.capitalize()} | initialization seed 525',xticks=BUDGETS)
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.savefig(out/'learning_curves.png',dpi=180); fig.savefig(out/'learning_curves.pdf'); plt.close(fig)
    result=dict(status='COMPLETE_STOP',methods=list(ACQUISITIONS),budgets=list(BUDGETS),initialization_seed=525,split_seed=partition['seed'],L0_seed=partition['l0_seed'],L0_population_std=scale,test_truth_reveals=1,frozen_method_budget_artifacts=20,round_0_shared_checkpoint=True,round_0_shared_raw_bank=True,unit_definition='unit-normalized sketched gradient; phi_raw / max(L2(phi_raw), 1e-12)',nrmse_definition='RMSE / frozen L333 population standard deviation',normalized_aulc_definition='trapezoid integral of NRMSE over budgets 333..429 / 96; lower is better',effective_rank_definition='exp(-sum(p*log(p))), p = singular values of centered selected bank / their sum; raw and unit spaces reported separately',percentile_reference='current unlabeled candidate pool U; outer-train reference also supplied',paired_difference_direction='method A minus method B; no inferential significance claimed for one seed',comparison_limitation='Raw vs Unit at round 0 shares predictor and pool; later geometry differences also include divergent label sets and models.',compute=compute,curves=curves,aulc=aulc,source_test_truth_sha256=sha(STUDY/'test_truth.npz'),label_access_audit_sha256=sha(STUDY/'label_access_audit.csv'))
    atomic_json(out/'summary.json',result)
    print('COMPLETE_STOP: 20 artifacts verified, one test reveal, report tables and plots written.')
    print(compute)
    for row in aulc:
        if row['split']=='test': print(row)


if __name__=='__main__':
    main()
