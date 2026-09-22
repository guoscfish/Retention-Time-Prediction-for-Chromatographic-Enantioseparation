"""Post-freeze diagnostics only: never trains, acquires or reads source targets."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from hplc_al.common import read_json, atomic_json, METHODS, sha
from hplc_al.protocol import verify_global_freeze, role_ids


def main():
    study = Path(__file__).resolve().parent
    verify_global_freeze(study)
    partition = read_json(study/'splits/partition.json')
    outer = sorted(role_ids(partition,'l0')+role_ids(partition,'u0'))
    lookup = {v:i for i,v in enumerate(outer)}
    geometries, fits = [], []
    for method in METHODS:
        for r in range(4):
            directory = study/f'runtime/{method}/round_{r}'
            record = read_json(directory/'round.json')
            fit_dir = study/'runtime/shared_round_0' if r==0 else directory/'fit'
            fit = read_json(fit_dir/'fit.json')
            curve_path = next(fit_dir/name for name in fit['files'] if name.endswith('training_curve.csv'))
            curve = list(csv.DictReader(curve_path.open()))
            scores = np.array([float(x['validation_mse']) for x in curve])
            fits.append(dict(method=method,round=r,budget=record['budget'],
                stopping_reason=fit['stopping_reason'],best_epoch=fit['best_epoch'],
                epochs=fit['epochs_run'],last_validation_rmse=float(np.sqrt(scores[-1])),
                best_validation_rmse=float(np.sqrt(scores.min())),
                last_to_best_mse_ratio=float(scores[-1]/scores.min()),
                last_50_mse_std=float(scores[-50:].std()),
                maximum_epochs_hit=fit['stopping_reason']=='maximum_epochs'))
            if method=='random' or r==3:
                continue
            gdir = study/'runtime/shared_round_0/gradient' if r==0 else directory/'gradient'
            audit = read_json(gdir/'audit.json')
            with np.load(gdir/'features.npz') as saved:
                x = saved['features'].astype(np.float64)
                assert saved['ids'].tolist()==outer
            lp=[lookup[i] for i in record['labeled_ids']]
            sp=[lookup[i] for i in record['selected']]
            active=set(lp)
            up=[i for i in range(len(outer)) if i not in active]
            pool,batch=x[up],x[sp]
            pn=np.linalg.norm(pool,axis=1)
            sn=np.linalg.norm(batch,axis=1)
            centered=pool-pool.mean(axis=0)
            eigen=np.maximum(np.linalg.eigvalsh(centered.T@centered/len(pool)),0)
            # Mean over distinct ordered pairs, no arbitrary random reference batch.
            def pair_sq(v):
                return float(2*(np.mean(np.sum(v*v,axis=1))-np.sum(v.mean(axis=0)**2))*len(v)/(len(v)-1))
            nearest=[]
            for start in range(0,len(pool),256):
                z=pool[start:start+256]
                d=np.sum(z*z,axis=1)[:,None]+np.sum(x[lp]**2,axis=1)[None,:]-2*z@x[lp].T
                nearest.extend(np.maximum(d,0).min(axis=1).tolist())
            up_lookup={v:i for i,v in enumerate(up)}
            selected_nearest=np.array(nearest)[[up_lookup[i] for i in sp]]
            geometries.append(dict(method=method,round=r,
                parameter_count=audit['parameter_count'],received_tensors=audit['tensors_receiving_gradient'],
                none_tensors=audit['tensors_without_gradient'],zero_gradient_tensors=audit['zero_gradient_tensors'],
                zero_norm_rows=audit['sketch_norm']['zero_count'],duplicate_feature_rows=audit['duplicate_feature_rows'],
                clamped_central_rows=audit['clamped_central_rows'],
                zero_feature_sample_ids=[outer[i] for i in np.flatnonzero(np.linalg.norm(x,axis=1)==0)],
                mapping_hash=audit['mapping_hash'],full_gradient_norm=audit['full_gradient_norm'],
                sketch_norm=audit['sketch_norm'],
                centered_participation_rank=float(eigen.sum()**2/np.sum(eigen**2)),
                centered_top_eigen_fraction=float(eigen[-1]/eigen.sum()),
                selected_mean_norm=float(sn.mean()),pool_mean_norm=float(pn.mean()),
                selected_median_pool_norm_percentile=float(np.median([(pn<=v).mean() for v in sn])),
                batch_mean_pair_sq=pair_sq(batch),pool_mean_pair_sq=pair_sq(pool),
                batch_to_pool_pair_sq_ratio=pair_sq(batch)/pair_sq(pool),
                selected_mean_nearest_L_sq=float(selected_nearest.mean()),
                pool_mean_nearest_L_sq=float(np.mean(nearest)),
                batch_duplicate_feature_rows=len(batch)-len(np.unique(batch,axis=0))))
    with (study/'label_access_audit.csv').open() as f:
        accesses=list(csv.DictReader(f))
    reveals=[a for a in accesses if a['purpose']=='final_test' and a['allowed']=='True']
    if len(reveals)!=1 or reveals[0]['test_frozen']!='True':
        raise RuntimeError('expected exactly one successful post-freeze test source reveal')
    # Use the already sealed post-freeze cache, never reread source test labels.
    with np.load(study/'results/test_truth.npz') as cache:
        truth=cache['truth'].astype(np.float64)
        test_ids=cache['ids'].tolist()
    test_diagnostics=[]
    for method in METHODS:
        for r in range(4):
            with np.load(study/f'runtime/{method}/round_{r}/predictions.npz') as cache:
                assert cache['ids'].tolist()[247:]==test_ids
                pred=cache['predictions'][247:,1].astype(np.float64)
            error=pred-truth
            squared=error**2
            test_diagnostics.append(dict(method=method,round=r,
                mean_signed_error=float(error.mean()),prediction_std=float(pred.std()),
                truth_std=float(truth.std()),pearson=float(np.corrcoef(pred,truth)[0,1]),
                largest_10_errors_fraction_SSE=float(np.sort(squared)[-10:].sum()/squared.sum()),
                note='All 247 rows retained in every primary metric; no diagnostic subgroup changes the protocol.'))
    result=dict(evidence='ENGINEERING / DEVELOPMENT EVIDENCE ONLY',
                global_freeze_verified=True,successful_test_reveals=len(reveals),
                diagnostic_script_hash=sha(Path(__file__)),training_stability=fits,geometry=geometries,
                diagnostic_test_residuals=test_diagnostics,
                note='Post-hoc diagnostics using frozen features/predictions and sealed test cache; no new source label reads, model fits or acquisition trajectories.')
    atomic_json(study/'results/critical_audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
