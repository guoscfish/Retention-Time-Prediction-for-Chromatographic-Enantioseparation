"""Fixed seed-525 IVR/Hybrid screen with a pre-test-truth artifact firewall."""

from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from .common import ROOT, atomic_json, code_hashes, ids_hash, metrics, read_json, sha, stable_hash, verify_files, write_once
from .data import load_graphs, predict
from .ensemble import member_seed
from .hybrid import select as hybrid_select
from .kernel_ivr import select as ivr_select
from .latent import extract_h_graph
from .protocol import RestrictedLabelStore, role_ids, transition
from .representation_runner import latent_bank, save_bank
from .representation_extension import PARENT, STUDY as EXTENSION
from .runner import assert_environment
from .training import fit, load_model
from .transfer_v2_runner import _gradient_bank, _save_predictions

STUDY = ROOT / 'studies/active_learning/odh_ivr_hybrid_screen_v1'
METHODS = ('kernel_ivr', 'hybrid_ensemble_latent')
COMPARATORS = ('random', 'raw_gradient_lcmd')
BUDGETS = (333, 365, 397, 429, 461, 493)
BATCH = 32
SCALE = 8.879240547556758
CC_COMMIT = 'e50ae7e3719e3563c1c4b11f1d2b7ff180461388'


def source_round(method, round_index):
    study = PARENT if round_index <= 3 else EXTENSION
    return study / f'runtime/{method}/round_{round_index}'


def audit_comparators(study=STUDY):
    """Verify sealed historical identity/training records without reading test metrics."""
    study = Path(study)
    parent_protocol = read_json(PARENT / 'protocol.json')
    extension_protocol = read_json(EXTENSION / 'protocol.json')
    partition = read_json(PARENT / 'splits/partition.json')
    if partition != read_json(EXTENSION / 'splits/partition.json'):
        raise RuntimeError('historical split identity mismatch')
    if stable_hash(partition) != parent_protocol['partition_hash'] != extension_protocol['partition_hash']:
        raise RuntimeError('historical partition hash mismatch')
    if parent_protocol['training'] != extension_protocol['training']:
        raise RuntimeError('historical training protocol mismatch')
    if not np.isclose(parent_protocol['scale'], SCALE, atol=1e-10, rtol=0):
        raise RuntimeError('historical L333 scale mismatch')
    if extension_protocol['batch_size'] != BATCH or extension_protocol['seed'] != 525:
        raise RuntimeError('historical seed or acquisition batch mismatch')
    if read_json(EXTENSION / 'global_pre_test_freeze.json')['stop_budget'] != 493:
        raise RuntimeError('historical extension did not stop at 493')
    parent_manifest = read_json(PARENT / 'completion_manifest.json')
    extension_manifest = read_json(EXTENSION / 'completion_manifest.json')
    verify_files(PARENT, parent_manifest['files'])
    verify_files(EXTENSION, extension_manifest['files'])
    valid, test = role_ids(partition, 'validation'), role_ids(partition, 'test')
    l0, u0 = role_ids(partition, 'l0'), role_ids(partition, 'u0')
    if len(l0) != 333 or len(u0) != 4114 or len(valid) != 247 or len(test) != 247:
        raise RuntimeError('historical role counts mismatch')
    audit = dict(parent_commit='3a4ee4589a1bbb159930d8f0e908afd90f98b7d8',
                 parent_completion_sha256=sha(PARENT / 'completion_manifest.json'),
                 extension_completion_sha256=sha(EXTENSION / 'completion_manifest.json'),
                 partition_sha256=sha(PARENT / 'splits/partition.json'),
                 partition_hash=stable_hash(partition),
                 training_hash=stable_hash(parent_protocol['training']),
                 l333_ids_hash=ids_hash(l0), validation_ids_hash=ids_hash(valid),
                 test_ids_hash=ids_hash(test), outer_ids_hash=ids_hash(l0 + u0),
                 scale=parent_protocol['scale'], budgets=list(BUDGETS),
                 metric='RMSE / frozen L333 population SD; central output[:,1]',
                 comparator_rounds={})
    shared_fit = read_json(PARENT / 'shared/fit/fit.json')
    verify_files(PARENT / 'shared/fit', shared_fit['files'])
    for method in COMPARATORS:
        labeled, unlabeled = l0, u0
        for r, budget in enumerate(BUDGETS):
            directory = source_round(method, r)
            record = read_json(directory / 'round.json')
            verify_files(PARENT if r <= 3 else EXTENSION, record['files'])
            if (record['method'], record['round'], record['budget'], record['labeled_ids'],
                    record['unlabeled_ids'], record['L_hash'], record['U_hash']) != (
                    method, r, budget, labeled, unlabeled, ids_hash(labeled), ids_hash(unlabeled)):
                raise RuntimeError(f'historical comparator state mismatch: {method}/{r}')
            if record['test_truth_access_count'] != 0:
                raise RuntimeError('historical pre-test round accessed truth')
            fit_dir = (PARENT / 'shared/fit' if r == 0 else directory / 'fit')
            fit_record = read_json(fit_dir / 'fit.json')
            verify_files(fit_dir, fit_record['files'])
            fit_input = read_json(fit_dir / 'input.json')
            if (fit_input['config'] != parent_protocol['training'] or
                    fit_input['labeled'] != labeled or fit_input['validation'] != valid or
                    fit_record['checkpoint_hash'] != record['checkpoint_hash']):
                raise RuntimeError(f'historical fit protocol mismatch: {method}/{r}')
            with np.load(directory / 'predictions.npz') as predictions:
                if predictions['ids'].tolist() != valid + test or predictions['predictions'].shape != (494, 3):
                    raise RuntimeError('historical prediction identity or shape mismatch')
            audit['comparator_rounds'][f'{method}/{r}'] = dict(
                round_sha256=sha(directory / 'round.json'),
                checkpoint_hash=record['checkpoint_hash'],
                prediction_hash=record['prediction_hash'],
                labeled_ids_hash=record['L_hash'], unlabeled_ids_hash=record['U_hash'])
            if r == 0 and (record['checkpoint_hash'] != shared_fit['checkpoint_hash'] or
                           record['prediction_hash'] != audit['comparator_rounds']['random/0']['prediction_hash']):
                raise RuntimeError('historical shared L333 checkpoint/prediction mismatch')
            if r < 5:
                selection_dir = EXTENSION / f'runtime/{method}/round_3' if r == 3 else directory
                selection = read_json(selection_dir / 'selection.json')
                if r == 3:
                    extension_start = read_json(selection_dir / 'round.json')
                    if extension_start['labeled_ids'] != labeled or extension_start['checkpoint_hash'] != record['checkpoint_hash']:
                        raise RuntimeError('parent-to-extension boundary mismatch')
                if selection['selected'] != record['selected'] or selection['selected_hash'] != stable_hash(record['selected']):
                    if r != 3 or selection['selected'] != extension_start['selected']:
                        raise RuntimeError('historical selection mismatch')
                labeled, unlabeled = transition(labeled, unlabeled, selection['selected'])
        if r != 5 or record['selected']:
            raise RuntimeError('historical terminal selection mismatch')
    write_once(study / 'comparator_audit.json', audit)
    return partition, audit


def prepare(study=STUDY):
    study = Path(study)
    study.mkdir(parents=True, exist_ok=True)
    environment = assert_environment()
    partition, audit = audit_comparators(study)
    training = read_json(PARENT / 'protocol.json')['training']
    protocol = dict(study=study.name, seed=525, methods=list(METHODS),
                    comparators=list(COMPARATORS), budgets=list(BUDGETS), batch_size=BATCH,
                    partition_hash=stable_hash(partition), training=training,
                    frozen_l333_target_population_sd=SCALE,
                    target='RTv = RT * Speed; central predictor output[:,1]',
                    evaluation_metric='trapezoidal NRMSE integral divided by interval width',
                    ivr=dict(reference='fixed original L0 union U0', feature='full-network central RTv gradient CountSketch 512D',
                             scale='single global RMS over reference', prior_precision=1,
                             noise_variance=1, arithmetic='float64', batch='greedy Sherman-Morrison',
                             tie_break='current sorted U order', refresh='current trajectory checkpoint'),
                    hybrid=dict(ensemble_size=3, member0='current evaluation checkpoint',
                                extra_member_seed='525000000 + 10000*round_index + member (member=1,2)',
                                uncertainty='ddof=0 std of output[:,1]',
                                shortlist_fraction=.25, shortlist_size='max(32,ceil(.25*current_U_size))',
                                latent='unit-normalized h_graph', distance='1-cosine',
                                centers='all current L', tie_break='shortlist order'),
                    shared_l333_checkpoint_hash=read_json(PARENT / 'shared/fit/fit.json')['checkpoint_hash'],
                    cc_source_commit=CC_COMMIT,
                    cc_transfer='scalar row-kernel IVR; three-member epistemic Top-25% then latent k-center',
                    hplc_adaptation='single central RTv output; unit h_graph cosine; CC Hybrid is not bitwise reproduced',
                    test_firewall='all new IDs/checkpoints/validation metrics/test-X predictions frozen before one source truth read',
                    stop='hard stop after L493; seed525 development only')
    write_once(study / 'splits/partition.json', partition)
    write_once(study / 'environment.json', environment)
    write_once(study / 'protocol.json', protocol)
    return partition, protocol


def state(study, partition, method, r):
    labeled, unlabeled = role_ids(partition, 'l0'), role_ids(partition, 'u0')
    for prior in range(r):
        selection = read_json(study / f'runtime/{method}/round_{prior}/selection.json')
        if (selection['method'], selection['round'], selection['L_hash'], selection['U_hash']) != (
                method, prior, ids_hash(labeled), ids_hash(unlabeled)):
            raise RuntimeError('new trajectory lineage mismatch')
        labeled, unlabeled = transition(labeled, unlabeled, selection['selected'])
    return labeled, unlabeled


def store_at(study, partition, method, r):
    store = RestrictedLabelStore(partition, study / 'label_access_audit.csv', method)
    for prior in range(r):
        store.commit_selection(study / f'runtime/{method}/round_{prior}/selection.json')
    return store


def _files(study, directory):
    return {str(p.relative_to(study)): sha(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'round.json'}


def evaluation_stage(study, partition, graphs, method, r, valid, test, valid_truth, protocol):
    directory = study / f'runtime/{method}/round_{r}'
    directory.mkdir(parents=True, exist_ok=True)
    labeled, unlabeled = state(study, partition, method, r)
    stage_path = directory / 'evaluation.json'
    if stage_path.exists():
        stage = read_json(stage_path)
        verify_files(study, stage['files'])
        if stage['labeled_ids'] != labeled or stage['unlabeled_ids'] != unlabeled:
            raise RuntimeError('resumed evaluation state mismatch')
        return stage
    if r == 0:
        fit_dir = PARENT / 'shared/fit'
        fitted = read_json(fit_dir / 'fit.json')
        source = source_round('raw_gradient_lcmd', 0) / 'predictions.npz'
        shutil.copyfile(source, directory / 'predictions.npz')
        with np.load(source) as bank:
            if bank['ids'].tolist() != valid + test:
                raise RuntimeError('shared L333 prediction ID mismatch')
            predictions = bank['predictions'].copy()
        fit_source = 'historical_shared_L333'
    else:
        fit_dir = directory / 'fit'
        training_labels = store_at(study, partition, method, r).reveal(labeled, 'fit')
        fitted = fit(graphs, labeled, training_labels, valid, valid_truth,
                     protocol['training'], fit_dir,
                     dict(study=study.name, method=method, round=r, role='evaluation'))
        model = load_model(fit_dir / fitted['checkpoint_path'])
        predictions = predict(model, graphs, valid + test)
        _save_predictions(directory / 'predictions.npz', valid + test, predictions)
        fit_source = 'new_scratch'
    stage = dict(method=method, round=r, budget=BUDGETS[r],
                 labeled_ids=labeled, unlabeled_ids=unlabeled,
                 L_hash=ids_hash(labeled), U_hash=ids_hash(unlabeled),
                 fit_directory=str(fit_dir), fit_source=fit_source,
                 checkpoint_path=fitted['checkpoint_path'],
                 checkpoint_hash=fitted['checkpoint_hash'],
                 checkpoint_state_hash=fitted['checkpoint_state_hash'],
                 prediction_hash=sha(directory / 'predictions.npz'),
                 validation_metrics=metrics(predictions[:len(valid), 1], valid_truth, SCALE),
                 evaluation_epochs=fitted['epochs_run'], evaluation_training_seconds=fitted['seconds'],
                 test_truth_access_count=0,
                 files=_files(study, directory))
    write_once(stage_path, stage)
    return stage


def _gradient(study, graphs, outer, stage, directory):
    if stage['round'] == 0:
        directory.mkdir(parents=True, exist_ok=True)
        source = source_round('raw_gradient_lcmd', 0) / 'raw_gradient_bank/features.npz'
        with np.load(source) as bank:
            if bank['ids'].tolist() != outer:
                raise RuntimeError('historical L333 gradient order mismatch')
            raw = bank['features'].copy()
        if not (directory / 'features.npz').exists():
            shutil.copyfile(source, directory / 'features.npz')
        if sha(directory / 'features.npz') != sha(source):
            raise RuntimeError('historical L333 gradient copy drift')
        write_once(directory / 'reuse.json', dict(source_sha256=sha(source),
                   checkpoint_hash=stage['checkpoint_hash']))
        return raw, dict(feature_hash=stable_hash(raw.shape), seconds=0., reused=True)
    model = load_model(Path(stage['fit_directory']) / stage['checkpoint_path'])
    raw, audit, reused = _gradient_bank(study, model, graphs, outer,
        dict(checkpoint_hash=stage['checkpoint_hash'], checkpoint_state_hash=stage['checkpoint_state_hash']), directory)
    return raw, dict(**audit, reused=reused)


def _latent(study, graphs, outer, stage, directory):
    if stage['round'] == 0:
        source = PARENT / 'diagnostics/lcmd_0/latent/features.npz'
        with np.load(source) as bank:
            if bank['ids'].tolist() != outer:
                raise RuntimeError('historical L333 latent order mismatch')
            z = bank['unit_latent'].copy()
        if not (directory / 'features.npz').exists():
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, directory / 'features.npz')
        if sha(directory / 'features.npz') != sha(source):
            raise RuntimeError('historical L333 latent copy drift')
        write_once(directory / 'reuse.json', dict(source_sha256=sha(source),
                   checkpoint_hash=stage['checkpoint_hash']))
        return z
    model = load_model(Path(stage['fit_directory']) / stage['checkpoint_path'])
    return latent_bank(model, graphs, outer, directory, stage['checkpoint_hash'])['unit_latent']


def _ensemble(study, partition, graphs, stage, valid, valid_truth, protocol):
    directory = study / f'runtime/hybrid_ensemble_latent/round_{stage["round"]}'
    path = directory / 'ensemble_predictions.npz'
    if path.exists():
        with np.load(path) as bank:
            if bank['ids'].tolist() != stage['unlabeled_ids'] or bank['central'].shape != (3, len(stage['unlabeled_ids'])):
                raise RuntimeError('resumed ensemble prediction alignment mismatch')
            return bank['central'].copy()
    unlabeled = stage['unlabeled_ids']
    values = []
    member0 = load_model(Path(stage['fit_directory']) / stage['checkpoint_path'])
    values.append(predict(member0, graphs, unlabeled)[:, 1])
    labeled = stage['labeled_ids']
    truth = store_at(study, partition, 'hybrid_ensemble_latent', stage['round']).reveal(labeled, 'fit')
    for member in (1, 2):
        config = dict(protocol['training'], initialization_seed=member_seed(stage['round'], member),
                      training_seed=member_seed(stage['round'], member))
        fit_dir = directory / f'member_{member}/fit'
        fitted = fit(graphs, labeled, truth, valid, valid_truth, config, fit_dir,
                     dict(study=study.name, method='hybrid_ensemble_latent',
                          round=stage['round'], role=f'ensemble_member_{member}'))
        model = load_model(fit_dir / fitted['checkpoint_path'])
        values.append(predict(model, graphs, unlabeled)[:, 1])
    save_bank(path, ids=np.asarray(unlabeled), central=np.asarray(values))
    return np.asarray(values)


def selection_stage(study, partition, graphs, outer, method, stage, valid, valid_truth, protocol):
    r = stage['round']
    directory = study / f'runtime/{method}/round_{r}'
    path = directory / 'selection.json'
    if path.exists():
        selection = read_json(path)
        verify_files(study, selection['files'])
        if (selection['L_hash'], selection['U_hash']) != (stage['L_hash'], stage['U_hash']):
            raise RuntimeError('resumed selection state mismatch')
        return selection
    index = {row: pos for pos, row in enumerate(outer)}
    labeled, unlabeled = stage['labeled_ids'], stage['unlabeled_ids']
    lp = [index[i] for i in labeled]
    up = [index[i] for i in unlabeled]
    started = time.perf_counter()
    if method == 'kernel_ivr':
        raw, gradient_audit = _gradient(study, graphs, outer, stage, directory / 'gradient')
        result = ivr_select(raw, lp, up, BATCH)
        selected = [unlabeled[i] for i in result['selected_candidate_positions']]
        nearest = cdist(raw[up].astype(np.float64), raw[lp].astype(np.float64)).min(axis=1)
        norms = np.linalg.norm(raw[up].astype(np.float64), axis=1)
        percentiles = (rankdata(norms, method='average') - .5) / len(up) * 100
        geometry = dict(gradient_norm_percentile=float(percentiles[result['selected_candidate_positions']].mean()),
                        initial_ivr_score_percentile=float(result['initial_score_percentiles'][result['selected_candidate_positions']].mean()),
                        nearest_l_gradient_distance=float(nearest[result['selected_candidate_positions']].mean()),
                        ivr=result['audit'], gradient=gradient_audit)
    else:
        z = _latent(study, graphs, outer, stage, directory / 'latent')
        predictions = _ensemble(study, partition, graphs, stage, valid, valid_truth, protocol)
        result = hybrid_select(z, lp, up, predictions, SCALE, BATCH)
        selected = [unlabeled[i] for i in result['selected_candidate_positions']]
        nearest = np.clip(1 - z[up] @ z[lp].T, 0, 2).min(axis=1)
        with np.load(PARENT / 'diagnostics/morgan.npz') as bank:
            if bank['ids'].tolist() != outer:
                raise RuntimeError('Morgan diagnostic order mismatch')
            morgan = bank['distance']
            novelty = morgan[np.ix_(up, lp)].min(axis=1)
        chemistry = read_json(PARENT / 'diagnostics/chemistry.json')
        known_scaffolds = {chemistry[i]['scaffold'] for i in lp}
        chosen = result['selected_candidate_positions']
        shortlist = result['shortlist_candidate_positions']
        ranks = np.empty(len(up), dtype=np.int64)
        ranks[result['uncertainty_ranking']] = np.arange(1, len(up) + 1)
        geometry = dict(uncertainty_percentile=float(result['uncertainty_percentiles'][chosen].mean()),
                        shortlist_uncertainty_percentile=float(result['uncertainty_percentiles'][shortlist].mean()),
                        selected_uncertainty_mean=float(result['uncertainty'][chosen].mean()),
                        selected_uncertainty_median=float(np.median(result['uncertainty'][chosen])),
                        shortlist_uncertainty_mean=float(result['uncertainty'][shortlist].mean()),
                        selected_latent_nearest_l_distance=float(nearest[chosen].mean()),
                        selected_morgan_novelty=float(novelty[chosen].mean()),
                        novel_scaffold_fraction=float(np.mean([chemistry[up[i]]['scaffold'] not in known_scaffolds for i in chosen])),
                        top32_uncertainty_retained=int(np.count_nonzero(ranks[chosen] <= 32)),
                        top_shortlist_quartile_retained=int(np.count_nonzero(ranks[chosen] <= max(1, len(shortlist)//4))),
                        hybrid=result['audit'])
    transition(labeled, unlabeled, selected)
    selection = dict(method=method, round=r, budget=stage['budget'],
                     L_hash=stage['L_hash'], U_hash=stage['U_hash'],
                     selected=selected, selected_hash=stable_hash(selected),
                     trace=result['trace'], geometry=geometry,
                     selection_seconds=time.perf_counter()-started,
                     test_truth_access_count=0,
                     files=_files(study, directory))
    write_once(path, selection)
    return selection


def verify_freeze(study=STUDY):
    study = Path(study)
    frozen = read_json(study / 'global_pre_test_freeze.json')
    if frozen['status'] != 'FROZEN_BEFORE_TEST_TRUTH' or frozen['test_truth_access_count'] != 0:
        raise PermissionError('invalid pre-test freeze')
    verify_files(study, frozen['files'])
    if sha(PARENT / 'completion_manifest.json') != frozen['parent_completion_sha256'] or \
       sha(EXTENSION / 'completion_manifest.json') != frozen['extension_completion_sha256']:
        raise PermissionError('historical comparator changed after freeze')
    partition = read_json(study / 'splits/partition.json')
    if stable_hash(partition) != read_json(study / 'protocol.json')['partition_hash']:
        raise PermissionError('frozen partition mismatch')
    expected = {f'{m}/{r}' for m in METHODS for r in range(6)}
    if set(frozen['entries']) != expected:
        raise PermissionError('incomplete new grid')
    for method in METHODS:
        labeled, unlabeled = role_ids(partition, 'l0'), role_ids(partition, 'u0')
        for r in range(6):
            rec = read_json(study / frozen['entries'][f'{method}/{r}'])
            if (rec['labeled_ids'], rec['unlabeled_ids'], rec['budget'], rec['checkpoint_hash']) != (
                    labeled, unlabeled, BUDGETS[r], rec['evaluation']['checkpoint_hash']):
                raise PermissionError('frozen new trajectory mismatch')
            if r < 5:
                labeled, unlabeled = transition(labeled, unlabeled, rec['selected'])
            elif rec['selected']:
                raise PermissionError('terminal L493 selected again')
    return frozen


def run(study=STUDY):
    study = Path(study)
    if (study / 'global_pre_test_freeze.json').exists():
        return verify_freeze(study)
    partition, protocol = prepare(study)
    write_once(study / 'execution_code.json', code_hashes())
    graphs = load_graphs(partition)
    valid, test = role_ids(partition, 'validation'), role_ids(partition, 'test')
    outer = sorted(role_ids(partition, 'l0') + role_ids(partition, 'u0'))
    valid_truth = RestrictedLabelStore(partition, study / 'label_access_audit.csv', 'shared').reveal(valid, 'validation')
    entries = {}
    for r in range(6):
        for method in METHODS:
            directory = study / f'runtime/{method}/round_{r}'
            stage = evaluation_stage(study, partition, graphs, method, r, valid, test, valid_truth, protocol)
            selection = (selection_stage(study, partition, graphs, outer, method, stage, valid, valid_truth, protocol)
                         if r < 5 else None)
            round_path = directory / 'round.json'
            record = {**stage, 'evaluation': stage,
                      'selected': selection['selected'] if selection else [],
                      'selection_hash': sha(directory / 'selection.json') if selection else None,
                      'files': {**stage['files'], **(selection['files'] if selection else {}),
                                **({str((directory/'selection.json').relative_to(study)): sha(directory/'selection.json')}
                                   if selection else {})}}
            write_once(round_path, record)
            entries[f'{method}/{r}'] = str(round_path.relative_to(study))
            print(dict(stage='round_complete', method=method, budget=BUDGETS[r]), flush=True)
    with (study / 'label_access_audit.csv').open() as stream:
        access = list(csv.DictReader(stream))
    if any(row['purpose'] == 'final_test' or row['allowed'] != 'True' for row in access):
        raise PermissionError('test truth or unauthorized label access pre-freeze')
    write_once(study / 'pre_test_label_access_audit.json', access)
    files = {str(p.relative_to(study)): sha(p) for p in study.rglob('*')
             if p.is_file() and p.name != 'label_access_audit.csv' and p.suffix != '.log'}
    write_once(study / 'global_pre_test_freeze.json', dict(
        status='FROZEN_BEFORE_TEST_TRUTH', test_truth_access_count=0,
        parent_completion_sha256=sha(PARENT / 'completion_manifest.json'),
        extension_completion_sha256=sha(EXTENSION / 'completion_manifest.json'),
        entries=entries, files=files))
    return verify_freeze(study)


def reveal(study=STUDY):
    study = Path(study)
    verify_freeze(study)
    receipt = study / 'test_reveal.json'
    if receipt.exists():
        verify_files(study, read_json(receipt)['files'])
        return
    if (study / 'label_access_audit.csv').exists():
        with (study / 'label_access_audit.csv').open() as stream:
            if any(row['purpose'] == 'final_test' for row in csv.DictReader(stream)):
                raise RuntimeError('interrupted reveal; refuse second source truth read')
    partition = read_json(study / 'splits/partition.json')
    store = RestrictedLabelStore(partition, study / 'label_access_audit.csv', 'final_test')
    store.test_frozen = True
    test = role_ids(partition, 'test')
    truth = store.reveal(test, 'final_test')
    save_bank(study / 'test_truth.npz', ids=np.asarray(test), truth=truth)
    write_once(receipt, dict(status='REVEALED_AFTER_GLOBAL_FREEZE', source_reveal_count=1,
                             files={'test_truth.npz': sha(study / 'test_truth.npz')}))
