"""Resumable, bounded representation comparison with a sixteen-record firewall."""

import csv
import itertools
import shutil
from pathlib import Path

import numpy as np

from .common import (ROOT, atomic_json, code_hashes, ids_hash, metrics, read_json,
                     sha, stable_hash, verify_files, write_once)
from .coreset import cosine_distance, kcenter, morgan_bank, nested_random_order
from .data import load_graphs
from .gradient import extract
from .latent import extract_h_graph
from .protocol import RestrictedLabelStore, make_partition, role_ids, transition
from .representation_diagnostics import coverage, representation_relation, stability
from .runner import assert_environment
from .training import fit, load_model
from .transfer_v2 import BUDGETS, transfer_partition
from .transfer_v2_runner import _fit_config, _gradient_bank, _save_predictions
from .data import predict

STUDY = ROOT / 'studies/active_learning/odh_representation_coreset_v1'
HISTORY = ROOT / 'studies/active_learning/odh_gradient_al_transfer_v2'
METHODS = ('random', 'raw_gradient_lcmd', 'latent_coreset', 'morgan_coreset')
RANDOM_SEED = 525_900_001


def tree_hashes(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '.DS_Store' not in p.parts and '__pycache__' not in p.parts}


def save_bank(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f, **arrays)
    path.with_suffix('.tmp').replace(path)


def copy_fit(source, target):
    """Copy only sealed artifacts; never invoke training on historical paths."""
    record = read_json(source / 'fit.json')
    verify_files(source, record['files'])
    for name in [*record['files'], 'fit.json']:
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if sha(dest) != sha(source / name):
                raise RuntimeError('copied fit drift')
        else:
            shutil.copyfile(source / name, dest)
    return record


def latent_bank(model, graphs, ids, directory, checkpoint_hash):
    directory.mkdir(parents=True, exist_ok=True)
    write_once(directory / 'input.json', dict(ids=ids, checkpoint_hash=checkpoint_hash))
    if (directory / 'complete.json').exists():
        verify_files(directory, read_json(directory / 'complete.json')['files'])
        with np.load(directory / 'features.npz') as saved:
            bank = {k: saved[k].copy() for k in saved.files}
        if bank['ids'].tolist() != ids:
            raise RuntimeError('latent ID order drift')
        return bank
    bank, audit = extract_h_graph(model, graphs, ids)
    if audit['zero_norm_rows']:
        raise RuntimeError('zero latent cannot define cosine coverage')
    save_bank(directory / 'features.npz', **bank)
    atomic_json(directory / 'audit.json', audit)
    write_once(directory / 'complete.json', dict(files={n: sha(directory / n) for n in
                ['input.json', 'features.npz', 'audit.json']}))
    return bank


def prepare(study=STUDY):
    study = Path(study)
    study.mkdir(parents=True, exist_ok=True)
    environment = assert_environment()
    partition = transfer_partition(make_partition())
    if partition != read_json(HISTORY / 'splits/partition.json'):
        raise RuntimeError('historical split mismatch')
    write_once(study / 'splits/partition.json', partition)
    write_once(study / 'environment.json', environment)
    historical = {}
    for name in ['odh_gradient_al_smoke', 'odh_training_protocol_diagnosis',
                 'odh_gradient_al_transfer_v2']:
        historical[name] = tree_hashes(study.parent / name)
    write_once(study / 'historical_artifacts.json', historical)
    # Only the sealed pre-test artifacts are consumed, never historical test truth.
    freeze = read_json(HISTORY / 'global_pre_test_freeze.json')
    verify_files(HISTORY, freeze['files'])
    for entry in freeze['entries'].values():
        verify_files(HISTORY, read_json(HISTORY / entry['path'])['files'])
    shared = copy_fit(HISTORY / 'duration_smoke/seed_525', study / 'shared/fit')
    if read_json(study / 'shared/fit/input.json')['config'] != _fit_config():
        raise RuntimeError('shared training protocol mismatch')
    u0 = role_ids(partition, 'u0')
    write_once(study / 'random_order.json', dict(seed=RANDOM_SEED,
               ids=nested_random_order(u0, RANDOM_SEED)))
    store = RestrictedLabelStore(partition, study / 'label_access_audit.csv', 'shared')
    truth = store.reveal(role_ids(partition, 'l0'), 'fit')
    protocol = dict(study=study.name, methods=list(METHODS), budgets=list(BUDGETS),
        acquisition_batch=32, initialization_seed=525, training_seed=525,
        training=_fit_config(), random_seed=RANDOM_SEED,
        random_definition='one frozen permutation of sorted U0; disjoint slices of 32',
        latent='task-aware QGeoGNN learned structure/geometry/condition representation',
        latent_distance='1 - cosine(unit h_graph, unit h_graph)',
        latent_refresh='current method checkpoint, each acquisition round',
        morgan=dict(radius=2, nBits=2048, useChirality=True, distance='1 - Tanimoto'),
        kcenter='farthest first, existing centers=current L, ties=smallest sample ID',
        historical_coverage_latent='shared L333 geometry for every historical batch',
        stability_candidates='512 fixed IDs sampled from historical LCMD U429, seed 525',
        stability_gradient_geometry='Euclidean in raw 512D CountSketch space',
        gradient_reuse='sealed transfer_v2 raw_gradient_lcmd trajectory; exact protocol and IDs',
        metadata_fix='future fit records use training_seed; historical seeds both 525, values unchanged',
        scale=float(np.std(truth.astype(np.float64), ddof=0)),
        shared_checkpoint_hash=shared['checkpoint_hash'], partition_hash=stable_hash(partition),
        aulc='trapezoidal NRMSE integral over 333..429 divided by 96',
        firewall='all four methods and four budgets frozen before unified test truth reveal',
        stop='one seed, four methods, three acquisitions; no expansion',
        diagnostics_gate='finite aligned state-preserving nonzero latent; valid Morgan; all diagnostics complete; no outcome threshold')
    write_once(study / 'protocol.json', protocol)
    return partition


def historical_raw(round_index, ids):
    directory = (HISTORY / 'runtime/shared_round_0/raw_gradient_bank' if round_index == 0
                 else HISTORY / f'runtime/raw_gradient_lcmd/round_{round_index}/raw_gradient_bank')
    verify_files(directory, read_json(directory / 'complete.json')['files'])
    with np.load(directory / 'features.npz') as saved:
        lookup = {int(i): p for p, i in enumerate(saved['ids'])}
        return saved['features'][[lookup[i] for i in ids]].copy()


def historical_fit(round_index):
    return (HISTORY / 'duration_smoke/seed_525' if round_index == 0 else
            HISTORY / f'runtime/raw_gradient_lcmd/round_{round_index}/fit')


def diagnostics(study=STUDY):
    study = Path(study)
    if (study / 'diagnostics/gate.json').exists():
        gate = read_json(study / 'diagnostics/gate.json')
        verify_files(study, gate['files'])
        return gate
    partition = prepare(study)
    graphs = load_graphs(partition)
    outer = sorted(role_ids(partition, 'l0') + role_ids(partition, 'u0'))
    bits, morgan, chemistry, audit = morgan_bank(outer)
    save_bank(study / 'diagnostics/morgan.npz', ids=np.asarray(outer), bits=bits, distance=morgan)
    atomic_json(study / 'diagnostics/chemistry.json', chemistry)
    atomic_json(study / 'diagnostics/chirality.json', audit)
    latent = []
    for r in range(4):
        source = historical_fit(r)
        rec = read_json(source / 'fit.json')
        model = load_model(source / rec['checkpoint_path'])
        latent.append(latent_bank(model, graphs, outer, study / f'diagnostics/lcmd_{r}/latent', rec['checkpoint_hash']))
        print(dict(stage='diagnostic_latent', budget=BUDGETS[r]), flush=True)
    relation = representation_relation(latent[0]['raw_latent'], latent[0]['unit_latent'],
        chemistry, morgan, [graphs[i][0].num_nodes for i in outer])
    atomic_json(study / 'diagnostics/relation.json', relation)
    historical_coverage = {}
    for method in ['random', 'raw_gradient_lcmd', 'raw_gradient_maxdet']:
        for r in range(3):
            rec = read_json(HISTORY / f'runtime/{method}/round_{r}/round.json')
            historical_coverage[f'{method}/{r}'] = coverage(outer, rec['labeled_ids'],
                rec['unlabeled_ids'], rec['selected'], latent[0]['unit_latent'], morgan, chemistry)
    atomic_json(study / 'diagnostics/historical_coverage.json', historical_coverage)
    terminal = read_json(HISTORY / 'runtime/raw_gradient_lcmd/round_3/round.json')
    candidates = sorted(np.random.default_rng(525).choice(terminal['unlabeled_ids'], 512, replace=False).tolist())
    write_once(study / 'diagnostics/stability_candidates.json', candidates)
    pos = [outer.index(i) for i in candidates]
    gradients = [historical_raw(r, candidates) for r in range(3)]
    source = historical_fit(3)
    rec = read_json(source / 'fit.json')
    model = load_model(source / rec['checkpoint_path'])
    last, grad_audit = extract(model, graphs, candidates,
        progress=lambda x: print(dict(stage='stability_gradient_429', **x), flush=True))
    gradients.append(last)
    save_bank(study / 'diagnostics/stability_gradients.npz', ids=np.asarray(candidates),
              **{f'round_{r}': g for r, g in enumerate(gradients)})
    atomic_json(study / 'diagnostics/stability_gradient_audit.json', grad_audit)
    comparisons = {f'{BUDGETS[r]}-{BUDGETS[r+1]}': dict(
        gradient=stability(gradients[r], gradients[r+1], 'raw_gradient_euclidean'),
        latent=stability(latent[r]['unit_latent'][pos], latent[r+1]['unit_latent'][pos], 'unit_latent_cosine'))
        for r in range(3)}
    atomic_json(study / 'diagnostics/stability.json', comparisons)
    gate = dict(status='PASS', test_truth_access_count=0, files={
        str(p.relative_to(study)): sha(p) for p in sorted((study / 'diagnostics').rglob('*')) if p.is_file()})
    write_once(study / 'diagnostics/gate.json', gate)
    return gate


def verify_records(records, partition):
    """Validate the complete state transitions before granting any test access."""
    if set(records) != {f'{m}/{r}' for m in METHODS for r in range(4)}:
        raise PermissionError('all four methods x four budgets required')
    initial = []
    for method in METHODS:
        labeled, unlabeled = role_ids(partition, 'l0'), role_ids(partition, 'u0')
        for r, budget in enumerate(BUDGETS):
            rec = records[f'{method}/{r}']
            if (rec['method'], rec['round'], rec['budget'], rec['labeled_ids'], rec['unlabeled_ids']) != (method, r, budget, labeled, unlabeled):
                raise PermissionError('trajectory identity/budget drift')
            if len(labeled) != budget or rec.get('test_truth_access_count') != 0:
                raise PermissionError('budget or firewall violation')
            if r == 0:
                initial.append(rec)
            if r < 3:
                labeled, unlabeled = transition(labeled, unlabeled, rec['selected'])
            elif rec['selected']:
                raise PermissionError('terminal acquisition forbidden')
    for field in ['checkpoint_hash', 'prediction_hash', 'L_hash']:
        if len({r[field] for r in initial}) != 1:
            raise PermissionError('round-zero fairness violated: ' + field)
    if len({r['initialization_hash'] for r in records.values()}) != 1:
        raise PermissionError('scratch initialization drift')


def verify_freeze(study=STUDY):
    study = Path(study)
    freeze = read_json(study / 'global_pre_test_freeze.json')
    if freeze['status'] != 'FROZEN_BEFORE_TEST_TRUTH' or freeze['test_truth_access_count'] != 0:
        raise PermissionError('invalid freeze')
    verify_files(study, freeze['files'])
    records = {k: read_json(study / v['path']) for k, v in freeze['entries'].items()}
    partition = read_json(study / 'splits/partition.json')
    verify_records(records, partition)
    for rec in records.values():
        verify_files(study, rec['files'])
        if rec['checkpoint_hash'] not in rec['files'].values():
            raise PermissionError('unbound checkpoint')
        prefix = f"runtime/{rec['method']}/round_{rec['round']}"
        if rec['files'].get(prefix + '/predictions.npz') != rec['prediction_hash']:
            raise PermissionError('unbound predictions')
        if rec['round'] < 3:
            sel = read_json(study / prefix / 'selection.json')
            if (sel['selected'], sel['L_hash'], sel['U_hash']) != (rec['selected'], ids_hash(rec['labeled_ids']), ids_hash(rec['unlabeled_ids'])):
                raise PermissionError('selection drift')
    for row in read_json(study / 'pre_test_label_access_audit.json'):
        if row['purpose'] == 'final_test' or row['allowed'] != 'True':
            raise PermissionError('invalid pre-test label access')
        ids = set(map(int, row['ids'].split(';')))
        if row['purpose'] == 'validation':
            allowed = set(role_ids(partition, 'validation'))
        elif row['purpose'] == 'fit':
            allowed = set(role_ids(partition, 'l0')) if row['method'] == 'shared' else set(records[f"{row['method']}/{row['round']}"]['labeled_ids'])
        else:
            raise PermissionError('unknown label purpose')
        if not ids <= allowed:
            raise PermissionError('unauthorized training labels')
    return records


def run(study=STUDY):
    study = Path(study)
    assert_environment()
    if (study / 'global_pre_test_freeze.json').exists():
        return verify_freeze(study)
    gate = read_json(study / 'diagnostics/gate.json')
    if gate['status'] != 'PASS':
        raise RuntimeError('diagnostics must pass before AL')
    verify_files(study, gate['files'])
    write_once(study / 'execution_code.json', code_hashes())
    partition = read_json(study / 'splits/partition.json')
    protocol = read_json(study / 'protocol.json')
    graphs = load_graphs(partition)
    valid, test = role_ids(partition, 'validation'), role_ids(partition, 'test')
    l0, u0 = role_ids(partition, 'l0'), role_ids(partition, 'u0')
    outer = sorted(l0 + u0)
    chemistry = read_json(study / 'diagnostics/chemistry.json')
    with np.load(study / 'diagnostics/morgan.npz') as bank:
        if bank['ids'].tolist() != outer:
            raise RuntimeError('Morgan ID mismatch')
        morgan = bank['distance'].copy()
    order = read_json(study / 'random_order.json')['ids']
    valid_truth = RestrictedLabelStore(partition, study / 'label_access_audit.csv', 'shared').reveal(valid, 'validation')
    records = {}
    for method in METHODS:
        store = RestrictedLabelStore(partition, study / 'label_access_audit.csv', method)
        labeled, unlabeled = l0[:], u0[:]
        for r, budget in enumerate(BUDGETS):
            directory = study / f'runtime/{method}/round_{r}'
            directory.mkdir(parents=True, exist_ok=True)
            record_path = directory / 'round.json'
            if record_path.exists():
                rec = read_json(record_path)
                verify_files(study, rec['files'])
                if rec['labeled_ids'] != labeled or rec['unlabeled_ids'] != unlabeled:
                    raise RuntimeError('resumed state drift')
            else:
                fit_dir = study / 'shared/fit' if r == 0 else directory / 'fit'
                reused = r == 0 or method == 'raw_gradient_lcmd'
                if r == 0:
                    fitted = read_json(fit_dir / 'fit.json')
                elif method == 'raw_gradient_lcmd':
                    old = read_json(HISTORY / f'runtime/{method}/round_{r}/round.json')
                    if old['labeled_ids'] != labeled or old['unlabeled_ids'] != unlabeled:
                        raise RuntimeError('historical reuse requires identical states')
                    fitted = copy_fit(historical_fit(r), fit_dir)
                    inp = read_json(fit_dir / 'input.json')
                    if inp['config'] != protocol['training'] or inp['labeled'] != labeled or inp['validation'] != valid:
                        raise RuntimeError('historical fit contract mismatch')
                else:
                    truth = store.reveal(labeled, 'fit')
                    fitted = fit(graphs, labeled, truth, valid, valid_truth, protocol['training'],
                                 fit_dir, dict(study=study.name, method=method, round=r))
                verify_files(fit_dir, fitted['files'])
                model = load_model(fit_dir / fitted['checkpoint_path'])
                # All methods use the exact same serialized L333 predictions.
                prediction_path = directory / 'predictions.npz'
                if r == 0:
                    source = HISTORY / 'runtime/raw_gradient_lcmd/round_0/predictions.npz'
                    shutil.copyfile(source, prediction_path)
                    with np.load(source) as saved:
                        if saved['ids'].tolist() != valid + test:
                            raise RuntimeError('shared prediction identities changed')
                        predictions = saved['predictions'].copy()
                else:
                    predictions = predict(model, graphs, valid + test)
                    _save_predictions(prediction_path, valid + test, predictions)
                selected = []
                if r < 3:
                    bank = latent_bank(model, graphs, outer, directory / 'latent', fitted['checkpoint_hash'])
                    if r == 0 or method == 'raw_gradient_lcmd':
                        raw = historical_raw(r, outer)
                        save_bank(directory / 'raw_gradient_bank/features.npz', ids=np.asarray(outer), features=raw)
                        atomic_json(directory / 'raw_gradient_bank/reuse.json', dict(source=str(HISTORY), round=r,
                                    checkpoint_hash=fitted['checkpoint_hash']))
                    else:
                        raw, _, _ = _gradient_bank(study, model, graphs, outer, fitted, directory / 'raw_gradient_bank')
                    if method == 'random':
                        selected, trace = order[r*32:(r+1)*32], []
                    elif method == 'raw_gradient_lcmd':
                        old = read_json(HISTORY / f'runtime/{method}/round_{r}/selection.json')
                        selected, trace = old['selected'], old['trace']
                    else:
                        distance = cosine_distance(bank['unit_latent']) if method == 'latent_coreset' else morgan
                        selected, trace = kcenter(distance, outer, labeled, unlabeled)
                    transition(labeled, unlabeled, selected)
                    write_once(directory / 'selection.json', dict(method=method, round=r,
                        L_hash=ids_hash(labeled), U_hash=ids_hash(unlabeled), selected=selected,
                        selected_hash=stable_hash(selected), trace=trace))
                    atomic_json(directory / 'coverage.json', coverage(outer, labeled, unlabeled,
                        selected, bank['unit_latent'], morgan, chemistry, raw))
                files = {str(p.relative_to(study)): sha(p) for p in directory.rglob('*') if p.is_file()}
                files.update({str((fit_dir / n).relative_to(study)): sha(fit_dir / n) for n in [*fitted['files'], 'fit.json']})
                rec = dict(method=method, round=r, budget=budget, labeled_ids=labeled, unlabeled_ids=unlabeled,
                    L_hash=ids_hash(labeled), U_hash=ids_hash(unlabeled), selected=selected,
                    checkpoint_hash=fitted['checkpoint_hash'], initialization_hash=fitted['initialization_hash'],
                    prediction_hash=sha(prediction_path), fit_reused=reused, training_seconds=fitted['seconds'],
                    epochs_run=fitted['epochs_run'], best_epoch=fitted['best_epoch'], test_truth_access_count=0,
                    validation_metrics=metrics(predictions[:len(valid), 1], valid_truth, protocol['scale']), files=files)
                write_once(record_path, rec)
            records[f'{method}/{r}'] = dict(path=str(record_path.relative_to(study)))
            if r < 3:
                store.commit_selection(directory / 'selection.json')
                labeled, unlabeled = transition(labeled, unlabeled, rec['selected'])
            print(dict(stage='round_frozen', method=method, budget=budget), flush=True)
    overlaps = {}
    for r in range(3):
        for a, b in itertools.combinations(METHODS, 2):
            x = set(read_json(study / records[f'{a}/{r}']['path'])['selected'])
            y = set(read_json(study / records[f'{b}/{r}']['path'])['selected'])
            overlaps[f'{a} vs {b}/{r}'] = dict(ids=sorted(x & y), count=len(x & y), jaccard=len(x & y)/len(x | y))
    atomic_json(study / 'acquisition_overlap.json', overlaps)
    with (study / 'label_access_audit.csv').open() as f:
        accesses = list(csv.DictReader(f))
    atomic_json(study / 'pre_test_label_access_audit.json', accesses)
    files = {str(p.relative_to(study)): sha(p) for p in study.rglob('*')
             if p.is_file() and p.name != 'label_access_audit.csv' and p.suffix != '.log'}
    write_once(study / 'global_pre_test_freeze.json', dict(status='FROZEN_BEFORE_TEST_TRUTH',
               test_truth_access_count=0, entries=records, files=files))
    return verify_freeze(study)


def reveal(study=STUDY):
    study = Path(study)
    records = verify_freeze(study)
    if (study / 'test_reveal.json').exists():
        verify_files(study, read_json(study / 'test_reveal.json')['files'])
        return records
    partition = read_json(study / 'splits/partition.json')
    store = RestrictedLabelStore(partition, study / 'label_access_audit.csv', 'final_test')
    store.test_frozen = True
    ids = role_ids(partition, 'test')
    truth = store.reveal(ids, 'final_test')
    save_bank(study / 'test_truth.npz', ids=np.asarray(ids), truth=truth)
    write_once(study / 'test_reveal.json', dict(status='REVEALED_AFTER_GLOBAL_FREEZE',
               test_truth_access_count=1, files={'test_truth.npz': sha(study / 'test_truth.npz')}))
    return records
