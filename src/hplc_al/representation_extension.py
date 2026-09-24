"""Prospective, validation-gated continuation of the frozen four-method study."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np

from .acquisition import lcmd_tp_select
from .common import ROOT, atomic_json, code_hashes, ids_hash, metrics, read_json, sha, stable_hash, verify_files, write_once
from .coreset import cosine_distance, kcenter
from .data import load_graphs, predict
from .protocol import RestrictedLabelStore, role_ids, transition
from .representation_diagnostics import coverage
from .representation_runner import (METHODS, STUDY as PARENT, _gradient_bank,
                                    _save_predictions, latent_bank, save_bank,
                                    verify_freeze as verify_parent_freeze)
from .runner import assert_environment
from .training import fit, load_model

STUDY = ROOT / 'studies/active_learning/odh_representation_coreset_extension_v1'
START_ROUND = 3
START_BUDGET = 429
STEP = 32
MAX_EXTRA_ROUNDS = 8
SIGNAL_THRESHOLD = 0.01
PAIR_SIZE = 2


def positive_signal(rows, threshold=SIGNAL_THRESHOLD):
    """A coreset must beat LCMD on validation NRMSE by at least threshold."""
    if len(rows) != PAIR_SIZE:
        raise ValueError('exactly two new budget points required')
    details = []
    for budget, values in rows:
        if set(values) != set(METHODS) or not all(np.isfinite(v) for v in values.values()):
            raise ValueError('complete finite validation metrics required')
        gains = {m: float(values['raw_gradient_lcmd'] - values[m])
                 for m in ('latent_coreset', 'morgan_coreset')}
        details.append(dict(budget=int(budget), nrmse=values, gains_vs_lcmd=gains,
                            positive_methods=[m for m, gain in gains.items() if gain >= threshold]))
    return any(row['positive_methods'] for row in details), details


def _source_round(method):
    return read_json(PARENT / f'runtime/{method}/round_{START_ROUND}/round.json')


def _source_fit(method):
    return PARENT / f'runtime/{method}/round_{START_ROUND}/fit'


def prepare(study=STUDY):
    study = Path(study)
    study.mkdir(parents=True, exist_ok=True)
    assert_environment()
    verify_parent_freeze(PARENT)
    completion = read_json(PARENT / 'completion_manifest.json')
    verify_files(PARENT, completion['files'])
    parent_protocol = read_json(PARENT / 'protocol.json')
    partition = read_json(PARENT / 'splits/partition.json')
    if parent_protocol['methods'] != list(METHODS) or parent_protocol['budgets'] != [333, 365, 397, 429]:
        raise RuntimeError('frozen parent protocol differs')
    for method in METHODS:
        old = _source_round(method)
        if old['budget'] != START_BUDGET or old['selected'] or len(old['labeled_ids']) != START_BUDGET:
            raise RuntimeError('parent terminal state drift')
    write_once(study / 'splits/partition.json', partition)
    source = dict(parent_study=PARENT.name, completion_hash=sha(PARENT / 'completion_manifest.json'),
                  protocol_hash=sha(PARENT / 'protocol.json'), terminal_round_hashes={
                  m: sha(PARENT / f'runtime/{m}/round_{START_ROUND}/round.json') for m in METHODS})
    write_once(study / 'source.json', source)
    for old, new in [(PARENT / 'diagnostics/morgan.npz', study / 'representation/morgan.npz'),
                     (PARENT / 'diagnostics/chemistry.json', study / 'representation/chemistry.json'),
                     (PARENT / 'random_order.json', study / 'representation/random_order.json')]:
        new.parent.mkdir(parents=True, exist_ok=True)
        if new.exists():
            if sha(new) != sha(old):
                raise RuntimeError('copied representation changed')
        else:
            shutil.copyfile(old, new)
    protocol = dict(study=study.name, source=source, methods=list(METHODS),
        seed=525, training=parent_protocol['training'], partition_hash=stable_hash(partition),
        first_budget=START_BUDGET, batch_size=STEP, first_new_budgets=[461, 493],
        signal=dict(metric='validation NRMSE with frozen L333 SD',
                    definition='at either of the latest two NEW budgets, latent_coreset or morgan_coreset improves on raw_gradient_lcmd by >= 0.01 absolute NRMSE',
                    threshold=SIGNAL_THRESHOLD, window=PAIR_SIZE,
                    check_every='after two newly completed budgets for all four methods'),
        continuation='if positive, run the next pair of budgets; otherwise stop',
        maximum_extra_rounds=MAX_EXTRA_ROUNDS, maximum_budget=START_BUDGET + MAX_EXTRA_ROUNDS * STEP,
        cap_policy='stop at 685 even if signal persists; report cap explicitly',
        random_order='continue frozen U0 permutation slices 96:128, 128:160, ...',
        training_continuation='scratch re-training after each acquisition, seed 525, same parent protocol',
        representation_refresh='current method checkpoint at every acquisition; Morgan fixed',
        test_firewall='new test-X predictions only until all extensions and stop decisions are frozen; one common test truth reveal afterward',
        evidence_limit='previous study already revealed this test set; extension stopping uses validation only')
    write_once(study / 'protocol.json', protocol)
    return protocol


def _state(study, partition, method, round_index):
    labeled, unlabeled = role_ids(partition, 'l0'), role_ids(partition, 'u0')
    for r in range(round_index):
        path = (PARENT if r < START_ROUND else study) / f'runtime/{method}/round_{r}/selection.json'
        selection = read_json(path)
        if (selection['method'], selection['round'], selection['L_hash'], selection['U_hash']) != (
                method, r, ids_hash(labeled), ids_hash(unlabeled)):
            raise RuntimeError('selection lineage drift')
        labeled, unlabeled = transition(labeled, unlabeled, selection['selected'])
    if len(labeled) != 333 + STEP * round_index:
        raise RuntimeError('budget lineage drift')
    return labeled, unlabeled


def _store(study, partition, method, round_index):
    store = RestrictedLabelStore(partition, study / 'label_access_audit.csv', method)
    for r in range(round_index):
        path = (PARENT if r < START_ROUND else study) / f'runtime/{method}/round_{r}/selection.json'
        store.commit_selection(path)
    if store.round != round_index:
        raise RuntimeError('label boundary state drift')
    return store


def _data(study):
    partition = read_json(study / 'splits/partition.json')
    outer = sorted(role_ids(partition, 'l0') + role_ids(partition, 'u0'))
    with np.load(study / 'representation/morgan.npz') as bank:
        if bank['ids'].tolist() != outer:
            raise RuntimeError('Morgan row alignment drift')
        morgan = bank['distance'].copy()
    chemistry = read_json(study / 'representation/chemistry.json')
    if [r['id'] for r in chemistry] != outer:
        raise RuntimeError('chemistry row alignment drift')
    order = read_json(study / 'representation/random_order.json')['ids']
    if len(order) != len(role_ids(partition, 'u0')) or set(order) != set(role_ids(partition, 'u0')):
        raise RuntimeError('random order drift')
    return partition, outer, morgan, chemistry, order


def fit_stage(study, partition, graphs, method, r, valid, test, valid_truth, protocol):
    directory = study / f'runtime/{method}/round_{r}'
    directory.mkdir(parents=True, exist_ok=True)
    stage_path = directory / 'fit_stage.json'
    labeled, unlabeled = _state(study, partition, method, r)
    if stage_path.exists():
        stage = read_json(stage_path)
        verify_files(study, stage['files'])
        if stage['labeled_ids'] != labeled or stage['unlabeled_ids'] != unlabeled:
            raise RuntimeError('resumed fit state drift')
        return stage
    if r == START_ROUND:
        source = _source_round(method)
        if source['labeled_ids'] != labeled or source['unlabeled_ids'] != unlabeled:
            raise RuntimeError('extension start differs from parent')
        fit_dir = _source_fit(method)
        fitted = read_json(fit_dir / 'fit.json')
        verify_files(fit_dir, fitted['files'])
        inp = read_json(fit_dir / 'input.json')
        if inp['config'] != protocol['training'] or inp['labeled'] != labeled or inp['validation'] != valid:
            raise RuntimeError('source fit protocol drift')
        source_pred = PARENT / f'runtime/{method}/round_{r}/predictions.npz'
        with np.load(source_pred) as source_bank:
            if source_bank['ids'].tolist() != valid + test:
                raise RuntimeError('source prediction IDs differ')
            predictions = source_bank['predictions'].copy()
        if source['checkpoint_hash'] != fitted['checkpoint_hash'] or source['prediction_hash'] != sha(source_pred):
            raise RuntimeError('source checkpoint/prediction drift')
        if not (directory / 'predictions.npz').exists():
            shutil.copyfile(source_pred, directory / 'predictions.npz')
        if sha(directory / 'predictions.npz') != sha(source_pred):
            raise RuntimeError('copied predictions changed')
    else:
        store = _store(study, partition, method, r)
        truth = store.reveal(labeled, 'fit')
        fit_dir = directory / 'fit'
        fitted = fit(graphs, labeled, truth, valid, valid_truth, protocol['training'], fit_dir,
                     dict(study=study.name, method=method, round=r,
                          source_completion_hash=protocol['source']['completion_hash']))
        model = load_model(fit_dir / fitted['checkpoint_path'])
        predictions = predict(model, graphs, valid + test)
        _save_predictions(directory / 'predictions.npz', valid + test, predictions)
    scale = read_json(PARENT / 'protocol.json')['scale']
    stage = dict(method=method, round=r, budget=333 + STEP*r,
        labeled_ids=labeled, unlabeled_ids=unlabeled, L_hash=ids_hash(labeled), U_hash=ids_hash(unlabeled),
        fit_source='parent' if r == START_ROUND else 'extension',
        fit_directory=str(fit_dir), checkpoint_path=fitted['checkpoint_path'],
        checkpoint_hash=fitted['checkpoint_hash'], initialization_hash=fitted['initialization_hash'],
        prediction_hash=sha(directory / 'predictions.npz'),
        validation_metrics=metrics(predictions[:len(valid), 1], valid_truth, scale),
        epochs_run=fitted['epochs_run'], best_epoch=fitted['best_epoch'],
        test_truth_access_count=0,
        files={str(p.relative_to(study)): sha(p) for p in directory.rglob('*') if p.is_file()})
    if r > START_ROUND:
        stage['files'].update({str((fit_dir / name).relative_to(study)): sha(fit_dir / name)
                               for name in [*fitted['files'], 'fit.json']})
    write_once(stage_path, stage)
    return stage


def select_stage(study, partition, graphs, outer, morgan, chemistry, order, method, r, stage):
    directory = study / f'runtime/{method}/round_{r}'
    path = directory / 'selection.json'
    labeled, unlabeled = stage['labeled_ids'], stage['unlabeled_ids']
    if path.exists():
        result = read_json(path)
        if (result['method'], result['round'], result['L_hash'], result['U_hash']) != (
                method, r, ids_hash(labeled), ids_hash(unlabeled)):
            raise RuntimeError('resumed acquisition state drift')
        verify_files(study, result['files'])
        return result
    model = load_model(Path(stage['fit_directory']) / stage['checkpoint_path'])
    latent = latent_bank(model, graphs, outer, directory / 'latent', stage['checkpoint_hash'])
    raw, gradient_audit, _ = _gradient_bank(study, model, graphs, outer,
        dict(checkpoint_hash=stage['checkpoint_hash'], checkpoint_state_hash=read_json(
            Path(stage['fit_directory']) / 'fit.json')['checkpoint_state_hash']),
        directory / 'raw_gradient_bank')
    index = {i: p for p, i in enumerate(outer)}
    if method == 'random':
        selected, trace = order[r*STEP:(r+1)*STEP], []
    elif method == 'raw_gradient_lcmd':
        pool = [index[i] for i in unlabeled]
        centers = [index[i] for i in labeled]
        result = lcmd_tp_select(raw[pool], raw[centers], STEP)
        selected = [unlabeled[int(i)] for i in result.selected_pool_positions]
        trace = list(result.trace)
    else:
        distance = cosine_distance(latent['unit_latent']) if method == 'latent_coreset' else morgan
        selected, trace = kcenter(distance, outer, labeled, unlabeled, STEP)
    transition(labeled, unlabeled, selected)
    atomic_json(directory / 'coverage.json', coverage(outer, labeled, unlabeled, selected,
                 latent['unit_latent'], morgan, chemistry, raw))
    files = {str(p.relative_to(study)): sha(p) for p in directory.rglob('*') if p.is_file() and p != path}
    record = dict(method=method, round=r, budget=stage['budget'], L_hash=ids_hash(labeled),
        U_hash=ids_hash(unlabeled), selected=selected, selected_hash=stable_hash(selected),
        trace=trace, latent_state_hash=read_json(directory / 'latent/audit.json')['state_after'],
        gradient_feature_hash=gradient_audit['feature_hash'],
        checkpoint_hash=stage['checkpoint_hash'], files=files)
    write_once(path, record)
    return record


def _commit_round(study, method, r, stage, selection=None):
    directory = study / f'runtime/{method}/round_{r}'
    record = {**stage, 'selected': selection['selected'] if selection else [],
              'selection_hash': sha(directory / 'selection.json') if selection else None,
              'files': {**stage['files'], **(selection['files'] if selection else {}),
                        **({str((directory / 'selection.json').relative_to(study)): sha(directory / 'selection.json')}
                           if selection else {})}}
    write_once(directory / 'round.json', record)
    return record


def _decision(study, budget, staged):
    path = study / f'decisions/after_{budget}.json'
    rows = []
    for b in (budget - STEP, budget):
        r = (b - 333) // STEP
        rows.append((b, {m: staged[(m, r)]['validation_metrics']['nrmse'] for m in METHODS}))
    positive, details = positive_signal(rows)
    record = dict(checked_after_budget=budget, positive=positive, details=details,
                  threshold=SIGNAL_THRESHOLD, decision='continue_next_two' if positive else 'stop_no_signal',
                  test_truth_access_count=0)
    write_once(path, record)
    return record


def _freeze(study, last_round, stop_reason):
    entries = {}
    for method in METHODS:
        for r in range(START_ROUND, last_round+1):
            path = study / f'runtime/{method}/round_{r}/round.json'
            if not path.is_file():
                raise PermissionError('incomplete method/budget grid')
            record = read_json(path)
            verify_files(study, record['files'])
            if (record['method'], record['round'], record['budget']) != (method,r,333+STEP*r):
                raise PermissionError('round identity drift')
            if (r == last_round) != (not record['selected']):
                raise PermissionError('terminal/continuing selection mismatch')
            entries[f'{method}/{r}'] = dict(path=str(path.relative_to(study)), hash=sha(path))
    for r in range(START_ROUND, last_round):
        for method in METHODS:
            current = read_json(study / entries[f'{method}/{r}']['path'])
            following = read_json(study / entries[f'{method}/{r+1}']['path'])
            l,u = transition(current['labeled_ids'], current['unlabeled_ids'], current['selected'])
            if (l,u) != (following['labeled_ids'], following['unlabeled_ids']):
                raise PermissionError('trajectory transition drift')
    for r in range(START_ROUND, last_round):
        random = read_json(study / entries[f'random/{r}']['path'])['selected']
        order = read_json(study / 'representation/random_order.json')['ids']
        if random != order[r*STEP:(r+1)*STEP]:
            raise PermissionError('nested random order drift')
    decision_budgets = list(range(493, 333+STEP*last_round+1, 64))
    decisions = [read_json(study / f'decisions/after_{b}.json') for b in decision_budgets]
    if not decisions or any(not d['positive'] for d in decisions[:-1]):
        raise PermissionError('stop decision history drift')
    if stop_reason == 'no_signal' and decisions[-1]['positive']:
        raise PermissionError('stopped while signal positive')
    if stop_reason == 'cap' and (last_round != START_ROUND+MAX_EXTRA_ROUNDS or not decisions[-1]['positive']):
        raise PermissionError('cap policy drift')
    with (study / 'label_access_audit.csv').open() as f:
        access = list(csv.DictReader(f))
    if any(row['purpose'] == 'final_test' or row['allowed'] != 'True' for row in access):
        raise PermissionError('test or unauthorized label access before freeze')
    atomic_json(study / 'pre_test_label_access_audit.json', access)
    files = {str(p.relative_to(study)): sha(p) for p in study.rglob('*')
             if p.is_file() and p.name != 'label_access_audit.csv' and p.suffix != '.log'}
    write_once(study / 'global_pre_test_freeze.json',
        dict(status='FROZEN_BEFORE_TEST_TRUTH', entries=entries, files=files,
             last_round=last_round, stop_budget=333+STEP*last_round,
             stop_reason=stop_reason, test_truth_access_count=0))
    return verify_freeze(study)


def verify_freeze(study=STUDY):
    study = Path(study)
    freeze = read_json(study / 'global_pre_test_freeze.json')
    if freeze['status'] != 'FROZEN_BEFORE_TEST_TRUTH' or freeze['test_truth_access_count'] != 0:
        raise PermissionError('invalid extension freeze')
    verify_files(study, freeze['files'])
    if read_json(study / 'source.json')['completion_hash'] != sha(PARENT / 'completion_manifest.json'):
        raise PermissionError('parent completion changed')
    records = {}
    for key, entry in freeze['entries'].items():
        path = study / entry['path']
        if sha(path) != entry['hash']:
            raise PermissionError('round hash drift')
        rec = read_json(path)
        verify_files(study, rec['files'])
        records[key] = rec
    expected = {f'{m}/{r}' for m in METHODS for r in range(START_ROUND,freeze['last_round']+1)}
    if set(records) != expected:
        raise PermissionError('incomplete frozen extension grid')
    return freeze, records


def run(study=STUDY):
    study = Path(study)
    if (study / 'global_pre_test_freeze.json').exists():
        return verify_freeze(study)
    protocol = prepare(study)
    write_once(study / 'execution_code.json', code_hashes())
    partition, outer, morgan, chemistry, order = _data(study)
    graphs = load_graphs(partition)
    valid, test = role_ids(partition,'validation'),role_ids(partition,'test')
    valid_truth = RestrictedLabelStore(partition,study/'label_access_audit.csv','shared').reveal(valid,'validation')
    staged = {}
    last_round = START_ROUND + MAX_EXTRA_ROUNDS
    for r in range(START_ROUND,last_round+1):
        for method in METHODS:
            staged[(method,r)] = fit_stage(study,partition,graphs,method,r,valid,test,valid_truth,protocol)
        if r > START_ROUND and (r-START_ROUND) % PAIR_SIZE == 0:
            decision = _decision(study,333+STEP*r,staged)
            print(dict(stage='validation_gate',budget=333+STEP*r,positive=decision['positive'],
                       gains=[d['gains_vs_lcmd'] for d in decision['details']]),flush=True)
            if not decision['positive']:
                for method in METHODS:
                    _commit_round(study,method,r,staged[(method,r)])
                return _freeze(study,r,'no_signal')
        if r == last_round:
            for method in METHODS:
                _commit_round(study,method,r,staged[(method,r)])
            return _freeze(study,r,'cap')
        for method in METHODS:
            selection = select_stage(study,partition,graphs,outer,morgan,chemistry,order,
                                     method,r,staged[(method,r)])
            _commit_round(study,method,r,staged[(method,r)],selection)
            print(dict(stage='round_frozen',method=method,budget=333+STEP*r),flush=True)


def reveal(study=STUDY):
    study = Path(study)
    freeze, records = verify_freeze(study)
    if (study / 'test_reveal.json').exists():
        verify_files(study,read_json(study/'test_reveal.json')['files'])
        return freeze,records
    partition = read_json(study/'splits/partition.json')
    store = RestrictedLabelStore(partition,study/'label_access_audit.csv','final_test')
    store.test_frozen = True
    ids = role_ids(partition,'test')
    truth = store.reveal(ids,'final_test')
    save_bank(study/'test_truth.npz',ids=np.asarray(ids),truth=truth)
    write_once(study/'test_reveal.json',dict(status='REVEALED_AFTER_GLOBAL_FREEZE',
        test_truth_access_count=1,files={'test_truth.npz':sha(study/'test_truth.npz')}))
    return freeze,records
